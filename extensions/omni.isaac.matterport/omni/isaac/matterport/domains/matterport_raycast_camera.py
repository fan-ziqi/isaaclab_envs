from omni.isaac.orbit.sensors.ray_caster import RayCasterCamera
from omni.isaac.orbit.sensors.ray_caster.ray_caster_cfg import RayCasterCfg

import os
import trimesh
import warp as wp
import numpy as np
import pandas as pd
import torch
from typing import Sequence

import carb
import omni.isaac.orbit.utils.math as math_utils
from omni.isaac.orbit.utils.warp import raycast_mesh

from omni.isaac.matterport.scripts import DATA_DIR


class MatterportRayCasterCamera(RayCasterCamera):
    
    UNSUPPORTED_TYPES={
        "rgb",
        "instance_id_segmentation",
        "instance_segmentation",
        "skeleton_data",
        "motion_vectors",
        "bounding_box_2d_tight",
        "bounding_box_2d_loose",
        "bounding_box_3d",
    }
        
    def __init__(self, cfg: RayCasterCfg):
        super().__init__(cfg)

        # matterport specific parameters
        self.trimesh: trimesh.Trimesh = None
        self.face_id_category_mapping: torch.Tensor = None

        # load categort id to class mapping (name and id of mpcat40 redcued class set)
        # More Information: https://github.com/niessner/Matterport/blob/master/data_organization.md#house_segmentations
        mapping = pd.read_csv(DATA_DIR + "/mappings/category_mapping.tsv", sep="\t")
        self.mapping_mpcat40 = torch.tensor(mapping["mpcat40index"].to_numpy(), device=self._device)
    
    def _color_mapping(self):
        # load defined colors for mpcat40
        mapping_40 = pd.read_csv(DATA_DIR + "/mappings/mpcat40.tsv", sep="\t")
        color = mapping_40["hex"].to_numpy()
        self.color = torch.tensor(
            [(int(color[i][1:3], 16), int(color[i][3:5], 16), int(color[i][5:7], 16)) for i in range(len(color))],
            device=self._device,
        )

    def _initialize_warp_meshes(self):
        # check if mesh is already loaded
        if self.cfg.mesh_prim_path in self.meshes:
            return
        
        # find ply
        if os.path.isabs(self.cfg.mesh_prim_path):
            file_path = self.cfg.mesh_prim_path
            assert os.path.isfile(self.cfg.mesh_prim_path), f"No .ply file found under absolute path: {self.cfg.mesh_prim_path}"
        else:
            file_path = os.path.join(DATA_DIR, self.cfg.mesh_prim_path)
            assert os.path.isfile(file_path), f"No .ply file found under relative path to extension data: {file_path}"

        # load ply
        self.trimesh = trimesh.load(file_path)

        # create mapping from face id to semantic categroy id
        # get raw face information
        faces_raw = self.trimesh.metadata["_ply_raw"]["face"]["data"]
        carb.log_info(f"Raw face information of type {faces_raw.dtype}")
        # get face categories
        self.face_id_category_mapping = torch.tensor([single_face[3] for single_face in faces_raw], device=self._device)

        # Convert trimesh into wp mesh
        mesh_wp = wp.Mesh(
            points=wp.array(self.trimesh.vertices.astype(np.float32), dtype=wp.vec3, device=self._device),
            indices=wp.array(self.trimesh.faces.astype(np.int32).flatten(), dtype=int, device=self._device),
        )
        # save mesh
        self.meshes[self.cfg.mesh_prim_path] = {"mesh": mesh_wp, "id": mesh_wp.id, "device": mesh_wp.device}

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Fills the buffers of the sensor data."""
        # check camera prim exists
        if not self._is_initialized:
            raise RuntimeError("Camera is not initialized. Please call 'sim.play()' first.")
        # Increment frame count
        self._frame[env_ids] += 1
        # update poses
        pos_w, quat_w = self._update_poses(env_ids)
        # full orientation is considered
        ray_starts_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
        ray_starts_w += pos_w.unsqueeze(1)
        ray_directions_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])
        # ray cast and store the hits
        # TODO: Make this work for multiple meshes?
        self._ray_hits_w, ray_depth, ray_normal, ray_face_ids = raycast_mesh(
            ray_starts_w,
            ray_directions_w,
            mesh_id=self.meshes[self.cfg.mesh_prim_paths[0]]["id"],
            mesh_device=self.meshes[self.cfg.mesh_prim_paths[0]]["device"],
            max_depth=self.cfg.max_distance,
            return_distance=True,
            return_normal=True,
            return_face_id=True,
        )

        # update buffers
        if "distance_to_image_plane" in self._data.output.keys():  # noqa: SIM118
            distance_to_image_plane = torch.abs(
                math_utils.quat_apply(
                    math_utils.quat_inv(quat_w).repeat(1, self.num_rays),
                    (ray_depth[:, :, None] * ray_directions_w - self._pixel_offset),
                )[:, :, 0]
            )
            self._data.output["distance_to_image_plane"][env_ids] = distance_to_image_plane.view(
                self._view.count, self.cfg.pattern_cfg.width, self.cfg.pattern_cfg.height
            ).permute(0, 2, 1)
        if "distance_to_camera" in self._data.output.keys():  # noqa: SIM118
            self._data.output["distance_to_camera"][env_ids] = ray_depth.view(
                self._view.count, self.cfg.pattern_cfg.width, self.cfg.pattern_cfg.height
            ).permute(0, 2, 1)
        if "normals" in self._data.output.keys():  # noqa: SIM118
            # to comply with the replicator annotator format, a forth channel exists but it is unused
            self._data.output["normals"][env_ids, :, :, :3] = ray_normal.view(
                self._view.count, self.cfg.pattern_cfg.width, self.cfg.pattern_cfg.height, 3
            ).permute(0, 2, 1, 3)
            self._data.output["normals"][env_ids, :, :, 3] = 1.0
        if "semantic_segementation" in self._data.output.keys():  # noqa: SIM118
            # get the category index of the hit faces (category index from unreduced set = ~1600 classes)
            face_id = self.face_id_category_mapping[ray_face_ids]
            face_id = face_id.astype(int)

            # map category index to reduced set
            face_id_mpcat40 = self.mapping_mpcat40[face_id - 1]

            # get the color of the face
            face_color = self.color[face_id_mpcat40]

            # reshape and transpose to get the correct orientation
            self._data.output["semantic_segementation"][env_ids] = face_color.reshape(self.cfg.pattern_cfg.width, self.cfg.pattern_cfg.height, 3).permute(0, 2, 1, 3)