#!/usr/bin/python
"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      Load Semantics from Matterport3D and make them available to Isaac Sim
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import scipy.spatial.transform as tf
import torch
import yaml

# Python
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import carb
import cv2

# omni
import omni
import pandas as pd
import trimesh
import warp as wp
from omni.isaac.core.simulation_context import SimulationContext

# omni-isaac-core
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.debug_draw import _debug_draw

# omni-isaac-matterport
from omni.isaac.matterport.config import MatterportConfig

try:
    from viplanner.config.viplanner_sem_meta import VIPlannerSemMetaHandler
except ImportError:
    print("VIPlanner Semantic Meta Handler not available.")

from omni.isaac.matterport.utils import DATA_DIR
from pxr import Usd

# omni-isaac-orbit
from omni.isaac.orbit.sensors.camera import Camera
from omni.isaac.orbit.utils.math import convert_quat

wp.init()
wp.ScopedTimer(name="Matterport Warp")

OPENGL_TO_ISAAC_MAT = tf.Rotation.from_euler("XYZ", [-90, 0, 90], degrees=True).as_matrix()
ISAAC_TO_OPENGL_MAT = tf.Rotation.from_euler("XYZ", [90, -90, 0], degrees=True).as_matrix()


@dataclass
class CameraData:
    """Camera data class"""

    # Camera prim
    prim: Usd.Prim = None
    omni_camera: Camera = None
    # image plane size
    width: int = 640
    height: int = 360
    # Camera pose
    pos: torch.tensor = None
    rot: torch.tensor = None
    poses: np.ndarray = np.zeros((1, 7))  # [x, y, z, qx, qy, qz, qw], save history of poses
    # Camera intrinsics
    cam_focal_length: float = 0.0  # in mm
    cam_horizontal_aperture: float = 0.0  # in mm
    cam_vertical_aperture: float = 0.0  # in mm
    intrinsics: torch.tensor = None
    intrinsics_inv: torch.tensor = None
    # Domains of the camera
    depth: bool = False
    semantic: bool = False
    rgb: bool = False
    visualize: bool = False  # visualize all domains of the camera
    # pixel coordinates in camera frame
    pixel_coords: torch.tensor = None
    # pixel offset in camera frame
    pixel_offset: torch.tensor = None
    # ray directions in world frame
    ray_directions: torch.tensor = None
    ray_directions_length: int = 0
    # ray hit distances and hit coordinates in world frame
    ray_distances: torch.tensor = None
    ray_hit_coords: torch.tensor = None
    # ray face indices
    ray_face_indices: torch.tensor = None
    # visualization parameters
    vis_init: bool = False
    fg_sem: plt.Figure = None
    fg_depth: plt.Figure = None
    fg_rgb: plt.Figure = None
    ax_sem: plt.Axes = None
    ax_depth: plt.Axes = None
    ax_rgb: plt.Axes = None
    img_sem = None
    img_depth = None
    img_rgb = None
    id_sem: int = 0
    id_depth: int = 0
    id_rgb: int = 0
    # render images
    render_rgb: np.ndarray = None
    render_sem: np.ndarray = None
    render_depth: np.ndarray = None

    def init_buffers(self):
        # camera pose
        self.pos = torch.zeros((3, 1))
        self.rot = torch.zeros((3, 3))
        # camera parameters
        self.intrinsics = torch.zeros((3, 3))
        self.intrinsics_inv = torch.zeros((3, 3))
        self.pixel_coords = torch.zeros((self.width * self.height, 3))
        self.pixel_offset = torch.zeros((self.width * self.height, 3))
        # raycasting
        self.ray_directions = torch.zeros((self.width * self.height, 3))
        self.ray_hit_coords = torch.zeros((self.width * self.height, 3))
        self.ray_distances = torch.zeros((self.width * self.height, 1))
        self.ray_face_indices = torch.zeros((self.width * self.height, 1))
        # render products
        self.render_rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.render_sem = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.render_depth = np.zeros((self.height, self.width, 1), dtype=np.uint16)
        return

    def update_ray_direction_length(self):
        self.ray_directions_length = self.width * self.height
        return

    @staticmethod
    def get_default_width() -> int:
        return CameraData.width

    @staticmethod
    def get_default_height() -> int:
        return CameraData.height


class MatterportWarp:
    """
    Load Matterport3D Semantics and make them available to Isaac Sim
    """

    debug = False

    def __init__(
        self,
        cfg: MatterportConfig,
    ):
        """
        Initialize MatterportSemWarp

        Args:
            path (str): path to Matterport3D Semantics
        """
        self._cfg: MatterportConfig = cfg

        # get internal parameters
        self._device = "cuda" if wp.is_cuda_available() else "cpu"
        self._stage = get_current_stage()
        self._fg_idx_max = 0

        # setup mesh
        print("Loading mesh...")
        self.mesh: trimesh.Trimesh = None
        self.mesh_wp: wp.Mesh = None
        self.face_id_category_mapping: np.ndarray = None
        self.vertex_to_rgb_mapping: np.ndarray = None
        self._load_mesh()  # load mesh as trimesh
        self._get_mapping_face_to_category()  # get mapping from face to category id
        self._get_mapping_vertex_to_rgb()  # get mapping from vertex to rgb color
        self._convert_to_wp_mesh()  # convert to wp mesh
        print("Mesh loaded.")

        # check simulation context
        if SimulationContext.instance():
            self.sim: SimulationContext = SimulationContext.instance()
        else:
            self.sim = None
            carb.log_warn("SimulationContext not available. Rendering for RGB images not possible.")

        # load categort id to class mapping (name and id of mpcat40 redcued class set)
        # More Information: https://github.com/niessner/Matterport/blob/master/data_organization.md#house_segmentations
        mapping = pd.read_csv(DATA_DIR + "/mappings/category_mapping.tsv", sep="\t")
        self.mapping_mpcat40 = mapping["mpcat40index"].to_numpy()
        self.mapping_mpcat40_names = mapping["mpcat40"].to_numpy()

        if self._cfg.viplanner_meta:
            viplanner_sem = VIPlannerSemMetaHandler()
            map_mpcat40_to_vip_sem = yaml.safe_load(open(DATA_DIR + "/mappings/mpcat40_to_vip_sem.yml"))
            color = viplanner_sem.get_colors_for_names(list(map_mpcat40_to_vip_sem.values()))
            self.color = np.array(color)
        else:
            # load defined colors for mpcat40
            mapping_40 = pd.read_csv(DATA_DIR + "/mappings/mpcat40.tsv", sep="\t")
            color = mapping_40["hex"].to_numpy()
            self.color = np.array(
                [(int(color[i][1:3], 16), int(color[i][3:5], 16), int(color[i][5:7], 16)) for i in range(len(color))]
            )

        # setup camera list
        self.cameras: List[CameraData] = []

        # check if visualization is enabled
        self._vis_enabled = True
        try:
            matplotlib.use("Qt5Agg")
        except:
            self._vis_enabled = False
            carb.log_warning(
                "Could not set matplotlib backend to Qt5Agg\n"
                "For visualization, need different matplotlib backend. Please install 'pyqt5' by running: \n"
                "``` python.sh -m pip install pyqt5 ```"
            )

        self.draw = _debug_draw.acquire_debug_draw_interface()
        if self.debug:
            self.num_points_network_return = 5000
            self.point_list = [(1, 0, 0.5)] * self.num_points_network_return
            self.colors_1 = [(1, 1, 1, 1)] * self.num_points_network_return
            self.colors_2 = [(1, 1, 255, 1)] * self.num_points_network_return
            self.colors_3 = [(1, 255, 1, 1)] * self.num_points_network_return

            self.sizes = [5] * self.num_points_network_return
        return

    ##
    # Public Methods
    ##

    def register_camera(
        self,
        cam_prim: Optional[Union[str, Usd.Prim]],
        width: int,
        height: int,
        semantics: bool = False,
        depth: bool = False,
        rgb: bool = False,
        visualization: bool = False,
        omni_cam: Optional[Camera] = None,
    ) -> None:
        """
        Register a camera to the MatterportSemWarp
        """
        # get Usd.Prim is string is passed
        if isinstance(cam_prim, str):
            cam_prim = self._stage.GetPrimAtPath(cam_prim)

        # CameraData --> init object and set basic parameters
        cam_data = CameraData(
            prim=cam_prim,
            width=width,
            height=height,
            semantic=semantics,
            depth=depth,
            rgb=rgb,
            omni_camera=omni_cam,
        )
        cam_data.init_buffers()  # reserve buffers
        cam_data.update_ray_direction_length()  # set length of ray directions

        # CameraData --> set further parameters
        self._get_cam_intrinsics(cam_data)
        cam_data.intrinsics_inv = torch.inverse(cam_data.intrinsics)
        self._get_pix_in_cam_frame(cam_data)

        # append to camera list
        self.cameras.append(cam_data)

        # init visualiztion plane
        if visualization and self._vis_enabled:
            self._init_visualization(cam_data)

        # if self.debug:
        #     self.draw.clear_points()
        #     self.draw.draw_points(cam_data.pixel_coords.cpu().tolist()[:1000], self.colors, self.sizes)

        print("Registered camera DONE")
        return

    async def compute_domains(self, time_step: int) -> torch.tensor:
        """
        Compute Matterport3D Semantics

        Returns:
            torch.tensor: Matterport3D Semantics
        """
        assert len(self.cameras) > 0, "No camera registered"

        for cam_idx, cam_data in enumerate(self.cameras):
            # update pos and rot
            cam_data.pos, cam_data.rot = self._get_cam_pose(cam_data.prim)
            # update ray_directions in world frame
            cam_data.ray_directions = self._get_ray_directions(cam_data.pos, cam_data.rot, cam_data.pixel_coords)
            # update ray_distances and get ray_face_indices
            cam_data.ray_hit_coords, cam_data.ray_face_indices, cam_data.ray_distances = self._raycast(
                cam_data.pos.repeat(len(cam_data.ray_directions)),
                cam_data.ray_directions,
                cam_rot=cam_data.rot,
                pix_offset=cam_data.pixel_offset,
            )

            # set all inf or nan values to 0
            cam_data.ray_distances[~torch.isfinite(cam_data.ray_distances)] = 0

            carb.log_verbose(f"Matterport Semantic Callback {cam_idx} DONE")

            await self._render_async(cam_data)

            if cam_data.omni_camera is not None:
                # TODO: find way to access the Orbit Camera class from the prim
                # TODO: check if the same as depth generated by isaac first
                # if "distance_to_image_plane" in cam_data.omni_camera.data.output.keys() and cam_data.depth:
                #     cam_data.omni_camera.data.output["distance_to_image_plane"] = cam_data.redner_depth
                if "semantic_segmentation" in cam_data.omni_camera.data.output.keys() and cam_data.semantic:
                    cam_data.omni_camera.data.output["semantic_segmentation"] = cam_data.render_sem
            else:
                carb.log_warn(
                    "No Omni Camera found! Matterport Domains not available in Omni Camera! Callback not used!"
                )

            if cam_data.visualize:
                await self._update_visualization_async(cam_data=cam_data)

            if self._cfg.save:
                await self._save_data_async(cam_data=cam_data, idx=time_step, cam_idx=cam_idx)

            if self.debug:
                self.draw.clear_points()
                self.draw.draw_points(
                    random.choices(cam_data.ray_hit_coords.cpu().tolist(), k=5000), self.colors, self.sizes
                )

        return

    ##
    # Private Methods (Helper Functions)
    ##

    # Mesh helpers ###

    def _load_mesh(self) -> None:
        self.mesh = trimesh.load(self._cfg.import_file_ply)
        return

    def _get_mapping_face_to_category(self) -> None:
        # get raw face information
        faces_raw = self.mesh.metadata["_ply_raw"]["face"]["data"]
        carb.log_info(f"Raw face information of type {faces_raw.dtype}")
        # get face categories
        self.face_id_category_mapping = np.asarray([single_face[3] for single_face in faces_raw])
        return

    def _get_mapping_vertex_to_rgb(self) -> None:
        # get raw vertex information
        vertices_raw = self.mesh.metadata["_ply_raw"]["vertex"]["data"]
        # get vertex colors
        self.vertex_to_rgb_mapping = np.asarray(
            [[single_vertex[8], single_vertex[9], single_vertex[10]] for single_vertex in vertices_raw]
        )
        return

    def _convert_to_wp_mesh(self) -> None:
        """Convert trimesh into wp mesh"""
        assert type(self.mesh) == trimesh.base.Trimesh
        self.mesh_wp = wp.Mesh(
            points=wp.array(self.mesh.vertices.astype(np.float32), dtype=wp.vec3, device=self._device),
            indices=wp.array(self.mesh.faces.astype(np.int32).flatten(), dtype=int, device=self._device),
        )
        return

    # Camera helpers ###

    def _get_cam_pose(self, cam_prim: Usd.Prim) -> Tuple[torch.tensor, torch.tensor]:
        """Get camera pose from prim

        Args:
            cam_prim (Usd.Prim): camera
        """
        cam_transform = omni.usd.utils.get_world_transform_matrix(cam_prim)
        cam_pos = torch.tensor(cam_transform.ExtractTranslation(), dtype=torch.float32)
        # transfrom from opengl convention (-z forward, y-up, x-right) to isaac convention (x-forward, y-left, z-up)
        # remove standard isaac rotation (cameras oriented in -z direction)
        cam_rot = np.asarray(cam_transform.ExtractRotationMatrix()).T
        cam_rot = torch.tensor(cam_rot @ OPENGL_TO_ISAAC_MAT, dtype=torch.float32)
        return cam_pos, cam_rot

    def _get_cam_intrinsics(
        self,
        cam_data: CameraData,
    ) -> None:
        """Returns the camera parameters."""
        # focal length in mm
        cam_data.cam_focal_length = torch.tensor(cam_data.prim.GetAttribute("focalLength").Get())
        # horizontal aperture size in mm
        cam_data.cam_horizontal_aperture = torch.tensor(cam_data.prim.GetAttribute("horizontalAperture").Get())
        # determine vertical aperture size in mm and set it accordingly
        cam_data.cam_vertical_aperture = cam_data.cam_horizontal_aperture * cam_data.height / cam_data.width
        cam_data.prim.GetAttribute("verticalAperture").Set(cam_data.cam_vertical_aperture.item())

        # compute camera intrinsics
        alpha_u = cam_data.width * cam_data.cam_focal_length / cam_data.cam_horizontal_aperture
        alpha_v = cam_data.height * cam_data.cam_focal_length / cam_data.cam_vertical_aperture
        cam_data.intrinsics = torch.tensor(
            [[alpha_u, 0, cam_data.width * 0.5], [0, alpha_v, cam_data.height * 0.5], [0, 0, 1]]
        )

        return

    def _get_pix_in_cam_frame(self, cam_data: CameraData) -> None:
        """Get pixel coordinates in camera frame"""
        # get image plane mesh grid
        pix_u = torch.arange(start=0, end=cam_data.width, dtype=torch.int32)
        pix_v = torch.arange(start=0, end=cam_data.height, dtype=torch.int32)
        grid = torch.meshgrid(pix_u, pix_v, indexing="ij")
        pixels = torch.vstack(list(map(torch.ravel, grid))).T
        pixels = torch.hstack([pixels, torch.ones((len(pixels), 1))])  # add ones for 3D coordinates

        # get pixel coordinates in camera frame
        pix_in_cam_frame = torch.matmul(cam_data.intrinsics_inv, pixels.T)  # transform to camera frame

        # in robotics camera frame is x forward, y left, z up from camera frame with x right, y down, z forward
        cam_data.pixel_coords = pix_in_cam_frame[[2, 0, 1], :].T * torch.tensor(
            [1, -1, -1]
        )  # transform to robotics camera frame

        # get pixel offset from camera center
        pixels_centered = pixels - torch.tensor([[cam_data.width / 2, cam_data.height / 2, 0]])
        pixels_centered[:, 0] *= cam_data.cam_vertical_aperture / 1000 / cam_data.width
        pixels_centered[:, 1] *= cam_data.cam_horizontal_aperture / 1000 / cam_data.height
        pixels_centered[:, 2] *= cam_data.cam_focal_length / 1000
        cam_data.pixel_offset = pixels_centered[:, [2, 0, 1]] * torch.tensor([1, -1, -1])
        return

    def _get_ray_directions(
        self,
        cam_pos: torch.Tensor,
        cam_rot: torch.Tensor,
        pix_in_cam_frame: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the ray directions in world space."""
        # compute ray directions in world space
        # cam_rot_inv = torch.inverse(cam_rot)  # inverse rotation
        ray_directions_world = torch.matmul(cam_rot, pix_in_cam_frame.T).T  # get pixel in world frame

        # normalize ray directions
        ray_directions_world = ray_directions_world / torch.norm(ray_directions_world, dim=1, keepdim=True)
        if self.debug:
            self.draw.clear_points()
            self.draw.draw_points(ray_directions_world.cpu().tolist()[:5000], self.colors_1, self.sizes)
        return ray_directions_world

    # Raycasting helpers ###

    def _raycast(
        self,
        ray_starts_world: torch.Tensor,
        ray_directions_world: torch.Tensor,
        cam_rot: torch.Tensor,
        pix_offset: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Performs ray casting on the terrain mesh.

        Args:
            ray_starts_world (Torch.tensor): The starting position of the ray.
            ray_directions_world (Torch.tensor): The ray direction.

        Returns:
            [Torch.tensor]: The ray hit position. Returns float('inf') for missed hits.
        """
        # move tensors to device
        ray_starts_world = ray_starts_world.to(self._device)
        ray_directions_world = ray_directions_world.to(self._device)
        cam_rot = cam_rot.to(self._device)
        pix_offset = pix_offset.to(self._device)

        ray_starts_world = ray_starts_world.reshape(-1, 3).contiguous()
        ray_directions_world = ray_directions_world.reshape(-1, 3).contiguous()
        num_rays = len(ray_starts_world)
        ray_starts_world_wp = wp.types.array(
            ptr=ray_starts_world.data_ptr(),
            dtype=wp.vec3,
            shape=(num_rays,),
            copy=False,
            owner=False,
            device=self.mesh_wp.device,
        )
        ray_directions_world_wp = wp.types.array(
            ptr=ray_directions_world.data_ptr(),
            dtype=wp.vec3,
            shape=(num_rays,),
            copy=False,
            owner=False,
            device=self.mesh_wp.device,
        )
        ray_hits_world = torch.zeros((num_rays, 3), device=ray_starts_world.device)
        ray_hits_world[:] = float("inf")
        ray_hits_world_wp = wp.types.array(
            ptr=ray_hits_world.data_ptr(),
            dtype=wp.vec3,
            shape=(num_rays,),
            copy=False,
            owner=False,
            device=self.mesh_wp.device,
        )
        ray_face_idxs = torch.zeros(num_rays, dtype=torch.int, device=ray_starts_world.device)
        ray_face_idxs[:] = -1
        ray_face_idxs_wp = wp.types.array(
            ptr=ray_face_idxs.data_ptr(),
            dtype=wp.int32,
            shape=(num_rays,),
            copy=False,
            owner=False,
            device=self.mesh_wp.device,
        )
        ray_hit_depth = torch.zeros(num_rays, device=ray_starts_world.device)
        ray_hit_depth[:] = float("inf")
        ray_hit_depth_wp = wp.types.array(
            ptr=ray_hit_depth.data_ptr(),
            dtype=wp.float32,
            shape=(num_rays,),
            copy=False,
            owner=False,
            device=self.mesh_wp.device,
        )
        max_depth = float(self._cfg.max_depth)
        wp.launch(
            kernel=self._raycast_kernel,
            dim=num_rays,
            inputs=[
                self.mesh_wp.id,
                ray_starts_world_wp,
                ray_directions_world_wp,
                ray_hits_world_wp,
                ray_face_idxs_wp,
                ray_hit_depth_wp,
                max_depth,
            ],
            device=self.mesh_wp.device,
        )
        wp.synchronize()

        ray_hit_depth = torch.abs(
            torch.matmul(cam_rot.T, (ray_hit_depth.unsqueeze(1) * ray_directions_world - pix_offset).T)[0]
        )
        return ray_hits_world.to("cpu"), ray_face_idxs.to("cpu"), ray_hit_depth.to("cpu")

    @staticmethod
    @wp.kernel
    def _raycast_kernel(
        mesh: wp.uint64,
        ray_starts_world: wp.array(dtype=wp.vec3),
        ray_directions_world: wp.array(dtype=wp.vec3),
        ray_hits_world: wp.array(dtype=wp.vec3),
        ray_face_idxs: wp.array(dtype=wp.int32),
        ray_hit_depth: wp.array(dtype=wp.float32),
        max_depth: float,
    ):
        tid = wp.tid()

        t = float(0.0)  # hit distance along ray
        u = float(0.0)  # hit face barycentric u
        v = float(0.0)  # hit face barycentric v
        sign = float(0.0)  # hit face sign
        n = wp.vec3()  # hit face normal
        f = int(0)  # hit face index
        # ray cast against the mesh
        if wp.mesh_query_ray(mesh, ray_starts_world[tid], ray_directions_world[tid], max_depth, t, u, v, sign, n, f):
            ray_hits_world[tid] = ray_starts_world[tid] + t * ray_directions_world[tid]
            ray_face_idxs[tid] = f
            ray_hit_depth[tid] = t
        return

    # Visualization helpers ###

    def _init_visualization(
        self,
        cam_data: CameraData,
    ) -> None:
        """Initializes the visualization plane."""
        if cam_data.semantic:
            # init semantics figure
            cam_data.fg_sem = plt.figure(num=self._fg_idx_max)
            cam_data.ax_sem = cam_data.fg_sem.gca()
            cam_data.ax_sem.set_title(f"Semantics {cam_data.prim.GetPrimPath().pathString}")
            cam_data.img_sem = cam_data.ax_sem.imshow(np.zeros((cam_data.height, cam_data.width, 3), dtype=np.uint8))
            cam_data.id_sem = self._fg_idx_max
            self._fg_idx_max += 1

        if cam_data.depth:
            # init depth figure
            cam_data.fg_depth = plt.figure(num=self._fg_idx_max)
            cam_data.ax_depth = cam_data.fg_depth.gca()
            cam_data.ax_depth.set_title(f"Depth {cam_data.prim.GetPrimPath().pathString}")
            cam_data.img_depth = cam_data.ax_depth.imshow(
                np.zeros((cam_data.height, cam_data.width, 1), dtype=np.float32)
            )
            cam_data.id_depth = self._fg_idx_max + 1
            self._fg_idx_max += 1

        if cam_data.rgb:
            # init depth figure
            cam_data.fg_rgb = plt.figure(num=self._fg_idx_max)
            cam_data.ax_rgb = cam_data.fg_rgb.gca()
            cam_data.ax_rgb.set_title(f"RGB {cam_data.prim.GetPrimPath().pathString}")
            cam_data.img_rgb = cam_data.ax_rgb.imshow(np.zeros((cam_data.height, cam_data.width, 3), dtype=np.float32))
            self._fg_idx_max += 1

        if any((cam_data.semantic, cam_data.depth, cam_data.rgb)):
            plt.ion()
            # update flag
            cam_data.vis_init = True

        return

    async def _update_visualization_async(self, cam_data: CameraData) -> None:
        self._update_visualization(cam_data)
        return

    def _update_visualization(self, cam_data: CameraData) -> None:
        """
        Updates the visualization plane.
        """
        # SEMANTICS
        if self._cfg.semantic:
            cam_data.img_sem.set_array(cam_data.render_sem)
            cam_data.fg_sem.canvas.draw()
            cam_data.fg_sem.canvas.flush_events()

        # DEPTH
        if self._cfg.depth:
            # cam_data.img_depth.set_array(cam_data.render_depth)
            cam_data.ax_depth.imshow(cam_data.render_depth)
            cam_data.fg_depth.canvas.draw()
            cam_data.fg_depth.canvas.flush_events()

        # RGB
        if self._cfg.rgb:
            cam_data.ax_rgb.imshow(cam_data.render_rgb)
            cam_data.fg_rgb.canvas.draw()
            cam_data.fg_rgb.canvas.flush_events()

        plt.pause(0.001)

        return

    async def _render_async(self, cam_data: CameraData) -> None:
        self._render(cam_data)
        return

    def _render(self, cam_data: CameraData) -> None:
        # SEMANTICS
        if cam_data.semantic:
            # get the category index of the hit faces (category index from unreduced set = ~1600 classes)
            face_id = self.face_id_category_mapping[cam_data.ray_face_indices.cpu().numpy()]
            face_id = face_id.astype(int)

            # map category index to reduced set
            face_id_mpcat40 = self.mapping_mpcat40[face_id - 1]
            face_name_mpcat40 = self.mapping_mpcat40_names[face_id - 1]

            # get the color of the face
            face_color = self.color[face_id_mpcat40]

            # reshape and transpose to get the correct orientation
            cam_data.render_sem = face_color.reshape(cam_data.width, cam_data.height, 3).transpose(1, 0, 2)

        # DEPTH
        if cam_data.depth:
            cam_data.render_depth = (
                cam_data.ray_distances.cpu().numpy().reshape(cam_data.width, cam_data.height, 1).transpose(1, 0, 2)
            )
            cam_data.render_depth = np.uint16(cam_data.render_depth * self._cfg.depth_scale)

        # RGB
        if cam_data.rgb and cam_data.omni_camera:
            # only generates RGB image if the simulation is running
            try:
                if not self.sim.is_playing():
                    self.sim.play()
            except AttributeError:
                carb.log_error("AttributeError: SimulationContext not loaded, no RGB image rendered")
                return

            # set camera pose
            rot = cam_data.rot.cpu().numpy() @ ISAAC_TO_OPENGL_MAT
            rot_quad = tf.Rotation.from_matrix(rot).as_quat()
            cam_data.omni_camera._sensor_xform.set_world_pose(cam_data.pos, convert_quat(rot_quad, "wxyz"))
            # render once to load data into buffer
            for _ in range(5):
                self.sim.render()
            # get image
            cam_data.omni_camera.update(dt=0.0)
            cam_data.render_rgb = cam_data.omni_camera.data.output["rgb"][:, :, :3]

        return

    # save data helpers ###

    def init_save(self, save_path: Optional[str] = None) -> None:
        if save_path is not None:
            self._cfg.save_path = save_path
        # create directories
        os.makedirs(self._cfg.save_path, exist_ok=True)
        os.makedirs(os.path.join(self._cfg.save_path, "semantics"), exist_ok=True)
        os.makedirs(os.path.join(self._cfg.save_path, "depth"), exist_ok=True)
        os.makedirs(os.path.join(self._cfg.save_path, "rgb"), exist_ok=True)

        # save camera configurations
        dim = 12 if self._cfg.ros_p_mat else 9
        intrinsics = np.zeros((len(self.cameras), dim))
        for idx, cam_data in enumerate(self.cameras):
            cam_intrinsics = cam_data.intrinsics.cpu().numpy()
            if self._cfg.ros_p_mat:
                p_mat = np.zeros((3, 4))
                p_mat[:3, :3] = cam_intrinsics
                intrinsics[idx] = p_mat.flatten()
            else:
                intrinsics[idx] = cam_intrinsics.flatten()
        np.savetxt(os.path.join(self._cfg.save_path, "intrinsics.txt"), intrinsics, delimiter=",")

        return

    async def _save_data_async(self, cam_data: CameraData, idx: int, cam_idx: int) -> None:
        self._save_data(cam_data, idx, cam_idx)
        return

    def _save_data(self, cam_data: CameraData, idx: int, cam_idx: int) -> None:
        cam_suffix = f"_cam{cam_idx}" if len(self.cameras) > 1 else ""

        # SEMANTICS
        if cam_data.semantic:
            cv2.imwrite(
                os.path.join(self._cfg.save_path, "semantics", f"{idx}".zfill(4) + cam_suffix + ".png"),
                cv2.cvtColor(cam_data.render_sem.astype(np.uint8), cv2.COLOR_RGB2BGR),
            )

        # DEPTH
        if cam_data.depth:
            cv2.imwrite(
                os.path.join(self._cfg.save_path, "depth", f"{idx}".zfill(4) + cam_suffix + ".png"),
                cam_data.render_depth,
            )

        # RGB
        if cam_data.rgb:
            cv2.imwrite(
                os.path.join(self._cfg.save_path, "rgb", f"{idx}".zfill(4) + cam_suffix + ".png"),
                cv2.cvtColor(cam_data.render_rgb, cv2.COLOR_RGB2BGR),
            )

        # camera pose in robotics frame (x forward, y left, z up)
        rot_quat = tf.Rotation.from_matrix(cam_data.rot.cpu().numpy()).as_quat()  # get quat as (x, y, z, w) format
        pose = np.hstack((cam_data.pos.cpu().numpy(), rot_quat))
        cam_data.poses = np.append(cam_data.poses, pose.reshape(1, -1), axis=0)
        return

    def _end_save(self) -> None:
        # save camera poses
        for idx, cam_data in enumerate(self.cameras):
            np.savetxt(
                os.path.join(self._cfg.save_path, f"camera_extrinsic_cam{idx}.txt"),
                cam_data.poses[1:],
                delimiter=",",
            )
        return


# EoF
