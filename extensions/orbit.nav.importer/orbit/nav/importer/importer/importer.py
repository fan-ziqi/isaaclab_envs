# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins
import os
from typing import TYPE_CHECKING

import carb
import numpy as np
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.lab.sim as sim_utils
import trimesh
from omni.isaac.lab.terrains import TerrainImporter
from omni.isaac.lab.utils.warp import convert_to_warp_mesh
from pxr import UsdGeom

if TYPE_CHECKING:
    from .importer_cfg import MatterportImporterCfg

# omniverse
from omni.isaac.core.utils import extensions

extensions.enable_extension("omni.kit.asset_converter")
import omni.kit.asset_converter as converter


class MatterportConverter:
    def __init__(self, input_obj: str, context: converter.impl.AssetConverterContext):
        self._input_obj = input_obj
        self._context = context

        # setup converter
        self.task_manager = converter.extension.AssetImporterExtension()
        return

    async def convert_asset_to_usd(self):
        # get usd file path and create directory
        base_path, _ = os.path.splitext(self._input_obj)
        # set task
        task = self.task_manager.create_converter_task(
            self._input_obj, base_path + ".usd", asset_converter_context=self._context
        )
        success = await task.wait_until_finished()

        # print error
        if not success:
            detailed_status_code = task.get_status()
            detailed_status_error_string = task.get_error_message()
            carb.log_error(
                f"Failed to convert {self._input_obj} to {base_path + '.usd'} "
                f"with status {detailed_status_code} and error {detailed_status_error_string}"
            )
        return


class MatterportImporter(TerrainImporter):
    """
    Default stairs environment for testing
    """

    cfg: MatterportImporterCfg

    def __init__(self, cfg: MatterportImporterCfg):
        """
        :param
        """
        # store inputs
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device

        # create a dict of meshes
        self.meshes = dict()
        self.warp_meshes = dict()
        self.env_origins = None
        self.terrain_origins = None

        # import the world
        if not self.cfg.terrain_type == "matterport":
            raise ValueError(
                "MatterportImporter can only import 'matterport' data. Given terrain type "
                f"'{self.cfg.terrain_type}'is not supported."
            )

        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            self.load_world()

            if isinstance(self.cfg.num_envs, int):
                self.configure_env_origins()

            # set initial state of debug visualization
            self.set_debug_vis(self.cfg.debug_vis)

        else:
            carb.log_info("[INFO]: Loading in extension mode requires calling 'load_world_async'")

            # Converter
            self.converter: MatterportConverter = MatterportConverter(self.cfg.obj_filepath, self.cfg.asset_converter)

    async def load_world_async(self):
        """Function called when clicking load button"""
        # create world
        await self.load_matterport()
        # update stage for any remaining process.
        await stage_utils.update_stage_async()
        # get environment origins
        if isinstance(self.cfg.num_envs, int):
            self.configure_env_origins()
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)
        # Now we are ready!
        carb.log_info("[INFO]: Setup complete...")

    def load_world(self):
        """Function called when clicking load button"""
        # create world
        self.load_matterport_sync()
        # update stage for any remaining process.
        stage_utils.update_stage()
        # Now we are ready!
        carb.log_info("[INFO]: Setup complete...")

    async def load_matterport(self):
        _, ext = os.path.splitext(self.cfg.obj_filepath)
        # if obj mesh --> convert to usd
        if ext == ".obj":
            await self.converter.convert_asset_to_usd()
        # add mesh to stage
        self.load_matterport_sync()

    def load_matterport_sync(self):
        base_path, _ = os.path.splitext(self.cfg.obj_filepath)
        assert os.path.exists(base_path + ".usd"), (
            "Matterport load sync can only handle '.usd' files not obj files. Please use the async function to convert"
            " the obj file to usd first (accessed over the extension in the GUI)"
        )

        self._xform_prim = prim_utils.create_prim(
            prim_path=self.cfg.prim_path + "/Matterport", translation=(0.0, 0.0, 0.0), usd_path=base_path + ".usd"
        )

        # apply collider properties
        collider_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        sim_utils.define_collision_properties(self._xform_prim.GetPrimPath(), collider_cfg)

        # create physics material
        physics_material_cfg: sim_utils.RigidBodyMaterialCfg = self.cfg.physics_material
        # spawn the material
        physics_material_cfg.func(f"{self.cfg.prim_path}/physicsMaterial", self.cfg.physics_material)
        sim_utils.bind_physics_material(self._xform_prim.GetPrimPath(), f"{self.cfg.prim_path}/physicsMaterial")

        # traverse the prim and get the collision mesh
        # THINK: Should the user specify the collision mesh?
        mesh_prim = sim_utils.get_first_matching_child_prim(
            self.cfg.prim_path + "/Matterport", lambda prim: prim.GetTypeName() == "Mesh"
        )
        # check if the mesh is valid
        if mesh_prim is None:
            raise ValueError(f"Could not find any collision mesh in {self.cfg.obj_filepath}. Please check asset.")
        # cast into UsdGeomMesh
        mesh_prim = UsdGeom.Mesh(mesh_prim)
        # store the mesh
        vertices = np.asarray(mesh_prim.GetPointsAttr().Get())
        faces = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        self.meshes["matterport"] = trimesh.Trimesh(vertices=vertices, faces=faces)
        # create a warp mesh
        device = "cuda" if "cuda" in self.device else "cpu"
        self.warp_meshes["matterport"] = convert_to_warp_mesh(vertices, faces, device=device)

        # add colliders and physics material
        if self.cfg.groundplane:
            ground_plane_cfg = sim_utils.GroundPlaneCfg(physics_material=self.cfg.physics_material)
            ground_plane = ground_plane_cfg.func("/World/GroundPlane", ground_plane_cfg)
            ground_plane.visible = False
