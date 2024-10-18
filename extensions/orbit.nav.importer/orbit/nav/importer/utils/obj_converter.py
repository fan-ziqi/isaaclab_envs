# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import asyncio

import carb
import omni.kit.commands
import omni.usd
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.lab.sim import SimulationCfg, SimulationContext
from omni.isaac.lab.sim.converters.asset_converter_base import AssetConverterBase
from pxr import Usd

from .obj_converter_cfg import ObjConverterCfg

# Enable asset converter extension
enable_extension("omni.kit.asset_converter")
import omni.kit.asset_converter as converter


class ObjConverter(AssetConverterBase):
    """Converter for a URDF description file to a USD file.

    This class wraps around the `omni.isaac.urdf_importer`_ extension to provide a lazy implementation
    for URDF to USD conversion. It stores the output USD file in an instanceable format since that is
    what is typically used in all learning related applications.

    .. caution::
        The current lazy conversion implementation does not automatically trigger USD generation if
        only the mesh files used by the URDF are modified. To force generation, either set
        :obj:`AssetConverterBaseCfg.force_usd_conversion` to True or delete the output directory.

    .. note::
        From Isaac Sim 2023.1 onwards, the extension name changed from ``omni.isaac.urdf`` to
        ``omni.importer.urdf``. This converter class automatically detects the version of Isaac Sim
        and uses the appropriate extension.

        The new extension supports a custom XML tag``"dont_collapse"`` for joints. Setting this parameter
        to true in the URDF joint tag prevents the child link from collapsing when the associated joint type
        is "fixed".

    .. _omni.isaac.urdf_importer: https://docs.omniverse.nvidia.com/isaacsim/latest/ext_omni_isaac_urdf.html
    """

    cfg: ObjConverterCfg
    """The configuration instance for URDF to USD conversion."""

    def __init__(self, cfg: ObjConverterCfg):
        """Initializes the class.

        Args:
            cfg: The configuration instance for URDF to USD conversion.
        """
        super().__init__(cfg=cfg)

    """
    Implementation specific methods.
    """

    def _convert_asset(self):
        """Calls underlying Omniverse command to convert obj to USD.

        Args:
            cfg: The obj conversion configuration.
        """
        # setup converter
        self.task_manager = converter.extension.AssetImporterExtension()

        self.success = False
        print("START CONVERSION")
        asyncio.ensure_future(self.convert_asset_to_usd)

        # init the simulation context
        print("START RENDERING")
        sim = SimulationCfg(SimulationContext())
        while not self.success:
            sim.render()
        print("END RENDERING")

        # fix the issue that material paths are not relative
        # note: This issue seems to have popped up in Isaac Sim 2023.1.1
        stage = Usd.Stage.Open(self.usd_path)
        # resolve all paths relative to layer path
        source_layer = stage.GetRootLayer()
        omni.usd.resolve_paths(source_layer.identifier, source_layer.identifier)
        stage.Save()

    async def convert_asset_to_usd(self):
        # get config
        print("GET CONFIG")
        import_config = self._get_obj_import_config(self.cfg)

        # set task
        print("CREATE CONVERTER TASK")
        task = self.task_manager.create_converter_task(self.usd_path, asset_converter_context=import_config)
        self.success = await task.wait_until_finished()

        print("PROCESSES HAS FINISHED")
        # print error
        if not self.success:
            detailed_status_code = task.get_status()
            detailed_status_error_string = task.get_error_message()
            carb.log_error(
                f"Failed to convert {self.cfg.asset_path} to {self.usd_path} "
                f"with status {detailed_status_code} and error {detailed_status_error_string}"
            )

    """
    Helper methods.
    """

    def _get_obj_import_config(self, cfg: ObjConverterCfg) -> converter.AssetConverterContext:
        """Create and fill AssetConverterContext with desired settings

        Args:
            cfg: The OBJ conversion configuration.

        Returns:
            The constructed ``AssetConverterContext`` object containing the desired settings.
        """
        # NOTE: hopefully will be soon changed to dataclass, then initialization can be improved
        asset_converter_cfg: converter.AssetConverterContext = converter.AssetConverterContext()
        asset_converter_cfg.ignore_materials = cfg.ignore_materials
        # Don't import/export materials
        asset_converter_cfg.ignore_animations = cfg.ignore_animations
        # Don't import/export animations
        asset_converter_cfg.ignore_camera = cfg.ignore_camera
        # Don't import/export cameras
        asset_converter_cfg.ignore_light = cfg.ignore_light
        # Don't import/export lights
        asset_converter_cfg.single_mesh = cfg.single_mesh
        # By default, instanced props will be export as single USD for reference. If
        # this flag is true, it will export all props into the same USD without instancing.
        asset_converter_cfg.smooth_normals = cfg.smooth_normals
        # Smoothing normals, which is only for assimp backend.
        asset_converter_cfg.export_preview_surface = cfg.export_preview_surface
        # Imports material as UsdPreviewSurface instead of MDL for USD export
        asset_converter_cfg.use_meter_as_world_unit = cfg.use_meter_as_world_unit
        # Sets world units to meters, this will also scale asset if it's centimeters model.
        asset_converter_cfg.create_world_as_default_root_prim = cfg.create_world_as_default_root_prim
        # Creates /World as the root prim for Kit needs.
        asset_converter_cfg.embed_textures = cfg.embed_textures
        # Embedding textures into output. This is only enabled for FBX and glTF export.
        asset_converter_cfg.convert_fbx_to_y_up = cfg.convert_fbx_to_y_up
        # Always use Y-up for fbx import.
        asset_converter_cfg.convert_fbx_to_z_up = cfg.convert_fbx_to_z_up
        # Always use Z-up for fbx import.
        asset_converter_cfg.keep_all_materials = cfg.keep_all_materials
        # If it's to remove non-referenced materials.
        asset_converter_cfg.merge_all_meshes = cfg.merge_all_meshes
        # Merges all meshes to single one if it can.
        asset_converter_cfg.use_double_precision_to_usd_transform_op = cfg.use_double_precision_to_usd_transform_op
        # Uses double precision for all transform ops.
        asset_converter_cfg.ignore_pivots = cfg.ignore_pivots
        # Don't export pivots if assets support that.
        asset_converter_cfg.disabling_instancing = cfg.disabling_instancing
        # Don't export instancing assets with instanceable flag.
        asset_converter_cfg.export_hidden_props = cfg.export_hidden_props
        # By default, only visible props will be exported from USD exporter.
        asset_converter_cfg.baking_scales = cfg.baking_scales
        # Only for FBX. It's to bake scales into meshes.

        return asset_converter_cfg
