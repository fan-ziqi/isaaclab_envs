# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.sim.converters.asset_converter_base_cfg import (
    AssetConverterBaseCfg,
)
from omni.isaac.lab.utils import configclass


@configclass
class ObjConverterCfg(AssetConverterBaseCfg):
    """The configuration class for ObjConverter."""

    ignore_materials: bool = False
    """Don't import/export materials"""
    ignore_animations: bool = False
    """Don't import/export animations"""
    ignore_camera: bool = False
    """Don't import/export cameras"""
    ignore_light: bool = False
    """Don't import/export lights"""
    single_mesh: bool = False
    """By default, instanced props will be export as single USD for reference. If
    this flag is true, it will export all props into the same USD without instancing."""
    smooth_normals: bool = True
    """Smoothing normals, which is only for assimp backend."""
    export_preview_surface: bool = False
    """Imports material as UsdPreviewSurface instead of MDL for USD export"""
    use_meter_as_world_unit: bool = True
    """Sets world units to meters, this will also scale asset if it's centimeters model."""
    create_world_as_default_root_prim: bool = True
    """Creates /World as the root prim for Kit needs."""
    embed_textures: bool = True
    """Embedding textures into output. This is only enabled for FBX and glTF export."""
    convert_fbx_to_y_up: bool = False
    """Always use Y-up for fbx import."""
    convert_fbx_to_z_up: bool = True
    """Always use Z-up for fbx import."""
    keep_all_materials: bool = False
    """If it's to remove non-referenced materials."""
    merge_all_meshes: bool = False
    """Merges all meshes to single one if it can."""
    use_double_precision_to_usd_transform_op: bool = False
    """Uses double precision for all transform ops."""
    ignore_pivots: bool = False
    """Don't export pivots if assets support that."""
    disabling_instancing: bool = False
    """Don't export instancing assets with instanceable flag."""
    export_hidden_props: bool = False
    """By default, only visible props will be exported from USD exporter."""
    baking_scales: bool = False
    """Only for FBX. It's to bake scales into meshes."""
