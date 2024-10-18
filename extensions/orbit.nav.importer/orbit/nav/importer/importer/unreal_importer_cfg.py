# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass

from .unreal_importer import UnRealImporter


@configclass
class UnRealImporterCfg(TerrainImporterCfg):
    class_type: type = UnRealImporter
    """The class name of the terrain importer."""

    terrain_type = "usd"
    """The type of terrain to generate. Defaults to "matterport".

    """

    sem_mesh_to_class_map: str | None = None
    """Path to the mesh to semantic class mapping file.

    If set, semantic classes will be added to the scene. Default is None."""

    duplicate_cfg_file: str | list | None = None
    """Configuration file(s) to duplicate prims in the scene.

    Selected prims are clone by the provoided factor and moved to the defined location. Default is None."""

    people_config_file: str | None = None
    """Path to the people configuration file.

    If set, people define in the Nvidia Nuclues can be added to the scene. Default is None."""
