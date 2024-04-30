# Copyright (c) 2023 ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.terrains import TerrainImporterCfg
from omni.isaac.orbit.utils import configclass

from .unreal_importer import UnRealImporter


@configclass
class UnRealImporterCfg(TerrainImporterCfg):
    class_type: type = UnRealImporter
    """The class name of the terrain importer."""

    terrain_type = "usd"
    """The type of terrain to generate. Defaults to "matterport".

    """

    # scale
    scale: float = 0.01  # 0.01  # carla: 0.01 nomoko: 1  park: 0.01 warehouse: 1.0 # scale the scene to be in meters
    # up axis
    axis_up: str = "Y"  # carla, nomoko: "Y", park, warehouse: "Z"
    duplicate_cfg_file: str | list | None = None
    """Configuration file(s) to duplicate prims in the scene. 
    
    Selected prims are clone by the provoided factor and moved to the defined location. Default is None."""

    # mesh to semantic class mapping --> only if set, semantic classes will be added to the scene
    sem_mesh_to_class_map: str | None = None  # os.path.join(DATA_DIR, "park", "keyword_mapping.yml")  os.path.join(DATA_DIR, "town01", "keyword_mapping.yml")
    # add Groundplane to the scene
    groundplane: bool = True
    # add people to the scene
    people_config_file: str | None = None

    groundplane: bool = True
