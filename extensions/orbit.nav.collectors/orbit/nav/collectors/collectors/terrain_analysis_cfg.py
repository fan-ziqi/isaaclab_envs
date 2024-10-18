# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.lab.utils import configclass
from orbit.nav.collectors.configs import MatterportSemanticCostMapping


@configclass
class TerrainAnalysisCfg:
    robot_height: float = 0.6
    """Height of the robot"""
    wall_height: float = 1.0
    """Height of the walls.

    Wall filtering will start rays from that height and filter all that hit the mesh within 0.3m."""
    robot_buffer_spawn: float = 0.7
    """Robot buffer for spawn location"""
    sample_points: int = 1000
    """Number of nodes in the tree"""
    max_path_length: float = 10.0
    """Maximum distance from the start location to the goal location"""
    num_connections: int = 5
    """Number of connections to make in the graph"""
    raycaster_sensor: str | None = None
    """Name of the raycaster sensor to use for terrain analysis.

    If None, the terrain analysis will be done on the USD stage. For matterport environments,
    the Orbit raycaster sensor can be used as the ply mesh is a single mesh. On the contrary,
    for unreal engine meshes (as they consists out of multiple meshes), raycasting should be
    performed over the USD stage. Default is None."""
    grid_resolution: float = 0.1
    """Resolution of the grid to check for not traversable edges"""
    height_diff_threshold: float = 0.3
    """Threshold for height difference between two points"""
    viz_graph: bool = True
    """Visualize the graph after the construction for a short amount of time."""

    semantic_cost_mapping: object | None = MatterportSemanticCostMapping()
    """Mapping of semantic categories to costs for filtering edges and nodes"""
    semantic_cost_threshold: float = 0.5
    """Threshold for semantic cost filtering"""

    dim_limiter_prim: str | None = None
    """Prim name that should be used to limit the dimensions of the mesh.

    All meshes including this prim string are used to set the range in which the graph is constructed and samples are
    generated. If None, all meshes are considered."""
