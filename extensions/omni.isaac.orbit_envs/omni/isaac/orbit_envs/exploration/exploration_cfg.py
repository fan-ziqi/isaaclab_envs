# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.orbit.sim as sim_utils
import torch
from omni.isaac.orbit_envs.domains import MatterportRayCasterCameraCfg
from omni.isaac.orbit_envs.importer import MatterportImporterCfg
from omni.isaac.orbit.assets import AssetBaseCfg, RigidObjectCfg
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.sensors import patterns
from omni.isaac.orbit.utils import configclass

from .terrain_analysis_cfg import TerrainAnalysisCfg

OBJ_PATH = "/home/pascal/orbit/orbit/source/extensions/omni.isaac.orbit_assets/data/matterport/2n8kARJN3HM/2n8kARJN3HM/matterport_mesh/0c334eaabb844eaaad049cbbb2e0a4f2/0c334eaabb844eaaad049cbbb2e0a4f2.usd"
PLY_PATH = "/home/pascal/orbit/orbit/source/extensions/omni.isaac.orbit_assets/data/matterport/2n8kARJN3HM/2n8kARJN3HM/house_segmentations/2n8kARJN3HM.ply"


@configclass
class ExplorationSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = MatterportImporterCfg(
        obj_filepath=OBJ_PATH,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    # rigid object to attach the cameras to
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )

    # camera snesors
    camera_0 = MatterportRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        mesh_prim_paths=[PLY_PATH],
        update_period=0,
        max_distance=10.0,
        data_types=["semantic_segmentation"],
        debug_vis=True,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24,
            horizontal_aperture=20.955,
            height=720,
            width=1280,
        ),
    )
    camera_1 = MatterportRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        mesh_prim_paths=[PLY_PATH],
        update_period=0,
        max_distance=10.0,
        data_types=["distance_to_image_plane"],
        debug_vis=False,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24,
            horizontal_aperture=20.955,
            height=480,
            width=848,
        ),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0)),
    )
    sphere_left = AssetBaseCfg(
        prim_path="/World/light_indoor_left",
        spawn=sim_utils.SphereLightCfg(intensity=50000.0, color=(0.75, 0.75, 0.75)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(3.0, 0.0, 3.0)),
    )
    sphere_middle = AssetBaseCfg(
        prim_path="/World/light_indoor_middle",
        spawn=sim_utils.SphereLightCfg(intensity=50000.0, color=(0.75, 0.75, 0.75)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(8.0, 0.0, 3.0)),
    )
    sphere_front = AssetBaseCfg(
        prim_path="/World/light_indoor_front",
        spawn=sim_utils.SphereLightCfg(intensity=50000.0, color=(0.75, 0.75, 0.75)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(8.0, -5.0, 3.0)),
    )
    sphere_front_2 = AssetBaseCfg(
        prim_path="/World/light_indoor_front_2",
        spawn=sim_utils.SphereLightCfg(intensity=50000.0, color=(0.75, 0.75, 0.75)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(8.0, -11.0, 3.0)),
    )
    sphere_right = AssetBaseCfg(
        prim_path="/World/light_indoor_right",
        spawn=sim_utils.SphereLightCfg(intensity=50000.0, color=(0.75, 0.75, 0.75)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(15.0, 0.0, 3.0)),
    )


@configclass
class ExplorationCfg:
    # scene and import definition
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg()
    exploration_scene: InteractiveSceneCfg = ExplorationSceneCfg(num_envs=1, env_spacing=1.0)
    """Parameters to construct the matterport scene"""
    terrain_analysis: TerrainAnalysisCfg = TerrainAnalysisCfg(raycaster_sensor="camera_0")
    """Name of the camera object in the scene definition used for the terrain analysis."""

    # sampling
    sample_points: int = 10000
    """Number of random points to sample."""
    x_angle_range: tuple[float, float] = (-2.5, 2.5)
    y_angle_range: tuple[float, float] = (-2, 5)  # negative angle means in isaac convention: look down
    """Range of the x and y angle of the camera (in degrees), will be randomly selected according to a uniform distribution"""
    height: float = 0.5
    """Height to use for the random points."""

    # point filtering
    min_height: float = 0.2
    """Maximum height to be considered an accessible point for the robot"""
    ground_height: float = -0.1
    """Height of the ground plane"""
    min_wall_distance: float = 0.5
    """Minimum distance to a wall to be considered an accessible point for the robot"""
    min_hit_rate: float = 0.8
    """Don't use a point if the hit rate is below this value"""
    min_avg_hit_distance: float = 0.5
    """Don't use a point if the max hit distance is below this value"""
    min_std_hit_distance: float = 0.5
    """Don't use a point if the std hit distance is below this value"""

    # convergence
    conv_rate: float = 0.9
    """Rate of faces that are covered by three different images, used to terminate the exploration"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    """Device to use for computations."""

    # SAVING
    max_images: int = 2000
    """Maximum number of images recorded"""
    save_path: str = "/home/pascal/viplanner/imperative_learning/data"
    suffix: str | None = "cam_mount"
    """Path to save the data to (directly with env name will be created)"""
