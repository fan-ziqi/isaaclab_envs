import torch
import os
from typing import Optional, Tuple

import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.sensors import RayCasterCameraCfg, patterns
from omni.isaac.orbit.assets import AssetBaseCfg

from omni.isaac.orbit_assets import ORBIT_ASSETS_DATA_DIR

from omni.isaac.matterport.importer import MatterportImporterCfg
from omni.isaac.matterport.domains import MatterportRayCasterCameraCfg

@configclass
class ExplorationSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = MatterportImporterCfg(
        obj_filepath=os.path.join(
            ORBIT_ASSETS_DATA_DIR,
            "matterport/2n8kARJN3HM/2n8kARJN3HM/matterport_mesh/0c334eaabb844eaaad049cbbb2e0a4f2/0c334eaabb844eaaad049cbbb2e0a4f2.obj",
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # sensors
    depth_camera = RayCasterCameraCfg(
        prim_path="/cam_depth",
        mesh_prim_paths=["/World/Matterport"],
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

    sem_camera = MatterportRayCasterCameraCfg(
        prim_path="/cam_sem",
        mesh_prim_paths=["/World/Matterport"],
        update_period=0,
        max_distance=10.0,
        data_types=["semantic_segmentation"],
        debug_vis=False,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24,
            horizontal_aperture=20.955,
            height=720,
            width=1280,
        ),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0)),
    )


@configclass
class RandomExplorationCfg:

    # scene and import definition
    exploration_scene: InteractiveSceneCfg = ExplorationSceneCfg(num_envs=200)
    """Parameters to construct the matterport scene"""
    
    # sampling
    points_per_m2: int = 20
    """Number of random points per m2 of the mesh surface area."""
    x_angle_range: Tuple[float, float] = (-2.5, 2.5)
    y_angle_range: Tuple[float, float] = (-2, 5)  # negative angle means in isaac convention: look down
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
    suffix: Optional[str] = "cam_mount"
    """Path to save the data to (directy with env name will be created)"""
