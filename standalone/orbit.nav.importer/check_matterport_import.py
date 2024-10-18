# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the rigid objects class.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

# omni-isaac-orbit
from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the camera sensor.")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sensors import patterns
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass
from orbit.nav.importer.importer import MatterportImporterCfg
from orbit.nav.importer.sensors import (
    MatterportRayCasterCameraCfg,
    MatterportRayCasterCfg,
)

##
# Pre-defined configs
##
from omni.isaac.orbit_assets.anymal import ANYMAL_C_CFG  # isort: skip

"""
Environment Configuration
"""

OBJ_PATH = "/home/pascal/orbit/orbit/source/extensions/omni.isaac.orbit_assets/data/matterport/2n8kARJN3HM/2n8kARJN3HM/matterport_mesh/0c334eaabb844eaaad049cbbb2e0a4f2/0c334eaabb844eaaad049cbbb2e0a4f2.usd"
PLY_PATH = "/home/pascal/orbit/orbit/source/extensions/omni.isaac.orbit_assets/data/matterport/2n8kARJN3HM/2n8kARJN3HM/house_segmentations/2n8kARJN3HM.ply"


@configclass
class TestTerrainCfg(InteractiveSceneCfg):
    """Configuration for a matterport terrain scene with a camera."""

    # ground terrain
    terrain = MatterportImporterCfg(
        prim_path="/World/Matterport",
        terrain_type="matterport",
        obj_filepath=OBJ_PATH,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    # articulation
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # height scanner
    height_scanner = MatterportRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=MatterportRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 1.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[2.0, 2.0]),
        debug_vis=True,
        mesh_prim_paths=[PLY_PATH],
    )
    # camera
    semantic_camera = MatterportRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=[PLY_PATH],
        update_period=0,
        max_distance=10.0,
        data_types=["semantic_segmentation"],
        debug_vis=True,
        offset=MatterportRayCasterCameraCfg.OffsetCfg(
            pos=(0.419, -0.025, -0.020), rot=(0.992, 0.008, 0.127, 0.001), convention="world"
        ),
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24,
            horizontal_aperture=20.955,
            height=720,
            width=1280,
        ),
    )
    depth_camera = MatterportRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        mesh_prim_paths=[PLY_PATH],
        update_period=0,
        max_distance=10.0,
        data_types=["distance_to_image_plane"],
        debug_vis=False,
        offset=MatterportRayCasterCameraCfg.OffsetCfg(
            pos=(0.419, -0.025, -0.020), rot=(0.992, 0.008, 0.127, 0.001), convention="world"
        ),
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=24,
            horizontal_aperture=20.955,
            height=480,
            width=848,
        ),
    )
    # extras - light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 500.0)),
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

    def __post_init__(self):
        """Post initialization."""
        self.robot.init_state.pos = (8.0, 0.0, 0.6)


"""
Main
"""


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([10.0, 1.5, 2.0], [8.0, -1.0, 0.5])
    # Design scene
    scene_cfg = TestTerrainCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # Simulation loop
    while simulation_app.is_running():
        # set joint targets
        scene.articulations["robot"].set_joint_position_target(
            scene.articulations["robot"].data.default_joint_pos.clone()
        )
        # write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Update buffers
        scene.update(sim_dt)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
