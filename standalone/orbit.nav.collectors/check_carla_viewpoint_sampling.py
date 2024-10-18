# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
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
import os

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import CameraCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass
from orbit.nav.collectors.collectors import ViewpointSampling, ViewpointSamplingCfg
from orbit.nav.collectors.configs import CarlaSemanticCostMapping
from orbit.nav.importer.importer import UnRealImporterCfg
from orbit.nav.importer.sensors import DATA_DIR

"""
Main
"""


@configclass
class TestTerrainCfg(InteractiveSceneCfg):
    """Configuration for a matterport terrain scene with a camera."""

    # ground terrain
    terrain = UnRealImporterCfg(
        prim_path="/World/Carla",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        usd_path="/home/pascal/viplanner/env/carla_exp/carla.usd",
        duplicate_cfg_file=[
            os.path.join(DATA_DIR, "unreal", "town01", "cw_multiply_cfg.yml"),
            os.path.join(DATA_DIR, "unreal", "town01", "vehicle_cfg.yml"),
        ],
        sem_mesh_to_class_map=os.path.join(DATA_DIR, "unreal", "town01", "keyword_mapping.yml"),
        people_config_file=os.path.join(DATA_DIR, "unreal", "town01", "people_cfg.yml"),
    )
    # camera
    camera_0 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/sem_cam",
        update_period=0,
        data_types=["semantic_segmentation"],
        debug_vis=True,
        offset=CameraCfg.OffsetCfg(pos=(0.419, -0.025, -0.020), rot=(0.992, 0.008, 0.127, 0.001), convention="world"),
        height=720,
        width=1280,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24,
            horizontal_aperture=20.955,
        ),
    )
    camera_1 = CameraCfg(
        prim_path="{ENV_REGEX_NS}/depth_cam",
        update_period=0,
        data_types=["distance_to_image_plane"],
        debug_vis=False,
        offset=CameraCfg.OffsetCfg(pos=(0.419, -0.025, -0.020), rot=(0.992, 0.008, 0.127, 0.001), convention="world"),
        height=480,
        width=848,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24,
            horizontal_aperture=20.955,
        ),
    )
    # extras - light
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 500.0)),
    )


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view([130, -125, 30], [100, -130, 0.5])

    cfg = ViewpointSamplingCfg()
    # override the scene configuration
    cfg.exploration_scene = TestTerrainCfg(args_cli.num_envs, env_spacing=1.0)
    # overwrite semantic cost mapping and adjust parameters based on larger map
    cfg.terrain_analysis.semantic_cost_mapping = CarlaSemanticCostMapping()
    cfg.terrain_analysis.grid_resolution = 1.0
    cfg.terrain_analysis.sample_points = 10000
    # limit space to be within the road network
    cfg.terrain_analysis.dim_limiter_prim = "Road_Sidewalk"
    # enable debug visualization
    cfg.terrain_analysis.viz_graph = True

    explorer = ViewpointSampling(cfg)
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # sample and render viewpoints
    samples = explorer.sample_viewpoints(9560)
    explorer.render_viewpoints(samples)
    print(
        "[INFO]: Viewpoints sampled and rendered will continue to render the environment and visualize the last camera"
        " positions..."
    )

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # Simulation loop
    while simulation_app.is_running():
        # Perform step
        sim.render()
        # Update buffers
        explorer.scene.update(sim_dt)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
