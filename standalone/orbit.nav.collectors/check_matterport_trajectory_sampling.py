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
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.sim import SimulationContext
from orbit.nav.collectors.collectors import TrajectorySampling, TrajectorySamplingCfg

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

    cfg = TrajectorySamplingCfg()
    cfg.terrain_analysis.viz_graph = True
    explorer = TrajectorySampling(cfg)
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # sample viewpoints
    explorer.sample_paths([1000], [0.0], [10.0])

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # get default cube positions
    default_cube_pose = explorer.scene.rigid_objects["cube"].data.default_root_state
    # Simulation loop
    while simulation_app.is_running():
        # set cube position
        explorer.scene.rigid_objects["cube"].write_root_state_to_sim(default_cube_pose)
        explorer.scene.write_data_to_sim()
        # Perform step
        sim.step()
        # Update buffers
        explorer.scene.update(sim_dt)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
