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
from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the camera sensor.")
parser.add_argument("--headless", action="store_true", default=True, help="Force display off at all times.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""
import omni.isaac.orbit.sim as sim_utils
from omni.isaac.orbit.sim import SimulationContext

from omni.isaac.matterport.exploration import ViewpointSampling, ViewpointSamplingCfg

"""
Main
"""


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)

    cfg = ViewpointSamplingCfg()
    explorer = ViewpointSampling(cfg)
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # sample and render viewpoints
    samples = explorer.sample_viewpoints(1879)
    explorer.render_viewpoints(samples)
    
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    # Simulation loop
    while simulation_app.is_running():
        # Perform step
        sim.step()
        # Update buffers
        explorer.scene.update(sim_dt)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()