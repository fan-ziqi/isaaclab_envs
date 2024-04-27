# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
@author     Pascal Roth
@email      rothpa@ethz.ch

@brief      Dense Mapping of Carla Datasets
"""


"""
Launch Omniverse Toolkit first.
"""

# python
import argparse

# orbit
import carb

# omni-isaac-orbit
from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_false", default=True, help="Force display off at all times.")
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
launcher = SimulationApp(config)


"""
Rest everything follows.
"""


# isaac-carla
from omni.isaac.carla.configs import CarlaExplorerConfig, CarlaLoaderConfig
from omni.isaac.carla.scripts import CarlaExplorer, CarlaLoader

if __name__ == "__main__":
    cfg_loader = CarlaLoaderConfig()
    cfg_explorer = CarlaExplorerConfig()

    if cfg_loader.groundplane:
        carb.log_warn(
            "Groundplane is enabled. This will cause issues with the semantic segmentation, will be set to False."
        )
        cfg_loader.groundplane = False

    # Load Carla Scene
    loader = CarlaLoader(cfg_loader)
    loader.load()

    # Explore Carla
    explorer = CarlaExplorer(cfg_explorer, cfg_loader)
    explorer.explore()

    # Close the simulator
    launcher.close()

# EOF
