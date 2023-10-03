#!/usr/bin/python
"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      Dense Mapping of Carla Datasets
"""


"""
Launch Omniverse Toolkit first.
"""

# python
import argparse
import os

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
from omni.isaac.carla.configs import DATA_DIR, CarlaExplorerConfig, CarlaLoaderConfig
from omni.isaac.carla.scripts import CarlaExplorer, CarlaLoader

if __name__ == "__main__":
    cfg_loader = CarlaLoaderConfig(
        # carla map
        root_path="/home/pascal/viplanner/env/carla/town02",
        usd_name="Town02.usd",
        suffix="/Town02",
        # prim path for the carla map
        prim_path="/World/Town02",
        # multipy crosswalks
        cw_config_file=os.path.join(DATA_DIR, "town02", "cw_multiply_cfg.yml"),
        # mesh to semantic class mapping
        sem_mesh_to_class_map=os.path.join(DATA_DIR, "town02", "keyword_mapping.yml"),
        # multiply vehicles
        vehicle_config_file=os.path.join(DATA_DIR, "town02", "vehicle_cfg.yml"),
    )
    cfg_explorer = CarlaExplorerConfig(
        carla_filter=None,
        output_dir_name="town02",
    )

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
