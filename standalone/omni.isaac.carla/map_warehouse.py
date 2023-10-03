#!/usr/bin/python
"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      Dense Mapping of Warehouse Datasets
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
parser.add_argument("--headless", action="store_false", default=False, help="Force display off at all times.")
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
        root_path="/home/pascal/viplanner/env/warehouse",
        usd_name="warehouse_multiple_shelves_without_ppl_ext_sem_space.usd",
        suffix="",
        prim_path="/World/Warehouse",
        scale=1.0,
        axis_up="Z",
        cw_config_file=None,
        sem_mesh_to_class_map=os.path.join(DATA_DIR, "warehouse", "keyword_mapping.yml"),
        groundplane=False,
        people_config_file=os.path.join(DATA_DIR, "warehouse", "people_cfg.yml"),
        vehicle_config_file=None,
    )
    cfg_explorer = CarlaExplorerConfig(
        max_cam_recordings=500,
        space_limiter="sm_wall",
        indoor_filter=False,
        output_dir_name="warehouse_multiple_shelves_without_ppl_ext_sem_space",
        nb_more_people=0,
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
