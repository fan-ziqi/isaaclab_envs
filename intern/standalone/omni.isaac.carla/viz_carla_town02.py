# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
@author     Pascal Roth
@email      rothpa@ethz.ch

@brief      Vizualize Carla Dataset with Scale and Orientation defined in omni.isaac.carla.configs.configs.py
"""


"""
Launch Omniverse Toolkit first.
"""

# python
import argparse
import os

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

import json
from typing import Optional

# python
import numpy as np

# isaac-debug
import omni.isaac.debug_draw._debug_draw as omni_debug_draw

# isaac-carla
from omni.isaac.carla.configs import DATA_DIR, CarlaExplorerConfig, CarlaLoaderConfig
from omni.isaac.carla.scripts import CarlaExplorer, CarlaLoader

# isaac-orbit
from omni.isaac.core.simulation_context import SimulationContext

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
    cfg_explorer = CarlaExplorerConfig()

    # Load Carla Scene
    loader = CarlaLoader(cfg_loader)
    loader.load()

    # get SimulationContext
    sim = SimulationContext.instance()

    # show trajectories
    show_trajectories = False
    repeated_trajectories: int | None = 1
    if show_trajectories:
        # Acquire draw interface
        draw_interface = omni_debug_draw.acquire_debug_draw_interface()

        model_viplanner = "/home/pascal/viplanner/imperative_learning/models/plannernet_env2azQ1b91cZZ_cam_mount_ep100_inputDepSem_costSem_optimSGD_new_cam_mount_combi_lossWidthMod_wgoal4.0_warehouse/eval_Town01_Opt"
        model_iplanner = "/home/pascal/viplanner/imperative_learning/models/plannernet_env2azQ1b91cZZ_ep100_inputDep_costSem_optimSGD_depth_carla/eval_Town01_Opt"
        path_files = [
            "waypoint0_path.npy",
            "waypoint1_path.npy",
            "waypoint2_path.npy",
            "waypoint3_path.npy",
            "waypoint4_path.npy",
        ]  # ['cw1_waypoint0_path.npy', 'cw1_waypoint1_path.npy', 'cw1_waypoint2_path.npy']
        # path_files = ['cw2_waypoint0_path.npy']
        waypoint_file = "/home/pascal/viplanner/imperative_learning/data/waypoints/crosswalk_paper_extended_2.json"

        waypoints = json.load(open(waypoint_file))

        # apply scale
        waypoints["start"] = [x for x in waypoints["start"]]
        waypoints["end"] = [x for x in waypoints["end"]]
        waypoints["waypoints"] = [[x for x in waypoint] for waypoint in waypoints["waypoints"]]

        # draw waypoints
        draw_interface.draw_points(
            [(np.array(waypoints["start"]) * 0.01).tolist()], [(1.0, 0.0, 0.0, 1.0)], [(10)]
        )  # orange
        draw_interface.draw_points(
            [(np.array(waypoints["end"]) * 0.01).tolist()], [(1.0, 0.0, 0.0, 1.0)], [(10)]
        )  # green
        draw_interface.draw_points(
            (np.array(waypoints["waypoints"]) * 0.01).tolist(),
            [(1.0, 0.0, 0.0, 1.0)] * len(waypoints["waypoints"]),  # blue
            [(10)] * len(waypoints["waypoints"]),
        )

        if repeated_trajectories is not None:
            viplanner_traj = []
            for repeat_idx in range(repeated_trajectories):
                paths = []
                for file in path_files:
                    paths.append(np.load(os.path.join(model_viplanner, f"repeat_{repeat_idx}", file)))
                viplanner_traj.append(np.concatenate(paths, axis=0))
        else:
            paths = []
            for file in path_files:
                paths.append(np.load(os.path.join(model_viplanner, file)))
            viplanner_traj = np.concatenate(paths, axis=0)

        paths = []
        for file in ["waypoint0_path.npy", "waypoint1_path.npy", "waypoint2_path.npy"]:  # path_files:
            paths.append(np.load(os.path.join(model_iplanner, file)))
        iplanner_traj = np.concatenate(paths, axis=0)

        # iplanner in orange
        color_viplanner = [(0, 0, 1, 1)]  # blue
        color_iplanner = [(1, 165 / 255, 0, 1)]  # orange
        size = [5.0]

        if isinstance(viplanner_traj, list):
            for curr_traj in viplanner_traj:
                curr_traj[:, 2] = 0.5
                draw_interface.draw_lines(
                    curr_traj.tolist()[:-1],
                    curr_traj.tolist()[1:],
                    color_viplanner * len(curr_traj.tolist()[1:]),
                    size * len(curr_traj.tolist()[1:]),
                )
        else:
            viplanner_traj[:, 2] = 0.5
            draw_interface.draw_lines(
                viplanner_traj.tolist()[:-1],
                viplanner_traj.tolist()[1:],
                color_viplanner * len(viplanner_traj.tolist()[1:]),
                size * len(viplanner_traj.tolist()[1:]),
            )
        draw_interface.draw_lines(
            iplanner_traj.tolist()[:-1],
            iplanner_traj.tolist()[1:],
            color_iplanner * len(iplanner_traj.tolist()[1:]),
            size * len(iplanner_traj.tolist()[1:]),
        )

    sim.play()
    for count in range(100000):
        sim.step()
    sim.pause()

    launcher.close()

# EoF
