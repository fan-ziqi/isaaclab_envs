#!/usr/bin/python
"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

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
import matplotlib as mpl

# python
import numpy as np
from typing import Optional

# isaac-debug
import omni.isaac.debug_draw._debug_draw as omni_debug_draw
from omni.isaac.anymal.config import VIPlannerCfg

# isaac-carla
from omni.isaac.matterport.config import MatterportConfig
from omni.isaac.matterport.utils import MatterportWorld

# isaac-orbit
from omni.isaac.core.simulation_context import SimulationContext

from omni.isaac.orbit.sensors.camera import Camera, PinholeCameraCfg


if __name__ == "__main__":
    config_2n8kARJN3HM = MatterportConfig(
        import_file_obj="/home/pascal/viplanner/env/matterport/v1/scans/2n8kARJN3HM/2n8kARJN3HM/matterport_mesh/0c334eaabb844eaaad049cbbb2e0a4f2/0c334eaabb844eaaad049cbbb2e0a4f2.obj",
        import_file_ply="/home/pascal/viplanner/env/matterport/v1/scans/2n8kARJN3HM/2n8kARJN3HM/house_segmentations/2n8kARJN3HM.ply",
        save=True,
        save_path="/home/pascal/viplanner/imperative_learning/data/2n8kARJN3HM_cam_mounted",
    )
    matterport = MatterportWorld(config_2n8kARJN3HM)
    matterport.load_world()

    # get SimulationContext
    sim = SimulationContext.instance()

    model_viplanner = "/home/pascal/viplanner/imperative_learning/models/plannernet_env2azQ1b91cZZ_cam_mount_ep100_inputDepSem_costSem_optimSGD_new_cam_mount_combi_lossWidthMod_wgoal4.0_warehouse/eval_2n8kARJN3HM_repeat"
    model_iplanner = "/home/pascal/viplanner/imperative_learning/code/iPlanner/iplanner/models/eval_2n8kARJN3HM_repeat"
    path_files = [file for file in os.listdir(model_viplanner + "/repeat_0") if file.startswith("waypoint")]
    path_files.sort()
    waypoint_file = "/home/pascal/viplanner/imperative_learning/data/waypoints/matterport_paper.json"  # crosswalk_paper_extended_5.json

    # show trajectories
    show_trajectories = True
    scale_waypoints = False
    repeated_trajectories: Optional[int] = 4
    if show_trajectories:
        # Acquire draw interface
        draw_interface = omni_debug_draw.acquire_debug_draw_interface()

        # load waypoints
        waypoints = json.load(open(waypoint_file))
        waypoint_file_name, _ = os.path.splitext(os.path.basename(waypoint_file))

        if repeated_trajectories is not None:
            viplanner_traj = []
            iplanner_traj = []
            viplanner_obs_cost = []
            iplanner_obs_cost = []
            for repeat_idx in range(repeated_trajectories):
                vip_paths = []
                ip_paths = []
                for file in path_files:
                    vip_paths.append(np.load(os.path.join(model_viplanner, f"repeat_{repeat_idx}", file)))
                    ip_paths.append(np.load(os.path.join(model_iplanner, f"repeat_{repeat_idx}", file)))
                viplanner_traj.append(np.concatenate(vip_paths, axis=0))
                iplanner_traj.append(np.concatenate(ip_paths, axis=0))

                viplanner_obs_cost.append(
                    np.load(
                        os.path.join(
                            model_viplanner, f"repeat_{repeat_idx}", f"{waypoint_file_name}_loss_obstacles.npy"
                        )
                    )
                )
                iplanner_obs_cost.append(
                    np.load(
                        os.path.join(model_iplanner, f"repeat_{repeat_idx}", f"{waypoint_file_name}_loss_obstacles.npy")
                    )
                )

            mean_height = np.mean(np.concatenate(viplanner_traj, axis=0)[:, 2])
            vip_mean_obs_cost = np.mean(np.concatenate(viplanner_obs_cost, axis=0))
            iplanner_mean_obs_cost = np.mean(np.concatenate(iplanner_obs_cost, axis=0))
        else:
            vip_paths = []
            ip_paths = []
            for file in path_files:
                vip_paths.append(np.load(os.path.join(model_viplanner, file)))
                ip_paths.append(np.load(os.path.join(model_iplanner, file)))
            viplanner_obs_cost = np.load(os.path.join(model_viplanner, f"{waypoint_file_name}_loss_obstacles.npy"))
            iplanner_obs_cost = np.load(os.path.join(model_iplanner, f"{waypoint_file_name}_loss_obstacles.npy"))
            viplanner_traj = np.concatenate(vip_paths, axis=0)
            iplanner_traj = np.concatenate(ip_paths, axis=0)

            mean_height = np.mean(viplanner_traj[:, 2])
            vip_mean_obs_cost = np.mean(viplanner_obs_cost)
            iplanner_mean_obs_cost = np.mean(iplanner_obs_cost)

        # cost print
        print(
            f"VIP mean obs cost: {vip_mean_obs_cost} | IPL mean obs cost: {iplanner_mean_obs_cost}\n"
            f"Reduction: {100 * (vip_mean_obs_cost - iplanner_mean_obs_cost) / iplanner_mean_obs_cost}%"
        )

        # adjust height of waypoints
        waypoint_array = np.array(waypoints["waypoints"])
        waypoint_end = np.array(waypoints["end"])
        waypoint_array[:, 2] = mean_height
        waypoint_end[2] = mean_height
        waypoints["waypoints"] = waypoint_array.tolist()
        waypoints["end"] = waypoint_end.tolist()

        # draw waypoints
        draw_interface.draw_points([waypoints["start"]], [(1.0, 0.0, 0.0, 1.0)], [(10)])
        draw_interface.draw_points([waypoints["end"]], [(1.0, 0.0, 0.0, 1.0)], [(10)])
        draw_interface.draw_points(
            waypoints["waypoints"],
            [(1.0, 0.0, 0.0, 1.0)] * len(waypoints["waypoints"]),  # blue
            [(10)] * len(waypoints["waypoints"]),
        )

        # draw circle around waypoints
        cfg_vip = VIPlannerCfg()
        circ_coord = np.array(
            [
                [cfg_vip.conv_dist, 0, 0],
                [0, cfg_vip.conv_dist, 0],
                [-cfg_vip.conv_dist, 0, 0],
                [0, -cfg_vip.conv_dist, 0],
                [cfg_vip.conv_dist, 0, 0],
                [0, cfg_vip.conv_dist, 0],
                [-cfg_vip.conv_dist, 0, 0],
                [0, -cfg_vip.conv_dist, 0],
                [cfg_vip.conv_dist, 0, 0],
            ]
        )
        for waypoint in waypoints["waypoints"]:
            draw_interface.draw_lines_spline(
                [tuple(circ_curr) for circ_curr in (circ_coord + waypoint).tolist()],
                (1.0, 0.0, 0.0, 1.0),
                (5),
                False,
            )
        endpoint = waypoints["end"]
        draw_interface.draw_lines_spline(
            [tuple(circ_curr) for circ_curr in (circ_coord + endpoint).tolist()],
            (1.0, 0.0, 0.0, 1.0),
            (5),
            False,
        )

        # iplanner in orange
        n_bins = 100  # Number of bins in the colormap
        vip_cm = mpl.colormaps["autumn"]
        ip_cm = mpl.colormaps["winter"]
        size = [5.0]

        def draw_traj(traj, color_map, size, mean_height):
            # steps = np.arange(0, 1, 1 / len(traj.tolist()[1:]))
            # steps = steps[:-1] if len(steps) > len(traj.tolist()[1:]) else steps
            # colors = color_map(steps)
            colors = color_map * len(traj.tolist()[1:])

            traj[:, 2] = mean_height

            draw_interface.draw_lines(
                traj.tolist()[:-1],
                traj.tolist()[1:],
                colors,
                size * len(traj.tolist()[1:]),
            )

        if isinstance(viplanner_traj, list):
            for curr_traj in viplanner_traj:
                # draw_traj(curr_traj, vip_cm, size, mean_height)
                draw_traj(curr_traj, [[1.0, 0.50, 0, 1.0]], size, mean_height)
            for curr_traj in iplanner_traj:
                # draw_traj(curr_traj, ip_cm, size, mean_height)
                draw_traj(curr_traj, [[0.0, 0.0, 1.0, 1.0]], size, mean_height)
        else:
            draw_traj(viplanner_traj, vip_cm, size, mean_height)
            draw_traj(iplanner_traj, ip_cm, size, mean_height)

        # get image for paper
        camera_cfg = PinholeCameraCfg(
            width=1920,
            height=1080,
            usd_params=PinholeCameraCfg.UsdCameraCfg(
                focal_length=24.0,
                clipping_range=(38, 10000.0),
            ),
        )
        camera = Camera(camera_cfg)
        camera.spawn("/World/camera", translation=[9.5, -8.0, 40], orientation=[0.7071068, 0, 0, 0.7071068])
        camera.initialize()
        sim.reset()
        sim.play()
        for i in range(10):
            sim.step()
            sim.render()

        camera.update(0.0)
        img_array = camera.data.output["rgb"]
        import PIL

        img = PIL.Image.fromarray(img_array[:, :, :3])
        img.save(f"/home/pascal/viplanner/env/matterport/matterport_trajectories.pdf", quality=100)

    sim.play()
    for count in range(100000):
        sim.step()
    sim.pause()

    launcher.close()

# EoF
