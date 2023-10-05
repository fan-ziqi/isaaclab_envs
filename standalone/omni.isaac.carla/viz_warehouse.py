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
from typing import Optional

import matplotlib as mpl

# python
import numpy as np

# isaac-debug
import omni.isaac.debug_draw._debug_draw as omni_debug_draw
from omni.isaac.anymal.config import VIPlannerCfg

# isaac-carla
from omni.isaac.carla.configs import DATA_DIR, CarlaLoaderConfig
from omni.isaac.carla.scripts import CarlaLoader

# isaac-orbit
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.orbit.sensors.camera import Camera, PinholeCameraCfg

if __name__ == "__main__":
    cfg_loader = CarlaLoaderConfig(
        root_path="/home/pascal/viplanner/env/warehouse",
        usd_name="warehouse_multiple_shelves_without_ppl.usd",
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

    model_viplanner = "/home/pascal/viplanner/imperative_learning/models/plannernet_env2azQ1b91cZZ_cam_mount_ep100_inputDepSem_costSem_optimSGD_new_cam_mount_combi_lossWidthMod_wgoal4.0_warehouse/eval_warehouse_multiple_shelves_without_ppl"
    model_iplanner = "/home/pascal/viplanner/imperative_learning/code/iPlanner/iplanner/models/eval_warehouse_multiple_shelves_without_ppl"
    path_files = [file for file in os.listdir(model_viplanner + "/repeat_0") if file.startswith("waypoint")]
    path_files.sort()
    waypoint_file = "/home/pascal/viplanner/imperative_learning/data/waypoints/warehouse_paper.json"

    # Load Carla Scene
    loader = CarlaLoader(cfg_loader)
    loader.load()

    # get SimulationContext
    sim = SimulationContext.instance()

    # show trajectories
    show_trajectories = True
    repeated_trajectories: Optional[int] = 5
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

            # viplanner_traj = np.concatenate(viplanner_traj, axis=0)
            # iplanner_traj = np.concatenate(iplanner_traj, axis=0)
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
        waypoint_array[:, 2] = mean_height
        waypoints["waypoints"] = waypoint_array.tolist()

        # draw waypoints
        draw_interface.draw_points(
            [(np.array(waypoints["start"]) * cfg_loader.scale).tolist()], [(1.0, 0.0, 0.0, 1.0)], [(10)]
        )  # orange
        draw_interface.draw_points(
            [(np.array(waypoints["end"]) * cfg_loader.scale).tolist()], [(1.0, 0.0, 0.0, 1.0)], [(10)]
        )  # green
        draw_interface.draw_points(
            (np.array(waypoints["waypoints"]) * cfg_loader.scale).tolist(),
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
        draw_interface.draw_lines_spline(
            [tuple(circ_curr) for circ_curr in (circ_coord + waypoints["end"]).tolist()],
            (1.0, 0.0, 0.0, 1.0),
            (5),
            False,
        )

        # iplanner in orange
        n_bins = 100  # Number of bins in the colormap
        vip_cm = mpl.colormaps["autumn"]
        ip_cm = mpl.colormaps["winter"]
        # color_viplanner = [(0, 0, 1, 1)]  # blue
        # color_iplanner = [(1, 165 / 255, 0, 1)]  # orange
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
                clipping_range=(25, 10000.0),
            ),
        )
        camera = Camera(camera_cfg)
        camera.spawn("/World/camera", translation=[0, 0, 30], orientation=[0.7071068, 0, 0, 0.7071068])
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
        img.save(f"{cfg_loader.root_path}/warehouse_trajectories.pdf", quality=100)

    sim.play()
    for count in range(100000):
        sim.step()
    sim.pause()

    launcher.close()

# EoF
