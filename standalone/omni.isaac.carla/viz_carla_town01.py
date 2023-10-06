"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      Visualize Carla Dataset with Scale and Orientation defined in omni.isaac.carla.configs.configs.py
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

import cv2
import matplotlib as mpl

# python
import numpy as np

# isaac-debug
import omni.isaac.debug_draw._debug_draw as omni_debug_draw

# isaac-anymal
from omni.isaac.anymal.config import VIPlannerCfg

# isaac-carla
from omni.isaac.carla.configs import CarlaExplorerConfig, CarlaLoaderConfig
from omni.isaac.carla.scripts import CarlaExplorer, CarlaLoader

# isaac-orbit
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.orbit.sensors.camera import Camera, PinholeCameraCfg
from viplanner.config import TrainCfg, VIPlannerSemMetaHandler

viplanner_sem_meta = VIPlannerSemMetaHandler()


def sem_color_transfer(sem_image: np.ndarray, sem_idToLabels: dict) -> np.ndarray:
    """Convert semantic segmentation image to viplanner color space

    Args:
        sem_image (np.ndarray): sem_image as received by the simulation
        sem_idToLabels (dict): information about which class is which index in sem_image

    Returns:
        np.ndarray: sem_image in viplanner color space
    """
    for k, v in sem_idToLabels.items():
        if not dict(v):
            sem_idToLabels[k] = {"class": "static"}
        elif "BACKGROUND" == v["class"]:
            sem_idToLabels[k] = {"class": "static"}
        elif "UNLABELLED" == v["class"]:
            sem_idToLabels[k] = {"class": "static"}

    # color mapping
    sem_idToColor = np.array(
        [
            [
                int(k),
                viplanner_sem_meta.class_color[v["class"]][0],
                viplanner_sem_meta.class_color[v["class"]][1],
                viplanner_sem_meta.class_color[v["class"]][2],
            ]
            for k, v in sem_idToLabels.items()
        ]
    )

    # order colors by their id and necessary to account for missing indices (not guaranteed to be consecutive)
    sem_idToColorMap = np.zeros((max(sem_idToColor[:, 0]) + 1, 3), dtype=np.uint8)
    for cls_color in sem_idToColor:
        sem_idToColorMap[cls_color[0]] = cls_color[1:]
    # colorize semantic image
    try:
        sem_image = sem_idToColorMap[sem_image.reshape(-1)].reshape(sem_image.shape + (3,))
    except IndexError:
        print("IndexError: Semantic image contains unknown labels")
        return

    return sem_image


if __name__ == "__main__":
    cfg_loader = CarlaLoaderConfig()
    cfg_explorer = CarlaExplorerConfig()

    # Load Carla Scene
    loader = CarlaLoader(cfg_loader)
    loader.load()

    # get SimulationContext
    sim = SimulationContext.instance()

    # view on town
    rgb_view_paper = False
    if rgb_view_paper:
        camera_cfg = PinholeCameraCfg(
            width=1920,
            height=1080,
            usd_params=PinholeCameraCfg.UsdCameraCfg(
                focal_length=10.0,
            ),
        )
        camera = Camera(camera_cfg)
        camera.spawn("/World/camera")
        camera.initialize()
        camera.set_world_pose_from_view(eye=[150, 75, 65], target=[200, 150, 0])
        sim.reset()
        sim.play()
        for i in range(10):
            sim.step()
            sim.render()

        camera.update(0.0)
        img_array = camera.data.output["rgb"]
        import PIL

        img = PIL.Image.fromarray(img_array[:, :, :3])
        img.save("/home/pascal/viplanner/imperative_learning/data/town01_cam_mount_train/town01_rgb.pdf", quality=100)

    # record camera trajectory paper
    record_camera_trajectory = True
    if record_camera_trajectory:
        eye = [126, 100, 1]
        target_0 = [126, 84, 1]
        target_1 = [115, 73, 1]
        step_size = 0.05
        sensor_save_paths = "/home/pascal/viplanner/paper/video_motivation"
        depth_scale = 1000.0

        os.makedirs(sensor_save_paths, exist_ok=True)
        os.makedirs(os.path.join(sensor_save_paths, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(sensor_save_paths, "semantic_segmentation"), exist_ok=True)
        os.makedirs(os.path.join(sensor_save_paths, "distance_to_image_plane"), exist_ok=True)

        camera_cfg = PinholeCameraCfg(
            width=1920,
            height=1080,
            usd_params=PinholeCameraCfg.UsdCameraCfg(
                focal_length=10.0,
            ),
            data_types=["rgb", "semantic_segmentation", "distance_to_image_plane"],
        )
        camera = Camera(camera_cfg)
        camera.spawn("/World/camera")
        camera.initialize()
        camera.set_world_pose_from_view(eye=eye, target=target_0)
        sim.reset()
        sim.play()

        needed_steps = 0
        for curr_step in range(int((eye[1] - target_1[1]) / step_size)):
            if np.array(eye[1]) - np.array([curr_step * step_size]) > target_0[1]:
                target = target_0
                curr_eye = np.array(eye) - np.array([0, curr_step * step_size, 0])
                needed_steps += 1
            else:
                target = target_1
                curr_eye = np.array(eye) - np.array([(curr_step - needed_steps) * step_size, curr_step * step_size, 0])
            camera.set_world_pose_from_view(eye=curr_eye.tolist(), target=target)
            for i in range(5):
                sim.render()
            camera.update(0.0)

            # collect image and transfer it to viplanner color space
            sem_image = camera.data.output["semantic_segmentation"]["data"]
            sem_idToLabels = camera.data.output["semantic_segmentation"]["info"]["idToLabels"]
            data_array = sem_color_transfer(sem_image, sem_idToLabels)
            # overlaid rgb and semantics
            rgb_sem_array = cv2.addWeighted(
                camera.data.output["rgb"].astype(np.uint8)[:, :, :3],
                0.6,
                data_array.astype(np.uint8),
                0.4,
                0,
            )
            # save overlaid images
            assert cv2.imwrite(
                os.path.join(
                    os.path.join(sensor_save_paths, "semantic_segmentation"),
                    "overlayed_step" + f"{curr_step}".zfill(5) + ".png",
                ),
                cv2.cvtColor(rgb_sem_array, cv2.COLOR_BGR2RGB),
            )

            assert cv2.imwrite(
                os.path.join(os.path.join(sensor_save_paths, "rgb"), "rgb_step" + f"{curr_step}".zfill(5) + ".png"),
                cv2.cvtColor(camera.data.output["rgb"].astype(np.uint8), cv2.COLOR_BGR2RGB),
            )
            depth_array = camera.data.output["distance_to_image_plane"]
            depth_array = np.clip(depth_array, 0, 15)
            assert cv2.imwrite(
                os.path.join(
                    os.path.join(sensor_save_paths, "distance_to_image_plane"),
                    "depth_step" + f"{curr_step}".zfill(5) + ".png",
                ),
                (depth_array * depth_scale).astype(np.uint16),
            )

    # show trajectories
    show_trajectories = False
    scale_waypoints = False
    repeated_trajectories: Optional[int] = 5
    if show_trajectories:
        model_viplanner = "/home/pascal/viplanner/imperative_learning/models/plannernet_env2azQ1b91cZZ_cam_mount_ep100_inputDepSem_costSem_optimSGD_new_cam_mount_combi_lossWidthMod_wgoal4.0_warehouse/eval_Town01_Opt_paper"
        model_iplanner = (
            "/home/pascal/viplanner/imperative_learning/code/iPlanner/iplanner/models/eval_Town01_Opt_paper"
        )
        path_files = [file for file in os.listdir(model_viplanner + "/repeat_0") if file.startswith("waypoint")]
        path_files.sort()
        waypoint_file = "/home/pascal/viplanner/imperative_learning/data/waypoints/crosswalk_paper_changed.json"  # crosswalk_paper_extended_5.json

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
        if scale_waypoints:
            waypoint_array[:, 2] = mean_height / cfg_loader.scale
            waypoint_end[2] = mean_height / cfg_loader.scale
        else:
            waypoint_array[:, 2] = mean_height
            waypoint_end[2] = mean_height
        waypoints["waypoints"] = waypoint_array.tolist()
        waypoints["end"] = waypoint_end.tolist()

        # draw waypoints
        if scale_waypoints:
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
        else:
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
            if scale_waypoints:
                waypoint = (np.array(waypoint) * cfg_loader.scale).tolist()
            draw_interface.draw_lines_spline(
                [tuple(circ_curr) for circ_curr in (circ_coord + waypoint).tolist()],
                (1.0, 0.0, 0.0, 1.0),
                (5),
                False,
            )
        if scale_waypoints:
            endpoint = (np.array(waypoints["end"]) * cfg_loader.scale).tolist()
        else:
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
        size = [10.0]

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
                clipping_range=(22, 10000.0),
            ),
        )
        camera = Camera(camera_cfg)
        camera.spawn("/World/camera", translation=[130, 96, 55], orientation=[0.7071068, 0, 0, 0.7071068])
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
        img.save(f"{cfg_loader.root_path}/town01_trajectories_new.pdf", quality=100)
    sim.play()
    for count in range(100000):
        sim.step()
    sim.pause()

    launcher.close()

# EoF
