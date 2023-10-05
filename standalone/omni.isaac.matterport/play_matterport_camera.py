# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
This script shows how to use the camera sensor from the Orbit framework.

The camera sensor is created and interfaced through the Omniverse Replicator API. However, instead of using
the simulator or OpenGL convention for the camera, we use the robotics or ROS convention.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

# omni-isaac-orbit
from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the camera sensor.")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--gpu", action="store_true", default=False, help="Use GPU device for camera rendering output.")
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import math
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use("Qt5Agg")

import omni.isaac.core.utils.prims as prim_utils
from omni.isaac.core.utils.viewports import set_camera_view

from omni.isaac.orbit.sensors.ray_caster import RayCasterCfg, patterns
from omni.isaac.orbit.sim import SimulationContext, SimulationCfg
from omni.isaac.matterport.config import MatterportImporterCfg
from omni.isaac.orbit.sensors.camera import Camera, CameraCfg
from omni.isaac.orbit.sim.spawners import PinholeCameraCfg
from omni.isaac.matterport.domains.matterport_raycast_camera import MatterportRayCasterCamera


def main():
    """Runs a camera sensor from orbit."""

    # Load kit helper
    sim = SimulationContext(SimulationCfg())
    # Set main camera
    set_camera_view([2.5, 2.5, 3.5], [0.0, 0.0, 0.0])
    # load matterport secene
    importer_cfg = MatterportImporterCfg(
        obj_filepath="/home/pascal/viplanner/env/matterport/v1/scans/2n8kARJN3HM/2n8kARJN3HM/matterport_mesh/0c334eaabb844eaaad049cbbb2e0a4f2/0c334eaabb844eaaad049cbbb2e0a4f2.obj"
    )
    matterport = importer_cfg.cls_name(importer_cfg)
    sim.reset()
    sim.pause()

    # Setup Matterport Camera
    camera_pattern_cfg = patterns.PinholeCameraPatternCfg(
        focal_length=24.0,
        horizontal_aperture=20.955,
        height=480,
        width=640,
        data_types=[
            "distance_to_image_plane",
            "semantic_segmentation",
        ],
    )
    ply_filepath = "/home/pascal/viplanner/env/matterport/v1/scans/2n8kARJN3HM/2n8kARJN3HM/house_segmentations/2n8kARJN3HM.ply"
    camera_cfg = RayCasterCfg(
        prim_path="/World/Camera",
        mesh_prim_paths=[ply_filepath],
        update_period=0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        debug_vis=False,
        pattern_cfg=camera_pattern_cfg,
        max_distance=15,
    )
    # create xform because placement of camera directly under world is not supported
    prim_utils.create_prim("/World/Camera", "Xform")
    # Create camera
    camera = MatterportRayCasterCamera(cfg=camera_cfg)

    # Setup RGB Camera
    rgb_camera_cfg = CameraCfg(
        prim_path="/World/Camera_rgb",
        data_types=[
            "rgb",
        ],
        spawn=PinholeCameraCfg(),
        width=640,
        height=480,
    )
    rgb_camera = Camera(rgb_camera_cfg)

    # Play simulator
    sim.play()

    figures = {}
    # init semantics figure
    fg_sem = plt.figure()
    ax_sem = fg_sem.gca()
    ax_sem.set_title("Semantic Segmentation")
    img_sem = ax_sem.imshow(camera.data.output["semantic_segmentation"][0].cpu().numpy())
    figures["semantics"] = {"fig": fg_sem, "axis": ax_sem, "img": img_sem}

    # init depth figure
    n_bins = 500  # Number of bins in the colormap
    colors = mpl.colormaps["jet"](np.linspace(0, 1, n_bins))  # Colormap

    def convert_depth_to_color(depth_img):
        depth_img = depth_img.cpu().numpy()
        depth_img_flattend = np.clip(depth_img.flatten(), a_min=0, a_max=15)
        depth_img_flattend = np.round(depth_img_flattend / 15 * (n_bins - 1)).astype(np.int32)
        depth_colors = colors[depth_img_flattend]
        depth_colors = depth_colors.reshape(depth_img.shape[0], depth_img.shape[1], 4)
        return depth_colors

    fg_depth = plt.figure()
    ax_depth = fg_depth.gca()
    ax_depth.set_title("Distance To Image Plane")
    img_depth = ax_depth.imshow(convert_depth_to_color(camera.data.output["distance_to_image_plane"][0]))
    figures["depth"] = {"fig": fg_depth, "axis": ax_depth, "img": img_depth}
    
    plt.ion()

    # Simulate physics
    while simulation_app.is_running():
        # Set pose
        eyes = torch.tensor([[5, -10, 1]], device=camera.device)  # [[2.5, 2.5, 2.5]]
        targets = torch.tensor(
            [[5 + math.cos(2 * math.pi * camera.frame[0] / 1000), -10 + math.sin(2 * math.pi * camera.frame[0] / 1000), 0.0]],
            device=camera.device
        )
        camera.set_world_poses_from_view(eyes, targets)
        rgb_camera.set_world_poses_from_view(eyes, targets)

        # Step simulation
        sim.step(render=app_launcher.RENDER)
        for i in range(5):
            sim.render()
        # Update camera data
        camera.update(dt=sim.get_physics_dt())
        rgb_camera.update(dt=sim.get_physics_dt())

        """
        Updates the visualization plane.
        """
        figures["semantics"]["img"].set_array(camera.data.output["semantic_segmentation"][0].cpu().numpy())
        figures["semantics"]["fig"].canvas.draw()
        figures["semantics"]["fig"].canvas.flush_events()

        figures["depth"]["img"].set_array(convert_depth_to_color(camera.data.output["distance_to_image_plane"][0]))
        figures["depth"]["fig"].canvas.draw()
        figures["depth"]["fig"].canvas.flush_events()

        plt.pause(0.000001)


if __name__ == "__main__":
    # Runs the main function
    main()
    # Close the simulator
    simulation_app.close()
