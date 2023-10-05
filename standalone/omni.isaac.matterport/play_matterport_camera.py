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

import os
import math
import torch

import omni.isaac.core.utils.prims as prim_utils
import omni.replicator.core as rep
from omni.isaac.core.utils.viewports import set_camera_view

from omni.isaac.orbit.sensors.ray_caster import RayCasterCfg, patterns
from omni.isaac.orbit.utils import convert_dict_to_backend
from omni.isaac.orbit.sim import SimulationContext, SimulationCfg
from omni.isaac.matterport.config import MatterportImporterCfg
from .matterport_raycast_camera import MatterportRayCasterCamera


def main():
    """Runs a camera sensor from orbit."""

    # Load kit helper
    sim = SimulationContext(SimulationCfg())
    # Set main camera
    set_camera_view([2.5, 2.5, 3.5], [0.0, 0.0, 0.0])
    # load matterport secene
    importer_cfg = MatterportImporterCfg(
        import_file_obj =  "/home/pascal/viplanner/env/matterport/v1/scans/2n8kARJN3HM/2n8kARJN3HM/matterport_mesh/0c334eaabb844eaaad049cbbb2e0a4f2/0c334eaabb844eaaad049cbbb2e0a4f2.obj",    )
    matterport = importer_cfg.cls_name(importer_cfg)
    sim.reset()
    sim.pause()
    # Setup camera sensor
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
        mesh_prim_paths=ply_filepath,
        update_period=0,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        debug_vis=True,
        pattern_cfg=camera_pattern_cfg,
    )
    # create xform because placement of camera directly under world is not supported
    prim_utils.create_prim("/World/Camera", "Xform")
    # Create camera
    camera = MatterportRayCasterCamera(cfg=camera_cfg)

    # Create replicator writer
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output_matterport", "camera")
    rep_writer = rep.BasicWriter(output_dir=output_dir, frame_padding=3)

    # Play simulator
    sim.play()

    # Simulate physics
    while simulation_app.is_running():
        # Set pose
        eyes = torch.tensor([[0, 0, 2]])  # [[2.5, 2.5, 2.5]]
        targets = torch.tensor([[math.cos(2*math.pi * camera.frame[0] / 1000), math.sin(2*math.pi * camera.frame[0] / 1000), 0.0]])
        camera.set_world_poses_from_view(eyes, targets)
        
        # Step simulation
        sim.step(render=app_launcher.RENDER)
        for i in range(5):
            sim.render()
        # Update camera data
        camera.update(dt=sim.get_physics_dt())

        # Print camera info
        print(camera)
        print("Received shape of depth image: ", camera.data.output["distance_to_image_plane"].shape)
        print("-------------------------------")

        # Extract camera data
        camera_index = 0
        # note: BasicWriter only supports saving data in numpy format, so we need to convert the data to numpy.
        if sim.backend == "torch":
            # tensordict allows easy indexing of tensors in the dictionary
            single_cam_data = convert_dict_to_backend(camera.data.output[camera_index], backend="numpy")
        else:
            # for numpy, we need to manually index the data
            single_cam_data = dict()
            for key, value in camera.data.output.items():
                single_cam_data[key] = value[camera_index]
        # Extract the other information
        single_cam_info = camera.data.info[camera_index]

        # Pack data back into replicator format to save them using its writer
        rep_output = dict()
        for key, data, info in zip(single_cam_data.keys(), single_cam_data.values(), single_cam_info.values()):
            if info is not None:
                rep_output[key] = {"data": data, "info": info}
            else:
                rep_output[key] = data
        # Save images
        rep_output["trigger_outputs"] = {"on_time": camera.frame[camera_index]}
        rep_writer.write(rep_output)

if __name__ == "__main__":
    # Runs the main function
    main()
    # Close the simulator
    simulation_app.close()
