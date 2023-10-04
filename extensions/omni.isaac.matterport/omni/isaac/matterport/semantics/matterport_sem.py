#!/usr/bin/python
"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      Load Semantics from Matterport3D and make them available to Isaac Sim
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import scipy.spatial.transform as tf
import torch
import yaml

# Python
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import carb
import cv2

# omni
import omni
import pandas as pd
import trimesh
import warp as wp
from omni.isaac.core.simulation_context import SimulationContext

# omni-isaac-core
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.debug_draw import _debug_draw

# omni-isaac-matterport
from omni.isaac.matterport.config import MatterportConfig
from .matterport_raycast_camera import MatterportRayCasterCamera

try:
    from viplanner.config.viplanner_sem_meta import VIPlannerSemMetaHandler
except ImportError:
    print("VIPlanner Semantic Meta Handler not available.")

from omni.isaac.matterport.utils import DATA_DIR
from pxr import Usd

# omni-isaac-orbit
from omni.isaac.orbit.sensors.ray_caster import RayCasterCfg
from omni.isaac.orbit.sensors.camera import CameraData

matplotlib.use("Qt5Agg")

wp.init()
wp.ScopedTimer(name="Matterport Warp")

OPENGL_TO_ISAAC_MAT = tf.Rotation.from_euler("XYZ", [-90, 0, 90], degrees=True).as_matrix()
ISAAC_TO_OPENGL_MAT = tf.Rotation.from_euler("XYZ", [90, -90, 0], degrees=True).as_matrix()


class MatterportWarp:
    """
    Load Matterport3D Semantics and make them available to Isaac Sim
    """

    def __init__(
        self,
        cfg: MatterportConfig,
    ):
        """
        Initialize MatterportSemWarp

        Args:
            path (str): path to Matterport3D Semantics
        """
        self._cfg: MatterportConfig = cfg

        # setup camera list
        self.cameras: List[MatterportRayCasterCamera] = []

        # setup camera visualization
        self.figures = {}
        return

    ##
    # Public Methods
    ##

    def register_camera(self, cfg: RayCasterCfg):
        """
        Register a camera to the MatterportSemWarp
        """
        # append to camera list
        self.cameras.append(MatterportRayCasterCamera(cfg, self._cfg.importer.import_file_ply))






    async def compute_domains(self, time_step: Optional[int] = None) -> torch.tensor:
        """
        Compute Matterport3D Semantics

        Returns:
            torch.tensor: Matterport3D Semantics
        """
        assert len(self.cameras) > 0, "No camera registered"

        for cam_idx, curr_camera in enumerate(self.cameras):
            curr_camera.data
            # set all inf or nan values to 0
            # TODO: check if this is necessary
            # cam_data.ray_distances[~torch.isfinite(cam_data.ray_distances)] = 0

            if self._cfg.visualize:
                await self._update_visualization_async(cam_data=cam_data)

            if self._cfg.save:
                await self._save_data_async(cam_data=cam_data, idx=time_step, cam_idx=cam_idx)

        return

    ##
    # Private Methods (Helper Functions)
    ##
    callback for sensor update, visualization and safe


    # Visualization helpers ###

    def _init_visualization(self, data: CameraData):
        """Initializes the visualization plane."""
        if "semantic_segmentation" in data.output.keys():  # noqa: SIM118
            height, width = data.output["semantic_segmentation"]
            # init semantics figure
            fg_sem = plt.figure(num=self._fg_idx_max)
            ax_sem = fg_sem.gca()
            ax_sem.set_title(f"Semantic Segmentation ")
            img_sem = ax_sem.imshow(np.zeros((height, width, 3), dtype=np.uint8))
            id_sem = self._fg_idx_max

        if cam_data.depth:
            # init depth figure
            cam_data.fg_depth = plt.figure(num=self._fg_idx_max)
            cam_data.ax_depth = cam_data.fg_depth.gca()
            cam_data.ax_depth.set_title(f"Depth {cam_data.prim.GetPrimPath().pathString}")
            cam_data.img_depth = cam_data.ax_depth.imshow(
                np.zeros((cam_data.height, cam_data.width, 1), dtype=np.float32)
            )
            cam_data.id_depth = self._fg_idx_max + 1

        if any((cam_data.semantic, cam_data.depth, cam_data.rgb)):
            plt.ion()
            # update flag
            self.vis_init = True

        return

    def _update_visualization(self, data: CameraData) -> None:
        """
        Updates the visualization plane.
        """
        if self.vis_init is False:
            self._init_visualization(data)

        # SEMANTICS
        if "semantic_segementation" in data.output.keys():  # noqa: SIM118
            cam_data.img_sem.set_array(cam_data.render_sem)
            cam_data.fg_sem.canvas.draw()
            cam_data.fg_sem.canvas.flush_events()

        # DEPTH
        if self._cfg.depth:
            # cam_data.img_depth.set_array(cam_data.render_depth)
            cam_data.ax_depth.imshow(cam_data.render_depth)
            cam_data.fg_depth.canvas.draw()
            cam_data.fg_depth.canvas.flush_events()

        plt.pause(0.000001)

        return

    # save data helpers ###

    def init_save(self, save_path: Optional[str] = None) -> None:
        if save_path is not None:
            self._cfg.save_path = save_path
        # create directories
        os.makedirs(self._cfg.save_path, exist_ok=True)
        os.makedirs(os.path.join(self._cfg.save_path, "semantics"), exist_ok=True)
        os.makedirs(os.path.join(self._cfg.save_path, "depth"), exist_ok=True)
        os.makedirs(os.path.join(self._cfg.save_path, "rgb"), exist_ok=True)

        # save camera configurations
        dim = 12 if self._cfg.ros_p_mat else 9
        intrinsics = np.zeros((len(self.cameras), dim))
        for idx, cam_data in enumerate(self.cameras):
            cam_intrinsics = cam_data.intrinsics.cpu().numpy()
            if self._cfg.ros_p_mat:
                p_mat = np.zeros((3, 4))
                p_mat[:3, :3] = cam_intrinsics
                intrinsics[idx] = p_mat.flatten()
            else:
                intrinsics[idx] = cam_intrinsics.flatten()
        np.savetxt(os.path.join(self._cfg.save_path, "intrinsics.txt"), intrinsics, delimiter=",")

        return

    async def _save_data_async(self, cam_data: CameraData, idx: int, cam_idx: int) -> None:
        self._save_data(cam_data, idx, cam_idx)
        return

    def _save_data(self, cam_data: CameraData, idx: int, cam_idx: int) -> None:
        cam_suffix = f"_cam{cam_idx}" if len(self.cameras) > 1 else ""

        # SEMANTICS
        if cam_data.semantic:
            cv2.imwrite(
                os.path.join(self._cfg.save_path, "semantics", f"{idx}".zfill(4) + cam_suffix + ".png"),
                cv2.cvtColor(cam_data.render_sem.astype(np.uint8), cv2.COLOR_RGB2BGR),
            )

        # DEPTH
        if cam_data.depth:
            cv2.imwrite(
                os.path.join(self._cfg.save_path, "depth", f"{idx}".zfill(4) + cam_suffix + ".png"),
                cam_data.render_depth,
            )

        # RGB
        if cam_data.rgb:
            cv2.imwrite(
                os.path.join(self._cfg.save_path, "rgb", f"{idx}".zfill(4) + cam_suffix + ".png"),
                cv2.cvtColor(cam_data.render_rgb, cv2.COLOR_RGB2BGR),
            )

        # camera pose in robotics frame (x forward, y left, z up)
        rot_quat = tf.Rotation.from_matrix(cam_data.rot.cpu().numpy()).as_quat()  # get quat as (x, y, z, w) format
        pose = np.hstack((cam_data.pos.cpu().numpy(), rot_quat))
        cam_data.poses = np.append(cam_data.poses, pose.reshape(1, -1), axis=0)
        return

    def _end_save(self) -> None:
        # save camera poses
        for idx, cam_data in enumerate(self.cameras):
            np.savetxt(
                os.path.join(self._cfg.save_path, f"camera_extrinsic_cam{idx}.txt"),
                cam_data.poses[1:],
                delimiter=",",
            )
        return


# EoF
