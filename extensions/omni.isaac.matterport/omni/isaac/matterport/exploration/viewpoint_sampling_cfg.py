# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.orbit.utils import configclass

from .exploration_cfg import ExplorationCfg


@configclass
class ViewpointSamplingCfg(ExplorationCfg):
    """Configuration for the viewpoint sampling."""

    # dict of cameras and corresponding annotators
    cameras: dict[str, str] = {
        "camera_0": "semantic_segmentation",
        "camera_1": "distance_to_image_plane",
    }
    """Dict of cameras and corresponding annotators to use for the viewpoint sampling."""
    depth_scale: float = 1000.0
    """Scaling factor for the depth values."""

    # sampling
    sample_points: int = 10000
    """Number of random points to sample."""
    x_angle_range: tuple[float, float] = (-2.5, 2.5)
    y_angle_range: tuple[float, float] = (-2, 5)  # negative angle means in isaac convention: look down
    """Range of the x and y angle of the camera (in degrees), will be randomly selected according to a uniform distribution"""
    height: float = 0.5
    """Height to use for the random points."""

    # point filtering
    min_height: float = 0.2
    """Maximum height to be considered an accessible point for the robot"""
    ground_height: float = -0.1
    """Height of the ground plane"""
    min_hit_rate: float = 0.8
    """Don't use a point if the hit rate is below this value"""
    min_avg_hit_distance: float = 0.5
    """Don't use a point if the max hit distance is below this value"""
    min_std_hit_distance: float = 0.5
    """Don't use a point if the std hit distance is below this value"""

    # convergence
    conv_rate: float = 0.9
    """Rate of faces that are covered by three different images, used to terminate the exploration"""

    # SAVING
    save_path: str | None = "/home/pascal/viplanner/imperative_learning/data/temp"
    """Directory to save the viewpoint samples, camera intrinsics and rendered images to.

    If None, the directory is the same as the one of the obj file. Default is None."""

    # debug
    debug_viz: bool = True
    """Whether to visualize the sampled points and orientations."""
