# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from .exploration_cfg import ExplorationCfg


@configclass
class TrajectorySamplingCfg(ExplorationCfg):
    """Configuration for the trajectory sampling."""

    # sampling
    sample_points: int = 10000
    """Number of random points to sample."""
    x_angle_range: tuple[float, float] = (-2.5, 2.5)
    y_angle_range: tuple[float, float] = (-2, 5)  # negative angle means in isaac convention: look down
    """Range of the x and y angle of the camera (in degrees), will be randomly selected according to a uniform distribution"""
    height: float = 0.5
    """Height to use for the random points."""

    # SAVING
    max_images: int = 2000
    """Maximum number of images recorded"""
    save_path: str = "/home/pascal/viplanner/imperative_learning/data"
    suffix: str | None = None
    """Path to save the data to (directly with env name will be created)"""
