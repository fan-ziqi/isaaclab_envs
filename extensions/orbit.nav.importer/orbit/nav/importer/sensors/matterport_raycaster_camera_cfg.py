# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.sensors.ray_caster import RayCasterCameraCfg
from omni.isaac.lab.utils import configclass

from .matterport_raycaster_camera import MatterportRayCasterCamera


@configclass
class MatterportRayCasterCameraCfg(RayCasterCameraCfg):
    """Configuration for the ray-cast camera for Matterport Environments."""

    class_type = MatterportRayCasterCamera
    """Name of the specific matterport ray caster camera class."""
