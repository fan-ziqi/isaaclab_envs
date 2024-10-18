# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from omni.isaac.lab.sensors.ray_caster import RayCasterData


class MatterportRayCasterData(RayCasterData):
    ray_class_ids: torch.Tensor = None
    """The class ids for each ray hit."""
