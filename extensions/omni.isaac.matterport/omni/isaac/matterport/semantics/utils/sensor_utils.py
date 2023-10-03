import numpy as np
import torch

# solves circular imports of LeggedRobot
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .sensors_cfg import BpearlPatternCfg, GridPatternCfg, RealSensePatternCfg


def grid_pattern(pattern_cfg: "GridPatternCfg", device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """A regular grid pattern for ray casting.

    Args:
        pattern_cfg (GridPatternCfg): The config for the pattern.
        device (str): The device
    Returns:
        ray_starts (torch.Tensor): The starting positions of the rays
        ray_directions (torch.Tensor): The ray directions

    """
    y = torch.arange(
        start=-pattern_cfg.width / 2, end=pattern_cfg.width / 2 + 1.0e-9, step=pattern_cfg.resolution, device=device
    )
    x = torch.arange(
        start=-pattern_cfg.length / 2, end=pattern_cfg.length / 2 + 1.0e-9, step=pattern_cfg.resolution, device=device
    )
    grid_x, grid_y = torch.meshgrid(x, y)

    num_rays = grid_x.numel()
    ray_starts = torch.zeros(num_rays, 3, device=device)
    ray_starts[:, 0] = grid_x.flatten()
    ray_starts[:, 1] = grid_y.flatten()

    ray_directions = torch.zeros_like(ray_starts)
    ray_directions[..., :] = torch.tensor(list(pattern_cfg.direction), device=device)
    return ray_starts, ray_directions


def bpearl_pattern(pattern_cfg: "BpearlPatternCfg", device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """The Bpearl pattern for ray casting.

    Args:
        pattern_cfg (BpearlPatternCfg): The config for the pattern.
        device (str): The device
    Returns:
        ray_starts (torch.Tensor): The starting positions of the rays
        ray_directions (torch.Tensor): The ray directions

    """
    h = torch.arange(
        -pattern_cfg.horizontal_fov / 2, pattern_cfg.horizontal_fov / 2, pattern_cfg.horizontal_res, device=device
    )
    v = torch.tensor(list(pattern_cfg.vertical_ray_angles), device=device)

    pitch, yaw = torch.meshgrid(v, h, indexing="xy")
    pitch, yaw = torch.deg2rad(pitch.reshape(-1)), torch.deg2rad(yaw.reshape(-1))
    pitch += torch.pi / 2
    x = torch.sin(pitch) * torch.cos(yaw)
    y = torch.sin(pitch) * torch.sin(yaw)
    z = torch.cos(pitch)

    ray_directions = -torch.stack([x, y, z], dim=1)
    ray_starts = torch.zeros_like(ray_directions)
    return ray_starts, ray_directions


def realsense_pattern(pattern_cfg: "RealSensePatternCfg", device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """The RealSense pattern for ray casting.
    Args:
        pattern_cfg (RealSensePatternCfg): The config for the pattern.
        device (str): The device
    Returns:
        ray_starts (torch.Tensor): The starting positions of the rays
        ray_directions (torch.Tensor): The ray directions
    """
    x_grid = torch.full((pattern_cfg.height, pattern_cfg.width), pattern_cfg.far_plane, device=device)
    y_range = np.tan(np.deg2rad(pattern_cfg.horizontal_fov) / 2.0) * pattern_cfg.far_plane
    y = torch.linspace(y_range, -y_range, pattern_cfg.width, device=device)
    z_range = y_range * pattern_cfg.height / pattern_cfg.width
    z = torch.linspace(z_range, -z_range, pattern_cfg.height, device=device)
    y_grid, z_grid = torch.meshgrid(y, z, indexing="xy")
    ray_directions = torch.cat([x_grid.unsqueeze(2), y_grid.unsqueeze(2), z_grid.unsqueeze(2)], dim=2)
    ray_directions = torch.nn.functional.normalize(ray_directions, p=2.0, dim=-1).view(-1, 3)
    ray_starts = torch.zeros_like(ray_directions)
    return ray_starts, ray_directions
