from typing import Callable, Tuple

from legged_gym.common.sensors.sensor_utils import bpearl_pattern, grid_pattern, realsense_pattern
from legged_gym.utils import configclass


@configclass
class GridPatternCfg:
    resolution: float = 0.1
    width: float = 1.0
    length: float = 1.6
    max_xy_drift: float = 0.05
    direction: Tuple = (0.0, 0.0, -1.0)
    pattern_func: Callable = grid_pattern


@configclass
class BpearlPatternCfg:
    horizontal_fov: float = 360.0
    horizontal_res: float = 10.0
    vertical_ray_angles: Tuple = [
        89.5,
        86.6875,
        83.875,
        81.0625,
        78.25,
        75.4375,
        72.625,
        69.8125,
        67.0,
        64.1875,
        61.375,
        58.5625,
        55.75,
        52.9375,
        50.125,
        47.3125,
        44.5,
        41.6875,
        38.875,
        36.0625,
        33.25,
        30.4375,
        27.625,
        24.8125,
        22,
        19.1875,
        16.375,
        13.5625,
        10.75,
        7.9375,
        5.125,
        2.3125,
    ]
    pattern_func: Callable = bpearl_pattern


@configclass
class RealSensePatternCfg:
    horizontal_fov: float = 80
    width: int = 128
    height: int = 128
    far_plane: float = 4.0
    pattern_func: Callable = realsense_pattern


# @configclass
# class RaycasterCfg:
#     class_name: str = "Raycaster"
#     terrain_mesh_name: str = "terrain"
#     robot_name: str = "robot"
#     body_attachement_name: str = "base"
#     attachement_pos: Tuple = (0.0, 0.0, 0.0)
#     attachement_quat: Tuple = (0.0, 0.0, 0.0, 1.0)
#     attach_yaw_only: bool = True  # do not use the roll and pitch of the robot to update the rays
#     default_hit_value: float = -10.0  # which value to return when a ray misses the hit
#     pattern_cfg = GridPatternCfg()


# @configclass
# class DepthCameraCfg:
#     pass


# @configclass
# class ImuCfg:
#     pass


# """ Ready to use sensors"""


# @configclass
# class SensorsCfg:
#     height_scanner = RaycasterCfg(attachement_pos=(0.0, 0.0, 20.0), attach_yaw_only=True, pattern_cfg=GridPatternCfg())


# @configclass
# class AnymalCSensors(SensorsCfg):
#     bpearl_front = RaycasterCfg(
#         attachement_pos=(0.524, 0.0, 0.0),
#         attachement_quat=(-0.271, 0.653, -0.271, 0.653),
#         attach_yaw_only=False,
#         pattern_cfg=BpearlPatternCfg(),
#     )
#     bpearl_rear = RaycasterCfg(
#         attachement_pos=(-0.524, 0.0, 0.0),
#         attachement_quat=(-0.653, -0.271, 0.653, 0.271),
#         attach_yaw_only=False,
#         pattern_cfg=BpearlPatternCfg(),
#     )
#     # TODO add realsenses


# @configclass
# class AnymalDSensors(SensorsCfg):
#     realsense_front = RaycasterCfg(
#         attachement_pos=(0.47, 0, 0.0),
#         attachement_quat=(0.0, 0.13052, 0.0, 0.99144),
#         attach_yaw_only=False,
#         pattern_cfg=RealSensePatternCfg(),
#     )

#     realsense_front_down = RaycasterCfg(
#         attachement_pos=(0.47, 0, -0.05),
#         attachement_quat=(0.0, 0.5, 0.0, 0.866025),
#         attach_yaw_only=False,
#         pattern_cfg=RealSensePatternCfg(),
#     )

#     realsense_rear = RaycasterCfg(
#         attachement_pos=(-0.47, 0, 0.0),
#         attachement_quat=(-0.13052, 0.0, 0.99144, 0.0),
#         attach_yaw_only=False,
#         pattern_cfg=RealSensePatternCfg(),
#     )

#     realsense_rear_down = RaycasterCfg(
#         attachement_pos=(-0.47, 0, -0.05),
#         attachement_quat=(-0.5, 0.0, 0.866025, 0.0),
#         attach_yaw_only=False,
#         pattern_cfg=RealSensePatternCfg(),
#     )

#     realsense_left = RaycasterCfg(
#         attachement_pos=(0.0, 0.1, 0.0),
#         attachement_quat=(-0.24184, 0.24184, 0.66446, 0.66446),
#         attach_yaw_only=False,
#         pattern_cfg=RealSensePatternCfg(),
#     )

#     realsense_right = RaycasterCfg(
#         attachement_pos=(0.0, -0.1, 0.0),
#         attachement_quat=(0.24184, 0.24184, -0.66446, 0.66446),
#         attach_yaw_only=False,
#         pattern_cfg=RealSensePatternCfg(),
#     )
