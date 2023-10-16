from omni.isaac.orbit.sensors.ray_caster import RayCasterCfg
from omni.isaac.orbit.utils import configclass

from .matterport_raycaster import MatterportRayCaster


@configclass
class MatterportRayCasterCfg(RayCasterCfg):
    cls_name = MatterportRayCaster
