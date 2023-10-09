from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.sensors.ray_caster import RayCasterCfg
from .matterport_raycaster import MatterportRayCaster


@configclass
class MatterportRayCasterCfg(RayCasterCfg):
    cls_name = MatterportRayCaster
