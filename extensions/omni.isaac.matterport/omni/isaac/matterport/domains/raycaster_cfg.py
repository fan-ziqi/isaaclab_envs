from omni.isaac.orbit.sensors.ray_caster import RayCasterCfg
from omni.isaac.orbit.utils import configclass

from .matterport_raycaster import MatterportRayCaster


@configclass
class MatterportRayCasterCfg(RayCasterCfg):
    """Configuration for the ray-cast sensor for Matterport Environments."""
    
    class_type = MatterportRayCaster
    """Name of the specfic matterport ray caster class."""
