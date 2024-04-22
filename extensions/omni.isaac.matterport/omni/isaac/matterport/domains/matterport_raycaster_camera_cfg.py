from omni.isaac.orbit.sensors.ray_caster import RayCasterCameraCfg
from omni.isaac.orbit.utils import configclass

from .matterport_raycaster_camera import MatterportRayCasterCamera


@configclass
class MatterportRayCasterCameraCfg(RayCasterCameraCfg):
    """Configuration for the ray-cast camera for Matterport Environments."""
    
    class_type = MatterportRayCasterCamera
    """Name of the specfic matterport ray caster camera class."""
