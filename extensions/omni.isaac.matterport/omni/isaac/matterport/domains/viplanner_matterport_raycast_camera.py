import torch
import yaml
from omni.isaac.matterport.utils import DATA_DIR
from omni.isaac.orbit.sensors.ray_caster import RayCasterCfg

from .matterport_raycast_camera import MatterportRayCasterCamera

try:
    from viplanner.config.viplanner_sem_meta import VIPlannerSemMetaHandler
except ImportError:
    print("VIPlanner Semantic Meta Handler not available.")


class VIPlannerMatterportRayCasterCamera(MatterportRayCasterCamera):
    def __init__(self, cfg: RayCasterCfg):
        super().__init__(cfg)

    def _color_mapping(self):
        viplanner_sem = VIPlannerSemMetaHandler()
        map_mpcat40_to_vip_sem = yaml.safe_load(open(DATA_DIR + "/mappings/mpcat40_to_vip_sem.yml"))
        color = viplanner_sem.get_colors_for_names(list(map_mpcat40_to_vip_sem.values()))
        self.color = torch.tensor(color)
