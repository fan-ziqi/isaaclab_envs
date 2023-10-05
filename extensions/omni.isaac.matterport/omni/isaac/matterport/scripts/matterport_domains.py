"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      Load Semantics from Matterport3D and make them available to Isaac Sim
"""

from typing import Dict

import carb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from omni.isaac.matterport.domains.matterport_raycast_camera import (
    MatterportRayCasterCamera,
)
from omni.isaac.orbit.sensors.camera import CameraData
from omni.isaac.orbit.sensors.ray_caster import RayCasterCfg
from omni.isaac.orbit.sim import SimulationContext

from .ext_cfg import MatterportExtConfig

mpl.use("Qt5Agg")


class MatterportDomains:
    """
    Load Matterport3D Semantics and make them available to Isaac Sim
    """

    def __init__(self, cfg: MatterportExtConfig):
        """
        Initialize MatterportSemWarp

        Args:
            path (str): path to Matterport3D Semantics
        """
        self._cfg: MatterportExtConfig = cfg

        # setup camera list
        self.cameras: Dict[str, MatterportRayCasterCamera] = {}

        # setup camera visualization
        self.figures = {}

        # internal parameters
        self._save_counter: int = 0
        return

    ##
    # Public Methods
    ##

    def register_camera(self, cfg: RayCasterCfg):
        """
        Register a camera to the MatterportSemWarp
        """
        # append to camera list
        self.cameras[cfg.prim_path] = MatterportRayCasterCamera(cfg)

    ##
    # Callback Setup
    ##

    def set_domain_callback(self, val):
        # check for camera
        if len(self.cameras) == 0:
            carb.log_warn("No cameras added! Add cameras first, then enable the callback!")
            return

        # get SimulationContext
        if SimulationContext.instance():
            self._sim: SimulationContext = SimulationContext.instance()
        else:
            carb.log_error("No Simulation Context found! Matterport Callback not attached!")

        # add callback
        if val:
            self._sim.pause()
            self._sim.add_render_callback("matterport_update", callback_fn=self._update)
        else:
            self._sim.add_render_callback("matterport_update")

    ##
    # Callback Function
    ##

    def _update(self, dt: float):
        for camera in self.cameras:
            camera.update(dt)

        if self._cfg.visualize:
            self._update_visualization(self.cameras[self._cfg.visualize_prim].data)
            self.max_depth = self.cameras[self._cfg.visualize_prim].cfg.max_distance

    ##
    # Private Methods (Helper Functions)
    ##

    # Visualization helpers ###

    def _init_visualization(self, data: CameraData):
        """Initializes the visualization plane."""
        # init depth figure
        self.n_bins = 100  # Number of bins in the colormap
        self.color_array = mpl.colormaps["jet"](np.linspace(0, 1, self.n_bins))  # Colormap

        if "semantic_segmentation" in data.output.keys():  # noqa: SIM118
            # init semantics figure
            fg_sem = plt.figure()
            ax_sem = fg_sem.gca()
            ax_sem.set_title("Semantic Segmentation")
            img_sem = ax_sem.imshow(data.output["semantic_segmentation"][0].cpu().numpy())
            self.figures["semantics"] = {"fig": fg_sem, "axis": ax_sem, "img": img_sem}

        if "distance_to_image_plane" in data.output.keys():  # noqa: SIM118
            # init semantics figure
            fg_depth = plt.figure()
            ax_depth = fg_depth.gca()
            ax_depth.set_title("Distance To Image Plane")
            img_depth = ax_depth.imshow(self.convert_depth_to_color(data.output["distance_to_image_plane"][0]))
            self.figures["depth"] = {"fig": fg_depth, "axis": ax_depth, "img": img_depth}

        if len(self.figures) > 0:
            plt.ion()
            # update flag
            self.vis_init = True

    def _update_visualization(self, data: CameraData) -> None:
        """
        Updates the visualization plane.
        """
        if self.vis_init is False:
            self._init_visualization(data)
        else:
            # SEMANTICS
            if "semantic_segmentation" in data.output.keys():  # noqa: SIM118
                self.figures["semantics"]["img"].set_array(data.output["semantic_segmentation"][0].cpu().numpy())
                self.figures["semantics"]["fig"].canvas.draw()
                self.figures["semantics"]["fig"].canvas.flush_events()

            # DEPTH
            if "distance_to_image_plane" in data.output.keys():  # noqa: SIM118
                # cam_data.img_depth.set_array(cam_data.render_depth)
                self.figures["depth"]["fig"].set_array(
                    self.convert_depth_to_color(data.output["distance_to_image_plane"][0])
                )
                self.figures["depth"]["img"].canvas.draw()
                self.figures["depth"]["img"].canvas.flush_events()

        plt.pause(0.000001)

    def convert_depth_to_color(self, depth_img):
        depth_img = depth_img.cpu().numpy()
        depth_img_flattend = np.clip(depth_img.flatten(), a_min=0, a_max=self.max_depth)
        depth_img_flattend = np.round(depth_img_flattend / self.max_depth * (self.n_bins - 1)).astype(np.int32)
        depth_colors = self.color_array[depth_img_flattend]
        depth_colors = depth_colors.reshape(depth_img.shape[0], depth_img.shape[1], 4)
        return depth_colors
