#!/usr/bin/python
"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      Load Semantics from Matterport3D and make them available to Isaac Sim
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict

import carb

from omni.isaac.matterport.config import MatterportConfig
from .matterport_raycast_camera import MatterportRayCasterCamera

from omni.isaac.orbit.sensors.ray_caster import RayCasterCfg
from omni.isaac.orbit.sensors.camera import CameraData
from omni.isaac.orbit.sim import SimulationContext

matplotlib.use("Qt5Agg")


class MatterportDomains:
    """
    Load Matterport3D Semantics and make them available to Isaac Sim
    """

    def __init__(self, cfg: MatterportConfig):
        """
        Initialize MatterportSemWarp

        Args:
            path (str): path to Matterport3D Semantics
        """
        self._cfg: MatterportConfig = cfg

        # setup camera list
        self.cameras: Dict[str, MatterportRayCasterCamera] = {}

        # setup camera visualization
        self.figures = {}

        # interal parameters
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

    def set_domain_callback(self, val) -> None:
        # check for camera
        if len(self.cameras) == 0:
            carb.log_warn("No cameras added! Add cameras first, then enable the callback!")
            return

        # get SimulationContext
        if SimulationContext.instance():
            self._sim: SimulationContext = SimulationContext.instance()
        else:
            carb.log_error("No Simulation Context found! Matterport Callback not attached!")
        
        # create dirs and save intrinsic matrices
        if self.cfg.save:
            self._init_save()
        
        # add callback
        if val:
            self._sim.pause()
            self._sim.add_render_callback("matterport_update", callback_fn=self._update)
        else:
            self._sim.add_render_callback("matterport_update")
        return

    ##
    # Callback Function
    ##

    def _update(self, dt: float) -> None:        
        for camera in self.cameras:
            camera.update(dt)
        
        if self._cfg.visualize:
            self._update_visualization(self.cameras[self._cfg.visualize_prim].data)

        return

    ##
    # Private Methods (Helper Functions)
    ##

    # Visualization helpers ###

    def _init_visualization(self, data: CameraData):
        """Initializes the visualization plane."""
        if "semantic_segmentation" in data.output.keys():  # noqa: SIM118
            # init semantics figure
            fg_sem = plt.figure()
            ax_sem = fg_sem.gca()
            ax_sem.set_title(f"Semantic Segmentation ")
            img_sem = ax_sem.imshow(data.output["semantic_segmentation"][0].cpu().numpy().dtype(np.uint8))
            self.figures["semantics"] = {"fig": fg_sem, "axis": ax_sem, "img": img_sem}

        if "distance_to_camera_plane" in data.output.keys():  # noqa: SIM118
            # init semantics figure
            fg_depth = plt.figure()
            ax_depth = fg_depth.gca()
            ax_depth.set_title(f"Distance To Image Plane")
            img_depth = ax_depth.imshow(data.output["distance_to_camera_plane"][0].cpu().numpy().dtype(np.float32))
            self.figures["depth"] = {"fig": fg_depth, "axis": ax_depth, "img": img_depth}

        if len(self.figures) > 0:
            plt.ion()
            # update flag
            self.vis_init = True

        return

    def _update_visualization(self, data: CameraData) -> None:
        """
        Updates the visualization plane.
        """
        if self.vis_init is False:
            self._init_visualization(data)
        else:
            # SEMANTICS
            if "semantic_segementation" in data.output.keys():  # noqa: SIM118
                self.figures["semantics"]["img"].set_array(data.output["semantic_segementation"][0].cpu().numpy())
                self.figures["semantics"]["fig"].canvas.draw()
                self.figures["semantics"]["fig"].canvas.flush_events()

            # DEPTH
            if "distance_to_camera_plane" in data.output.keys():  # noqa: SIM118
                # cam_data.img_depth.set_array(cam_data.render_depth)
                self.figures["depth"]["fig"].set_array(data.output["depth"][0].cpu().numpy())
                self.figures["depth"]["img"].canvas.draw()
                self.figures["depth"]["img"].canvas.flush_events()

        plt.pause(0.000001)
