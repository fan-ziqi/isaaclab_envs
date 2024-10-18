# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import asyncio
import weakref

import omni.isaac.ui.ui_utils as ui_utils
import omni.kit.app
import omni.kit.commands
import omni.ui
import omni.usd
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.sim import SimulationContext
from omni.kit.window.extensions import SimpleCheckBox


class SensorWindow:
    """Window manager for the ray caster camera window.

    This class creates a window that is used to display an image of the camera feed.
    """

    def __init__(self, sim: SimulationContext, scene: InteractiveScene, window_name: str = "MatterportExtension"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "Orbit".
        """
        # save scene
        self.scene = scene
        self.sim = sim

        # Listeners for environment selection changes
        self._env_selection_listeners: list = []

        print("Creating window for environment.")
        # create window for UI
        self.ui_window = omni.ui.Window(
            window_name, width=400, height=500, visible=True, dock_preference=omni.ui.DockPreference.RIGHT_TOP
        )
        # dock next to properties window
        asyncio.ensure_future(self._dock_window(window_title=self.ui_window.title))

        # keep a dictionary of stacks so that child environments can add their own UI elements
        # this can be done by using the `with` context manager
        self.ui_window_elements = dict()
        # create main frame
        self.ui_window_elements["main_frame"] = self.ui_window.frame
        with self.ui_window_elements["main_frame"]:
            # create main stack
            self.ui_window_elements["main_vstack"] = omni.ui.VStack(spacing=5, height=0)
            with self.ui_window_elements["main_vstack"]:
                # create collapsible frame for simulation
                self._build_viewer_frame()
                # create collapsible frame for debug visualization
                self._build_debug_vis_frame()

    def __del__(self):
        """Destructor for the window."""
        # destroy the window
        if self.ui_window is not None:
            self.ui_window.visible = False
            self.ui_window.destroy()
            self.ui_window = None

    """
    Build sub-sections of the UI.
    """

    def _build_viewer_frame(self):
        """Build the viewer-related control frame for the UI."""
        # create collapsible frame for viewer
        self.ui_window_elements["viewer_frame"] = omni.ui.CollapsableFrame(
            title="Viewer Settings",
            width=omni.ui.Fraction(1),
            height=0,
            collapsed=False,
            style=ui_utils.get_style(),
            horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with self.ui_window_elements["viewer_frame"]:
            # create stack for controls
            self.ui_window_elements["viewer_vstack"] = omni.ui.VStack(spacing=5, height=0)
            with self.ui_window_elements["viewer_vstack"]:
                # create a number slider to move to environment origin
                # NOTE: slider is 1-indexed, whereas the env index is 0-indexed
                viewport_origin_cfg = {
                    "label": "Environment Index",
                    "type": "button",
                    "default_val": 1,
                    "min": 1,
                    "max": self.scene.num_envs,
                    "tooltip": "The environment index to follow. Only effective if follow mode is not 'World'.",
                }
                self.ui_window_elements["viewer_env_index"] = ui_utils.int_builder(**viewport_origin_cfg)
                # create a number slider to move to environment origin
                self.ui_window_elements["viewer_env_index"].add_value_changed_fn(self._set_viewer_env_index_fn)

    def _build_debug_vis_frame(self):
        """Builds the debug visualization frame for the cameras.

        This function inquires the scene for all elements that have a debug visualization
        implemented and creates a checkbox to toggle the debug visualization for each element
        that has it implemented. If the element does not have a debug visualization implemented,
        a label is created instead.
        """
        # create collapsible frame for debug visualization
        self.ui_window_elements["debug_frame"] = omni.ui.CollapsableFrame(
            title="Scene Debug Visualization",
            width=omni.ui.Fraction(1),
            height=0,
            collapsed=False,
            style=ui_utils.get_style(),
            horizontal_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=omni.ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with self.ui_window_elements["debug_frame"]:
            # create stack for debug visualization
            self.ui_window_elements["debug_vstack"] = omni.ui.VStack(spacing=5, height=0)
            with self.ui_window_elements["debug_vstack"]:
                elements = [
                    self.scene.terrain,
                    *self.scene.rigid_objects.values(),
                    *self.scene.articulations.values(),
                    *self.scene.sensors.values(),
                ]
                names = [
                    "terrain",
                    *self.scene.rigid_objects.keys(),
                    *self.scene.articulations.keys(),
                    *self.scene.sensors.keys(),
                ]
                # create one for the terrain
                for elem, name in zip(elements, names):
                    if elem is not None:
                        self._create_debug_vis_ui_element(name, elem)

    """
    Helper functions - UI building.
    """

    def _set_viewer_env_index_fn(self, model: omni.ui.SimpleIntModel):
        """Sets the environment index and updates the camera if in 'env' origin mode."""
        # access the viewport camera controller (for brevity)
        vcc = self.scene.viewport_camera_controller
        if vcc is None:
            raise ValueError("Viewport camera controller is not initialized! Please check the rendering mode.")
        # store the desired env index, UI is 1-indexed
        vcc.set_view_env_index(model.as_int - 1)
        # notify additional listeners
        for listener in self._env_selection_listeners:
            listener.set_env_selection(model.as_int - 1)

    def _create_debug_vis_ui_element(self, name: str, elem: object):
        """Create a checkbox for toggling debug visualization for the given element."""
        with omni.ui.HStack():
            # create the UI element
            text = (
                "Toggle debug visualization."
                if elem.has_debug_vis_implementation
                else "Debug visualization not implemented."
            )
            omni.ui.Label(
                name.replace("_", " ").title(),
                width=ui_utils.LABEL_WIDTH - 12,
                alignment=omni.ui.Alignment.LEFT_CENTER,
                tooltip=text,
            )
            self.ui_window_elements[f"{name}_cb"] = SimpleCheckBox(
                model=omni.ui.SimpleBoolModel(),
                enabled=elem.has_debug_vis_implementation,
                checked=(hasattr(elem.cfg, "debug_vis") and elem.cfg.debug_vis)
                or (hasattr(elem, "debug_vis") and elem.debug_vis),
                on_checked_fn=lambda value, e=weakref.proxy(elem): e.set_debug_vis(value),
            )
            ui_utils.add_line_rect_flourish()

        elem.set_window(self.ui_window)

    async def _dock_window(self, window_title: str):
        """Docks the custom UI window to the property window."""
        # wait for the window to be created
        for _ in range(5):
            if omni.ui.Workspace.get_window(window_title):
                break
            await self.sim.app.next_update_async()

        # dock next to properties window
        custom_window = omni.ui.Workspace.get_window(window_title)
        property_window = omni.ui.Workspace.get_window("Property")
        if custom_window and property_window:
            custom_window.dock_in(property_window, omni.ui.DockPosition.SAME, 1.0)
            custom_window.focus()
