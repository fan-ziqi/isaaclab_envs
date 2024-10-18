# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import gc
import importlib
import os
from dataclasses import MISSING
from typing import Literal

import carb
import omni
import omni.client
import omni.ext
import omni.isaac.core.utils.stage as stage_utils
import omni.isaac.lab.sim as sim_utils
import omni.ui as ui
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.ui.ui_utils import (
    btn_builder,
    cb_builder,
    dropdown_builder,
    float_builder,
    get_style,
    setup_ui_headers,
    str_builder,
)
from orbit.nav.importer.importer import MatterportImporterCfg, UnRealImporterCfg
from orbit.nav.importer.utils.toggleable_window import ToggleableWindow

EXTENSION_NAME = "Orbit Navigation Environment Importer"


def is_mesh_file(path: str, allowed_ext: list[str] = [".obj", ".usd"]) -> bool:
    _, ext = os.path.splitext(path.lower())
    return ext in allowed_ext


def is_ply_file(path: str) -> bool:
    _, ext = os.path.splitext(path.lower())
    return ext in [".ply"]


def on_filter_mesh_item(item) -> bool:
    if not item or item.is_folder:
        return not (item.name == "Omniverse" or item.path.startswith("omniverse:"))
    return is_mesh_file(item.path)


def on_filter_ply_item(item) -> bool:
    if not item or item.is_folder:
        return not (item.name == "Omniverse" or item.path.startswith("omniverse:"))
    return is_ply_file(item.path)


def import_class(module_name, class_name) -> object | None:
    try:
        # Import the module dynamically
        module = importlib.import_module(module_name)

        # Get the class object from the module
        if hasattr(module, class_name):
            class_obj = getattr(module, class_name)
            return class_obj
        else:
            print(f"Class {class_name} not found in module {module_name}")
            return None
    except ImportError:
        print(f"Failed to import module {module_name}")
        return None


@configclass
class ImportSceneCfg(InteractiveSceneCfg):
    num_envs: int = 1
    """Number of environments to spawn."""
    env_spacing: float = 1.0
    """Spacing between environments."""

    terrain: MatterportImporterCfg | UnRealImporterCfg | TerrainImporterCfg = MatterportImporterCfg()
    """The terrain importer configuration."""

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0)),
    )
    """The light configuration."""


class OrbitNavImporterExtension(omni.ext.IExt):
    """Extension to load Matterport 3D Environments into Isaac Sim"""

    def on_startup(self, ext_id):
        self._ext_id = ext_id
        self._usd_context = omni.usd.get_context()
        self._window = ToggleableWindow(
            title=EXTENSION_NAME,
            menu_prefix="Window",
            dockPreference=ui.DockPreference.LEFT_BOTTOM,
            width=400,
            height=600,
        )
        # self._window = omni.ui.Window(
        #     EXTENSION_NAME, width=400, height=500, visible=True, dockPreference=ui.DockPreference.LEFT_BOTTOM
        # )
        # path to extension
        self._extension_path = omni.kit.app.get_app().get_extension_manager().get_extension_path(ext_id)

        # scene config constructed from the UI
        self._scene_cfg = ImportSceneCfg()
        # counter for the sensors
        self._camera_idx = 0
        # buffers
        self._mesh_origin: Literal["matterport", "multi-mesh-usd", "single-mesh-usd", "generator"] = "matterport"
        self._allowed_ext = [".obj", ".usd"]

        # set additional parameters
        self._input_fields: dict = {}  # dictionary to store values of buttion, float fields, etc.
        self._sensor_input_fields: dict = {}  # dictionary to store values of buttion, float fields, etc.
        self.ply_proposal: str = ""
        # build ui
        self.build_ui()
        return

    ##
    # UI Build functions
    ##

    def build_ui(self):
        self._window.frame.clear()

        with self._window.frame:
            with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                with ui.VStack(spacing=5, height=0):
                    self._build_info_ui()
                    self._build_data_origin_ui()
                    self._build_import_ui()
                    self._build_load_ui()

        async def dock_window():
            await omni.kit.app.get_app().next_update_async()

            def dock(space, name, location, pos=0.5):
                window = omni.ui.Workspace.get_window(name)
                if window and space:
                    window.dock_in(space, location, pos)
                return window

            tgt = ui.Workspace.get_window("Viewport")
            dock(tgt, EXTENSION_NAME, omni.ui.DockPosition.LEFT, 0.33)
            await omni.kit.app.get_app().next_update_async()

        self._task = asyncio.ensure_future(dock_window())

    def _build_info_ui(self):
        title = EXTENSION_NAME
        doc_link = "https://github.com/leggedrobotics/omni_isaac_orbit"

        overview = (
            "This utility is used to import photrealisitc meshes from e.g. Matterport3D or UnrealEngine into Isaac Sim."
            "It allows to access the semantic information and quickly sample trajectories and render images from them"
            "\n\nPress the 'Open in IDE' button to view the source code."
        )

        setup_ui_headers(self._ext_id, __file__, title, doc_link, overview)
        return

    def _build_data_origin_ui(self):
        frame = ui.CollapsableFrame(
            title="Data Source",
            height=0,
            collapsed=False,
            style=get_style(),
            style_type_name_override="CollapsableFrame",
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                # Question if matterport, usd mesh or generator terrain
                mesh_origin = ["matterport", "multi-mesh-usd", "single-mesh-usd", "generator"]
                dropdown_builder(
                    "Mesh Source",
                    items=mesh_origin,
                    default_val=mesh_origin.index(self._mesh_origin),
                    on_clicked_fn=self._set_data_origin_cfg,
                    tooltip=f"Mesh source of the environment (default: {self._mesh_origin})",
                )

    def _build_import_ui(self):
        frame = ui.CollapsableFrame(
            title="Terrain Parameters",
            height=0,
            collapsed=False,
            style=get_style(),
            style_type_name_override="CollapsableFrame",
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                # read import location
                def check_file_type(model=None):
                    path = model.get_value_as_string()
                    if is_mesh_file(path, self._allowed_ext):
                        # self._input_fields["load_btn"].enabled = True
                        if self._mesh_origin == "matterport":
                            self._make_ply_proposal(path)
                    else:
                        # self._input_fields["load_btn"].enabled = False
                        carb.log_warn(f"Invalid path to {self._allowed_ext} file: {path}")

                if self._mesh_origin == "generator":
                    # get the module and class name of the terrain generator
                    self._input_fields["module_name"] = str_builder(
                        label="Module Name",
                        default_val="omni.isaac.lab.terrains.config.rough",
                        tooltip="Module name of the terrain generator",
                    )
                    self._input_fields["class_name"] = str_builder(
                        label="Class Name",
                        default_val="ROUGH_TERRAINS_CFG",
                        tooltip="Class name of the terrain generator",
                    )
                else:
                    # get the mesh file location
                    kwargs = {
                        "label": "Input Mesh File",
                        "default_val": "/home/pascal/viplanner/env/matterport/v1/scans/2n8kARJN3HM/2n8kARJN3HM/matterport_mesh/0c334eaabb844eaaad049cbbb2e0a4f2/0c334eaabb844eaaad049cbbb2e0a4f2.usd",
                        "tooltip": "Click the Folder Icon to Set Filepath",
                        "use_folder_picker": True,
                        "item_filter_fn": on_filter_mesh_item,
                        "bookmark_label": "Select Mesh File that should be loaded",
                        "bookmark_path": f"{self._extension_path}/data",
                        "folder_dialog_title": "Select Mesh File",
                        "folder_button_title": "Select Mesh File",
                    }
                    self._input_fields["input_file"] = str_builder(**kwargs)
                    self._input_fields["input_file"].add_value_changed_fn(check_file_type)

                # for matterport also require ply file to access necessary information for camera sensors
                if self._mesh_origin == "matterport":
                    kwargs = {
                        "label": "Input ply File",
                        "default_val": "/home/pascal/viplanner/env/matterport/v1/scans/2n8kARJN3HM/2n8kARJN3HM/house_segmentations/2n8kARJN3HM.ply",
                        "tooltip": "Click the Folder Icon to Set Filepath",
                        "use_folder_picker": True,
                        "item_filter_fn": on_filter_ply_item,
                        "bookmark_label": "Included Matterport3D Point-Cloud with semantic labels",
                        "bookmark_path": f"{self._extension_path}/data",
                        "folder_dialog_title": "Select .ply Point-Cloud File",
                        "folder_button_title": "Select .ply Point-Cloud",
                    }
                    self._input_fields["input_ply_file"] = str_builder(**kwargs)

                # PhysicsMaterial
                self._input_fields["friction_dynamic"] = float_builder(
                    "Dynamic Friction",
                    default_val=self._scene_cfg.terrain.physics_material.dynamic_friction,
                    tooltip=(
                        "Sets the dynamic friction of the physics material (default:"
                        f" {self._scene_cfg.terrain.physics_material.dynamic_friction})"
                    ),
                )
                self._input_fields["friction_static"] = float_builder(
                    "Static Friction",
                    default_val=self._scene_cfg.terrain.physics_material.static_friction,
                    tooltip=(
                        "Sets the static friction of the physics material (default:"
                        f" {self._scene_cfg.terrain.physics_material.static_friction})"
                    ),
                )
                self._input_fields["restitution"] = float_builder(
                    "Restitution",
                    default_val=self._scene_cfg.terrain.physics_material.restitution,
                    tooltip=(
                        "Sets the restitution of the physics material (default:"
                        f" {self._scene_cfg.terrain.physics_material.restitution})"
                    ),
                )
                self.friction_restitution_options = ["average", "min", "multiply", "max"]
                self._input_fields["friction_combine_mode"] = dropdown_builder(
                    "Friction Combine Mode",
                    items=self.friction_restitution_options,
                    default_val=self.friction_restitution_options.index(
                        self._scene_cfg.terrain.physics_material.friction_combine_mode
                    ),
                    tooltip=(
                        "Sets the friction combine mode of the physics material (default:"
                        f" {self._scene_cfg.terrain.physics_material.friction_combine_mode})"
                    ),
                )
                self._input_fields["restitution_combine_mode"] = dropdown_builder(
                    "Restitution Combine Mode",
                    items=self.friction_restitution_options,
                    default_val=self.friction_restitution_options.index(
                        self._scene_cfg.terrain.physics_material.restitution_combine_mode
                    ),
                    tooltip=(
                        "Sets the friction combine mode of the physics material (default:"
                        f" {self._scene_cfg.terrain.physics_material.restitution_combine_mode})"
                    ),
                )
                self._input_fields["improve_patch_friction"] = cb_builder(
                    label="Improved Patch Friction",
                    tooltip=(
                        "Sets the improved patch friction of the physics material (default:"
                        f" {self._scene_cfg.terrain.physics_material.improve_patch_friction})"
                    ),
                    default_val=self._scene_cfg.terrain.physics_material.improve_patch_friction,
                )

                # Set prim path for environment
                self._input_fields["prim_path"] = str_builder(
                    "Prim Path of the Environment",
                    tooltip="Prim path of the environment",
                    default_val=(
                        self._scene_cfg.terrain.prim_path
                        if not isinstance(self._scene_cfg.terrain.prim_path, type(MISSING))
                        else "/World/terrain"
                    ),
                )

    def _build_load_ui(self):
        frame = ui.CollapsableFrame(
            title="Load Scene",
            height=0,
            collapsed=False,
            style=get_style(),
            style_type_name_override="CollapsableFrame",
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                self._input_fields["load_btn"] = btn_builder(
                    "Load Scene", text="Load", on_clicked_fn=self._start_loading
                )
                # self._input_fields["load_btn"].enabled = False
                self._input_fields["load_btn"] = btn_builder(
                    "Reset Scene", text="Reset", on_clicked_fn=self._reset_scene
                )
                # self._input_fields["load_btn"].enabled = False

    ##
    # Shutdown Helpers
    ##

    def on_shutdown(self):
        if self._window:
            self._window.shutdown()
            self._window.destroy()
            self._window = None
        gc.collect()
        stage_utils.clear_stage()

    ##
    # Path Helpers
    ##
    def _make_ply_proposal(self, path: str) -> None:
        """use default matterport datastructure to make proposal about point-cloud file

        - "env_id"
            - matterport_mesh
                - "id_nbr"from orbit.nav.collectors.collectors import ViewpointSampling, ViewpointSamplingCfg

                    - "id_nbr".obj
            - house_segmentations
                - "env_id".ply

        """
        file_dir, file_name = os.path.split(path)
        ply_dir = os.path.join(file_dir, "../..", "house_segmentations")
        env_id = file_dir.split("/")[-3]
        try:
            ply_file = os.path.join(ply_dir, f"{env_id}.ply")
            os.path.isfile(ply_file)
            carb.log_verbose(f"Found ply file: {ply_file}")
            self.ply_proposal = ply_file
        except FileNotFoundError:
            carb.log_verbose("No ply file found in default matterport datastructure")

    ##
    # Load Mesh and Point-Cloud
    ##

    def _set_data_origin_cfg(self, mesh_origin: str):
        self._mesh_origin = mesh_origin

        if mesh_origin == "matterport":
            self._scene_cfg.terrain = MatterportImporterCfg()
            self._allowed_ext = [".obj", ".usd"]
        elif mesh_origin == "multi-mesh-usd":
            self._scene_cfg.terrain = UnRealImporterCfg(terrain_type="usd")
            self._allowed_ext = [".usd"]
        elif mesh_origin == "single-mesh-usd":
            self._scene_cfg.terrain = TerrainImporterCfg(terrain_type="usd")
            self._allowed_ext = [".usd"]
        elif mesh_origin == "generator":
            self._scene_cfg.terrain = TerrainImporterCfg(terrain_type="generator")
            self._allowed_ext = None
        else:
            carb.log_warn(f"Invalid mesh origin: {mesh_origin}")
            return

        self.build_ui()

    async def load_scene(self):
        print("[INFO]: Start loading scene")
        # simulation settings
        # check if simulation context was created earlier or not.
        if sim_utils.SimulationContext.instance():
            sim_utils.SimulationContext.clear_instance()
            carb.log_warn("SimulationContext already loaded. Will clear now and init default SimulationContext")

        # create new simulation context
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg())
        self.sim.set_camera_view((20.0, 20.0, 20.0), (0.0, 0.0, 0.0))
        # initialize simulation
        await self.sim.initialize_simulation_context_async()
        # load scene
        self.scene = InteractiveScene(self._scene_cfg)
        # for matterport, allow for .obj to .usd conversion which is only available in an async workflow
        if self._mesh_origin == "matterport":
            await self.scene.terrain.load_world_async()
        # reset the simulator
        # note: this plays the simulator which allows setting up all the physics handles.
        await self.sim.reset_async()
        await self.sim.pause_async()

        print("[INFO]: Scene loaded")

    def _start_loading(self):
        # simulation settings
        if sim_utils.SimulationContext.instance():
            sim_utils.SimulationContext.clear_instance()
            carb.log_warn("SimulationContext already loaded. Will clear now and init default SimulationContext")

        # create new simulation context
        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg())

        # add terrain to scene config and load the interactive scene
        self._scene_cfg.terrain.prim_path = self._input_fields["prim_path"].get_value_as_string()
        self._scene_cfg.terrain.physics_material.improve_patch_friction = self._input_fields[
            "improve_patch_friction"
        ].get_value_as_bool()
        self._scene_cfg.terrain.physics_material.friction_combine_mode = self.friction_restitution_options[
            self._input_fields["friction_combine_mode"].get_item_value_model().as_int
        ]
        self._scene_cfg.terrain.physics_material.restitution_combine_mode = self.friction_restitution_options[
            self._input_fields["restitution_combine_mode"].get_item_value_model().as_int
        ]
        self._scene_cfg.terrain.physics_material.static_friction = self._input_fields[
            "friction_static"
        ].get_value_as_float()
        self._scene_cfg.terrain.physics_material.dynamic_friction = self._input_fields[
            "friction_dynamic"
        ].get_value_as_float()
        self._scene_cfg.terrain.physics_material.restitution = self._input_fields["restitution"].get_value_as_float()

        if self._mesh_origin == "matterport":
            self._scene_cfg.terrain.obj_filepath = self._input_fields["input_file"].get_value_as_string()
        elif self._mesh_origin == "multi-mesh-usd" or self._mesh_origin == "single-mesh-usd":
            self._scene_cfg.terrain.terrain_type = "usd"
            self._scene_cfg.terrain.usd_path = self._input_fields["input_file"].get_value_as_string()
        elif self._mesh_origin == "generator":
            self._scene_cfg.terrain.terrain_type = "generator"
            self._scene_cfg.terrain.terrain_generator = import_class(
                self._input_fields["module_name"].get_value_as_string(),
                self._input_fields["class_name"].get_value_as_string(),
            )
        else:
            carb.log_warn(f"Invalid mesh origin: {self._mesh_origin}")
            return

        # load scene and init simulationcontext in an async workflow
        asyncio.ensure_future(self.load_scene())

        # disable reloading
        # self._input_fields["load_btn"].enabled = False

    def _reset_scene(self):
        # create a new stage
        omni.usd.get_context().new_stage()
        # remove the scene
        self.scene = None
        # enable reloading
        # self._input_fields["load_btn"].enabled = True
