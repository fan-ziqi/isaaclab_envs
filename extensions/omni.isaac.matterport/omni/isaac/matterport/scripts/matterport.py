#!/usr/bin/python
"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief     MatterPort3D Extension in Omniverse-Isaac Sim
"""

import asyncio
import gc

# python
import os

import carb

# omni
import omni
import omni.client
import omni.ext

# isaac-core
import omni.isaac.core.utils.prims as prim_utils
import omni.ui as ui
from omni.isaac.matterport.config import MatterportConfig
from omni.isaac.matterport.exploration import RandomExplorer
from omni.isaac.matterport.semantics import CameraData, MatterportCallbackDomains, MatterportWarp
from omni.isaac.matterport.test import test_depth_warp

# omni-isaac-matterport
from omni.isaac.matterport.utils import MatterportWorld

# omni-isaac-ui
from omni.isaac.ui.ui_utils import (
    btn_builder,
    cb_builder,
    dropdown_builder,
    float_builder,
    get_style,
    int_builder,
    setup_ui_headers,
    state_btn_builder,
    str_builder,
)

EXTENSION_NAME = "Matterport Importer"


def is_mesh_file(path: str) -> bool:
    _, ext = os.path.splitext(path.lower())
    return ext in [".obj", ".usd"]


def is_ply_file(path: str) -> bool:
    _, ext = os.path.splitext(path.lower())
    return ext in [".ply"]


def on_filter_obj_item(item) -> bool:
    if not item or item.is_folder:
        return not (item.name == "Omniverse" or item.path.startswith("omniverse:"))
    return is_mesh_file(item.path)


def on_filter_ply_item(item) -> bool:
    if not item or item.is_folder:
        return not (item.name == "Omniverse" or item.path.startswith("omniverse:"))
    return is_ply_file(item.path)


class MatterPortExtension(omni.ext.IExt):
    """Extension to load Matterport 3D Environments into Isaac Sim"""

    def on_startup(self, ext_id):
        self._ext_id = ext_id
        self._usd_context = omni.usd.get_context()
        self._window = omni.ui.Window(
            EXTENSION_NAME, width=400, height=500, visible=True, dockPreference=ui.DockPreference.LEFT_BOTTOM
        )

        # init config class and get path to extension
        self._config = MatterportConfig()
        self._extension_path = omni.kit.app.get_app().get_extension_manager().get_extension_path(ext_id)

        # set additonal parameters
        self._input_fields: dict = {}  # dictionary to store values of buttion, float fields, etc.
        self.camera_list: list = []  # list of cameras prim for which semantic and depth are rendered
        self.domains: MatterportWarp = None  # callback class for semantic rendering

        # build ui
        self.build_ui()
        return

    ##
    # UI Build functiions
    ##

    def build_ui(self, task_callback: bool = False):
        with self._window.frame:
            with ui.VStack(spacing=5, height=0):
                self._build_info_ui()

                self._build_import_ui()

                if task_callback:
                    self._build_camera_ui()
                    self._build_writer_ui()
                    self._build_callback_ui()
                else:
                    self._build_task_ui()

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

        overview = "This utility is used to import Matterport3D Environments into Isaac Sim. "
        overview += "The environment and additional information are available at https://github.com/niessner/Matterport"
        overview += "\n\nPress the 'Open in IDE' button to view the source code."

        setup_ui_headers(self._ext_id, __file__, title, doc_link, overview)
        return

    def _build_import_ui(self):
        frame = ui.CollapsableFrame(
            title="Import Dataset",
            height=0,
            collapsed=False,
            style=get_style(),
            style_type_name_override="CollapsableFrame",
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                # colliders
                cb_builder(
                    label="Enable Colliders",
                    tooltip="Import Mesh with Colliders and Physics Materials",
                    on_clicked_fn=lambda m, config=self._config: config.set_colliders(m),
                    default_val=self._config.colliders,
                )
                # PhysicsMaterial
                self._input_fields["friction_dynamic"] = float_builder(
                    "Dynamic Friction",
                    default_val=self._config.friction_dynamic,
                    tooltip=f"Sets the dyanmic friction of the physics material (default: {self._config.friction_dynamic})",
                )
                self._input_fields["friction_dynamic"].add_value_changed_fn(
                    lambda m, config=self._config: config.set_friction_dynamic(m.get_value_as_float())
                )
                self._input_fields["friction_static"] = float_builder(
                    "Static Friction",
                    default_val=self._config.friction_static,
                    tooltip=f"Sets the static friction of the physics material (default: {self._config.friction_static})",
                )
                self._input_fields["friction_static"].add_value_changed_fn(
                    lambda m, config=self._config: config.set_friction_static(m.get_value_as_float())
                )
                self._input_fields["restitution"] = float_builder(
                    "Restitution",
                    default_val=self._config.restitution,
                    tooltip=f"Sets the restitution of the physics material (default: {self._config.restitution})",
                )
                self._input_fields["restitution"].add_value_changed_fn(
                    lambda m, config=self._config: config.set_restitution(m.get_value_as_float())
                )
                dropdown_builder(
                    "Friction Combine Mode",
                    items=["average", "min", "multiply", "max"],
                    default_val=self._config.friction_combine_mode,
                    on_clicked_fn=lambda i, config=self._config: config.set_friction_combine_mode(
                        0 if i == "average" else (1 if i == "min" else (2 if i == "multiply" else 3))
                    ),
                    tooltip="Sets the friction combine mode of the physics material (default: max)",
                )
                dropdown_builder(
                    "Restitution Combine Mode",
                    items=["average", "min", "multiply", "max"],
                    default_val=self._config.restitution_combine_mode,
                    on_clicked_fn=lambda i, config=self._config: config.set_restitution_combine_mode(
                        0 if i == "average" else (1 if i == "min" else (2 if i == "multiply" else 3))
                    ),
                    tooltip="Sets the friction combine mode of the physics material (default: max)",
                )
                cb_builder(
                    label="Improved Patch Friction",
                    tooltip=f"Sets the improved patch friction of the physics material (default: {self._config.improved_patch_friction})",
                    on_clicked_fn=lambda m, config=self._config: config.set_improved_patch_friction(m),
                    default_val=self._config.improved_patch_friction,
                )

                # Set prim path for environment
                self._input_fields["prim_path"] = str_builder(
                    "Prim Path of the Environment",
                    tooltip="Prim path of the environment",
                    default_val=self._config.prim_path,
                )
                self._input_fields["prim_path"].add_value_changed_fn(
                    lambda m, config=self._config: config.set_prim_path(m.get_value_as_string())
                )

                # read import location
                def check_file_type(model=None):
                    path = model.get_value_as_string()
                    if is_mesh_file(path):
                        self._input_fields["import_btn"].enabled = True
                        self._make_ply_proposal(path)
                        self._config.set_import_file_obj(path)
                    else:
                        self._input_fields["import_btn"].enabled = False
                        carb.log_warn(f"Invalid path to .obj file: {path}")

                kwargs = {
                    "label": "Input File",
                    "default_val": self._config.import_file_obj,
                    "tooltip": "Click the Folder Icon to Set Filepath",
                    "use_folder_picker": True,
                    "item_filter_fn": on_filter_obj_item,
                    "bookmark_label": "Included Matterport3D meshs",
                    "bookmark_path": f"{self._extension_path}/data/mesh",
                    "folder_dialog_title": "Select .obj File",
                    "folder_button_title": "*.obj, *.usd",
                }
                self._input_fields["input_file"] = str_builder(**kwargs)
                self._input_fields["input_file"].add_value_changed_fn(check_file_type)

                self._input_fields["import_btn"] = btn_builder(
                    "Import", text="Import", on_clicked_fn=self._load_matterport3d
                )
                self._input_fields["import_btn"].enabled = False

                # get import location and save directory
                def check_file_type_ply(model=None):
                    path = model.get_value_as_string()
                    if is_ply_file(path):
                        self._input_fields["import_ply_btn"].enabled = True
                        self._config.set_import_file_ply(path)
                    else:
                        carb.log_warn(f"Invalid path to .ply: {path}")

                kwargs = {
                    "label": "Input ply File",
                    "default_val": self._config.import_file_ply,
                    "tooltip": "Click the Folder Icon to Set Filepath",
                    "use_folder_picker": True,
                    "item_filter_fn": on_filter_ply_item,
                    "bookmark_label": "Included Matterport3D Point-Cloud with semantic labels",
                    "bookmark_path": f"{self._extension_path}/data/mesh",
                    "folder_dialog_title": "Select .ply Point-Cloud File",
                    "folder_button_title": "Select .ply Point-Cloud",
                }
                self._input_fields["input_ply_file"] = str_builder(**kwargs)
                self._input_fields["input_ply_file"].add_value_changed_fn(check_file_type_ply)

                # final button to load semantics
                self._input_fields["import_ply_btn"] = btn_builder(
                    "Import", text="Import", on_clicked_fn=self._load_domains
                )
                self._input_fields["import_ply_btn"].enabled = False
        return

    def _build_camera_ui(self):
        frame = ui.CollapsableFrame(
            title="Domains",
            height=0,
            collapsed=False,
            style=get_style(),
            style_type_name_override="CollapsableFrame",
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                # control parameters
                cb_builder(
                    label="Enable Semantics",
                    tooltip=f"Enable access to the semantics information of the mesh (default: {self._config.semantic})",
                    on_clicked_fn=lambda m, config=self._config: config.set_semantic(m),
                    default_val=self._config.semantic,
                )
                cb_builder(
                    label="Enable Depth",
                    tooltip=f"Enable access to the depth information of the mesh - no additional compute effort (default: {self._config.depth})",
                    on_clicked_fn=lambda m, config=self._config: config.set_depth(m),
                    default_val=self._config.depth,
                )
                cb_builder(
                    label="Enable RGB",
                    tooltip=f"Enable access to the rgb information of the mesh - no additional compute effort (default: {self._config.rgb})",
                    on_clicked_fn=lambda m, config=self._config: config.set_rgb(m),
                    default_val=self._config.rgb,
                )

                cb_builder(
                    label="Visualization",
                    tooltip=f"Visualize Semantics and/or Depth (default: {self._config.visualize})",
                    on_clicked_fn=lambda m, config=self._config: config.set_visualize(m),
                    default_val=self._config.visualize,
                )

                # add camera sensor for which semantics and depth should be rendered
                kwargs = {  # TODO: make dropdown or MultiStringField
                    "label": "Camera Prim Path",
                    "type": "stringfield",
                    "default_val": "",
                    "tooltip": "Enter Camera Prim Path",
                    "use_folder_picker": False,
                }
                self._input_fields["camera"] = str_builder(**kwargs)
                self._input_fields["camera"].add_value_changed_fn(self._check_cam_prim)

                self._input_fields["cam_height"] = int_builder(
                    "Camera Height in Pixels",
                    default_val=CameraData.get_default_height(),
                    tooltip=f"Set the height of the camera image plane in pixels (default: {CameraData.get_default_height()})",
                )
                self._input_fields["cam_height"].add_value_changed_fn(
                    lambda m, config=self._config: config.set_compute_frequency(m.get_value_as_int())
                )
                self._input_fields["cam_width"] = int_builder(
                    "Camera Width in Pixels",
                    default_val=CameraData.get_default_width(),
                    tooltip=f"Set the width of the camera image plane in pixels (default: {CameraData.get_default_width()})",
                )
                self._input_fields["cam_width"].add_value_changed_fn(
                    lambda m, config=self._config: config.set_compute_frequency(m.get_value_as_int())
                )

                self._input_fields["load_camera"] = btn_builder(
                    "Add Camera", text="Add Camera", on_clicked_fn=self._register_camera
                )
                self._input_fields["load_camera"].enabled = False
        return

    def _build_writer_ui(self):
        frame = ui.CollapsableFrame(
            title="Writer",
            height=0,
            collapsed=False,
            style=get_style(),
            style_type_name_override="CollapsableFrame",
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                # control parameters
                self._input_fields["compute_frequency"] = int_builder(
                    "Compute Frequency",
                    default_val=1,
                    tooltip=f"Sets the saving frequency, can be adjusted on the fly (default: {self._config.compute_frequency})",
                )
                self._input_fields["compute_frequency"].add_value_changed_fn(
                    lambda m, config=self._config: config.set_compute_frequency(m.get_value_as_int())
                )

                # get save directory
                kwargs = {
                    "label": "Save Directory",
                    "type": "stringfield",
                    "default_val": self._config.save_path,
                    "tooltip": "Click the Folder Icon to Set Filepath",
                    "use_folder_picker": True,
                }
                self._input_fields["save_path"] = str_builder(**kwargs)
                self._input_fields["save_path"].add_value_changed_fn(self._check_save_path)
                self._default_save_path = self._config.save_path

                self._input_fields["start_writer"] = btn_builder(
                    "Start Writer", text="Start Writer", on_clicked_fn=self._start_writer
                )
                self._input_fields["start_writer"].enabled = False
        return

    def _build_task_ui(self):
        frame = ui.CollapsableFrame(
            title="Tasks",
            height=0,
            collapsed=False,
            style=get_style(),
            style_type_name_override="CollapsableFrame",
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                # Random Exploration
                self._input_fields["annotator_callback"] = btn_builder(
                    "Configure Physics Callback", text="Configure", on_clicked_fn=self._rebuild_ui
                )
                self._input_fields["annotator_callback"].enabled = False

                # Random Exploration
                self._input_fields["explorer"] = btn_builder(
                    "Start Explorer",
                    text="Explore",
                    on_clicked_fn=self._start_explorer,
                )
                self._input_fields["explorer"].enabled = False

                # Depth Test
                self._input_fields["test"] = btn_builder("Test Warp", text="Start", on_clicked_fn=self._depth_test)
                self._input_fields["test"].enabled = False
        return

    def _build_callback_ui(self):
        frame = ui.CollapsableFrame(
            title="Callback",
            height=0,
            collapsed=False,
            style=get_style(),
            style_type_name_override="CollapsableFrame",
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with frame:
            # Attach Callback
            butten_dict = {
                "label": "Attach callback",
                "type": "button",
                "a_text": "Attach",
                "b_text": "Remove",
                "tooltip": "Attach callback to the camera",
                "on_clicked_fn": self._attach_callback,
            }

            self._input_fields["attach_callback"] = state_btn_builder(**butten_dict)
            self._input_fields["attach_callback"].enabled = False

        return

    ##
    # Shutdown Helpers
    ##

    def on_shutdown(self):
        if self._window:
            self._window = None
        gc.collect()

    ##
    # Path Helpers
    ##

    def _check_save_path(self, path):
        path = path.get_value_as_string()

        if not os.path.isfile(path):
            self._input_fields["start_writer"].enabled = True
            self._config.set_save_path(path)
        else:
            self._input_fields["start_writer"].enabled = False
            carb.log_warn(f"Directory at save path {path} does not exist!")

    def _check_cam_prim(self, path) -> None:
        # check if prim exists
        if not prim_utils.is_prim_path_valid(path.get_value_as_string()):
            carb.log_verbose(f"Invalid prim path: {path.get_value_as_string()}")
        elif path.get_value_as_string() in self.camera_list:
            carb.log_warn(f"Annotators already generated for this camera: {path.get_value_as_string()}")
        else:
            self._input_fields["load_camera"].enabled = True
        return

    # FIXME: currently cannot set value because of read-only property
    def _make_ply_proposal(self, path: str) -> None:
        """use default matterport datastructure to make proposal about point-cloud file

        - "env_id"
            - matterport_mesh
                - "id_nbr"
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
            # self._input_fields["input_ply_file"].set_value = ply_file
        except FileNotFoundError:
            carb.log_verbose("No ply file found in default matterport datastructure")
        return

    ##
    # Load Mesh and Point-Cloud
    ##

    def _load_matterport3d(self) -> None:
        path = self._config.import_file_obj
        if not path:
            return

        # find obj, usd file
        if os.path.isabs(path):
            file_path = path
            assert os.path.isfile(file_path), f"No .obj or .usd file found under absolute path: {file_path}"
        else:
            file_path = os.path.join(self._extension_path, "data", path)
            assert os.path.isfile(
                file_path
            ), f"No .obj or .usd file found under relative path to extension data: {file_path}"
            self._config.set_import_file_obj(file_path)  # update config

        carb.log_verbose("MatterPort 3D Mesh found, start loading...")

        # matterport class
        self._matterport = MatterportWorld(self._config)
        asyncio.ensure_future(self._matterport.load_world_async())

        carb.log_info("MatterPort 3D Mesh loaded")
        self._input_fields["import_btn"].enabled = False
        return

    def _load_domains(self):
        path = self._config.import_file_ply
        if not isinstance(path, str):
            path = path.get_value_as_string()

        # find ply, usd file
        if os.path.isabs(path):
            file_path = path
            assert os.path.isfile(file_path), f"No .ply file found under absolute path: {file_path}"
        else:
            file_path = os.path.join(self._extension_path, "data", path)
            assert os.path.isfile(file_path), f"No .ply file found under relative path to extension data: {file_path}"
            self._config.set_import_file_ply(file_path)  # update config

        self.domains = MatterportWarp(self._config)

        self._input_fields["import_ply_btn"].enabled = False
        self._input_fields["explorer"].enabled = True
        self._input_fields["annotator_callback"].enabled = True
        self._input_fields["test"].enabled = True
        carb.log_info("MatterPort 3D Semantics Callback successfully loaded!")
        return

    ##
    # Register Cameras and Writers
    ##

    def _register_camera(self):
        path = self._input_fields["camera"].get_value_as_string()

        # check that prim path is Camera by trying to get typical camera attribute
        prim = prim_utils.get_prim_at_path(path)
        if not prim.GetAttribute("focalLength").IsValid():
            carb.log_warn(f"Prim {path} is not a camera! Please select a valid camera.")
            return

        # register camera
        self.domains.register_camera(
            prim,
            height=self._input_fields["cam_height"].get_value_as_int(),
            width=self._input_fields["cam_width"].get_value_as_int(),
            semantics=self._config.semantic,
            depth=self._config.depth,
            rgb=self._config.rgb,
            visualization=self._config.visualize,
        )

        # after successful load, add camera to list
        self.camera_list.append(path)

        # allow for tasks
        self._input_fields["start_writer"].enabled = True
        self._input_fields["attach_callback"].enabled = True
        return

    def _start_writer(self):
        # update config
        self._config.set_save(True)
        if self._config.save_path == self._default_save_path:
            ply_dir = os.path.split(self._config.import_file_ply)[0]
            matterport_dir = os.path.split(ply_dir)[0]
            data_dir = os.path.join(matterport_dir, "matterport_data")
            self._config.set_save_path(data_dir)
        # init saving
        self.domains.init_save()
        # disable button and print info
        self._input_fields["start_writer"].enabled = False
        carb.log_info("Writer started!")
        return

    ##
    # Tasks
    ##
    # Annotator Physics Callback
    def _rebuild_ui(self):
        return self.build_ui(task_callback=True)

    def _attach_callback(self, val) -> None:
        callback = MatterportCallbackDomains(self._config, self.domains)
        callback.set_domain_callback(val)
        return

    # Random Exploration
    def _start_explorer(self):
        self._input_fields["annotator_callback"].enabled = False
        self._input_fields["test"].enabled = False
        # start random exploration
        self.explorer = RandomExplorer(self.domains)
        self.explorer.setup()
        asyncio.ensure_future(self.explorer.explore())
        return

    def _depth_test(self):
        asyncio.ensure_future(test_depth_warp(self.domains))
        return


# EoF
