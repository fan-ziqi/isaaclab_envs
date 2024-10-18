# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
from typing import Literal

import carb
import omni
import omni.client
import omni.ext
import omni.isaac.lab.sim as sim_utils
import omni.ui as ui
from omni.isaac.lab.assets import RigidObjectCfg
from omni.isaac.lab.sensors import CameraCfg, RayCasterCameraCfg, patterns
from omni.isaac.lab.utils import configclass
from omni.isaac.ui.ui_utils import (
    btn_builder,
    cb_builder,
    float_builder,
    get_style,
    int_builder,
    setup_ui_headers,
    str_builder,
)
from orbit.nav.collectors.collectors import (
    TrajectorySampling,
    TrajectorySamplingCfg,
    ViewpointSampling,
    ViewpointSamplingCfg,
)
from orbit.nav.importer.scripts import (
    ImportSceneCfg,
    OrbitNavImporterExtension,
    import_class,
)
from orbit.nav.importer.sensors import MatterportRayCasterCameraCfg
from orbit.nav.importer.utils.toggleable_window import ToggleableWindow

EXTENSION_NAME = "Orbit Navigation Data Collectors"


@configclass
class CollectorSceneCfg(ImportSceneCfg):
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            visible=False,
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
    """For the construction of the scene, need an articualted or rigid object"""


class OrbitNavCollectorExtension(OrbitNavImporterExtension):
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

        # path to extension
        self._extension_path = omni.kit.app.get_app().get_extension_manager().get_extension_path(ext_id)

        # scene config constructed from the UI
        self._scene_cfg = CollectorSceneCfg()
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

    ##
    # UI Build functions
    ##

    def build_ui(self, task_space=False):
        with self._window.frame:
            with ui.ScrollingFrame(horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                with ui.VStack(spacing=5, height=0):
                    self._build_info_ui()

                    if not task_space:
                        self._build_data_origin_ui()
                        self._build_import_ui()
                        self._build_camera_ui()
                        self._build_load_ui()
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

        overview = (
            "This utility is used to sample trajectories and viewpoints with the option to fast render images in the"
            " navigation environments. The environments are imported using the `orbit.nav.importer` extension."
            "\n\nPress the 'Open in IDE' button to view the source code."
        )

        setup_ui_headers(self._ext_id, __file__, title, doc_link, overview)

    def _build_camera_ui(self):
        frame = ui.CollapsableFrame(
            title="Camera Sensor Parameters",
            height=0,
            collapsed=False,
            style=get_style(),
            style_type_name_override="CollapsableFrame",
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                # define sensor parameters
                self._sensor_input_fields["camera_semantics"] = cb_builder(
                    label="Semantic Domain",
                    tooltip="Enable access to the semantics information of the mesh (default: True)",
                    default_val=True,
                )
                self._sensor_input_fields["camera_depth"] = cb_builder(
                    label="Distance to Camera Frame Domain",
                    tooltip=(
                        "Enable access to the depth information of the mesh - no additional compute effort (default:"
                        " True)"
                    ),
                    default_val=True,
                )
                self._sensor_input_fields["cam_height"] = int_builder(
                    "Camera Height in Pixels",
                    default_val=480,
                    tooltip="Set the height of the camera image plane in pixels (default: 480)",
                )
                self._sensor_input_fields["cam_width"] = int_builder(
                    "Camera Width in Pixels",
                    default_val=640,
                    tooltip="Set the width of the camera image plane in pixels (default: 640)",
                )
                self._sensor_input_fields["focal_length"] = float_builder(
                    "Focal Length",
                    default_val=24.0,
                    tooltip="Set the focal length of the camera in mm (default: 24.0)",
                )
                self._sensor_input_fields["horizontal_aperture"] = float_builder(
                    "Horizontal Aperture",
                    default_val=20.955,
                    tooltip="Set the horizontal aperture of the camera in mm (default: 20.955)",
                )

                self._sensor_input_fields["add_camera"] = btn_builder(
                    "Add Camera", text="Add Camera", on_clicked_fn=self._add_camera_to_scene
                )

    def _build_task_ui(self):
        frame = ui.CollapsableFrame(
            title="Sampling Tasks",
            height=0,
            collapsed=False,
            style=get_style(),
            style_type_name_override="CollapsableFrame",
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):

                self._input_fields["traj_sampling_module_name"] = str_builder(
                    label="Trajectory Module Name",
                    default_val="orbit.nav.collectors.collectors",
                    tooltip="Module name of the Trajectory Sampling Cfg class",
                )
                self._input_fields["traj_sampling_class_name"] = str_builder(
                    label="Trajectory Class Name",
                    default_val="TrajectorySamplingCfg",
                    tooltip="Class name of Trajectory Sampling Cfg",
                )
                self._input_fields["traj_sampling_nbr_samples"] = int_builder(
                    "Number Trajectory Samples",
                    default_val=1000,
                    tooltip="How many trajectories should be sampled (default: 1000)",
                )
                self._input_fields["traj_sampling_min_length"] = float_builder(
                    "Min Trajectory Length",
                    default_val=0.0,
                    tooltip="Minimum trajectory length in Meter (default: 0.0)",
                )
                self._input_fields["traj_sampling_max_length"] = float_builder(
                    "Max Trajectory Length",
                    default_val=10.0,
                    tooltip="Maximum trajectory length in Meter (default: 10.0)",
                )
                self._input_fields["traj_sampling_btn"] = btn_builder(
                    "Trajectory Sampling", text="Start", on_clicked_fn=self._execute_trajectory_sampling
                )

                self._input_fields["viewpoint_sampling_module_name"] = str_builder(
                    label="Viewpoint Module Name",
                    default_val="orbit.nav.collectors.collectors",
                    tooltip="Module name of the Viewpoint Sampling Cfg class",
                )
                self._input_fields["viewpoint_sampling_class_name"] = str_builder(
                    label="Viewpoint Class Name",
                    default_val="ViewpointSamplingCfg",
                    tooltip="Class name of Viewpoint Sampling Cfg",
                )
                self._input_fields["viewpoint_sampling_nbr_samples"] = int_builder(
                    "Number Viepoint Samples",
                    default_val=1000,
                    tooltip="How many viepoints should be sampled (default: 1000)",
                )

                self._input_fields["viewpoint_sampling_btn"] = btn_builder(
                    "Viewpoint Sampling", text="Start", on_clicked_fn=self._execute_viewpoint_sampling
                )
                self._input_fields["viewpoint_rendering_btn"] = btn_builder(
                    "Viewpoint Rendering", text="Start", on_clicked_fn=self._execute_viewpoint_rendering
                )
                # disable rendering button until viewpoints are sampled
                self._input_fields["viewpoint_rendering_btn"].enabled = False

    ##
    # Load Mesh and Point-Cloud
    ##

    def _start_loading(self):
        super()._start_loading()
        self.build_ui(task_space=True)
        self._sensor_input_fields["add_camera"].enabled = False

    ##
    # Add camera to scene
    ##

    def _add_camera_to_scene(self):
        data_types = []
        if self._sensor_input_fields["camera_semantics"].get_value_as_bool():
            data_types += ["semantic_segmentation"]
        if self._sensor_input_fields["camera_depth"].get_value_as_bool():
            data_types += ["distance_to_image_plane"]

        if self._mesh_origin == "matterport":
            assert self._input_fields["input_ply_file"].get_value_as_string(), "No ply file found"

            # add specialized matterport raycaster camera to scene
            setattr(
                self._scene_cfg,
                f"camera_{self._camera_idx}",
                MatterportRayCasterCameraCfg(
                    prim_path="{ENV_REGEX_NS}/cube",
                    mesh_prim_paths=[self._input_fields["input_ply_file"].get_value_as_string()],
                    update_period=0,
                    data_types=data_types,
                    debug_vis=True,
                    pattern_cfg=patterns.PinholeCameraPatternCfg(
                        focal_length=self._sensor_input_fields["focal_length"].get_value_as_float(),
                        horizontal_aperture=self._sensor_input_fields["horizontal_aperture"].get_value_as_float(),
                        height=self._sensor_input_fields["cam_height"].get_value_as_int(),
                        width=self._sensor_input_fields["cam_width"].get_value_as_int(),
                    ),
                ),
            )
        elif self._mesh_origin == "multi-mesh-usd":
            # add usd camera to scene
            setattr(
                self._scene_cfg,
                f"camera_{self._camera_idx}",
                CameraCfg(
                    prim_path=f"{self._input_fields['prim_path'].get_value_as_string()}/camera_{self._camera_idx}",
                    data_types=data_types,
                    debug_vis=True,
                    height=self._sensor_input_fields["cam_height"].get_value_as_int(),
                    width=self._sensor_input_fields["cam_width"].get_value_as_int(),
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=self._sensor_input_fields["focal_length"].get_value_as_float(),
                        horizontal_aperture=self._sensor_input_fields["horizontal_aperture"].get_value_as_float(),
                    ),
                ),
            )
        elif self._mesh_origin == "single-mesh-usd" or self._mesh_origin == "generator":
            assert (
                "semantic_segmentation" not in data_types
            ), "Semantic segmentation is currently not supported for single-mesh-usd and generator"

            # add orbit ray caster camera to scene
            setattr(
                self._scene_cfg,
                f"camera_{self._camera_idx}",
                RayCasterCameraCfg(
                    prim_path=f"{self._input_fields['prim_path'].get_value_as_string()}/cube",
                    mesh_prim_paths=[self._input_fields["prim_path"].get_value_as_string()],
                    update_period=0,
                    data_types=data_types,
                    debug_vis=True,
                    pattern_cfg=patterns.PinholeCameraPatternCfg(
                        focal_length=self._sensor_input_fields["focal_length"].get_value_as_float(),
                        horizontal_aperture=self._sensor_input_fields["horizontal_aperture"].get_value_as_float(),
                        height=self._sensor_input_fields["cam_height"].get_value_as_int(),
                        width=self._sensor_input_fields["cam_width"].get_value_as_int(),
                    ),
                ),
            )
        else:
            carb.log_warn(f"Invalid mesh origin: {self._mesh_origin}")
            return

        print(f"[INFO] Added camera_{self._camera_idx} to scene")
        self._camera_idx += 1

    ##
    # Sampling tasks function
    ##

    def _execute_trajectory_sampling(self):
        if not hasattr(self, "_traj_explorer"):
            # get the config
            traj_sampling_cfg: TrajectorySamplingCfg = import_class(
                self._input_fields["traj_sampling_module_name"].get_value_as_string(),
                self._input_fields["traj_sampling_class_name"].get_value_as_string(),
            )()
            # execute trajectory sampling
            self._traj_explorer = TrajectorySampling(traj_sampling_cfg, scene=self.scene)

        self.sim.play()
        self._traj_explorer.sample_paths(
            [self._input_fields["traj_sampling_nbr_samples"].get_value_as_int()],
            [self._input_fields["traj_sampling_min_length"].get_value_as_float()],
            [self._input_fields["traj_sampling_max_length"].get_value_as_float()],
        )

    def _execute_viewpoint_sampling(self):
        if not hasattr(self, "_viewpoint_explorer"):
            # get the config
            viewpoint_sampling_cfg: ViewpointSamplingCfg = import_class(
                self._input_fields["viewpoint_sampling_module_name"].get_value_as_string(),
                self._input_fields["viewpoint_sampling_class_name"].get_value_as_string(),
            )()
            # execute viewpoint sampling
            self._viewpoint_explorer = ViewpointSampling(viewpoint_sampling_cfg, scene=self.scene)

        self._viepoint_samples = self._viewpoint_explorer.sample_viewpoints(
            self._input_fields["viewpoint_sampling_nbr_samples"].get_value_as_int()
        )
        # enable rendering button
        self._input_fields["viewpoint_rendering_btn"].enabled = True

    def _execute_viewpoint_rendering(self):
        if not hasattr(self, "_viewpoint_explorer") or not hasattr(self, "_viepoint_samples"):
            carb.log_warn("No viewpoint explorer found. Please sample viewpoints first.")
            return

        self._viewpoint_explorer.render_viewpoints(self._viepoint_samples)
