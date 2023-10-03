#!/usr/bin/python
"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      World for Matterport3D Extension in Omniverse-Isaac Sim
"""

# python
import os

import carb

# omni
import omni
import omni.isaac.core.utils.prims as prim_utils

# isaac-core
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core.materials import PhysicsMaterial
from omni.isaac.core.objects.ground_plane import GroundPlane
from omni.isaac.core.prims import GeometryPrim, XFormPrim
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.matterport.config import MatterportConfig, SimCfg, ViewerCfg
from pxr import PhysxSchema

# isaac-orbit
from omni.isaac.orbit.utils.configclass import class_to_dict

# omni-isaac-matterport
from .usd_converter import MatterportConverter


class MatterportWorld:
    """
    Default stairs environment for testing
    """

    def __init__(
        self,
        matterport_cfg: MatterportConfig,
    ) -> None:
        """
        :param
        """
        # configs
        self._sim_cfg = matterport_cfg.sim
        self._viewer_cfg = matterport_cfg.viewer
        self._matterport_cfg = matterport_cfg

        # Simulation Context
        self.sim: SimulationContext = None

        # Converter
        self.converter: MatterportConverter = MatterportConverter(
            self._matterport_cfg.import_file_obj, self._matterport_cfg.asset_converter
        )
        return

    async def load_world_async(self) -> None:
        """Function called when clicking load buttton"""
        # create new stage
        await stage_utils.create_new_stage_async()

        # simulation settings
        # check if simulation context was created earlier or not.
        if SimulationContext.instance():
            SimulationContext.clear_instance()
            carb.log_warn("SimulationContext already loaded. Will clear now and init default SimulationContext")

        # create new simulation context

        self.sim = SimulationContext(
            stage_units_in_meters=1.0,
            physics_dt=self._sim_cfg.dt,
            rendering_dt=self._sim_cfg.dt * self._sim_cfg.substeps,
            backend="torch",
            sim_params=class_to_dict(self._sim_cfg.physx),
            device=self._sim_cfg.device,
        )
        # initialize simulation
        await self.sim.initialize_simulation_context_async()
        # create world
        await self.load_matterport()
        self._design_scene()
        # update stage for any remaining process.
        await stage_utils.update_stage_async()

        # reset the simulator
        # note: this plays the simulator which allows setting up all the physics handles.
        await self.sim.reset_async()
        await self.sim.pause_async()

        # Now we are ready!
        carb.log_info("[INFO]: Setup complete...")
        return

    def load_world(self) -> None:
        """Function called when clicking load buttton"""
        # create new stage
        stage_utils.create_new_stage()
        # simulation settings
        # check if simulation context was created earlier or not.
        if SimulationContext.instance():
            SimulationContext.clear_instance()
            carb.log_warn("SimulationContext already loaded. Will clear now and init default SimulationContext")

        # create new simulation context
        self.sim = SimulationContext(
            stage_units_in_meters=1.0,
            physics_dt=self._sim_cfg.dt,
            rendering_dt=self._sim_cfg.dt * self._sim_cfg.substeps,
            backend="torch",
            sim_params=class_to_dict(self._sim_cfg.physx),
            device=self._sim_cfg.device,
        )
        # create world
        self.load_matterport_sync()
        self._design_scene()
        # update stage for any remaining process.
        stage_utils.update_stage()
        # reset the simulator
        # note: this plays the simulator which allows setting up all the physics handles.
        self.sim.reset()
        self.sim.pause()
        # Now we are ready!
        carb.log_info("[INFO]: Setup complete...")
        return

    async def load_matterport(self) -> None:
        _, ext = os.path.splitext(self._matterport_cfg.import_file_obj)
        # if obj mesh --> convert to usd
        if ext == ".obj":
            await self.converter.convert_asset_to_usd()
        # add mesh to stage
        self.load_matterport_sync()

    def load_matterport_sync(self) -> None:
        base_path, _ = os.path.splitext(self._matterport_cfg.import_file_obj)
        name = "Matterport"

        assert os.path.exists(base_path + ".usd"), (
            "Matterport load sync can only handle '.usd' files not obj files. "
            "Please use the async function to convert the obj file to usd first (accessed over the extension in the GUI)"
        )

        self._xform_prim = prim_utils.create_prim(
            prim_path=self._matterport_cfg.prim_path, translation=(0.0, 0.0, 0.0), usd_path=base_path + ".usd"
        )

        collision_prim_path = (
            prim_utils.get_prim_children(self._xform_prim)[1].GetPrimPath().pathString
        )  # self._xfrom_prim.prim
        self._mesh_prim = GeometryPrim(
            prim_path=collision_prim_path,
            name=name + "_collision_plane",
            position=None,
            orientation=None,
            collision=True,
        )

        # add colliders and physics material
        if self._matterport_cfg.colliders:
            material = PhysicsMaterial(
                f"/World/PhysicsMaterial",
                static_friction=self._matterport_cfg.friction_static,
                dynamic_friction=self._matterport_cfg.friction_dynamic,
                restitution=self._matterport_cfg.restitution,
            )
            # enable patch-friction: yields better results!
            physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(material._prim)
            physx_material_api.CreateImprovePatchFrictionAttr().Set(self._matterport_cfg.improved_patch_friction)
            physx_material_api.CreateFrictionCombineModeAttr().Set(
                ["avg", "min", "mul", "max"][self._matterport_cfg.friction_combine_mode]
            )
            physx_material_api.CreateRestitutionCombineModeAttr().Set(
                ["avg", "min", "mul", "max"][self._matterport_cfg.restitution_combine_mode]
            )
            self._mesh_prim.apply_physics_material(material)

            if self._matterport_cfg.groundplane:
                _ = GroundPlane("/World/GroundPlane", z_position=0.0, physics_material=material, visible=False)

        return

    def _design_scene(self) -> None:
        """Design scene."""
        # Lights-1
        prim_utils.create_prim(
            "/World/Light/GreySphere",
            "SphereLight",
            translation=(4.5, 3.5, 10.0),
            attributes={"radius": 1.0, "intensity": 300.0, "color": (0.75, 0.75, 0.75)},
        )
        # Lights-2
        prim_utils.create_prim(
            "/World/Light/WhiteSphere",
            "SphereLight",
            translation=(-4.5, 3.5, 10.0),
            attributes={"radius": 1.0, "intensity": 300.0, "color": (1.0, 1.0, 1.0)},
        )

        # set camera view
        set_camera_view(eye=self._viewer_cfg.eye, target=self._viewer_cfg.lookat)

        return


# EoF
