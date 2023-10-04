#!/usr/bin/python
"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      Config for MatterPort3D Extension in Omniverse-Isaac Sim
"""

# python
from dataclasses import dataclass

# orbit
from omni.isaac.orbit.envs import ViewerCfg
from .importer_cfg import MatterportImporterCfg


@dataclass
class MatterportConfig:
    # config classes
    viewer: ViewerCfg = ViewerCfg()
    importer: MatterportImporterCfg = MatterportImporterCfg()
    # semantic and depth inforamtion (can be changed individually for each camera)
    visualize: bool = False
    # callback parameters
    compute_frequency: int = 16
    # depth parameters
    depth_scale: float = 1000.0  # scale depth for saving to have millimeter precision
    # saving
    save: bool = False
    save_path: str = "(same as source)"
    ros_p_mat: bool = True  # save intrinsics in ROS P matrix format

    # set value functions
    def set_friction_dynamic(self, value: float):
        self.importer.physics_material.dynamic_friction = value

    def set_friction_static(self, value: float):
        self.importer.physics_material.static_friction = value

    def set_restitution(self, value: float):
        self.importer.physics_material.restitution = value

    def set_friction_combine_mode(self, value: int):
        self.importer.physics_material.friction_combine_mode = value

    def set_restitution_combine_mode(self, value: int):
        self.importer.physics_material.restitution_combine_mode = value

    def set_improved_patch_friction(self, value: bool):
        self.importer.physics_material.improve_patch_friction = value

    def set_import_file_obj(self, value: str):
        self.importer.import_file_obj = value

    def set_import_file_ply(self, value: str):
        self.importer.import_file_ply = value

    def set_prim_path(self, value: str):
        self.importer.prim_path = value

    def set_visualize(self, value: bool):
        self.visualize = value

    def set_save(self, value: bool):
        self.save = value

    def set_save_frequency(self, value: int):
        self.compute_frequency = value

    def set_save_path(self, value: str):
        self.save_path = value
