#!/usr/bin/python
"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      Config for MatterPort3D Extension in Omniverse-Isaac Sim
"""

# python
from dataclasses import dataclass

# omni-kit
from omni.isaac.core.utils import extensions

# enable ROS bridge extension  --> otherwise rospy cannot be imported
extensions.enable_extension("omni.kit.asset_converter")
from omni.kit.asset_converter.impl import AssetConverterContext

# orbit
from omni.isaac.orbit.utils.configclass import configclass


@configclass
class SimCfg:
    """Simulation physics."""

    dt = 0.005  # physics-dt:(s)
    substeps = 8  # rendering-dt = physics-dt * substeps (s)
    gravity = [0.0, 0.0, -9.81]  # (m/s^2)

    enable_scene_query_support = False  # disable scene query for more speed-up
    use_flatcache = True  # output from simulation to flat cache
    use_gpu_pipeline = True  # direct GPU access functionality
    device = "cpu"  # device on which to run simulation/environment

    @configclass
    class PhysxCfg:
        """PhysX solver parameters."""

        worker_thread_count = 10  # note: unused
        solver_position_iteration_count = 4  # note: unused
        solver_velocity_iteration_count = 1  # note: unused
        enable_sleeping = True  # note: unused
        max_depenetration_velocity = 1.0  # note: unused
        contact_offset = 0.002  # note: unused
        rest_offset = 0.0  # note: unused

        use_gpu = True  # GPU dynamics pipeline and broad-phase type
        solver_type = 1  # 0: PGS, 1: TGS
        enable_stabilization = True  # additional stabilization pass in solver

        # (m/s): contact with relative velocity below this will not bounce
        bounce_threshold_velocity = 0.5
        # (m): threshold for contact point to experience friction force
        friction_offset_threshold = 0.04
        # (m): used to decide if contacts are close enough to merge into a single friction anchor point
        friction_correlation_distance = 0.025

        # GPU buffers parameters
        gpu_max_rigid_contact_count = 512 * 1024
        gpu_max_rigid_patch_count = 80 * 1024 * 2
        gpu_found_lost_pairs_capacity = 1024 * 1024 * 2
        gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 32
        gpu_total_aggregate_pairs_capacity = 1024 * 1024 * 2
        gpu_max_soft_body_contacts = 1024 * 1024
        gpu_max_particle_contacts = 1024 * 1024
        gpu_heap_capacity = 128 * 1024 * 1024
        gpu_temp_buffer_capacity = 32 * 1024 * 1024
        gpu_max_num_partitions = 8

    physx: PhysxCfg = PhysxCfg()


@configclass
class ViewerCfg:
    """Configuration of the scene viewport camera."""

    debug_vis = True  # view debug prims in the scene
    eye = [2.5, 2.5, 2.5]  # camera eye position(in m)
    lookat = [0.0, 0.0, 0.0]  # target position (in m)


# NOTE: hopefully will be soon changed to dataclass, then initialization can be improved
asset_converter_cfg: AssetConverterContext = AssetConverterContext()
asset_converter_cfg.ignore_materials = False
# Don't import/export materials
asset_converter_cfg.ignore_animations = False
# Don't import/export animations
asset_converter_cfg.ignore_camera = False
# Don't import/export cameras
asset_converter_cfg.ignore_light = False
# Don't import/export lights
asset_converter_cfg.single_mesh = False
# By default, instanced props will be export as single USD for reference. If
# this flag is true, it will export all props into the same USD without instancing.
asset_converter_cfg.smooth_normals = True
# Smoothing normals, which is only for assimp backend.
asset_converter_cfg.export_preview_surface = False
# Imports material as UsdPreviewSurface instead of MDL for USD export
asset_converter_cfg.use_meter_as_world_unit = True
# Sets world units to meters, this will also scale asset if it's centimeters model.
asset_converter_cfg.create_world_as_default_root_prim = True
# Creates /World as the root prim for Kit needs.
asset_converter_cfg.embed_textures = True
# Embedding textures into output. This is only enabled for FBX and glTF export.
asset_converter_cfg.convert_fbx_to_y_up = False
# Always use Y-up for fbx import.
asset_converter_cfg.convert_fbx_to_z_up = True
# Always use Z-up for fbx import.
asset_converter_cfg.keep_all_materials = False
# If it's to remove non-referenced materials.
asset_converter_cfg.merge_all_meshes = False
# Merges all meshes to single one if it can.
asset_converter_cfg.use_double_precision_to_usd_transform_op = False
# Uses double precision for all transform ops.
asset_converter_cfg.ignore_pivots = False
# Don't export pivots if assets support that.
asset_converter_cfg.disabling_instancing = False
# Don't export instancing assets with instanceable flag.
asset_converter_cfg.export_hidden_props = False
# By default, only visible props will be exported from USD exporter.
asset_converter_cfg.baking_scales = False
# Only for FBX. It's to bake scales into meshes.


@dataclass
class MatterportConfig:
    # config classes
    sim: SimCfg = SimCfg()
    asset_converter: AssetConverterContext = asset_converter_cfg
    viewer: ViewerCfg = ViewerCfg()
    # physics material
    colliders: bool = True
    friction_dynamic: float = 0.7
    friction_static: float = 0.7
    restitution: float = 0.0
    friction_combine_mode: int = 3  # 0: Average, 1: Minimum, 2: Multiply, 3: Maximum
    restitution_combine_mode: int = 3  # 0: Average, 1: Minimum, 2: Multiply, 3: Maximum
    improved_patch_friction: bool = True
    groundplane: bool = True
    # file locations
    # FIXME: remove default when released
    import_file_obj: str = "/home/pascal/viplanner/env/matterport/v1/scans/2n8kARJN3HM/2n8kARJN3HM/matterport_mesh/0c334eaabb844eaaad049cbbb2e0a4f2/0c334eaabb844eaaad049cbbb2e0a4f2.obj"
    import_file_ply: str = (
        "/home/pascal/viplanner/env/matterport/v1/scans/2n8kARJN3HM/2n8kARJN3HM/house_segmentations/2n8kARJN3HM.ply"
    )
    # prim path of environment
    prim_path: str = "/World/Matterport"
    # semantic and depth inforamtion (can be changed individually for each camera)
    semantic: bool = True
    depth: bool = True
    rgb: bool = False
    visualize: bool = False
    # callback parameters
    compute_frequency: int = 16
    # depth parameters
    max_depth: float = 15.0
    depth_scale: float = 1000.0  # scale depth for saving to have millimeter precision
    # saving
    save: bool = False
    save_path: str = "(same as source)"
    ros_p_mat: bool = True  # save intrinsics in ROS P matrix format

    # VIPlanner connection
    viplanner_meta: bool = True
    # NOTE: include fle in config directory: TODO: add file github link

    # set value functions
    def set_colliders(self, value: bool):
        self.colliders = value

    def set_friction_dynamic(self, value: float):
        self.friction_dynamic = value

    def set_friction_static(self, value: float):
        self.friction_static = value

    def set_friction_combine_mode(self, value: int):
        self.friction_combine_mode = value

    def set_restitution_combine_mode(self, value: int):
        self.restitution_combine_mode = value

    def set_improved_patch_friction(self, value: bool):
        self.improved_patch_friction = value

    def set_import_file_obj(self, value: str):
        self.import_file_obj = value

    def set_import_file_ply(self, value: str):
        self.import_file_ply = value

    def set_prim_path(self, value: str):
        self.prim_path = value

    def set_semantic(self, value: bool):
        self.semantic = value

    def set_depth(self, value: bool):
        self.depth = value

    def set_rgb(self, value: bool):
        self.rgb = value

    def set_visualize(self, value: bool):
        self.visualize = value

    def set_save(self, value: bool):
        self.save = value

    def set_save_frequency(self, value: int):
        self.compute_frequency = value

    def set_save_path(self, value: str):
        self.save_path = value


# EoF
