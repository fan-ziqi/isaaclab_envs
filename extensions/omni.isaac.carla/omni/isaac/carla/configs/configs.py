"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief      Configs
"""

# python
import os
from dataclasses import dataclass
from typing import Optional, Tuple

# isaac-orbit
from omni.isaac.orbit.sensors.camera import PinholeCameraCfg

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../data"))


@dataclass
class SimCfg:
    """Simulation physics."""

    dt = 0.005  # physics-dt:(s)
    substeps = 8  # rendering-dt = physics-dt * substeps (s)
    gravity = [0.0, 0.0, -9.81]  # (m/s^2)

    enable_scene_query_support = False  # disable scene query for more speed-up
    use_flatcache = True  # output from simulation to flat cache
    use_gpu_pipeline = True  # direct GPU access functionality
    device = "cpu"  # device on which to run simulation/environment

    @dataclass
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


@dataclass
class CarlaLoaderConfig:
    # carla map
    root_path: str = "/home/pascal/viplanner/env/carla/town01"  #  "/home/pascal/viplanner/env/nomoko/zurich" "/home/orbit/Downloads/park2_nice"
    usd_name: str = "Town01_Opt_paper.usd"  # "Zurich_3Dmodel.obj.usd"  "Showcase.usd"
    suffix: str = "/Town01_Opt"  # carla: "/Town01_Opt"  nomoko: "/Zurich"  park, warehouse: ""
    # prim path for the carla map
    prim_path: str = "/World/Carla"  # "/World/nomoko"  "/World/Park"  "/World/Carla"
    # SimCfg
    sim_cfg: SimCfg = SimCfg()
    # scale
    scale: float = 0.01  # 0.01  # carla: 0.01 nomoko: 1  park: 0.01 warehouse: 1.0 # scale the scene to be in meters
    # up axis
    axis_up: str = "Y"  # carla, nomoko: "Y", park, warehouse: "Z"
    # multipy crosswalks
    cw_config_file: Optional[str] = os.path.join(
        DATA_DIR, "town01", "cw_multiply_cfg.yml"
    )  # if None, no crosswalks are added
    # mesh to semantic class mapping --> only if set, semantic classes will be added to the scene
    sem_mesh_to_class_map: Optional[str] = os.path.join(
        DATA_DIR, "town01", "keyword_mapping.yml"
    )  # os.path.join(DATA_DIR, "park", "keyword_mapping.yml")  os.path.join(DATA_DIR, "town01", "keyword_mapping.yml")
    # add Groundplane to the scene
    groundplane: bool = True
    # add people to the scene
    people_config_file: Optional[str] = os.path.join(
        DATA_DIR, "town01", "people_cfg.yml"
    )  # if None, no people are added
    # multiply vehicles
    vehicle_config_file: Optional[str] = os.path.join(
        DATA_DIR, "town01", "vehicle_cfg.yml"
    )  # if None, no vehicles are added

    @property
    def usd_path(self) -> str:
        return os.path.join(self.root_path, self.usd_name)


@dataclass
class CarlaExplorerConfig:
    """Configuration for the CarlaMap class."""

    # coverage parameters
    points_per_m2: float = 0.5
    obs_loss_threshold: float = 0.8
    max_cam_recordings: Optional[int] = 10000  # if None, not limitation is applied
    # indoor filter (for outdoor maps filter inside of buildings as traversable, for inside maps set to False)
    indoor_filter: bool = True
    carla_filter: Optional[str] = os.path.join(DATA_DIR, "town01", "area_filter_cfg.yml")
    # nomoko model
    nomoko_model: bool = False
    # are limiter --> only select area within the defined prim names (e.g. "Road_SideWalk")
    space_limiter: Optional[str] = "Road_Sidewalk"  # carla: "Road_Sidewalk"  nomoko None  park: MergedRoad05
    # robot height
    robot_height = 0.7  # m
    # depth camera
    camera_cfg_depth: PinholeCameraCfg = PinholeCameraCfg(
        sensor_tick=0,
        height=480,
        width=848,
        data_types=["distance_to_image_plane"],
        usd_params=PinholeCameraCfg.UsdCameraCfg(
            focal_length=1.93, clipping_range=(0.01, 1.0e5), horizontal_aperture=3.8
        ),
    )
    camera_intrinsics_depth: Optional[Tuple[float]] = None
    # ANYmal D/C realsense:         (423.54608, 0.0, 427.69815, 0.0, 423.54608, 240.17773, 0.0, 0.0, 1.0)
    # RealSense D455:               (430.31607, 0.0, 428.28408, 0.0, 430.31607, 244.00695, 0.0, 0.0, 1.0)
    # ANYmal D wide_angle_camera: 1.0 <-> ANYmal C realsense: 1.93 <-> RealSense D455: 1.93
    camera_prim_depth: str = "/World/CameraSensor_depth"
    # semantic camera
    camera_cfg_sem: PinholeCameraCfg = PinholeCameraCfg(
        sensor_tick=0,
        height=720,  # 480,  # 1080
        width=1280,  # 848,  # 1440
        data_types=["rgb", "semantic_segmentation"],
        usd_params=PinholeCameraCfg.UsdCameraCfg(
            focal_length=1.93, clipping_range=(0.01, 1.0e5), horizontal_aperture=3.8
        ),
    )
    # ANYmal D wide_angle_camera: (1440, 1080)  <-> ANYmal C realsense (848, 480) <-> RealSense D455 (1280, 720)
    # ANYmal D wide_angle_camera: 1.93 <-> ANYmal C realsense: 1.93 <-> RealSense D455: 1.93
    camera_intrinsics_sem: Optional[Tuple[float]] = None
    # ANYmal D wide_angle_camera:   (575.60504, 0.0, 745.73121, 0.0, 578.56484, 519.52070, 0.0, 0.0, 1.0)
    # ANYmal C realsense:           (423.54608, 0.0, 427.69815, 0.0, 423.54608, 240.17773, 0.0, 0.0, 1.0)
    # RealSense D455:               (644.15496, 0.0, 639.53125, 0.0, 643.49212, 366.30880, 0.0, 0.0, 1.0)
    camera_prim_sem: str = "/World/CameraSensor_sem"
    x_angle_range: Tuple[float, float] = (-5, 5)  # downtilt angle of the camera in degree
    y_angle_range: Tuple[float, float] = (
        -2,
        5,
    )  # downtilt angle of the camera in degree  --> isaac convention, positive is downwards
    # image suffix
    depth_suffix = "_cam0"
    sem_suffix = "_cam1"
    # transformation from depth (src) to semantic camera (target)
    tf_pos: tuple = (0.0, 0.0, 0.0)  # (translation in depth frame)
    # ANYmal D: (-0.002, 0.025, 0.042)  <-> ANYmal C and RealSense D455: (0.0, 0.0, 0.0)
    tf_quat: tuple = (0.0, 0.0, 0.0, 1.0)  # xyzw quaternion format (rotation in depth frame)
    # ANYmal D: (0.001, 0.137, -0.000, 0.991)  <-> ANYmal C and RealSense D455: (0.0, 0.0, 0.0, 1.0)
    tf_quat_convention: str = "roll-pitch-yaw"  # or "isaac"
    # NOTE: if the quat follows the roll-pitch-yaw convention, i.e. x-forward, y-right, z-down, will be converted to the isaac convention
    # high resolution depth for reconstruction (in cityenviroment can otherwise lead to artifcats)
    # will now also take the depth image of the rgb camera and use its depth images for reconstruction
    high_res_depth: bool = False
    # output_dir
    output_root: Optional[
        str
    ] = "/home/pascal/viplanner/imperative_learning/data"  # if None, output dir is stored under root_dir
    output_dir_name: str = "town01_cam_mount_train_red"  # "nomoko_zurich"
    ros_p_mat: bool = True  # save intrinsic matrix in ros P-matrix format
    depth_scale: float = 1000.0  # scale depth values before saving s.t. mm resolution can be achieved

    # add more people to the scene
    nb_more_people: Optional[int] = 1200  # if None, no people are added
    random_seed: Optional[int] = 42  # if None, no seed is set

    @property
    def output_dir(self) -> str:
        if self.output_root is not None:
            return os.path.join(self.output_root, self.output_dir_name)
        else:
            return os.path.join(self.root_path, self.output_dir_name)
