from omni.isaac.orbit.utils import configclass


@configclass
class ReconstructionCfg:
    """
    Arguments for 3D reconstruction using depth maps
    """

    # directory where the environment with the depth (and semantic) images is located
    data_dir: str = "${USER_PATH_TO_DATA}"
    # environment name
    env: str = "town01"
    # image suffix
    depth_suffix = "_cam0"
    sem_suffix = "_cam1"
    # higher resolution depth images available for reconstruction  (meaning that the depth images are also taked by the semantic camera)
    high_res_depth: bool = False

    # reconstruction parameters
    voxel_size: float = 0.05  # [m] 0.05 for matterport 0.1 for carla
    start_idx: int = 0  # start index for reconstruction
    max_images: Optional[int] = 1000  # maximum number of images to reconstruct, if None, all images are used
    depth_scale: float = 1000.0  # depth scale factor
    # semantic reconstruction
    semantics: bool = True

    # speed vs. memory trade-off parameters
    point_cloud_batch_size: int = (
        200  # 3d points of nbr images added to point cloud at once (higher values use more memory but faster)
    )

    """ Internal functions """

    def get_data_path(self) -> str:
        return os.path.join(self.data_dir, self.env)

    def get_out_path(self) -> str:
        return os.path.join(self.out_dir, self.env)