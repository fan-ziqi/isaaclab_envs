# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
@author     Pascal Roth
@email      rothpa@ethz.ch

@brief      Exploration of Matterport3D Dataset
"""


"""
Launch Omniverse Toolkit first.
"""

# python
import argparse
import os

# omni-isaac-orbit
from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_true", default=True, help="Force display off at all times.")
args_cli = parser.parse_args()

# load cheaper kit config in headless
if args_cli.headless:
    app_experience = f"{os.environ['ISAACSIM_PATH']}/apps/omni.isaac.sim.python.gym.headless.render.kit"
else:
    app_experience = f"{os.environ['ISAACSIM_PATH']}/apps/omni.isaac.sim.python.kit"

# launch omniverse app
config = {"headless": args_cli.headless}
simulation_app = SimulationApp(config, experience=app_experience)


"""
Rest everything follows.
"""

# python
import torch

# omni
if args_cli.headless:
    from omni.isaac.core.utils import extensions

    extensions.enable_extension("omni.isaac.debug_draw")
    extensions.enable_extension("omni.replicator.core")
    extensions.enable_extension("omni.kit.manipulator.viewport")

from omni.isaac.matterport.config import MatterportConfig, SamplerCfg
from omni.isaac.matterport.exploration import RandomExplorer

# omni-isaac-matterport
from omni.isaac.matterport.semantics import MatterportWarp
from omni.isaac.matterport.utils import MatterportWorld


def explore(config: MatterportConfig, sample_cfg: SamplerCfg = SamplerCfg()) -> None:
    # load world --> enable RGB image recording
    world = MatterportWorld(config)
    world.load_world()

    # create warp domain
    domains = MatterportWarp(config)

    # start random exploration
    explorer = RandomExplorer(domains, sample_cfg)
    explorer.setup()
    explorer.explore()


if __name__ == "__main__":
    # ENV 2n8kARJN3HM
    config_2n8kARJN3HM = MatterportConfig(
        obj_filepath="/home/pascal/viplanner/env/matterport/v1/scans/2n8kARJN3HM/2n8kARJN3HM/matterport_mesh/0c334eaabb844eaaad049cbbb2e0a4f2/0c334eaabb844eaaad049cbbb2e0a4f2.obj",
        import_file_ply=(
            "/home/pascal/viplanner/env/matterport/v1/scans/2n8kARJN3HM/2n8kARJN3HM/house_segmentations/2n8kARJN3HM.ply"
        ),
        save=True,
        save_path="/home/pascal/viplanner/imperative_learning/data/2n8kARJN3HM_cam_mounted",
    )
    explore(config_2n8kARJN3HM)
    torch.cuda.empty_cache()

    # ENV 2azQ1b91cZZ
    config_2azQ1b91cZZ = MatterportConfig(
        obj_filepath="/home/pascal/viplanner/env/matterport/v1/scans/2azQ1b91cZZ/2azQ1b91cZZ/matterport_mesh/7812e14df5e746388ff6cfe8b043950a/7812e14df5e746388ff6cfe8b043950a.obj",
        import_file_ply=(
            "/home/pascal/viplanner/env/matterport/v1/scans/2azQ1b91cZZ/2azQ1b91cZZ/house_segmentations/2azQ1b91cZZ.ply"
        ),
        save=True,
        save_path="/home/pascal/viplanner/imperative_learning/data/2azQ1b91cZZ_cam_mounted",
    )
    sample_cfg_2azQ1b91cZZ = SamplerCfg(
        points_per_m2=15,
        min_wall_distance=0.3,
        max_images=3500,
    )
    explore(config_2azQ1b91cZZ, sample_cfg_2azQ1b91cZZ)
    torch.cuda.empty_cache()

    # ENV JeFG25nYj2p
    config_JeFG25nYj2p = MatterportConfig(
        obj_filepath="/home/pascal/viplanner/env/matterport/v1/scans/JeFG25nYj2p/JeFG25nYj2p/matterport_mesh/7b285021f3114c4cb66675cbd139cd17/7b285021f3114c4cb66675cbd139cd17.obj",
        import_file_ply=(
            "/home/pascal/viplanner/env/matterport/v1/scans/JeFG25nYj2p/JeFG25nYj2p/house_segmentations/JeFG25nYj2p.ply"
        ),
        save=True,
        save_path="/home/pascal/viplanner/imperative_learning/data/JeFG25nYj2p_cam_mounted",
    )
    sample_cfg_JeFG25nYj2p = SamplerCfg(
        points_per_m2=15,
        min_wall_distance=0.3,
    )
    explore(config_JeFG25nYj2p, sample_cfg_JeFG25nYj2p)
    torch.cuda.empty_cache()

    # ENV Vvot9Ly1tCj
    config_Vvot9Ly1tCj = MatterportConfig(
        obj_filepath="/home/pascal/viplanner/env/matterport/v1/scans/Vvot9Ly1tCj/Vvot9Ly1tCj/matterport_mesh/8caaade0a587493ca329937a41be44fc/8caaade0a587493ca329937a41be44fc.obj",
        import_file_ply=(
            "/home/pascal/viplanner/env/matterport/v1/scans/Vvot9Ly1tCj/Vvot9Ly1tCj/house_segmentations/Vvot9Ly1tCj.ply"
        ),
        save=True,
        save_path="/home/pascal/viplanner/imperative_learning/data/Vvot9Ly1tCj_cam_mounted",
    )
    sample_cfg_Vvot9Ly1tCj = SamplerCfg(
        points_per_m2=10,
    )
    explore(config_Vvot9Ly1tCj, sample_cfg_Vvot9Ly1tCj)
    torch.cuda.empty_cache()

    # ENV ur6pFq6Qu1A
    config_ur6pFq6Qu1A = MatterportConfig(
        obj_filepath="/home/pascal/viplanner/env/matterport/v1/scans/ur6pFq6Qu1A/ur6pFq6Qu1A/matterport_mesh/b693ef1b45de41a6a51bdbf5ee631907/b693ef1b45de41a6a51bdbf5ee631907.obj",
        import_file_ply=(
            "/home/pascal/viplanner/env/matterport/v1/scans/ur6pFq6Qu1A/ur6pFq6Qu1A/house_segmentations/ur6pFq6Qu1A.ply"
        ),
        save=True,
        save_path="/home/pascal/viplanner/imperative_learning/data/ur6pFq6Qu1A_cam_mounted",
    )
    sample_cfg_ur6pFq6Qu1A = SamplerCfg(
        points_per_m2=5,
        max_images=2500,
        min_wall_distance=0.3,
    )
    explore(config_ur6pFq6Qu1A, sample_cfg_ur6pFq6Qu1A)
    torch.cuda.empty_cache()

    # ENV B6ByNegPMKs
    config_B6ByNegPMKs = MatterportConfig(
        obj_filepath="/home/pascal/viplanner/env/matterport/v1/scans/B6ByNegPMKs/B6ByNegPMKs/matterport_mesh/85cef4a4c3c244479c56e56d9a723ad2/85cef4a4c3c244479c56e56d9a723ad2.obj",
        import_file_ply=(
            "/home/pascal/viplanner/env/matterport/v1/scans/B6ByNegPMKs/B6ByNegPMKs/house_segmentations/B6ByNegPMKs.ply"
        ),
        save=True,
        save_path="/home/pascal/viplanner/imperative_learning/data/B6ByNegPMKs_cam_mounted",
    )
    sample_cfg_B6ByNegPMKs = SamplerCfg(
        points_per_m2=5,
        max_images=4000,
    )
    explore(config_B6ByNegPMKs, sample_cfg_B6ByNegPMKs)
    # torch.cuda.empty_cache()

    # ENV 8WUmhLawc2A
    config_8WUmhLawc2A = MatterportConfig(
        obj_filepath="/home/pascal/viplanner/env/matterport/v1/scans/8WUmhLawc2A/8WUmhLawc2A/matterport_mesh/caef338e1683434ba3a471ead89008cc/caef338e1683434ba3a471ead89008cc.obj",
        import_file_ply=(
            "/home/pascal/viplanner/env/matterport/v1/scans/8WUmhLawc2A/8WUmhLawc2A/house_segmentations/8WUmhLawc2A.ply"
        ),
        save=True,
        save_path="/home/pascal/viplanner/imperative_learning/data/8WUmhLawc2A_cam_mounted",
    )
    sample_cfg_8WUmhLawc2A = SamplerCfg(
        points_per_m2=15,
        min_wall_distance=0.3,
    )
    explore(config_8WUmhLawc2A, sample_cfg_8WUmhLawc2A)
    torch.cuda.empty_cache()

    # ENV E9uDoFAP3SH
    config_E9uDoFAP3SH = MatterportConfig(
        obj_filepath="/home/pascal/viplanner/env/matterport/v1/scans/E9uDoFAP3SH/E9uDoFAP3SH/matterport_mesh/e996abcc45ad411fa7f406025fcf2a63/e996abcc45ad411fa7f406025fcf2a63.obj",
        import_file_ply=(
            "/home/pascal/viplanner/env/matterport/v1/scans/E9uDoFAP3SH/E9uDoFAP3SH/house_segmentations/E9uDoFAP3SH.ply"
        ),
        save=True,
        save_path="/home/pascal/viplanner/imperative_learning/data/E9uDoFAP3SH_cam_mounted",
    )
    sample_cfg_E9uDoFAP3SH = SamplerCfg(
        points_per_m2=7,
    )
    explore(config_E9uDoFAP3SH, sample_cfg_E9uDoFAP3SH)
    torch.cuda.empty_cache()

    # ENV QUCTc6BB5sX
    config_QUCTc6BB5sX = MatterportConfig(
        obj_filepath="/home/pascal/viplanner/env/matterport/v1/scans/QUCTc6BB5sX/QUCTc6BB5sX/matterport_mesh/0685d2c5313948bd94e920b5b9e1a7b2/0685d2c5313948bd94e920b5b9e1a7b2.obj",
        import_file_ply=(
            "/home/pascal/viplanner/env/matterport/v1/scans/QUCTc6BB5sX/QUCTc6BB5sX/house_segmentations/QUCTc6BB5sX.ply"
        ),
        save=True,
        save_path="/home/pascal/viplanner/imperative_learning/data/QUCTc6BB5sX_cam_mounted",
    )
    sample_cfg_QUCTc6BB5sX = SamplerCfg(
        points_per_m2=15,
    )
    explore(config_QUCTc6BB5sX, sample_cfg_QUCTc6BB5sX)
    torch.cuda.empty_cache()

    # ENV YFuZgdQ5vWj
    config_YFuZgdQ5vWj = MatterportConfig(
        obj_filepath="/home/pascal/viplanner/env/matterport/v1/scans/YFuZgdQ5vWj/YFuZgdQ5vWj/matterport_mesh/d4f8bb56230e4695860deffe751adf20/d4f8bb56230e4695860deffe751adf20.obj",
        import_file_ply=(
            "/home/pascal/viplanner/env/matterport/v1/scans/YFuZgdQ5vWj/YFuZgdQ5vWj/house_segmentations/YFuZgdQ5vWj.ply"
        ),
        save=True,
        save_path="/home/pascal/viplanner/imperative_learning/data/YFuZgdQ5vWj_cam_mounted",
    )
    sample_cfg_YFuZgdQ5vWj = SamplerCfg(
        min_wall_distance=0.3,
    )
    explore(config_YFuZgdQ5vWj, sample_cfg_YFuZgdQ5vWj)
    torch.cuda.empty_cache()

    # Close the simulator
    simulation_app.close()

# EOF
