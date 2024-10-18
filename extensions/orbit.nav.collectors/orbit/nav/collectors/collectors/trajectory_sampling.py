# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import pickle
import random

import torch
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.sim import SimulationContext

from .terrain_analysis import TerrainAnalysis
from .trajectory_sampling_cfg import TrajectorySamplingCfg


class TrajectorySampling:
    def __init__(self, cfg: TrajectorySamplingCfg, scene: InteractiveScene | None = None):
        # save cfg and env
        self.cfg = cfg

        # get or setup scene
        if scene:
            self.scene = scene
        else:
            self.scene = InteractiveScene(self.cfg.exploration_scene)
            # reset simulation to initialize buffers
            sim = SimulationContext.instance()
            sim.reset()

        # analyse terrains
        self.terrain_analyser = TerrainAnalysis(self.cfg.terrain_analysis, scene=self.scene)

    def sample_paths(self, num_paths, min_path_length, max_path_length, seed: int = 1) -> torch.Tensor:
        # check dimensions
        assert (
            len(num_paths) == len(min_path_length) == len(max_path_length)
        ), "Number of paths, min path length and max path length must be equal"

        # the data is stored in torch tensors with the structure
        # [start_x, start_y, start_z, goal_x, goal_y, goal_z, path_length]
        data = torch.empty(0, 7)

        # load paths if they exist
        num_paths_to_explore = []
        min_path_length_to_explore = []
        max_path_length_to_explore = []
        for num_path, min_len, max_len in zip(num_paths, min_path_length, max_path_length):
            filename = self._get_save_path_trajectories(seed, num_path, min_len, max_len)
            if os.path.isfile(filename):
                with open(filename, "rb") as f:
                    saved_paths = pickle.load(f)
                # add loaded path dict to data dict
                data = torch.concatenate((data, saved_paths))
                print(f"[INFO] Loaded {num_path} with [{min_len},{max_len}] length generated with seed {seed}.")
            else:
                num_paths_to_explore.append(num_path)
                min_path_length_to_explore.append(min_len)
                max_path_length_to_explore.append(max_len)

        if len(num_paths_to_explore) == 0:
            return data

        # analyse terrain if not done yet
        if not self.terrain_analyser.complete:
            self.terrain_analyser.analyse()

        # map distance to idx pairs
        random.seed(seed)

        for num_path, min_len, max_len in zip(
            num_paths_to_explore, min_path_length_to_explore, max_path_length_to_explore
        ):
            # get index of samples within length
            within_length = (self.terrain_analyser.samples[:, 2] > min_len) & (
                self.terrain_analyser.samples[:, 2] <= max_len
            )

            # randomly select certain pairs
            rand_idx = torch.randperm(self.terrain_analyser.samples.shape[0])

            # select the samples
            selected_samples = self.terrain_analyser.samples[rand_idx[within_length]][:num_path]

            # filter edge cases
            if selected_samples.shape[0] == 0:
                print(f"[WARNING] No paths found with length [{min_len},{max_len}]")
                continue
            if selected_samples.shape[0] < num_path:
                print(
                    f"[WARNING] Only {selected_samples.shape[0]} paths found with length [{min_len},{max_len}] instead"
                    f" of {num_path}"
                )

            # get start, goal and path length
            curr_data = torch.zeros((selected_samples.shape[0], 7))
            curr_data[:, :3] = self.terrain_analyser.points[selected_samples[:, 0].type(torch.int64)]
            curr_data[:, 3:6] = self.terrain_analyser.points[selected_samples[:, 1].type(torch.int64)]
            curr_data[:, 6] = selected_samples[:, 2]

            # save curr_data as pickle
            filename = self._get_save_path_trajectories(seed, num_path, min_len, max_len)
            with open(filename, "wb") as f:
                pickle.dump(curr_data, f)

            # update data buffer
            data = torch.concatenate((data, curr_data), dim=0)

        # define start points
        return data

    ###
    # Safe paths
    ###

    def _get_save_path_trajectories(self, seed, num_path: int, min_len: float, max_len: float) -> str:
        filename = f"paths_seed{seed}_paths{num_path}_min{min_len}_max{max_len}.pkl"
        # get env name
        if hasattr(self.scene.terrain.cfg, "obj_filepath"):
            terrain_file_path = self.scene.terrain.cfg.obj_filepath
        elif hasattr(self.scene.terrain.cfg, "usd_path") and isinstance(self.scene.terrain.cfg.usd_path, str):
            terrain_file_path = self.scene.terrain.cfg.usd_path
        else:
            raise KeyError("Only implemented for terrains loaded from usd and matterport")
        env_name = os.path.splitext(terrain_file_path)[0]
        # create directory if necessary
        filedir = os.path.join(terrain_file_path, env_name)
        os.makedirs(filedir, exist_ok=True)
        return os.path.join(filedir, filename)
