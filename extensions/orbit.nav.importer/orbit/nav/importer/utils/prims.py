# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.core.utils.prims as prim_utils
from pxr import Usd


def get_all_meshes(env_prim: str) -> tuple[list[Usd.Prim], list[str]]:
    def recursive_mesh_search(start_prim: str, mesh_prims: list):
        for curr_prim in prim_utils.get_prim_at_path(start_prim).GetChildren():
            if curr_prim.GetTypeName() == "Mesh":
                mesh_prims.append(curr_prim)
            else:
                mesh_prims = recursive_mesh_search(start_prim=curr_prim.GetPath().pathString, mesh_prims=mesh_prims)

        return mesh_prims

    assert prim_utils.is_prim_path_valid(env_prim), f"Prim path '{env_prim}' is not valid"

    mesh_prims = recursive_mesh_search(env_prim, [])

    # mesh_prims: dict = prim_utils.get_prim_at_path(self.cfg.prim_path + "/" + self.cfg.usd_name.split(".")[0]).GetChildren()
    mesh_prims_name = [mesh_prim_single.GetName() for mesh_prim_single in mesh_prims]

    return mesh_prims, mesh_prims_name


def get_mesh_prims(env_prim: str) -> tuple[list[Usd.Prim], list[str]]:
    def recursive_search_mesh(start_prim: str, mesh_prims: list):
        for curr_prim in prim_utils.get_prim_at_path(start_prim).GetChildren():
            if curr_prim.GetTypeName() == "Xform" or curr_prim.GetTypeName() == "Mesh":
                mesh_prims.append(curr_prim)
            elif curr_prim.GetTypeName() == "Scope":
                mesh_prims = recursive_search_mesh(start_prim=curr_prim.GetPath().pathString, mesh_prims=mesh_prims)

        return mesh_prims

    assert prim_utils.is_prim_path_valid(env_prim), f"Prim path '{env_prim}' is not valid"

    mesh_prims = recursive_search_mesh(env_prim, [])
    mesh_prims_name = [mesh_prim_single.GetName() for mesh_prim_single in mesh_prims]

    return mesh_prims, mesh_prims_name


def get_all_prims_including_str(start_prim: str, path: str) -> list[Usd.Prim]:
    """Get all prims that include the given path str.

    This function recursively searches for all prims that include the given path str.

    Args:
        start_prim: The environment prim path from which to begin the search.
        path: The path string to search for.

    Returns:
        A list of all prims that include the given path str.
    """

    def recursive_search(start_prim: Usd.Prim, prim_name: str, found_prims: list) -> list[Usd.Prim]:
        for curr_prim in start_prim.GetChildren():
            if prim_name.lower() in curr_prim.GetPath().pathString.lower():
                found_prims.append(curr_prim)
            else:
                found_prims = recursive_search(start_prim=curr_prim, prim_name=prim_name, found_prims=found_prims)

        return found_prims

    # Raise error if the start prim is not valid
    assert prim_utils.is_prim_path_valid(start_prim), f"Prim path '{start_prim}' is not valid"

    start_prim = prim_utils.get_prim_at_path(start_prim)
    final_prim = recursive_search(start_prim=start_prim, prim_name=path, found_prims=[])
    return final_prim
