# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import carb
import omni
import omni.isaac.core.utils.prims as prim_utils
import yaml
from omni.isaac.core.utils.semantics import add_update_semantics, remove_all_semantics
from omni.isaac.lab.terrains import TerrainImporter
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from orbit.nav.importer.utils.prims import get_all_prims_including_str, get_mesh_prims
from pxr import Gf, UsdGeom

if TYPE_CHECKING:
    from .unreal_importer_cfg import UnRealImporterCfg


class UnRealImporter(TerrainImporter):
    """
    Default stairs environment for testing
    """

    cfg: UnRealImporterCfg

    def __init__(self, cfg: UnRealImporterCfg) -> None:
        """
        :param
        """
        super().__init__(cfg)

        # modify mesh
        if self.cfg.duplicate_cfg_file and isinstance(self.cfg.duplicate_cfg_file, str):
            self.mesh_duplicator(self.cfg.duplicate_cfg_file)
        elif self.cfg.duplicate_cfg_file and isinstance(self.cfg.duplicate_cfg_file, list):
            [self.mesh_duplicator(duplicate_cfg_file) for duplicate_cfg_file in self.cfg.duplicate_cfg_file]
        else:
            print("[INFO] No mesh duplication executed.")

        # add usd meshes (e.g. for people)
        if self.cfg.people_config_file:
            self._insert_people()

        # assign semantic labels
        if self.cfg.sem_mesh_to_class_map:
            self._add_semantics()

    """ Assign Semantic Labels """

    def _add_semantics(self):
        # remove all previous semantic labels
        remove_all_semantics(prim_utils.get_prim_at_path(self.cfg.prim_path + "/terrain"), recursive=True)

        # get mesh prims
        mesh_prims, mesh_prims_name = get_mesh_prims(self.cfg.prim_path + "/terrain")

        carb.log_info(f"Total of {len(mesh_prims)} meshes in the scene, start assigning semantic class ...")

        # mapping from prim name to class
        with open(self.cfg.sem_mesh_to_class_map) as stream:
            class_keywords = yaml.safe_load(stream)

        # make all the string lower case
        mesh_prims_name = [mesh_prim_single.lower() for mesh_prim_single in mesh_prims_name]
        keywords_class_mapping_lower = {
            key: [value_single.lower() for value_single in value] for key, value in class_keywords.items()
        }

        # assign class to mesh in ISAAC
        def recursive_semUpdate(prim, sem_class_name: str, update_submesh: bool) -> bool:
            # Necessary for Park Mesh
            if (
                prim.GetName() == "HierarchicalInstancedStaticMesh"
            ):  # or "FoliageInstancedStaticMeshComponent" in prim.GetName():
                add_update_semantics(prim, sem_class_name)
                update_submesh = True
            children = prim.GetChildren()
            if len(children) > 0:
                for child in children:
                    update_submesh = recursive_semUpdate(child, sem_class_name, update_submesh)
            return update_submesh

        def recursive_meshInvestigator(mesh_idx, mesh_name, mesh_prim_list) -> bool:
            success = False
            for class_name, keywords in keywords_class_mapping_lower.items():
                if any([keyword in mesh_name for keyword in keywords]):
                    update_submesh = recursive_semUpdate(mesh_prim_list[mesh_idx], class_name, False)
                    if not update_submesh:
                        add_update_semantics(mesh_prim_list[mesh_idx], class_name)
                    success = True
                    break

            if not success:
                success_child = []
                mesh_prims_children, mesh_prims_name_children = get_mesh_prims(
                    mesh_prim_list[mesh_idx].GetPrimPath().pathString
                )
                mesh_prims_name_children = [mesh_prim_single.lower() for mesh_prim_single in mesh_prims_name_children]
                for mesh_idx_child, mesh_name_child in enumerate(mesh_prims_name_children):
                    success_child.append(
                        recursive_meshInvestigator(mesh_idx_child, mesh_name_child, mesh_prims_children)
                    )
                success = any(success_child)

            return success

        mesh_list = []
        for mesh_idx, mesh_name in enumerate(mesh_prims_name):
            success = recursive_meshInvestigator(mesh_idx=mesh_idx, mesh_name=mesh_name, mesh_prim_list=mesh_prims)
            if success:
                mesh_list.append(mesh_idx)

        missing = [i for x, y in zip(mesh_list, mesh_list[1:]) for i in range(x + 1, y) if y - x > 1]
        assert len(mesh_list) > 0, "No mesh is assigned a semantic class!"
        assert len(mesh_list) == len(mesh_prims_name), (
            "Not all meshes are assigned a semantic class! Following mesh names are included yet:"
            f" {[mesh_prims_name[miss_idx] for miss_idx in missing]}"
        )
        carb.log_info("Semantic mapping done.")

        return

    """ Modify Mesh """

    def mesh_duplicator(self, duplicate_cfg_filepath: str):
        """Duplicate prims in the scene."""

        with open(duplicate_cfg_filepath) as stream:
            multipy_cfg: dict = yaml.safe_load(stream)

        # get the stage
        stage = omni.usd.get_context().get_stage()

        # init counter
        add_counter = 0

        for value in multipy_cfg.values():
            # get the prim that should be duplicated
            prims = get_all_prims_including_str(self.cfg.prim_path + "/terrain", value["prim"])

            if len(prims) == 0:
                print(f"[WARNING] Could not find prim {value['prim']}, no replication possible!")
                continue

            if value.get("only_first_match", True):
                prims = [prims[0]]

            # make translations a list of lists in the case only a single translation is given
            if not isinstance(value["translation"][0], list):
                value["translation"] = [value["translation"]]

            # iterate over translations and their factor
            for translation_idx, curr_translation in enumerate(value["translation"]):
                for copy_idx in range(value.get("factor", 1)):
                    for curr_prim in prims:
                        # get the path of the current prim
                        curr_prim_path = curr_prim.GetPath().pathString
                        # copy path
                        new_prim_path = os.path.join(
                            curr_prim_path + f"_tr{translation_idx}_cp{copy_idx}" + value.get("suffix", "")
                        )

                        success = omni.usd.duplicate_prim(
                            stage=stage,
                            prim_path=curr_prim_path,
                            path_to=new_prim_path,
                            duplicate_layers=True,
                        )
                        assert success, f"Failed to duplicate prim '{curr_prim_path}'"

                        # get crosswalk prim
                        prim = prim_utils.get_prim_at_path(new_prim_path)
                        xform = UsdGeom.Mesh(prim).AddTranslateOp()
                        xform.Set(
                            Gf.Vec3d(curr_translation[0], curr_translation[1], curr_translation[2]) * (copy_idx + 1)
                        )

                        # update counter
                        add_counter += 1

        print(f"Number of added prims: {add_counter} from file {duplicate_cfg_filepath}")

    def _insert_people(self):
        # load people config file
        with open(self.cfg.people_config_file) as stream:
            people_cfg: dict = yaml.safe_load(stream)

        for key, person_cfg in people_cfg.items():
            self.insert_single_person(
                person_cfg["prim_name"],
                person_cfg["translation"],
                scale_people=person_cfg.get("scale", 1.0),
                usd_path=person_cfg.get("usd_path", "People/Characters/F_Business_02/F_Business_02.usd"),
            )
            # TODO: allow for movement of the people

        print(f"Number of people added: {len(people_cfg)}")

        return

    @staticmethod
    def insert_single_person(
        prim_name: str,
        translation: list,
        scale_people: float = 1.0,
        usd_path: str = "People/Characters/F_Business_02/F_Business_02.usd",
    ) -> None:
        person_prim = prim_utils.create_prim(
            prim_path=os.path.join("/World/People", prim_name),
            translation=tuple(translation),
            usd_path=os.path.join(ISAAC_NUCLEUS_DIR, usd_path),
            scale=(scale_people, scale_people, scale_people),
        )

        if isinstance(person_prim.GetAttribute("xformOp:orient").Get(), Gf.Quatd):
            person_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
        else:
            person_prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        add_update_semantics(person_prim, "person")

        # add collision body
        UsdGeom.Mesh(person_prim)

        return
