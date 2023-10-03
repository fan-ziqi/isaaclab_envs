#!/usr/bin/python
"""
@author     Pascal Roth
@email      rothpa@student.ethz.ch

@brief     MatterPort3D Extension in Omniverse-Isaac Sim
"""
# python
import os

# omniverse
import carb
import omni.kit.asset_converter as converter

# omni.isaac.core
from omni.isaac.core.utils import extensions

# omni-isaac-matterport
from .progress_popup import ProgressPopup

# enable ROS bridge extension
extensions.enable_extension("omni.kit.asset_converter")


class MatterportConverter:
    def __init__(self, input_obj: str, context: converter.impl.AssetConverterContext) -> None:
        self._input_obj = input_obj
        self._context = context

        # setup converter
        self.task_manager = converter.extension.AssetImporterExtension()

        # setup progress popup
        self.progress = ProgressPopup(
            "Converting Matterport3D to USD", cancel_button_fn=lambda: self.task_manager.cancel()
        )
        return

    def progress_callback(self, current_step: int, total: int):
        # Show progress
        # TODO: get progress running without crash
        # self.progress.set_progress(current_step / total)
        # return
        pass

    async def convert_asset_to_usd(self) -> None:
        # get usd file path and create directory
        base_path, _ = os.path.splitext(self._input_obj)
        # set task
        task = self.task_manager.create_converter_task(
            self._input_obj, base_path + ".usd", self.progress_callback, self._context
        )
        success = await task.wait_until_finished()

        # print error
        if not success:
            detailed_status_code = task.get_status()
            detailed_status_error_string = task.get_error_message()
            carb.log_error(
                f"Failed to convert {self._input_obj} to {base_path + '.usd'} "
                f"with status {detailed_status_code} and error {detailed_status_error_string}"
            )
        return


# EoF
