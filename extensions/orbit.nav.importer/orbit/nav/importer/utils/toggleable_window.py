# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.kit.ui
from omni import ui


class ToggleableWindow(ui.Window):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deferred_dock_in("Stage", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
        self._window_name = kwargs["title"]
        self._menu_prefix = kwargs["menu_prefix"]
        self.visible = False
        self._visibility_menu_item_name = f"{self._menu_prefix}/{self._window_name}"
        editor_menu = omni.kit.ui.get_editor_menu()
        self._visibility_menu_item = editor_menu.add_item(
            self._visibility_menu_item_name,
            lambda _, visible: self._change_window_visibility(visible),
            toggle=True,
            value=self._window_is_visible(),
            priority=200,
        )
        self.set_visibility_changed_fn(self._on_window_visibility_changed)

    def shutdown(self):
        print("Window shutdown.")
        self.frame.clear()
        self.set_visibility_changed_fn(None)
        editor_menu = omni.kit.ui.get_editor_menu()
        editor_menu.remove_item(self._visibility_menu_item_name)
        self._visibility_menu_item = None

    def _on_window_visibility_changed(self, visible):
        omni.kit.ui.get_editor_menu().set_value(self._visibility_menu_item_name, visible)

    def _change_window_visibility(self, visible):
        self.visible = visible

    def _window_is_visible(self):
        return self.visible
