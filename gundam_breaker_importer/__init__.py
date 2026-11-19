# SPDX-License-Identifier: GPL-3.0-or-later

bl_info = {
    "name": "Gundam Breaker Importer",
    "author": "Gundam Breaker Importer contributors",
    "version": (0, 2, 0),
    "blender": (5, 2, 0),
    "location": "File > Import; 3D View > Sidebar > Gundam Breaker",
    "description": "Import Gundam Breaker models, rigs, materials, textures, and animations",
    "category": "Import-Export",
}


def register() -> None:
    from .blender_ui import register as register_ui

    register_ui()


def unregister() -> None:
    from .blender_ui import unregister as unregister_ui

    unregister_ui()
