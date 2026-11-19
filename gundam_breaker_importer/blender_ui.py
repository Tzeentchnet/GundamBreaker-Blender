# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from pathlib import Path

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)

from .animation import (
    SOURCE_FPS,
    discover_motion_assets,
    motion_compatibility_error,
    parse_motion_package,
)
from .archive import (
    MOTION_TYPE_HASH,
    FormatError,
    GameSource,
)
from .blender_scene import create_motion_action, import_model_asset
from .model import parse_materials, parse_model
from .profiles import PROFILES


_PROFILE_ITEMS = [
    ("AUTO", "Auto Detect", "Detect the game from the selected source"),
    *[
        (
            profile.identifier,
            profile.label,
            f"Use the {profile.platform} archive and resource formats",
        )
        for profile in PROFILES.values()
    ],
]


class GBI_PG_ModelItem(bpy.types.PropertyGroup):
    name: StringProperty()
    resource_hash: StringProperty()
    selected: BoolProperty(name="Import", default=False)


class GBI_PG_AnimationItem(bpy.types.PropertyGroup):
    name: StringProperty()
    resource_hash: StringProperty()
    clip_count: IntProperty()
    selected: BoolProperty(name="Import", default=False)


class GBI_PG_Settings(bpy.types.PropertyGroup):
    source_path: StringProperty(
        name="Game Source",
        description="Game folder, ISO image, or GxArchivedFile DAT archive",
        subtype="FILE_PATH",
    )
    profile: EnumProperty(
        name="Game",
        description="Game-specific import path",
        items=_PROFILE_ITEMS,
        default="AUTO",
    )
    model_filter: StringProperty(
        name="Filter",
        description="Filter models by archive name or resource hash",
    )
    models: CollectionProperty(type=GBI_PG_ModelItem)
    active_model: IntProperty(default=0)
    import_textures: BoolProperty(
        name="Import Textures",
        description="Decode and pack referenced textures into the blend file",
        default=True,
    )
    scale: FloatProperty(
        name="Scale",
        description="Scale imported geometry and armatures",
        default=1.0,
        min=0.0001,
        soft_max=100.0,
    )
    animations: CollectionProperty(type=GBI_PG_AnimationItem)
    active_animation: IntProperty(default=0)
    animation_model_hash: StringProperty()
    animation_clip_mode: EnumProperty(
        name="Clips",
        description="Choose which clips to create as Blender Actions",
        items=(
            ("ALL", "All", "Import every clip in each checked package"),
            ("SINGLE", "Single", "Import one numbered clip"),
            ("RANGE", "Range", "Import an inclusive range of clips"),
        ),
        default="SINGLE",
    )
    animation_clip_start: IntProperty(
        name="Clip",
        description="One-based source clip number",
        default=1,
        min=1,
    )
    animation_clip_end: IntProperty(
        name="End",
        description="Last one-based source clip number",
        default=1,
        min=1,
    )
    animation_status: StringProperty(name="Animation Status")
    status: StringProperty(name="Status")


def _chosen_profile(settings: GBI_PG_Settings):
    return None if settings.profile == "AUTO" else PROFILES[settings.profile]


def _active_imported_armature(
    context: bpy.types.Context,
) -> bpy.types.Object | None:
    active = context.active_object
    if active is not None and active.type == "ARMATURE":
        return active
    if (
        active is not None
        and active.parent is not None
        and active.parent.type == "ARMATURE"
    ):
        return active.parent
    for selected in context.selected_objects:
        if selected.type == "ARMATURE":
            return selected
        if (
            selected.parent is not None
            and selected.parent.type == "ARMATURE"
        ):
            return selected.parent
    return None


def _source_model_asset(
    source: GameSource,
    armature: bpy.types.Object,
):
    raw_hash = armature.get("gundam_breaker_resource_hash")
    if not isinstance(raw_hash, str):
        raise FormatError(
            "The active armature was not imported by Gundam Breaker Importer"
        )
    resource_hash = int(raw_hash, 16)
    for asset in source.model_assets():
        if asset.resource_hash == resource_hash:
            return asset
    raise FormatError(
        f"Model {resource_hash:08X} is no longer present in the game source"
    )


class GBI_OT_BrowseSource(bpy.types.Operator):
    bl_idname = "gundam_breaker.browse_source"
    bl_label = "Choose Gundam Breaker Source"
    bl_description = "Choose a game folder, ISO image, or archive and scan it"
    bl_options = {"REGISTER"}

    filepath: StringProperty(subtype="FILE_PATH")
    filter_glob: StringProperty(
        default="*.iso;*.dat", options={"HIDDEN"}
    )

    def invoke(
        self, context: bpy.types.Context, _event: bpy.types.Event
    ) -> set[str]:
        settings = context.scene.gundam_breaker_importer
        if settings.source_path:
            self.filepath = settings.source_path
        context.window_manager.fileselect_add(self)
        return {"RUNNING_MODAL"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        context.scene.gundam_breaker_importer.source_path = self.filepath
        return bpy.ops.gundam_breaker.scan_source()


class GBI_OT_ScanSource(bpy.types.Operator):
    bl_idname = "gundam_breaker.scan_source"
    bl_label = "Scan Models"
    bl_description = "Index model assets in the selected game source"
    bl_options = {"REGISTER"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = context.scene.gundam_breaker_importer
        source_path = bpy.path.abspath(settings.source_path)
        if not source_path:
            self.report({"ERROR"}, "Choose a game source first")
            return {"CANCELLED"}

        context.window_manager.progress_begin(0, 1)
        source = None
        try:
            source = GameSource.open(
                Path(source_path), _chosen_profile(settings)
            )
            assets = source.model_assets()
            settings.models.clear()
            settings.animations.clear()
            settings.animation_model_hash = ""
            settings.animation_status = ""
            for asset in assets:
                item = settings.models.add()
                item.name = asset.display_name
                item.resource_hash = f"{asset.resource_hash:08X}"
            settings.active_model = 0
            settings.status = (
                f"{source.profile.label}: {len(assets):,} models"
            )
        except (FormatError, OSError) as error:
            settings.status = str(error)
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        finally:
            if source is not None:
                source.close()
            context.window_manager.progress_end()
        return {"FINISHED"}


class GBI_OT_SetSelection(bpy.types.Operator):
    bl_idname = "gundam_breaker.set_selection"
    bl_label = "Set Model Selection"
    bl_options = {"INTERNAL"}

    action: EnumProperty(
        items=(
            ("ALL", "All", "Select all visible models"),
            ("NONE", "None", "Clear all model selections"),
        )
    )

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = context.scene.gundam_breaker_importer
        query = settings.model_filter.casefold().strip()
        for item in settings.models:
            if query and query not in (
                f"{item.name} {item.resource_hash}".casefold()
            ):
                continue
            item.selected = self.action == "ALL"
        return {"FINISHED"}


class GBI_OT_ImportSelected(bpy.types.Operator):
    bl_idname = "gundam_breaker.import_selected"
    bl_label = "Import Selected Models"
    bl_description = "Import checked models, or the active model if none are checked"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = context.scene.gundam_breaker_importer
        selected_hashes = {
            int(item.resource_hash, 16)
            for item in settings.models
            if item.selected
        }
        if not selected_hashes and settings.models:
            index = min(settings.active_model, len(settings.models) - 1)
            selected_hashes.add(
                int(settings.models[index].resource_hash, 16)
            )
        if not selected_hashes:
            self.report({"ERROR"}, "Scan and select at least one model")
            return {"CANCELLED"}

        source = None
        imported = []
        warnings: list[str] = []
        context.window_manager.progress_begin(0, len(selected_hashes))
        try:
            source = GameSource.open(
                Path(bpy.path.abspath(settings.source_path)),
                _chosen_profile(settings),
            )
            assets = {
                asset.resource_hash: asset
                for asset in source.model_assets()
                if asset.resource_hash in selected_hashes
            }
            missing = selected_hashes - assets.keys()
            if missing:
                hashes = ", ".join(f"{value:08X}" for value in missing)
                raise FormatError(
                    f"Selected models are no longer in the source: {hashes}"
                )

            for progress, resource_hash in enumerate(
                sorted(selected_hashes), 1
            ):
                asset = assets[resource_hash]
                model = parse_model(
                    asset.model.read_decoded(),
                    source.profile.endian_hint,
                )
                definitions = (
                    parse_materials(
                        asset.material.read_decoded(),
                        source.profile.endian_hint,
                    )
                    if asset.material is not None
                    else []
                )
                if asset.material is None:
                    warnings.append(
                        f"{asset.display_name}: material resource not found"
                    )
                result = import_model_asset(
                    context,
                    source,
                    asset,
                    model,
                    definitions,
                    import_textures=settings.import_textures,
                    scale=settings.scale,
                )
                warnings.extend(result.warnings)
                imported.append(result.mesh_object)
                context.window_manager.progress_update(progress)
        except (FormatError, OSError, RuntimeError) as error:
            settings.status = str(error)
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        finally:
            if source is not None:
                source.close()
            context.window_manager.progress_end()

        for selected in context.selected_objects:
            selected.select_set(False)
        for mesh_object in imported:
            mesh_object.select_set(True)
        if imported:
            context.view_layer.objects.active = imported[-1]
        settings.status = f"Imported {len(imported):,} model(s)"
        if warnings:
            settings.status += f" with {len(warnings):,} warning(s)"
            self.report(
                {"WARNING"},
                f"{len(warnings)} warning(s); first: {warnings[0]}",
            )
        else:
            self.report(
                {"INFO"}, f"Imported {len(imported):,} model(s)"
            )
        return {"FINISHED"}


class GBI_OT_ScanAnimations(bpy.types.Operator):
    bl_idname = "gundam_breaker.scan_animations"
    bl_label = "Find Compatible Animations"
    bl_description = "Find motion packages compatible with the active imported rig"
    bl_options = {"REGISTER"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = context.scene.gundam_breaker_importer
        armature = _active_imported_armature(context)
        if armature is None:
            self.report({"ERROR"}, "Select an imported armature or its mesh")
            return {"CANCELLED"}
        source_path = bpy.path.abspath(settings.source_path)
        if not source_path:
            self.report({"ERROR"}, "Choose the matching game source first")
            return {"CANCELLED"}

        source = None
        context.window_manager.progress_begin(0, 1)
        try:
            source = GameSource.open(
                Path(source_path), _chosen_profile(settings)
            )
            if source.profile.endian_hint != ">":
                raise FormatError(
                    "Animation import currently supports Gundam Breaker PS3 "
                    "MPK packages"
                )
            asset = _source_model_asset(source, armature)
            model = parse_model(
                asset.model.read_decoded(),
                source.profile.endian_hint,
            )
            discovery = discover_motion_assets(
                source, asset.display_name, model
            )
            settings.animations.clear()
            for index, motion in enumerate(discovery.assets):
                item = settings.animations.add()
                item.name = motion.name
                item.resource_hash = f"{motion.resource_hash:08X}"
                item.clip_count = motion.clip_count
                item.selected = index == 0
            settings.active_animation = 0
            settings.animation_model_hash = (
                f"{asset.resource_hash:08X}"
            )
            settings.animation_status = (
                f"{len(discovery.assets):,} compatible package(s)"
            )
            if discovery.warnings:
                settings.animation_status += (
                    f"; {len(discovery.warnings):,} skipped"
                )
                self.report(
                    {"WARNING"},
                    (
                        f"{len(discovery.warnings)} motion warning(s); "
                        f"first: {discovery.warnings[0]}"
                    ),
                )
            elif discovery.assets:
                self.report(
                    {"INFO"},
                    f"Found {len(discovery.assets):,} motion package(s)",
                )
            else:
                self.report(
                    {"WARNING"},
                    "No compatible motion packages were found",
                )
        except (FormatError, OSError, RuntimeError, ValueError) as error:
            settings.animation_status = str(error)
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        finally:
            if source is not None:
                source.close()
            context.window_manager.progress_end()
        return {"FINISHED"}


class GBI_OT_SetAnimationSelection(bpy.types.Operator):
    bl_idname = "gundam_breaker.set_animation_selection"
    bl_label = "Set Animation Selection"
    bl_options = {"INTERNAL"}

    action: EnumProperty(
        items=(
            ("ALL", "All", "Select all motion packages"),
            ("NONE", "None", "Clear all motion package selections"),
        )
    )

    def execute(self, context: bpy.types.Context) -> set[str]:
        selected = self.action == "ALL"
        for item in context.scene.gundam_breaker_importer.animations:
            item.selected = selected
        return {"FINISHED"}


class GBI_OT_ImportAnimations(bpy.types.Operator):
    bl_idname = "gundam_breaker.import_animations"
    bl_label = "Import Selected Animations"
    bl_description = "Bake selected source clips into Blender 5.2 Actions"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context: bpy.types.Context) -> set[str]:
        settings = context.scene.gundam_breaker_importer
        armature = _active_imported_armature(context)
        if armature is None:
            self.report({"ERROR"}, "Select an imported armature or its mesh")
            return {"CANCELLED"}
        armature_hash = armature.get("gundam_breaker_resource_hash")
        if armature_hash != settings.animation_model_hash:
            self.report(
                {"ERROR"},
                "Find animations again for the active imported rig",
            )
            return {"CANCELLED"}

        selected_items = [
            item for item in settings.animations if item.selected
        ]
        if not selected_items and settings.animations:
            index = min(
                settings.active_animation,
                len(settings.animations) - 1,
            )
            selected_items = [settings.animations[index]]
        if not selected_items:
            self.report({"ERROR"}, "Find and select a motion package first")
            return {"CANCELLED"}

        source = None
        created_results = []
        context.window_manager.progress_begin(0, 1)
        try:
            source = GameSource.open(
                Path(bpy.path.abspath(settings.source_path)),
                _chosen_profile(settings),
            )
            asset = _source_model_asset(source, armature)
            model = parse_model(
                asset.model.read_decoded(),
                source.profile.endian_hint,
            )
            packages = []
            total_actions = 0
            for item in selected_items:
                resource_hash = int(item.resource_hash, 16)
                resource = source.find(resource_hash, MOTION_TYPE_HASH)
                if resource is None:
                    raise FormatError(
                        f"Motion package {resource_hash:08X} is no longer "
                        "present in the game source"
                    )
                package = parse_motion_package(resource.read_decoded())
                compatibility_error = motion_compatibility_error(
                    package, model
                )
                if compatibility_error is not None:
                    raise FormatError(
                        f"{item.name} is not compatible: "
                        f"{compatibility_error}"
                    )
                clip_indices = self._clip_indices(
                    settings, item.name, len(package.clips)
                )
                packages.append((item, package, clip_indices))
                total_actions += len(clip_indices)

            context.window_manager.progress_end()
            context.window_manager.progress_begin(0, total_actions)
            progress = 0
            import_scale = float(
                armature.get("gundam_breaker_scale", 1.0)
            )
            for item, package, clip_indices in packages:
                resource_hash = int(item.resource_hash, 16)
                for clip_index in clip_indices:
                    result = create_motion_action(
                        armature,
                        package,
                        item.name,
                        resource_hash,
                        clip_index,
                        scale=import_scale,
                    )
                    created_results.append(result)
                    progress += 1
                    context.window_manager.progress_update(progress)

            if not created_results:
                raise FormatError("The selected motion range contains no clips")
            animation_data = armature.animation_data_create()
            animation_data.action = created_results[0].action
            animation_data.action_slot = created_results[0].slot
            context.scene.render.fps = int(SOURCE_FPS)
            context.scene.render.fps_base = 1.0
            context.scene.frame_start = 1
            context.scene.frame_end = int(
                max(
                    result.action.frame_range[1]
                    for result in created_results
                )
            )
            context.scene.frame_set(1)
            settings.animation_status = (
                f"Created {len(created_results):,} Action(s) at "
                f"{SOURCE_FPS:g} FPS"
            )
            self.report({"INFO"}, settings.animation_status)
        except (FormatError, OSError, RuntimeError, ValueError) as error:
            if (
                armature.animation_data is not None
                and any(
                    armature.animation_data.action == result.action
                    for result in created_results
                )
            ):
                armature.animation_data.action = None
            for result in created_results:
                bpy.data.actions.remove(result.action)
            settings.animation_status = str(error)
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        finally:
            if source is not None:
                source.close()
            context.window_manager.progress_end()
        return {"FINISHED"}

    @staticmethod
    def _clip_indices(
        settings: GBI_PG_Settings,
        package_name: str,
        clip_count: int,
    ) -> range:
        if settings.animation_clip_mode == "ALL":
            return range(clip_count)
        start = settings.animation_clip_start
        end = (
            start
            if settings.animation_clip_mode == "SINGLE"
            else settings.animation_clip_end
        )
        if end < start:
            raise FormatError("Animation clip range ends before it starts")
        if start > clip_count or end > clip_count:
            raise FormatError(
                f"{package_name} contains clips 1-{clip_count}, not "
                f"{start}-{end}"
            )
        return range(start - 1, end)


class GBI_UL_Models(bpy.types.UIList):
    def draw_item(
        self,
        _context: bpy.types.Context,
        layout: bpy.types.UILayout,
        _data: object,
        item: GBI_PG_ModelItem,
        _icon: int,
        _active_data: object,
        _active_property: str,
        _index: int,
    ) -> None:
        row = layout.row(align=True)
        row.prop(item, "selected", text="")
        row.label(text=item.name, icon="MESH_DATA")
        row.label(text=item.resource_hash)

    def filter_items(
        self,
        _context: bpy.types.Context,
        data: GBI_PG_Settings,
        property_name: str,
    ) -> tuple[list[int], list[int]]:
        items = getattr(data, property_name)
        query = data.model_filter.casefold().strip()
        if not query:
            return [self.bitflag_filter_item] * len(items), []
        flags = [
            self.bitflag_filter_item
            if query in f"{item.name} {item.resource_hash}".casefold()
            else 0
            for item in items
        ]
        return flags, []


class GBI_UL_Animations(bpy.types.UIList):
    def draw_item(
        self,
        _context: bpy.types.Context,
        layout: bpy.types.UILayout,
        _data: object,
        item: GBI_PG_AnimationItem,
        _icon: int,
        _active_data: object,
        _active_property: str,
        _index: int,
    ) -> None:
        row = layout.row(align=True)
        row.prop(item, "selected", text="")
        row.label(text=item.name, icon="ACTION")
        row.label(text=f"{item.clip_count} clips")


class GBI_PT_Importer(bpy.types.Panel):
    bl_label = "Gundam Breaker Importer"
    bl_idname = "GBI_PT_importer"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Gundam Breaker"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        settings = context.scene.gundam_breaker_importer
        layout.prop(settings, "profile")
        layout.prop(settings, "source_path")
        row = layout.row(align=True)
        row.operator(
            GBI_OT_BrowseSource.bl_idname,
            text="Browse",
            icon="FILE_FOLDER",
        )
        row.operator(
            GBI_OT_ScanSource.bl_idname,
            text="Scan",
            icon="VIEWZOOM",
        )
        if settings.status:
            layout.label(text=settings.status, icon="INFO")
        if not settings.models:
            return

        layout.separator()
        layout.prop(settings, "model_filter", icon="VIEWZOOM")
        layout.template_list(
            GBI_UL_Models.__name__,
            "",
            settings,
            "models",
            settings,
            "active_model",
            rows=10,
        )
        row = layout.row(align=True)
        select_all = row.operator(
            GBI_OT_SetSelection.bl_idname, text="All"
        )
        select_all.action = "ALL"
        select_none = row.operator(
            GBI_OT_SetSelection.bl_idname, text="None"
        )
        select_none.action = "NONE"
        layout.prop(settings, "import_textures")
        layout.prop(settings, "scale")
        layout.operator(
            GBI_OT_ImportSelected.bl_idname,
            icon="IMPORT",
        )

        layout.separator()
        box = layout.box()
        box.label(text="Animations", icon="ACTION")
        armature = _active_imported_armature(context)
        if armature is None:
            box.label(text="Select an imported rig", icon="INFO")
        else:
            box.label(text=armature.name, icon="ARMATURE_DATA")
        box.operator(
            GBI_OT_ScanAnimations.bl_idname,
            icon="VIEWZOOM",
        )
        if settings.animation_status:
            box.label(text=settings.animation_status, icon="INFO")
        if settings.animations:
            box.template_list(
                GBI_UL_Animations.__name__,
                "",
                settings,
                "animations",
                settings,
                "active_animation",
                rows=6,
            )
            row = box.row(align=True)
            select_all = row.operator(
                GBI_OT_SetAnimationSelection.bl_idname, text="All"
            )
            select_all.action = "ALL"
            select_none = row.operator(
                GBI_OT_SetAnimationSelection.bl_idname, text="None"
            )
            select_none.action = "NONE"
            box.prop(settings, "animation_clip_mode")
            if settings.animation_clip_mode in {"SINGLE", "RANGE"}:
                row = box.row(align=True)
                row.prop(settings, "animation_clip_start")
                if settings.animation_clip_mode == "RANGE":
                    row.prop(settings, "animation_clip_end")
            box.operator(
                GBI_OT_ImportAnimations.bl_idname,
                icon="ACTION",
            )


def _draw_import_menu(
    self: bpy.types.Menu, _context: bpy.types.Context
) -> None:
    self.layout.operator(
        GBI_OT_BrowseSource.bl_idname,
        text="Gundam Breaker Game Source (.iso/.dat/folder)",
    )


_CLASSES = (
    GBI_PG_ModelItem,
    GBI_PG_AnimationItem,
    GBI_PG_Settings,
    GBI_OT_BrowseSource,
    GBI_OT_ScanSource,
    GBI_OT_SetSelection,
    GBI_OT_ImportSelected,
    GBI_OT_ScanAnimations,
    GBI_OT_SetAnimationSelection,
    GBI_OT_ImportAnimations,
    GBI_UL_Models,
    GBI_UL_Animations,
    GBI_PT_Importer,
)


def register() -> None:
    for cls in _CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.gundam_breaker_importer = PointerProperty(
        type=GBI_PG_Settings
    )
    bpy.types.TOPBAR_MT_file_import.append(_draw_import_menu)


def unregister() -> None:
    bpy.types.TOPBAR_MT_file_import.remove(_draw_import_menu)
    del bpy.types.Scene.gundam_breaker_importer
    for cls in reversed(_CLASSES):
        bpy.utils.unregister_class(cls)
