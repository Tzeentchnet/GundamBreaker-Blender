# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gundam_breaker_importer.archive import FormatError, GameSource
from gundam_breaker_importer.blender_scene import import_model_asset
from gundam_breaker_importer.model import parse_materials, parse_model


@dataclass(frozen=True, slots=True)
class Sample:
    label: str
    resource_hash: int
    coverage: str


@dataclass(slots=True)
class Result:
    label: str
    resource_hash: str
    coverage: str
    source_name: str
    vertices: int
    triangles: int
    materials: int
    images: int
    bones: int
    warnings: list[str]
    blend_file: str
    front_screenshot: str


SAMPLES = (
    Sample("chr5490_compact_rig", 0x2A53B678, "compact 3-bone rig"),
    Sample("chr4175_typical_rig", 0x4C7C78BE, "typical 33-bone rig"),
    Sample("chr4670_complex_rig", 0x34FCFF2E, "complex 164-bone rig"),
    Sample("2A835EA9_static_multipart", 0x2A835EA9, "static 53-part asset"),
)


def _arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--iso",
        type=Path,
        default=Path(r"F:\Recomp_GameSource\Gundam Breaker (Japan).iso"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"E:\tests\GundamBreakerImporter"),
    )
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--quit-after", action="store_true")
    return parser.parse_args(arguments)


def _look_at(obj: bpy.types.Object, target: np.ndarray) -> None:
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _add_area_light(
    collection: bpy.types.Collection,
    name: str,
    location: tuple[float, float, float],
    target: np.ndarray,
    energy: float,
    size: float,
) -> None:
    light = bpy.data.lights.new(name, "AREA")
    light.energy = energy
    light.shape = "DISK"
    light.size = size
    light.use_shadow = False
    obj = bpy.data.objects.new(name, light)
    collection.objects.link(obj)
    obj.location = location
    _look_at(obj, target)


def _configure_preview(
    mesh_object: bpy.types.Object,
    resolution: int,
) -> tuple[bpy.types.Scene, np.ndarray, np.ndarray]:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = 0.35

    world = bpy.data.worlds.new("Front Preview World")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.018, 0.022, 0.032, 1.0)
    background.inputs["Strength"].default_value = 0.22
    scene.world = world

    coordinates = np.empty(len(mesh_object.data.vertices) * 3, dtype=np.float32)
    mesh_object.data.vertices.foreach_get("co", coordinates)
    coordinates = coordinates.reshape(-1, 3)
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    center = (minimum + maximum) * 0.5
    extent = maximum - minimum
    span = max(float(extent.max()), 0.25)

    preview = bpy.data.collections.new("Front Preview")
    scene.collection.children.link(preview)
    camera_data = bpy.data.cameras.new("Front Camera")
    camera = bpy.data.objects.new("Front Camera", camera_data)
    preview.objects.link(camera)
    front_distance = max(span * 1.75, 1.0)
    camera.location = (
        float(center[0]),
        float(minimum[1] - front_distance),
        float(center[2]),
    )
    _look_at(camera, center)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = max(
        float(extent[0]) * 1.16,
        float(extent[2]) * 1.16,
        0.5,
    )
    camera_data.clip_start = max(span * 0.0001, 0.001)
    camera_data.clip_end = front_distance + float(extent[1]) + span * 4.0
    scene.camera = camera

    light_scale = max(span, 1.0)
    _add_area_light(
        preview,
        "Key Light",
        (
            float(center[0] - span * 0.65),
            float(minimum[1] - span * 0.85),
            float(center[2] + span * 0.8),
        ),
        center,
        850.0 * light_scale * light_scale,
        span * 1.15,
    )
    _add_area_light(
        preview,
        "Fill Light",
        (
            float(center[0] + span * 0.9),
            float(minimum[1] - span * 0.45),
            float(center[2] + span * 0.1),
        ),
        center,
        420.0 * light_scale * light_scale,
        span * 1.35,
    )
    _add_area_light(
        preview,
        "Rim Light",
        (
            float(center[0]),
            float(maximum[1] + span * 0.5),
            float(center[2] + span * 0.65),
        ),
        center,
        600.0 * light_scale * light_scale,
        span,
    )
    return scene, minimum, maximum


def _capture_material_preview(
    scene: bpy.types.Scene,
    screenshot_path: Path,
) -> None:
    if bpy.app.background:
        raise RuntimeError(
            "Material Preview capture requires Blender with an OpenGL window"
        )
    window = bpy.context.window_manager.windows[0]
    area = next(
        area for area in window.screen.areas if area.type == "VIEW_3D"
    )
    region = next(
        region for region in area.regions if region.type == "WINDOW"
    )
    space = area.spaces.active
    scene.display.shading.type = "MATERIAL"
    scene.display.shading.use_scene_world = False
    scene.display.shading.use_scene_lights = False
    scene.render.filepath = str(screenshot_path)
    space.shading.type = "MATERIAL"
    space.shading.use_scene_world = False
    space.shading.use_scene_lights = False
    space.shading.studiolight_rotate_z = 0.0
    space.shading.studiolight_background_alpha = 1.0
    space.overlay.show_overlays = False
    with bpy.context.temp_override(
        window=window,
        screen=window.screen,
        area=area,
        region=region,
        space_data=space,
        scene=scene,
    ):
        space.region_3d.view_perspective = "CAMERA"
        space.region_3d.view_camera_zoom = 0.0
        bpy.ops.render.opengl(write_still=True, view_context=True)


def _render_sample(
    source: GameSource,
    asset_by_hash: dict[int, object],
    sample: Sample,
    output_root: Path,
    resolution: int,
) -> Result:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    asset = asset_by_hash.get(sample.resource_hash)
    if asset is None:
        raise FormatError(
            f"Sample resource {sample.resource_hash:08X} was not found"
        )

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
    imported = import_model_asset(
        bpy.context,
        source,
        asset,
        model,
        definitions,
        import_textures=True,
    )
    scene, minimum, maximum = _configure_preview(
        imported.mesh_object, resolution
    )

    sample_directory = output_root / sample.label
    sample_directory.mkdir(parents=True, exist_ok=True)
    screenshot_path = sample_directory / f"{sample.label}_front.png"
    blend_path = sample_directory / f"{sample.label}.blend"
    scene["gundam_test_view"] = "Front (-Y looking +Y), orthographic"
    scene["gundam_test_shading"] = "Material Preview"
    scene["gundam_test_bounds_min"] = minimum.tolist()
    scene["gundam_test_bounds_max"] = maximum.tolist()
    scene["gundam_test_coverage"] = sample.coverage

    _capture_material_preview(scene, screenshot_path)
    bpy.ops.wm.save_as_mainfile(
        filepath=str(blend_path),
        check_existing=False,
        compress=True,
    )

    return Result(
        label=sample.label,
        resource_hash=f"{sample.resource_hash:08X}",
        coverage=sample.coverage,
        source_name=asset.display_name,
        vertices=len(imported.mesh_object.data.vertices),
        triangles=len(imported.mesh_object.data.polygons),
        materials=len(imported.mesh_object.data.materials),
        images=sum(
            1 for image in bpy.data.images if image.type == "IMAGE"
        ),
        bones=(
            len(imported.armature_object.data.bones)
            if imported.armature_object is not None
            else 0
        ),
        warnings=imported.warnings,
        blend_file=str(blend_path),
        front_screenshot=str(screenshot_path),
    )


def main() -> None:
    options = _arguments()
    bpy.context.preferences.filepaths.save_version = 0
    options.output.mkdir(parents=True, exist_ok=True)
    source = GameSource.open(options.iso)
    results = []
    try:
        asset_by_hash = {
            asset.resource_hash: asset for asset in source.model_assets()
        }
        for sample in SAMPLES:
            result = _render_sample(
                source,
                asset_by_hash,
                sample,
                options.output,
                options.resolution,
            )
            results.append(result)
            print(
                "GUNDAM_SAMPLE_OK",
                result.label,
                result.vertices,
                result.triangles,
                result.bones,
                result.images,
            )
    finally:
        source.close()

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "blender_version": bpy.app.version_string,
        "source": str(options.iso),
        "view": "Front orthographic (-Y looking +Y)",
        "shading": "Material Preview",
        "resolution": [options.resolution, options.resolution],
        "results": [asdict(result) for result in results],
    }
    manifest_path = options.output / "import_results.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print("GUNDAM_SAMPLE_MANIFEST", manifest_path)
    if options.quit_after:
        bpy.ops.wm.quit_blender()


if __name__ == "__main__":
    main()
