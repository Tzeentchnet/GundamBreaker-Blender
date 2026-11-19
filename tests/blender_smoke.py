# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import bpy
import numpy as np
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gundam_breaker_importer
from gundam_breaker_importer.animation import (
    LocalTransform,
    MotionBase,
    MotionClip,
    MotionPackage,
    NodeTracks,
    ScalarCurve,
    VectorTrack,
    parse_motion_package,
)
from gundam_breaker_importer.blender_scene import (
    _animated_motion_matrices,
    create_motion_action,
    import_model_asset,
)
from gundam_breaker_importer.model import (
    Bone,
    MaterialDefinition,
    Model,
    parse_model,
)
from gundam_breaker_importer.profiles import GUNDAM_BREAKER_3_VITA
from test_formats import _minimal_motion_package, _minimal_vita_model


def _rest_tracks() -> NodeTracks:
    return NodeTracks(VectorTrack(), VectorTrack(), VectorTrack())


def _constant_vector_track(
    values: tuple[float, float, float],
) -> VectorTrack:
    return VectorTrack(
        tuple(
            ScalarCurve(kind=0, values=(value,))
            for value in values
        )
    )


def _test_two_bone_ik() -> None:
    identity = (0.0, 0.0, 0.0, 1.0)
    unit_scale = (1.0, 1.0, 1.0)
    base = MotionBase(
        parents=(-1, 0, 1, 2, 3),
        node_types=(0, 1, 2, 3, 0),
        joint_ids=(0, 1, 2, 3),
        transforms=(
            LocalTransform((0.0, 0.0, 0.0), identity, unit_scale),
            LocalTransform((0.0, 0.0, 0.0), identity, unit_scale),
            LocalTransform((1.0, 0.0, 0.0), identity, unit_scale),
            LocalTransform((1.0, 0.0, 0.0), identity, unit_scale),
            LocalTransform((0.25, 0.0, 0.0), identity, unit_scale),
        ),
    )
    target = (1.0, 1.0, 0.0)
    rest = _rest_tracks()
    clip = MotionClip(
        frame_count=1,
        special_curves=(
            ScalarCurve(kind=0, values=(0.0,)),
            ScalarCurve(kind=0, values=(0.0,)),
        ),
        global_tracks=rest,
        bone_tracks=(
            rest,
            rest,
            NodeTracks(
                _constant_vector_track(target),
                VectorTrack(),
                VectorTrack(),
            ),
            rest,
        ),
        curves=(),
    )
    package = MotionPackage(base=base, clips=(clip,))
    sampled = [
        clip.sample_node(base, node_index, 0.0)
        for node_index in range(base.node_count)
    ]
    matrices = _animated_motion_matrices(
        package,
        clip,
        sampled,
        1.0,
    )
    positions = [matrix.translation for matrix in matrices]
    assert (positions[3] - Vector(target)).length < 1.0e-5
    assert abs((positions[2] - positions[1]).length - 1.0) < 1.0e-5
    assert abs((positions[3] - positions[2]).length - 1.0) < 1.0e-5
    assert abs((positions[4] - positions[3]).length - 0.25) < 1.0e-5


def main() -> None:
    gundam_breaker_importer.register()
    try:
        _test_two_bone_ik()
        model = parse_model(_minimal_vita_model(), "<")
        source = SimpleNamespace(profile=GUNDAM_BREAKER_3_VITA)
        asset = SimpleNamespace(
            display_name="synthetic_triangle",
            resource_hash=0x12345678,
        )
        result = import_model_asset(
            bpy.context,
            source,
            asset,
            model,
            [MaterialDefinition(name="synthetic_material")],
            import_textures=False,
        )
        assert len(result.mesh_object.data.vertices) == 3
        assert len(result.mesh_object.data.polygons) == 1
        assert len(result.mesh_object.data.materials) == 1
        assert result.mesh_object.data.has_custom_normals is False

        rigged_model = Model(
            header=model.header,
            bones=[
                Bone(
                    name="jointroot_000",
                    parent_index=-1,
                    position=np.zeros(3, dtype=np.float32),
                    bind_matrix=np.identity(4, dtype=np.float32),
                )
            ],
            submeshes=model.submeshes,
        )
        rigged_asset = SimpleNamespace(
            display_name="chr0000",
            resource_hash=0x87654321,
        )
        rigged_result = import_model_asset(
            bpy.context,
            source,
            rigged_asset,
            rigged_model,
            [MaterialDefinition(name="animated_material")],
            import_textures=False,
        )
        package = parse_motion_package(_minimal_motion_package())
        action_result = create_motion_action(
            rigged_result.armature_object,
            package,
            "chr0000",
            0x11223344,
            0,
        )
        animation_data = rigged_result.armature_object.animation_data_create()
        animation_data.action = action_result.action
        animation_data.action_slot = action_result.slot
        bpy.context.scene.frame_set(3)
        bpy.context.view_layer.update()
        pose_bone = rigged_result.armature_object.pose.bones[0]
        assert abs(pose_bone.location.x - 1.0) < 1.0e-5
        assert tuple(action_result.action.frame_range) == (1.0, 3.0)
        strip = action_result.action.layers[0].strips[0]
        assert len(strip.channelbags[0].fcurves) == 10
        print("GUNDAM_BREAKER_BLENDER_SMOKE_OK")
    finally:
        gundam_breaker_importer.unregister()


if __name__ == "__main__":
    main()
