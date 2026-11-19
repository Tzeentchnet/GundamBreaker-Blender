# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import math
import tempfile
from dataclasses import dataclass
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Quaternion, Vector

from .animation import (
    LocalTransform,
    MotionClip,
    MotionPackage,
    SOURCE_FPS,
)
from .archive import (
    FormatError,
    GameSource,
    ModelAsset,
    TEXTURE_TYPE_HASH,
)
from .model import MaterialDefinition, Model, SubMesh
from .texture import make_dds, parse_texture


@dataclass(slots=True)
class ImportResult:
    collection: bpy.types.Collection
    mesh_object: bpy.types.Object
    armature_object: bpy.types.Object | None
    warnings: list[str]


@dataclass(slots=True)
class MotionActionResult:
    action: bpy.types.Action
    slot: object


_AXIS_CONVERSION = np.array(
    (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ),
    dtype=np.float32,
)


def import_model_asset(
    context: bpy.types.Context,
    source: GameSource,
    asset: ModelAsset,
    model: Model,
    material_definitions: list[MaterialDefinition],
    import_textures: bool = True,
    scale: float = 1.0,
) -> ImportResult:
    name = _asset_name(asset)
    collection = bpy.data.collections.new(name)
    context.scene.collection.children.link(collection)
    warnings: list[str] = []
    image_cache: dict[tuple[int, bool], bpy.types.Image] = {}

    material_count = max(
        len(material_definitions),
        max(
            (submesh.material_index + 1 for submesh in model.submeshes),
            default=0,
        ),
    )
    materials = []
    for index in range(material_count):
        definition = (
            material_definitions[index]
            if index < len(material_definitions)
            else MaterialDefinition(name=f"material_{index:03d}")
        )
        materials.append(
            _create_material(
                source,
                definition,
                image_cache,
                warnings,
                import_textures,
            )
        )

    armature_object, bone_names = _create_armature(
        context, collection, name, model, scale
    )
    mesh_object = _create_mesh_object(
        collection,
        name,
        model.submeshes,
        materials,
        bone_names,
        armature_object,
        scale,
    )
    collection["gundam_breaker_profile"] = source.profile.identifier
    collection["gundam_breaker_resource_hash"] = (
        f"{asset.resource_hash:08X}"
    )
    mesh_object["gundam_breaker_resource_hash"] = (
        f"{asset.resource_hash:08X}"
    )
    if armature_object is not None:
        armature_object["gundam_breaker_profile"] = source.profile.identifier
        armature_object["gundam_breaker_resource_hash"] = (
            f"{asset.resource_hash:08X}"
        )
        armature_object["gundam_breaker_model_name"] = asset.display_name
        armature_object["gundam_breaker_scale"] = scale
    return ImportResult(
        collection=collection,
        mesh_object=mesh_object,
        armature_object=armature_object,
        warnings=warnings,
    )


def _asset_name(asset: ModelAsset) -> str:
    name = asset.display_name.replace("\\", "/").rsplit("/", 1)[-1]
    return Path(name).stem or f"{asset.resource_hash:08X}"


def _convert_vectors(values: np.ndarray, scale: float = 1.0) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float32)
    output[:, 0] = values[:, 0] * scale
    output[:, 1] = -values[:, 2] * scale
    output[:, 2] = values[:, 1] * scale
    return output


def _create_mesh_object(
    collection: bpy.types.Collection,
    name: str,
    submeshes: list[SubMesh],
    materials: list[bpy.types.Material],
    bone_names: list[str],
    armature_object: bpy.types.Object | None,
    scale: float,
) -> bpy.types.Object:
    nonempty = [
        submesh
        for submesh in submeshes
        if len(submesh.positions) and len(submesh.faces)
    ]
    if not nonempty:
        raise FormatError("Model contains no nonempty triangle meshes")

    vertex_counts = np.fromiter(
        (len(submesh.positions) for submesh in nonempty),
        dtype=np.int64,
    )
    vertex_offsets = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(vertex_counts[:-1]))
    )
    positions = _convert_vectors(
        np.concatenate([submesh.positions for submesh in nonempty]), scale
    )
    faces = np.concatenate(
        [
            submesh.faces.astype(np.int64, copy=False) + offset
            for submesh, offset in zip(nonempty, vertex_offsets)
        ]
    )
    material_indices = np.concatenate(
        [
            np.full(
                len(submesh.faces),
                submesh.material_index,
                dtype=np.int32,
            )
            for submesh in nonempty
        ]
    )
    submesh_indices = np.concatenate(
        [
            np.full(len(submesh.faces), index, dtype=np.int32)
            for index, submesh in enumerate(nonempty)
        ]
    )
    lods = np.concatenate(
        [
            np.full(len(submesh.faces), submesh.lod, dtype=np.int32)
            for submesh in nonempty
        ]
    )

    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(len(positions))
    mesh.vertices.foreach_set("co", positions.ravel())
    mesh.loops.add(faces.size)
    mesh.loops.foreach_set("vertex_index", faces.ravel())
    mesh.polygons.add(len(faces))
    mesh.polygons.foreach_set(
        "loop_start", np.arange(len(faces), dtype=np.int32) * 3
    )
    mesh.polygons.foreach_set(
        "loop_total", np.full(len(faces), 3, dtype=np.int32)
    )
    mesh.polygons.foreach_set("material_index", material_indices)
    mesh.polygons.foreach_set(
        "use_smooth", np.ones(len(faces), dtype=np.bool_)
    )
    for material in materials:
        mesh.materials.append(material)

    _set_face_attribute(mesh, "gundam_submesh", submesh_indices)
    _set_face_attribute(mesh, "gundam_lod", lods)
    _set_uvs(mesh, nonempty, faces)
    _set_colors(mesh, nonempty, faces)
    mesh.update(calc_edges=True)
    _set_normals(mesh, nonempty)

    mesh_object = bpy.data.objects.new(name, mesh)
    collection.objects.link(mesh_object)
    if bone_names:
        _create_vertex_groups(
            mesh_object, nonempty, vertex_offsets, bone_names
        )
    if armature_object is not None:
        modifier = mesh_object.modifiers.new("Armature", "ARMATURE")
        modifier.object = armature_object
        mesh_object.parent = armature_object
    return mesh_object


def _set_face_attribute(
    mesh: bpy.types.Mesh, name: str, values: np.ndarray
) -> None:
    attribute = mesh.attributes.new(name=name, type="INT", domain="FACE")
    attribute.data.foreach_set("value", values)


def _set_uvs(
    mesh: bpy.types.Mesh,
    submeshes: list[SubMesh],
    faces: np.ndarray,
) -> None:
    if not any(submesh.uvs is not None for submesh in submeshes):
        return
    vertices = []
    for submesh in submeshes:
        if submesh.uvs is None:
            vertices.append(
                np.zeros((len(submesh.positions), 2), dtype=np.float32)
            )
        else:
            vertices.append(
                np.asarray(submesh.uvs[:, :2], dtype=np.float32)
            )
    all_uvs = np.concatenate(vertices)
    loop_uvs = all_uvs[faces.ravel()].copy()
    loop_uvs[:, 1] = 1.0 - loop_uvs[:, 1]
    layer = mesh.uv_layers.new(name="UVMap")
    layer.uv.foreach_set("vector", loop_uvs.ravel())


def _set_colors(
    mesh: bpy.types.Mesh,
    submeshes: list[SubMesh],
    faces: np.ndarray,
) -> None:
    if not any(submesh.colors is not None for submesh in submeshes):
        return
    vertices = []
    for submesh in submeshes:
        if submesh.colors is None:
            vertices.append(
                np.ones((len(submesh.positions), 4), dtype=np.float32)
            )
            continue
        colors = np.asarray(submesh.colors, dtype=np.float32)
        if colors.shape[1] == 3:
            colors = np.column_stack(
                (colors, np.ones(len(colors), dtype=np.float32))
            )
        vertices.append(colors[:, :4])
    all_colors = np.concatenate(vertices)
    loop_colors = all_colors[faces.ravel()]
    attribute = mesh.color_attributes.new(
        name="Color", type="FLOAT_COLOR", domain="CORNER"
    )
    attribute.data.foreach_set("color", loop_colors.ravel())


def _set_normals(
    mesh: bpy.types.Mesh, submeshes: list[SubMesh]
) -> None:
    if not all(submesh.normals is not None for submesh in submeshes):
        return
    normals = _convert_vectors(
        np.concatenate([submesh.normals[:, :3] for submesh in submeshes])
    )
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    np.divide(normals, lengths, out=normals, where=lengths > 0)
    mesh.normals_split_custom_set_from_vertices(normals)


def _create_vertex_groups(
    mesh_object: bpy.types.Object,
    submeshes: list[SubMesh],
    vertex_offsets: np.ndarray,
    bone_names: list[str],
) -> None:
    groups = [
        mesh_object.vertex_groups.new(name=bone_name)
        for bone_name in bone_names
    ]
    for submesh, vertex_offset in zip(submeshes, vertex_offsets):
        if submesh.bone_indices is None or submesh.bone_weights is None:
            continue
        indices = np.asarray(submesh.bone_indices, dtype=np.int32)
        weights = np.asarray(submesh.bone_weights, dtype=np.float32)
        component_count = min(indices.shape[1], weights.shape[1])
        invalid = (
            (indices[:, :component_count] < 0)
            | (indices[:, :component_count] >= len(groups))
        ) & (weights[:, :component_count] > 0.0)
        if np.any(invalid):
            raise FormatError(
                f"{submesh.name} has a weighted bone index outside the armature"
            )
        for component in range(component_count):
            component_indices = indices[:, component]
            component_weights = weights[:, component]
            for bone_index in np.unique(component_indices):
                if not 0 <= bone_index < len(groups):
                    continue
                mask = (component_indices == bone_index) & (
                    component_weights > 0.0
                )
                if not np.any(mask):
                    continue
                selected_weights = component_weights[mask]
                selected_vertices = np.flatnonzero(mask) + vertex_offset
                for weight in np.unique(selected_weights):
                    vertices = selected_vertices[
                        selected_weights == weight
                    ]
                    groups[bone_index].add(
                        vertices.tolist(), float(weight), "ADD"
                    )


def _create_armature(
    context: bpy.types.Context,
    collection: bpy.types.Collection,
    name: str,
    model: Model,
    scale: float,
) -> tuple[bpy.types.Object | None, list[str]]:
    if not model.bones:
        return None, []

    armature = bpy.data.armatures.new(f"{name}_Armature")
    armature_object = bpy.data.objects.new(f"{name}_Armature", armature)
    collection.objects.link(armature_object)
    armature_object.show_in_front = True
    armature_object.display_type = "WIRE"

    for selected in context.view_layer.objects:
        if selected.select_get():
            selected.select_set(False)
    armature_object.select_set(True)
    context.view_layer.objects.active = armature_object
    bpy.ops.object.mode_set(mode="EDIT")
    edit_bones: list[bpy.types.EditBone] = []
    matrices: list[np.ndarray] = []
    heads: list[np.ndarray] = []

    for index, bone in enumerate(model.bones):
        game_matrix = np.asarray(bone.bind_matrix, dtype=np.float32).copy()
        game_head = game_matrix[:3, 3].copy()
        if (
            bone.parent_index >= 0
            and np.linalg.norm(game_head) < 1.0e-7
            and np.linalg.norm(bone.position) > 1.0e-7
        ):
            parent_matrix = matrices[bone.parent_index]
            game_head = (
                heads[bone.parent_index]
                + parent_matrix[:3, :3] @ bone.position
            )
            game_matrix[:3, 3] = game_head
        matrices.append(game_matrix)
        heads.append(game_head)

        converted = (
            _AXIS_CONVERSION
            @ game_matrix
            @ _AXIS_CONVERSION.T
        )
        converted[:3, 3] *= scale
        edit_bone = armature.edit_bones.new(bone.name)
        head = Vector(converted[:3, 3])
        local_length = float(np.linalg.norm(bone.position) * scale)
        length = max(local_length, 0.01 * scale)
        direction = Vector(converted[:3, 1])
        if direction.length_squared < 1.0e-12:
            direction = Vector((0.0, 0.0, 1.0))
        direction.normalize()
        edit_bone.head = head
        edit_bone.tail = head + direction * length
        edit_bone.matrix = Matrix(converted.tolist())
        edit_bones.append(edit_bone)

    for index, bone in enumerate(model.bones):
        if bone.parent_index >= 0:
            edit_bones[index].parent = edit_bones[bone.parent_index]

    bone_names = [bone.name for bone in edit_bones]
    bpy.ops.object.mode_set(mode="OBJECT")
    return armature_object, bone_names


def create_motion_action(
    armature_object: bpy.types.Object,
    package: MotionPackage,
    package_name: str,
    resource_hash: int,
    clip_index: int,
    scale: float = 1.0,
) -> MotionActionResult:
    if armature_object.type != "ARMATURE":
        raise FormatError("Motion Actions require an armature object")
    if not 0 <= clip_index < len(package.clips):
        raise FormatError(
            f"Motion clip {clip_index + 1} is outside package "
            f"{package_name}"
        )

    motion_bone_count = package.base.node_count - 1
    pose_bones = list(armature_object.pose.bones)
    if len(pose_bones) < motion_bone_count:
        raise FormatError(
            f"{package_name} requires {motion_bone_count} bones, but "
            f"{armature_object.name} has {len(pose_bones)}"
        )

    clip = package.clips[clip_index]
    action = bpy.data.actions.new(
        name=f"{package_name}_clip{clip_index + 1:03d}"
    )
    action.use_fake_user = True
    action["gundam_breaker_motion_hash"] = f"{resource_hash:08X}"
    action["gundam_breaker_clip"] = clip_index + 1
    action["gundam_breaker_fps"] = SOURCE_FPS

    slot = action.slots.new(
        id_type="OBJECT",
        name=armature_object.name,
    )
    layer = action.layers.new(name="Motion")
    strip = layer.strips.new(type="KEYFRAME")
    channelbag = strip.channelbag(slot, ensure=True)

    full_frames = np.arange(clip.frame_count, dtype=np.float32) + 1.0
    static_frames = np.array(
        (1.0,)
        if clip.frame_count == 1
        else (1.0, float(clip.frame_count)),
        dtype=np.float32,
    )
    motion_locations, motion_rotations, motion_scales = _sample_motion_basis(
        armature_object,
        package,
        clip_index,
        scale,
    )
    for bone_index, pose_bone in enumerate(pose_bones):
        pose_bone.rotation_mode = "QUATERNION"
        if bone_index < motion_bone_count:
            frames = full_frames
            locations = motion_locations[bone_index]
            rotations = motion_rotations[bone_index]
            scales = motion_scales[bone_index]
        else:
            frames = static_frames
            count = len(frames)
            locations = np.zeros((count, 3), dtype=np.float32)
            rotations = np.zeros((count, 4), dtype=np.float32)
            rotations[:, 0] = 1.0
            scales = np.ones((count, 3), dtype=np.float32)

        _write_action_curves(
            channelbag,
            pose_bone.path_from_id("location"),
            locations,
            frames,
            pose_bone.name,
        )
        _write_action_curves(
            channelbag,
            pose_bone.path_from_id("rotation_quaternion"),
            rotations,
            frames,
            pose_bone.name,
        )
        _write_action_curves(
            channelbag,
            pose_bone.path_from_id("scale"),
            scales,
            frames,
            pose_bone.name,
        )
    return MotionActionResult(action=action, slot=slot)


def _sample_motion_basis(
    armature_object: bpy.types.Object,
    package: MotionPackage,
    clip_index: int,
    scale: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    clip = package.clips[clip_index]
    motion_bone_count = package.base.node_count - 1
    locations = [
        np.empty((clip.frame_count, 3), dtype=np.float32)
        for _ in range(motion_bone_count)
    ]
    rotations = [
        np.empty((clip.frame_count, 4), dtype=np.float32)
        for _ in range(motion_bone_count)
    ]
    scales = [
        np.empty((clip.frame_count, 3), dtype=np.float32)
        for _ in range(motion_bone_count)
    ]
    axis = Matrix(_AXIS_CONVERSION.tolist())
    data_bones = list(armature_object.data.bones)
    bone_indices = {
        bone.name: index for index, bone in enumerate(data_bones)
    }
    base_absolute = _absolute_motion_matrices(
        package.base.transforms,
        package.base.parents,
        scale,
    )
    base_inverses = [matrix.inverted() for matrix in base_absolute]
    previous_rotations: list[np.ndarray | None] = [
        None
    ] * motion_bone_count

    for frame in range(clip.frame_count):
        sampled = list(
            clip.sample_node(package.base, node_index, frame)
            for node_index in range(package.base.node_count)
        )
        animated_absolute = _animated_motion_matrices(
            package,
            clip,
            sampled,
            scale,
        )
        pose_matrices: list[Matrix] = []
        for bone_index in range(motion_bone_count):
            node_index = bone_index + 1
            data_bone = data_bones[bone_index]
            deformation = (
                animated_absolute[node_index]
                @ base_inverses[node_index]
            )
            converted_deformation = (
                axis @ deformation @ axis.transposed()
            )
            pose_matrix = (
                converted_deformation @ data_bone.matrix_local
            )
            pose_matrices.append(pose_matrix)

            if data_bone.parent is None:
                basis = data_bone.matrix_local.inverted() @ pose_matrix
            else:
                parent_index = bone_indices[data_bone.parent.name]
                if parent_index >= bone_index:
                    raise FormatError(
                        f"Bone {data_bone.name} has an invalid animation parent"
                    )
                basis = (
                    data_bone.matrix_local.inverted()
                    @ data_bone.parent.matrix_local
                    @ pose_matrices[parent_index].inverted()
                    @ pose_matrix
                )
            location, rotation, basis_scale = basis.decompose()
            rotation.normalize()
            rotation_array = np.asarray(
                tuple(rotation), dtype=np.float32
            )
            previous_rotation = previous_rotations[bone_index]
            if (
                previous_rotation is not None
                and np.dot(previous_rotation, rotation_array) < 0.0
            ):
                rotation_array = -rotation_array
            locations[bone_index][frame] = location
            rotations[bone_index][frame] = rotation_array
            scales[bone_index][frame] = basis_scale
            previous_rotations[bone_index] = rotation_array
    return locations, rotations, scales


def _animated_motion_matrices(
    package: MotionPackage,
    clip: MotionClip,
    transforms: list[LocalTransform],
    scale: float,
) -> list[Matrix]:
    base = package.base
    chains = {}
    for end_index, node_type in enumerate(base.node_types):
        if node_type < 3:
            continue
        middle_index = base.parents[end_index]
        if middle_index < 0 or base.node_types[middle_index] != 2:
            continue
        root_index = base.parents[middle_index]
        if root_index >= 0 and base.node_types[root_index] == 1:
            chains[root_index] = (middle_index, end_index)

    matrices: list[Matrix | None] = [None] * base.node_count
    for node_index, transform in enumerate(transforms):
        if matrices[node_index] is not None:
            continue
        parent_index = base.parents[node_index]
        parent_matrix = (
            None if parent_index < 0 else matrices[parent_index]
        )
        if parent_index >= 0 and parent_matrix is None:
            raise FormatError(
                f"Motion node {node_index} has an unevaluated parent"
            )
        local = _local_transform_matrix(transform, scale)
        absolute = local if parent_matrix is None else parent_matrix @ local
        matrices[node_index] = absolute

        chain = chains.get(node_index)
        if chain is None:
            continue
        middle_index, end_index = chain
        end_tracks = clip.bone_tracks[end_index - 1]
        if end_tracks.translation.is_rest:
            continue

        middle_local = _local_transform_matrix(
            transforms[middle_index], scale
        )
        end_transform = transforms[end_index]
        end_local = _local_transform_matrix(
            LocalTransform(
                translation=base.transforms[end_index].translation,
                rotation=end_transform.rotation,
                scale=end_transform.scale,
            ),
            scale,
        )
        middle_absolute = absolute @ middle_local
        end_absolute = middle_absolute @ end_local
        target = Vector(end_transform.translation) * scale
        (
            matrices[node_index],
            matrices[middle_index],
            matrices[end_index],
        ) = _solve_two_bone_chain(
            absolute,
            middle_absolute,
            end_absolute,
            middle_local,
            end_local,
            target,
            base.node_types[end_index] == 4,
        )

    if any(matrix is None for matrix in matrices):
        raise FormatError("Motion hierarchy evaluation is incomplete")
    return [matrix for matrix in matrices if matrix is not None]


def _solve_two_bone_chain(
    root: Matrix,
    middle: Matrix,
    end: Matrix,
    middle_local: Matrix,
    end_local: Matrix,
    target: Vector,
    reverse_bend: bool,
) -> tuple[Matrix, Matrix, Matrix]:
    root_position = root.translation
    middle_position = middle.translation
    end_position = end.translation
    upper_length = (middle_position - root_position).length
    lower_length = (end_position - middle_position).length
    if upper_length < 1.0e-8 or lower_length < 1.0e-8:
        return root, middle, end

    target_offset = target - root_position
    target_distance = target_offset.length
    if target_distance < 1.0e-8:
        target_offset = end_position - root_position
        target_distance = target_offset.length
    if target_distance < 1.0e-8:
        return root, middle, end

    target_direction = target_offset.normalized()
    minimum_distance = abs(upper_length - lower_length) + 1.0e-7
    maximum_distance = upper_length + lower_length - 1.0e-7
    solved_distance = min(
        max(target_distance, minimum_distance),
        maximum_distance,
    )
    solved_target = root_position + target_direction * solved_distance

    pole = middle_position - root_position
    pole -= target_direction * pole.dot(target_direction)
    if pole.length_squared < 1.0e-12:
        pole = root.to_3x3() @ Vector((0.0, 1.0, 0.0))
        pole -= target_direction * pole.dot(target_direction)
    if pole.length_squared < 1.0e-12:
        pole = root.to_3x3() @ Vector((0.0, 0.0, 1.0))
        pole -= target_direction * pole.dot(target_direction)
    if pole.length_squared < 1.0e-12:
        return root, middle, end
    pole.normalize()
    if reverse_bend:
        pole.negate()

    along = (
        upper_length * upper_length
        - lower_length * lower_length
        + solved_distance * solved_distance
    ) / (2.0 * solved_distance)
    height = math.sqrt(
        max(upper_length * upper_length - along * along, 0.0)
    )
    solved_middle_position = (
        root_position + target_direction * along + pole * height
    )

    current_upper = middle_position - root_position
    solved_upper = solved_middle_position - root_position
    root_location, root_rotation, root_scale = root.decompose()
    root_rotation = (
        current_upper.normalized().rotation_difference(
            solved_upper.normalized()
        )
        @ root_rotation
    )
    solved_root = Matrix.LocRotScale(
        root_location,
        root_rotation,
        root_scale,
    )

    swung_middle = solved_root @ middle_local
    current_lower = (swung_middle @ end_local).translation
    current_lower -= swung_middle.translation
    solved_lower = solved_target - solved_middle_position
    middle_location, middle_rotation, middle_scale = (
        swung_middle.decompose()
    )
    middle_rotation = (
        current_lower.normalized().rotation_difference(
            solved_lower.normalized()
        )
        @ middle_rotation
    )
    solved_middle = Matrix.LocRotScale(
        solved_middle_position,
        middle_rotation,
        middle_scale,
    )
    solved_end = solved_middle @ end_local
    solved_end.translation = solved_target
    return solved_root, solved_middle, solved_end


def _absolute_motion_matrices(
    transforms,
    parents: tuple[int, ...],
    scale: float,
) -> list[Matrix]:
    matrices = []
    for node_index, transform in enumerate(transforms):
        local = _local_transform_matrix(transform, scale)
        parent_index = parents[node_index]
        matrices.append(
            local
            if parent_index < 0
            else matrices[parent_index] @ local
        )
    return matrices


def _local_transform_matrix(transform, scale: float) -> Matrix:
    x, y, z, w = transform.rotation
    return Matrix.LocRotScale(
        Vector(transform.translation) * scale,
        Quaternion((w, x, y, z)),
        Vector(transform.scale),
    )


def _write_action_curves(
    channelbag: object,
    data_path: str,
    values: np.ndarray,
    frames: np.ndarray,
    group_name: str,
) -> None:
    coordinates = np.empty((len(frames), 2), dtype=np.float32)
    coordinates[:, 0] = frames
    for component in range(values.shape[1]):
        coordinates[:, 1] = values[:, component]
        fcurve = channelbag.fcurves.new(
            data_path,
            index=component,
            group_name=group_name,
        )
        fcurve.keyframe_points.add(len(frames))
        fcurve.keyframe_points.foreach_set("co", coordinates.ravel())
        for point in fcurve.keyframe_points:
            point.interpolation = "LINEAR"
        fcurve.update()


def _create_material(
    source: GameSource,
    definition: MaterialDefinition,
    image_cache: dict[tuple[int, bool], bpy.types.Image],
    warnings: list[str],
    import_textures: bool,
) -> bpy.types.Material:
    material = bpy.data.materials.new(definition.name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    output.location = (640.0, 0.0)
    shader = nodes.new("ShaderNodeBsdfPrincipled")
    shader.location = (360.0, 0.0)
    links.new(shader.outputs["BSDF"], output.inputs["Surface"])

    color = definition.parameters.get(
        "gMaterialColor", (0.8, 0.8, 0.8, 1.0)
    )
    material.diffuse_color = color
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Alpha"].default_value = color[3]
    specular = definition.parameters.get("gSpecularParameter")
    if specular:
        shader.inputs["Roughness"].default_value = min(
            1.0, math.sqrt(2.0 / max(specular[0] + 2.0, 2.0))
        )
    for parameter_name, value in definition.parameters.items():
        material[f"gundam_{parameter_name}"] = value

    if not import_textures:
        return material

    y = 280.0
    for slot_name, texture_name in definition.textures.items():
        non_color = slot_name in {
            "gNormalMapSampler",
            "gNormalMapSecondSampler",
            "gSpecularMapSampler",
            "gControlMapSampler",
        }
        record = source.find_name(texture_name, TEXTURE_TYPE_HASH)
        if record is None:
            warnings.append(
                f"{definition.name}: texture not found: {texture_name}"
            )
            material[f"missing_{slot_name}"] = texture_name
            continue
        cache_key = (record.resource_hash, non_color)
        image = image_cache.get(cache_key)
        if image is None:
            try:
                image = _create_image(
                    record.display_name,
                    record.read_decoded(),
                    source.profile.endian_hint,
                    non_color,
                )
            except FormatError as error:
                warnings.append(
                    f"{definition.name}: {texture_name}: {error}"
                )
                continue
            image["gundam_resource_hash"] = f"{record.resource_hash:08X}"
            image_cache[cache_key] = image

        texture_node = nodes.new("ShaderNodeTexImage")
        texture_node.name = slot_name
        texture_node.label = slot_name
        texture_node.image = image
        texture_node.location = (-520.0, y)
        y -= 260.0

        if slot_name == "gColorMapSampler":
            tint = nodes.new("ShaderNodeMixRGB")
            tint.blend_type = "MULTIPLY"
            tint.inputs["Fac"].default_value = 1.0
            tint.inputs["Color2"].default_value = color
            tint.location = (80.0, 120.0)
            links.new(texture_node.outputs["Color"], tint.inputs["Color1"])
            links.new(tint.outputs["Color"], shader.inputs["Base Color"])
            links.new(texture_node.outputs["Alpha"], shader.inputs["Alpha"])
        elif slot_name == "gNormalMapSampler":
            normal = nodes.new("ShaderNodeNormalMap")
            normal.location = (80.0, -180.0)
            links.new(texture_node.outputs["Color"], normal.inputs["Color"])
            links.new(normal.outputs["Normal"], shader.inputs["Normal"])
        elif slot_name == "gSpecularMapSampler":
            specular_input = shader.inputs.get("Specular IOR Level")
            if specular_input is not None:
                links.new(texture_node.outputs["Color"], specular_input)
    return material


def _create_image(
    name: str,
    data: bytes,
    endian_hint: str,
    non_color: bool,
) -> bpy.types.Image:
    texture = parse_texture(data, endian_hint)
    if texture.pixels is not None:
        image = bpy.data.images.new(
            name=name,
            width=texture.width,
            height=texture.height,
            alpha=True,
        )
        pixels = np.flip(texture.pixels, axis=0)
        image.pixels.foreach_set(pixels.ravel())
        image.update()
        image.pack()
    else:
        dds_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".dds", delete=False
            ) as temporary:
                temporary.write(make_dds(texture))
                dds_path = Path(temporary.name)
            image = bpy.data.images.load(str(dds_path), check_existing=False)
            image.name = name
            image.pack()
        finally:
            if dds_path is not None:
                dds_path.unlink(missing_ok=True)
    if non_color:
        image.colorspace_settings.name = "Non-Color"
    image.alpha_mode = "STRAIGHT"
    return image
