# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import bisect
import math
import re
import struct
from dataclasses import dataclass

import numpy as np

from .archive import (
    MOTION_TYPE_HASH,
    FormatError,
    GameSource,
    ResourceRecord,
    require,
)
from .model import Model

SOURCE_FPS = 30.0
MAX_CONTROL_COUNT = 0x900
_REST_SENTINEL = 9.999999680285692e37
_IMPLICIT_VALUES = (
    0.0,
    _REST_SENTINEL,
    math.pi / 2.0,
    math.pi,
    3.0 * math.pi / 2.0,
)
_CHARACTER_PATTERN = re.compile(r"chr\d{4}", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ScalarCurve:
    kind: int
    interior_frames: tuple[int, ...] = ()
    values: tuple[float, ...] = ()
    tangents: tuple[float, ...] = ()

    @property
    def is_rest(self) -> bool:
        return self.kind == 1

    def evaluate(
        self,
        frame: float,
        frame_count: int,
        fps: float = SOURCE_FPS,
    ) -> float:
        require(not self.is_rest, "A rest-sentinel curve has no scalar value")
        if self.kind <= 5:
            return self.values[0]
        if frame_count <= 1:
            return self.values[0]

        last_frame = frame_count - 1
        sample_frame = min(max(float(frame), 0.0), float(last_frame))
        key_frames = (0, *self.interior_frames, last_frame)
        right_index = bisect.bisect_left(key_frames, sample_frame)
        if right_index < len(key_frames) and key_frames[right_index] == sample_frame:
            return self.values[right_index]
        right_index = min(max(right_index, 1), len(key_frames) - 1)
        left_index = right_index - 1
        left_frame = key_frames[left_index]
        right_frame = key_frames[right_index]
        fraction = (sample_frame - left_frame) / (right_frame - left_frame)
        smooth = fraction * fraction * (3.0 - 2.0 * fraction)
        value = self.values[left_index] + (
            self.values[right_index] - self.values[left_index]
        ) * smooth
        if self.kind == 7:
            duration = (right_frame - left_frame) / fps
            tangent = (
                self.tangents[left_index] * (1.0 - fraction)
                - self.tangents[right_index] * fraction
            )
            value += (
                duration
                * fraction
                * (1.0 - fraction)
                * tangent
            )
        return value


@dataclass(frozen=True, slots=True)
class VectorTrack:
    curves: tuple[ScalarCurve, ScalarCurve, ScalarCurve] | None = None

    @property
    def is_rest(self) -> bool:
        return self.curves is None

    def evaluate(
        self,
        base_value: tuple[float, float, float],
        frame: float,
        frame_count: int,
    ) -> tuple[float, float, float]:
        if self.curves is None:
            return base_value
        return tuple(
            curve.evaluate(frame, frame_count) for curve in self.curves
        )


@dataclass(frozen=True, slots=True)
class NodeTracks:
    translation: VectorTrack
    rotation: VectorTrack
    scale: VectorTrack

    @property
    def is_expanded(self) -> bool:
        return not (
            self.translation.is_rest
            and self.rotation.is_rest
            and self.scale.is_rest
        )


@dataclass(frozen=True, slots=True)
class LocalTransform:
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    scale: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class MotionBase:
    parents: tuple[int, ...]
    node_types: tuple[int, ...]
    joint_ids: tuple[int, ...]
    transforms: tuple[LocalTransform, ...]

    @property
    def node_count(self) -> int:
        return len(self.transforms)


@dataclass(frozen=True, slots=True)
class MotionClip:
    frame_count: int
    special_curves: tuple[ScalarCurve, ScalarCurve]
    global_tracks: NodeTracks
    bone_tracks: tuple[NodeTracks, ...]
    curves: tuple[ScalarCurve, ...]

    def sample_node(
        self,
        base: MotionBase,
        node_index: int,
        frame: float,
    ) -> LocalTransform:
        require(
            0 <= node_index < base.node_count,
            f"Motion node {node_index} is out of range",
        )
        tracks = (
            self.global_tracks
            if node_index == 0
            else self.bone_tracks[node_index - 1]
        )
        rest = base.transforms[node_index]
        translation = tracks.translation.evaluate(
            rest.translation, frame, self.frame_count
        )
        if tracks.rotation.is_rest:
            rotation = rest.rotation
        else:
            angles = tracks.rotation.evaluate(
                (0.0, 0.0, 0.0), frame, self.frame_count
            )
            rotation = euler_xyz_quaternion(angles)
        scale = tracks.scale.evaluate(
            (1.0, 1.0, 1.0), frame, self.frame_count
        )
        return LocalTransform(translation, rotation, scale)


@dataclass(frozen=True, slots=True)
class MotionPackage:
    base: MotionBase
    clips: tuple[MotionClip, ...]


@dataclass(frozen=True, slots=True)
class MotionAsset:
    name: str
    resource: ResourceRecord
    clip_count: int

    @property
    def resource_hash(self) -> int:
        return self.resource.resource_hash


@dataclass(frozen=True, slots=True)
class MotionDiscovery:
    assets: tuple[MotionAsset, ...]
    warnings: tuple[str, ...]


def euler_xyz_quaternion(
    angles: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    x, y, z = (value * 0.5 for value in angles)
    sx, cx = math.sin(x), math.cos(x)
    sy, cy = math.sin(y), math.cos(y)
    sz, cz = math.sin(z), math.cos(z)
    return (
        cz * cy * sx - sz * sy * cx,
        cz * sy * cx + sz * cy * sx,
        sz * cy * cx - cz * sy * sx,
        cz * cy * cx + sz * sy * sx,
    )


def parse_motion_package(data: bytes) -> MotionPackage:
    require(len(data) >= 12, "MPK package is too small")
    section_count = struct.unpack_from(">I", data)[0]
    require(2 <= section_count <= 100_000, "Invalid MPK section count")
    table_size = 4 * (section_count + 1)
    require(
        table_size <= len(data) - 4,
        "MPK section table is truncated",
    )
    offsets = struct.unpack_from(f">{section_count + 1}I", data, 4)
    table_end = 4 + table_size
    require(offsets[0] >= table_end, "MPK first section overlaps its table")
    require(
        all(left < right for left, right in zip(offsets, offsets[1:])),
        "MPK section offsets are not strictly increasing",
    )
    require(offsets[-1] == len(data), "MPK package has trailing or missing data")

    base = _parse_base_section(data, offsets[0], offsets[1])
    clips = tuple(
        _parse_clip_section(
            data,
            offsets[index],
            offsets[index + 1],
            base.node_count,
        )
        for index in range(1, section_count)
    )
    return MotionPackage(base=base, clips=clips)


def _parse_base_section(data: bytes, start: int, end: int) -> MotionBase:
    require(end - start >= 4, "MPK base section is truncated")
    node_count = data[start]
    require(node_count >= 2, "MPK base section has fewer than two nodes")
    cursor = start + 4

    parents = [-1, 0]
    node_types = [0, 0]
    for node_index in range(2, node_count):
        require(cursor + 2 <= end, "MPK parent table is truncated")
        parent_index = data[cursor]
        node_type = data[cursor + 1]
        require(
            parent_index < node_index,
            f"MPK node {node_index} has invalid parent {parent_index}",
        )
        require(
            node_type <= 4,
            f"MPK node {node_index} has invalid type {node_type}",
        )
        parents.append(parent_index)
        node_types.append(node_type)
        cursor += 2
    cursor = _align(cursor, 4)

    joint_count = node_count - 1
    require(cursor + joint_count <= end, "MPK joint ID table is truncated")
    raw_joint_ids = tuple(data[cursor : cursor + joint_count])
    cursor = _align(cursor + joint_count, 4)
    joint_ids = (
        (raw_joint_ids[-1], *raw_joint_ids[:-1])
        if raw_joint_ids
        else ()
    )

    transform_size = 28 * node_count
    require(
        cursor + transform_size == end,
        "MPK base transform section has an unexpected size",
    )
    raw_transforms = []
    for node_index in range(node_count):
        values = struct.unpack_from(">7f", data, cursor + node_index * 28)
        quaternion = _normalize_quaternion(values[3:7])
        raw_transforms.append(
            LocalTransform(
                translation=values[:3],
                rotation=quaternion,
                scale=(1.0, 1.0, 1.0),
            )
        )
    transforms = []
    for node_index, transform in enumerate(raw_transforms):
        parent_index = parents[node_index]
        rotation = transform.rotation
        if parent_index >= 0:
            rotation = _normalize_quaternion(
                _multiply_quaternions(
                    _conjugate_quaternion(
                        raw_transforms[parent_index].rotation
                    ),
                    rotation,
                )
            )
        transforms.append(
            LocalTransform(
                translation=transform.translation,
                rotation=rotation,
                scale=transform.scale,
            )
        )
    return MotionBase(
        parents=tuple(parents),
        node_types=tuple(node_types),
        joint_ids=tuple(joint_ids),
        transforms=tuple(transforms),
    )


def _parse_clip_section(
    data: bytes,
    start: int,
    end: int,
    node_count: int,
) -> MotionClip:
    require(end - start >= 4, "MPK clip section is truncated")
    control_count, frame_count = struct.unpack_from(">HH", data, start)
    require(
        0 < control_count <= MAX_CONTROL_COUNT,
        f"Invalid MPK control count {control_count}",
    )
    require(frame_count > 0, "MPK clip has no frames")
    cursor = start + 4

    packed_size = 3 * ((control_count + 7) // 8)
    require(cursor + packed_size <= end, "MPK control type stream is truncated")
    kinds = []
    for group_offset in range(cursor, cursor + packed_size, 3):
        packed = int.from_bytes(data[group_offset : group_offset + 3], "big")
        kinds.extend((packed >> (3 * index)) & 7 for index in range(8))
    kinds = kinds[:control_count]
    cursor += packed_size

    frame_width = 1 if frame_count <= 255 else 2
    if frame_width == 2:
        cursor = _align(cursor, 2)
    interior_frames = []
    for kind in kinds:
        if kind < 6:
            interior_frames.append(())
            continue
        require(
            cursor + frame_width <= end,
            "MPK curve key count is truncated",
        )
        count = int.from_bytes(
            data[cursor : cursor + frame_width], "big"
        )
        cursor += frame_width
        byte_count = count * frame_width
        require(cursor + byte_count <= end, "MPK curve frame table is truncated")
        frames = tuple(
            int.from_bytes(
                data[
                    cursor + index * frame_width :
                    cursor + (index + 1) * frame_width
                ],
                "big",
            )
            for index in range(count)
        )
        require(
            all(
                0 < frame < frame_count - 1
                for frame in frames
            ),
            "MPK interior key frame lies outside the clip",
        )
        require(
            all(left < right for left, right in zip(frames, frames[1:])),
            "MPK interior key frames are not strictly increasing",
        )
        interior_frames.append(frames)
        cursor += byte_count

    cursor = _align(cursor, 4)
    curves = []
    for kind, frames in zip(kinds, interior_frames):
        if kind <= 4:
            curves.append(
                ScalarCurve(kind=kind, values=(_IMPLICIT_VALUES[kind],))
            )
            continue
        if kind == 5:
            values, cursor = _read_floats(data, cursor, end, 1)
            curves.append(ScalarCurve(kind=kind, values=values))
            continue

        key_count = len(frames) + 2
        value_count = key_count * (2 if kind == 7 else 1)
        raw_values, cursor = _read_floats(
            data, cursor, end, value_count
        )
        if kind == 7:
            values = raw_values[::2]
            tangents = raw_values[1::2]
        else:
            values = raw_values
            tangents = ()
        curves.append(
            ScalarCurve(
                kind=kind,
                interior_frames=frames,
                values=values,
                tangents=tangents,
            )
        )
    require(cursor == end, "MPK clip payload has an unexpected size")

    curve_tuple = tuple(curves)
    require(control_count >= 4, "MPK clip has no global node controls")
    special_curves = (curve_tuple[0], curve_tuple[1])
    cursor_index = 2
    global_translation, cursor_index = _take_vector_track(
        curve_tuple, cursor_index
    )
    global_rotation, cursor_index = _take_vector_track(
        curve_tuple, cursor_index
    )
    global_tracks = NodeTracks(
        translation=global_translation,
        rotation=global_rotation,
        scale=VectorTrack(),
    )

    bone_tracks = []
    for _node_index in range(1, node_count):
        translation, cursor_index = _take_vector_track(
            curve_tuple, cursor_index
        )
        rotation, cursor_index = _take_vector_track(
            curve_tuple, cursor_index
        )
        scale, cursor_index = _take_vector_track(
            curve_tuple, cursor_index
        )
        bone_tracks.append(NodeTracks(translation, rotation, scale))
    require(
        cursor_index == len(curve_tuple),
        "MPK sparse control stream does not match its node count",
    )
    return MotionClip(
        frame_count=frame_count,
        special_curves=special_curves,
        global_tracks=global_tracks,
        bone_tracks=tuple(bone_tracks),
        curves=curve_tuple,
    )


def _take_vector_track(
    curves: tuple[ScalarCurve, ...],
    cursor: int,
) -> tuple[VectorTrack, int]:
    require(cursor < len(curves), "MPK sparse vector stream is truncated")
    if curves[cursor].is_rest:
        return VectorTrack(), cursor + 1
    require(
        cursor + 3 <= len(curves),
        "MPK expanded vector stream is truncated",
    )
    components = curves[cursor : cursor + 3]
    require(
        not any(curve.is_rest for curve in components),
        "MPK rest sentinel appears inside an expanded vector",
    )
    return VectorTrack(components), cursor + 3


def _read_floats(
    data: bytes,
    cursor: int,
    end: int,
    count: int,
) -> tuple[tuple[float, ...], int]:
    byte_count = count * 4
    require(cursor + byte_count <= end, "MPK curve value payload is truncated")
    values = struct.unpack_from(f">{count}f", data, cursor)
    require(
        all(math.isfinite(value) for value in values),
        "MPK curve contains a non-finite value",
    )
    return values, cursor + byte_count


def _normalize_quaternion(
    values: tuple[float, ...],
) -> tuple[float, float, float, float]:
    length = math.sqrt(sum(value * value for value in values))
    require(length > 1.0e-8, "MPK base quaternion has zero length")
    return tuple(value / length for value in values)


def _conjugate_quaternion(
    value: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x, y, z, w = value
    return -x, -y, -z, w


def _multiply_quaternions(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & -alignment


def motion_compatibility_error(
    package: MotionPackage,
    model: Model,
) -> str | None:
    bone_count = package.base.node_count - 1
    if len(model.bones) < bone_count:
        return (
            f"motion requires {bone_count} bones, model has "
            f"{len(model.bones)}"
        )
    for bone_index in range(bone_count):
        model_parent = model.bones[bone_index].parent_index
        expected_parent = 0 if model_parent < 0 else model_parent + 1
        actual_parent = package.base.parents[bone_index + 1]
        if actual_parent != expected_parent:
            return (
                f"motion node {bone_index + 1} has parent {actual_parent}, "
                f"expected {expected_parent}"
            )
        base_translation = package.base.transforms[
            bone_index + 1
        ].translation
        if not np.allclose(
            base_translation,
            model.bones[bone_index].position,
            rtol=1.0e-4,
            atol=2.0e-4,
        ):
            return f"motion node {bone_index + 1} has a different rest position"
    return None


def discover_motion_assets(
    source: GameSource,
    model_name: str,
    model: Model,
) -> MotionDiscovery:
    match = _CHARACTER_PATTERN.search(model_name)
    if match is None:
        return MotionDiscovery(
            assets=(),
            warnings=(
                f"Cannot derive a character motion path from {model_name!r}",
            ),
        )
    character = match.group(0).lower()
    names = [
        (
            character,
            f"GP/motion/{character}/BigEndian/{character}",
        )
    ]
    names.extend(
        (
            f"{character}_gmp{index:03d}",
            (
                f"GP/motion/{character}/{character}_gmp{index:03d}/"
                f"BigEndian/{character}_gmp{index:03d}"
            ),
        )
        for index in range(1000)
    )

    assets = []
    warnings = []
    seen_hashes = set()
    for display_name, resource_name in names:
        resource = source.find_name(resource_name, MOTION_TYPE_HASH)
        if resource is None or resource.resource_hash in seen_hashes:
            continue
        seen_hashes.add(resource.resource_hash)
        try:
            package = parse_motion_package(resource.read_decoded())
        except FormatError as error:
            warnings.append(f"{display_name}: {error}")
            continue
        compatibility_error = motion_compatibility_error(package, model)
        if compatibility_error is not None:
            continue
        assets.append(
            MotionAsset(
                name=display_name,
                resource=resource,
                clip_count=len(package.clips),
            )
        )
    return MotionDiscovery(tuple(assets), tuple(warnings))
