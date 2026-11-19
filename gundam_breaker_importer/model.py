# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from .archive import FormatError, require


@dataclass(frozen=True, slots=True)
class ModelHeader:
    layout: str
    endian: str
    field_offset: int
    bone_data_offset: int
    bone_name_size: int
    bone_map_size: int
    bone_count: int
    count3: int
    count4: int
    count5: int
    material_count: int
    mesh_count: int
    buffer_count: int
    vertex_size: int
    face_size: int
    fvf_size: int


@dataclass(frozen=True, slots=True)
class Bone:
    name: str
    parent_index: int
    position: np.ndarray
    bind_matrix: np.ndarray


@dataclass(frozen=True, slots=True)
class MaterialRange:
    material_id: int
    lod: int
    unknown: int
    vertex_count: int
    index_start: int
    index_count: int
    base_vertex: int
    name: str


@dataclass(frozen=True, slots=True)
class MeshInfo:
    material_range_start: int
    material_range_count: int
    vertex_buffer_start: int
    vertex_buffer_size: int
    vertex_stride: int
    index_buffer_start: int
    index_buffer_size: int
    index_type: int
    unknown_offset: int
    bone_palette_start: int
    bone_palette_count: int
    material_index: int
    unknown2: int
    unknown3: int


@dataclass(frozen=True, slots=True)
class VertexElement:
    end: int
    offset: int
    component_count: int
    data_type: int
    semantic: int


@dataclass(slots=True)
class SubMesh:
    name: str
    material_index: int
    lod: int
    positions: np.ndarray
    faces: np.ndarray
    normals: np.ndarray | None = None
    tangents: np.ndarray | None = None
    uvs: np.ndarray | None = None
    colors: np.ndarray | None = None
    bone_indices: np.ndarray | None = None
    bone_weights: np.ndarray | None = None


@dataclass(slots=True)
class Model:
    header: ModelHeader
    bones: list[Bone]
    submeshes: list[SubMesh]


@dataclass(slots=True)
class MaterialDefinition:
    name: str
    textures: dict[str, str] = field(default_factory=dict)
    parameters: dict[str, tuple[float, float, float, float]] = field(
        default_factory=dict
    )


def _decode_string(value: bytes) -> str:
    for encoding in ("utf-8", "shift_jis"):
        try:
            return value.decode(encoding)
        except UnicodeDecodeError:
            pass
    return value.decode("latin-1")


def _read_c_string(data: bytes, offset: int, limit: int | None = None) -> str:
    require(0 <= offset < len(data), f"String offset 0x{offset:X} is invalid")
    end_limit = len(data) if limit is None else min(len(data), offset + limit)
    end = data.find(b"\x00", offset, end_limit)
    require(end >= 0, f"String at 0x{offset:X} is not terminated")
    return _decode_string(data[offset:end])


def _valid_bone_names(data: bytes, offset: int, size: int, count: int) -> bool:
    if offset < 0 or size < 0 or offset + size > len(data):
        return False
    cursor = offset
    limit = offset + size
    try:
        for _ in range(count):
            end = data.index(0, cursor, limit)
            raw = data[cursor:end]
            if not raw or any(byte < 0x20 or byte >= 0x7F for byte in raw):
                return False
            cursor = end + 1
    except ValueError:
        return False
    return cursor <= limit


def _ps3_header(data: bytes) -> ModelHeader | None:
    if len(data) < 60:
        return None
    fields = struct.unpack_from(">14I", data)
    (
        bone_name_size,
        bone_map_size,
        bone_count,
        count3,
        count4,
        count5,
        material_count,
        mesh_count,
        _unknown8,
        _unknown9,
        fvf_size,
        buffer_count,
        face_size,
        vertex_size,
    ) = fields
    if (
        bone_count > 4096
        or material_count > 100_000
        or mesh_count > 100_000
        or mesh_count == 0
    ):
        return None

    vertex_base = len(data) - vertex_size
    fvf_base = vertex_base - fvf_size
    mesh_records_start = fvf_base - mesh_count * 60
    material_records_start = mesh_records_start - material_count * 128
    bone_section_size = (
        bone_name_size
        + bone_map_size
        + bone_count * (80 + 64 + 64)
    )
    bone_data_offset = material_records_start - bone_section_size
    face_base = bone_data_offset - face_size
    face_end = bone_data_offset
    if buffer_count < mesh_count or face_base < 56:
        return None
    expected_bone_end = bone_data_offset + bone_section_size
    if not (
        60 <= face_base <= face_end <= expected_bone_end
        <= material_records_start
        <= mesh_records_start
        <= fvf_base
        <= vertex_base
        <= len(data)
    ):
        return None
    if not _valid_bone_names(data, face_end, bone_name_size, bone_count):
        return None

    return ModelHeader(
        layout="ps3",
        endian=">",
        field_offset=0,
        bone_data_offset=bone_data_offset,
        bone_name_size=bone_name_size,
        bone_map_size=bone_map_size,
        bone_count=bone_count,
        count3=count3,
        count4=count4,
        count5=count5,
        material_count=material_count,
        mesh_count=mesh_count,
        buffer_count=buffer_count,
        vertex_size=vertex_size,
        face_size=face_size,
        fvf_size=fvf_size,
    )


def _vita_header(data: bytes) -> ModelHeader | None:
    if len(data) < 52:
        return None
    fields = struct.unpack_from("<12I", data, 4)
    (
        bone_name_size,
        bone_map_size,
        bone_count,
        count3,
        count4,
        count5,
        material_count,
        mesh_count,
        vertex_size,
        face_size,
        fvf_size,
        _expression_count,
    ) = fields
    if (
        bone_count > 4096
        or not 0 < material_count <= 100_000
        or not 0 < mesh_count <= 100_000
        or vertex_size == 0
        or face_size == 0
        or fvf_size == 0
    ):
        return None

    bone_data_offset = 52
    bone_records_start = bone_data_offset + bone_name_size + bone_map_size
    material_records_start = bone_records_start + bone_count * (80 + 64 + 64)
    mesh_records_start = material_records_start + material_count * 112
    vertex_base = mesh_records_start + mesh_count * 56
    face_base = vertex_base + vertex_size
    fvf_base = face_base + face_size
    if not (
        bone_data_offset
        <= bone_records_start
        <= material_records_start
        <= mesh_records_start
        <= vertex_base
        <= face_base
        <= fvf_base
        <= fvf_base + fvf_size
        <= len(data)
    ):
        return None
    if not _valid_bone_names(
        data, bone_data_offset, bone_name_size, bone_count
    ):
        return None

    return ModelHeader(
        layout="vita",
        endian="<",
        field_offset=4,
        bone_data_offset=bone_data_offset,
        bone_name_size=bone_name_size,
        bone_map_size=bone_map_size,
        bone_count=bone_count,
        count3=count3,
        count4=count4,
        count5=count5,
        material_count=material_count,
        mesh_count=mesh_count,
        buffer_count=mesh_count,
        vertex_size=vertex_size,
        face_size=face_size,
        fvf_size=fvf_size,
    )


def _select_header(data: bytes, endian_hint: str | None) -> ModelHeader:
    candidates = [candidate for candidate in (_ps3_header(data), _vita_header(data)) if candidate]
    require(bool(candidates), "No supported console MDL layout matched the file")
    if endian_hint in ("<", ">"):
        preferred = [
            candidate for candidate in candidates if candidate.endian == endian_hint
        ]
        if preferred:
            return preferred[0]
    return candidates[0]


def _section_offsets(
    data: bytes, header: ModelHeader
) -> tuple[int, int, int, int, int, int]:
    if header.layout == "ps3":
        vertex_base = len(data) - header.vertex_size
        fvf_base = vertex_base - header.fvf_size
        mesh_records_start = fvf_base - header.mesh_count * 60
        material_records_start = (
            mesh_records_start - header.material_count * 128
        )
        face_base = header.bone_data_offset - header.face_size
        bone_map_start = (
            header.bone_data_offset + header.bone_name_size
        )
    else:
        bone_map_start = (
            header.bone_data_offset + header.bone_name_size
        )
        bone_records_start = bone_map_start + header.bone_map_size
        material_records_start = (
            bone_records_start + header.bone_count * (80 + 64 + 64)
        )
        mesh_records_start = material_records_start + header.material_count * 112
        vertex_base = mesh_records_start + header.mesh_count * 56
        face_base = vertex_base + header.vertex_size
        fvf_base = face_base + header.face_size
    return (
        bone_map_start,
        material_records_start,
        mesh_records_start,
        vertex_base,
        face_base,
        fvf_base,
    )


def _parse_bones(
    data: bytes,
    header: ModelHeader,
    bone_map_start: int,
) -> list[Bone]:
    names = []
    cursor = header.bone_data_offset
    names_end = cursor + header.bone_name_size
    for _ in range(header.bone_count):
        end = data.index(0, cursor, names_end)
        names.append(_decode_string(data[cursor:end]))
        cursor = end + 1

    records_start = bone_map_start + header.bone_map_size
    first_matrices = records_start + header.bone_count * 80
    second_matrices = first_matrices + header.bone_count * 64
    float_dtype = np.dtype(f"{header.endian}f4")
    bones = []

    for index, name in enumerate(names):
        record_offset = records_start + index * 80
        parent_index = struct.unpack_from(
            f"{header.endian}h", data, record_offset + 60
        )[0]
        require(
            parent_index < index,
            f"Bone {index} has invalid parent {parent_index}",
        )
        position = np.frombuffer(
            data, dtype=float_dtype, count=3, offset=record_offset + 32
        ).astype(np.float32)
        inverse_bind = np.frombuffer(
            data,
            dtype=float_dtype,
            count=16,
            offset=second_matrices + index * 64,
        ).astype(np.float64).reshape(4, 4)
        inverse_bind[:3, 3] = 0.0
        inverse_bind[3, 3] = 1.0
        try:
            bind_row = np.linalg.inv(inverse_bind)
        except np.linalg.LinAlgError as error:
            raise FormatError(
                f"Bone {index} has a singular inverse bind matrix"
            ) from error
        bones.append(
            Bone(
                name=name,
                parent_index=parent_index,
                position=position,
                bind_matrix=bind_row.T.astype(np.float32),
            )
        )
    return bones


def _parse_material_ranges(
    data: bytes, header: ModelHeader, start: int
) -> list[MaterialRange]:
    record_size = 128 if header.layout == "ps3" else 112
    ranges = []
    for index in range(header.material_count):
        offset = start + index * record_size
        require(
            offset + record_size <= len(data),
            f"Material range {index} is truncated",
        )
        values = struct.unpack_from(f"{header.endian}IhhIIII", data, offset)
        ranges.append(
            MaterialRange(
                material_id=values[0],
                lod=values[1],
                unknown=values[2],
                vertex_count=values[3],
                index_start=values[4],
                index_count=values[5],
                base_vertex=values[6],
                name=f"part_{index:03d}",
            )
        )
    return ranges


def _parse_mesh_infos(
    data: bytes, header: ModelHeader, start: int
) -> list[MeshInfo]:
    meshes = []
    for index in range(header.mesh_count):
        record_size = 60 if header.layout == "ps3" else 56
        offset = start + index * record_size
        require(
            offset + record_size <= len(data),
            f"Mesh info {index} is truncated",
        )
        meshes.append(
            MeshInfo(
                *struct.unpack_from(
                    f"{header.endian}14I", data, offset
                )
            )
        )
    return meshes


def _parse_vita_layouts(
    data: bytes, header: ModelHeader, fvf_base: int
) -> list[list[VertexElement]]:
    layouts = []
    cursor = fvf_base
    limit = fvf_base + header.fvf_size
    for mesh_index in range(header.mesh_count):
        layout = []
        for _ in range(128):
            require(
                cursor + 8 <= limit,
                f"FVF layout {mesh_index} is truncated",
            )
            values = struct.unpack_from("<hhBBh", data, cursor)
            cursor += 8
            element = VertexElement(*values)
            layout.append(element)
            if element.end == -1:
                break
        else:
            raise FormatError(f"FVF layout {mesh_index} has no terminator")
        layouts.append(layout)
    return layouts


def _ps3_type_size(data_type: int, component_count: int) -> int:
    base_type = data_type & 0x0F
    if base_type == 2:
        return component_count * 4
    if base_type in (1, 3, 5):
        return component_count * 2
    if base_type in (4, 7):
        return component_count
    if base_type == 6:
        return 4
    raise FormatError(f"Unsupported PS3 vertex data type {data_type}")


def _plausible_ps3_descriptor(data: bytes, offset: int, limit: int) -> bool:
    if offset + 8 > limit:
        return False
    data_type, count, semantic, end_offset = struct.unpack_from(
        ">BBHI", data, offset
    )
    return (
        1 <= (data_type & 0x0F) <= 7
        and 1 <= count <= 4
        and semantic <= 32
        and (
            end_offset & 0xFFFF0000 == 0xFFFF0000
            or end_offset <= 256
        )
    )


def _parse_ps3_layouts(
    data: bytes, header: ModelHeader, fvf_base: int
) -> list[list[VertexElement]]:
    layouts = []
    cursor = fvf_base
    limit = fvf_base + header.fvf_size
    for mesh_index in range(header.mesh_count):
        descriptor = None
        search_limit = max(cursor, limit - 7)
        for candidate in range(cursor, search_limit, 4):
            if _plausible_ps3_descriptor(data, candidate, limit):
                descriptor = candidate
                break
        require(
            descriptor is not None,
            f"PS3 FVF layout {mesh_index} was not found",
        )

        layout = []
        element_offset = 0
        cursor = descriptor
        for _ in range(128):
            require(
                cursor + 8 <= limit,
                f"PS3 FVF layout {mesh_index} is truncated",
            )
            data_type, count, semantic, end_offset = struct.unpack_from(
                ">BBHI", data, cursor
            )
            require(
                _plausible_ps3_descriptor(data, cursor, limit),
                f"PS3 FVF layout {mesh_index} has an invalid element",
            )
            terminated = end_offset & 0xFFFF0000 == 0xFFFF0000
            layout.append(
                VertexElement(
                    end=-1 if terminated else 0,
                    offset=element_offset,
                    component_count=count,
                    data_type=data_type,
                    semantic=semantic,
                )
            )
            cursor += 8
            if terminated:
                break
            expected_end = element_offset + _ps3_type_size(data_type, count)
            require(
                end_offset == expected_end,
                (
                    f"PS3 FVF element {semantic} ends at 0x{end_offset:X}, "
                    f"expected 0x{expected_end:X}"
                ),
            )
            element_offset = end_offset
        else:
            raise FormatError(f"PS3 FVF layout {mesh_index} has no terminator")
        layouts.append(layout)
    return layouts


def _layout_stride(layout: list[VertexElement], ps3: bool) -> int:
    elements = [element for element in layout if element.end != -1 or ps3]
    require(bool(elements), "Vertex layout is empty")
    last = elements[-1]
    if ps3:
        return last.offset + _ps3_type_size(
            last.data_type, last.component_count
        )
    sizes = {2: 4, 3: 2, 4: 1, 7: 1, 20: 1}
    size = sizes.get(last.data_type)
    require(size is not None, f"Unsupported Vita vertex type {last.data_type}")
    return last.offset + last.component_count * size


def _mesh_ranges(
    mesh: MeshInfo, material_ranges: list[MaterialRange], mesh_index: int
) -> list[MaterialRange]:
    end = mesh.material_range_start + mesh.material_range_count
    require(
        end <= len(material_ranges),
        f"Mesh {mesh_index} references missing material ranges",
    )
    return material_ranges[mesh.material_range_start:end]


def _range_groups(
    ranges: list[MaterialRange],
) -> list[list[MaterialRange]]:
    groups: list[list[MaterialRange]] = []
    for material_range in ranges:
        if (
            not groups
            or (
                material_range.index_start == 0
                and material_range.base_vertex == 0
            )
        ):
            groups.append([])
        groups[-1].append(material_range)
    return groups


def _bone_palette(
    data: bytes,
    header: ModelHeader,
    bone_map_start: int,
    mesh: MeshInfo,
) -> np.ndarray:
    if mesh.bone_palette_count == 0:
        return np.empty(0, dtype=np.int32)
    offset = bone_map_start + mesh.bone_palette_start * 2
    require(
        offset + mesh.bone_palette_count * 2 <= bone_map_start + header.bone_map_size,
        "Mesh bone palette exceeds the bone map",
    )
    return np.frombuffer(
        data,
        dtype=np.dtype(f"{header.endian}u2"),
        count=mesh.bone_palette_count,
        offset=offset,
    ).astype(np.int32)


def _attribute_components(semantic: int) -> int:
    return {0: 3, 1: 4, 2: 3, 3: 4, 7: 4, 8: 2, 14: 3, 15: 3}.get(
        semantic, 0
    )


def _strided_values(
    data: bytes,
    offset: int,
    count: int,
    stride: int,
    dtype: np.dtype,
    components: int,
) -> np.ndarray:
    if count == 0:
        return np.empty((0, components), dtype=dtype)
    require(
        offset >= 0
        and offset + (count - 1) * stride + components * dtype.itemsize
        <= len(data),
        "Vertex attribute data is truncated",
    )
    return np.ndarray(
        shape=(count, components),
        dtype=dtype,
        buffer=data,
        offset=offset,
        strides=(stride, dtype.itemsize),
    ).copy()


def _decode_cmp(
    data: bytes, offset: int, count: int, stride: int
) -> np.ndarray:
    words = _strided_values(
        data, offset, count, stride, np.dtype(">u4"), 1
    ).reshape(-1).astype(np.uint32)
    x = (words & 0x7FF).astype(np.int32)
    y = ((words >> 11) & 0x7FF).astype(np.int32)
    z = ((words >> 22) & 0x3FF).astype(np.int32)
    x[x >= 0x400] -= 0x800
    y[y >= 0x400] -= 0x800
    z[z >= 0x200] -= 0x400
    return np.stack(
        (
            x.astype(np.float32) / 1023.0,
            y.astype(np.float32) / 1023.0,
            z.astype(np.float32) / 511.0,
        ),
        axis=1,
    )


def _decode_ps3_vertices(
    data: bytes,
    base_offset: int,
    count: int,
    stride: int,
    layout: list[VertexElement],
    palette: np.ndarray,
) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for element in layout:
        components = _attribute_components(element.semantic)
        if components == 0:
            continue
        base_type = element.data_type & 0x0F
        available = element.component_count
        wanted = min(components, available if base_type != 6 else 3)
        element_offset = base_offset + element.offset

        if base_type == 6:
            values = _decode_cmp(data, element_offset, count, stride)[:, :wanted]
        elif base_type == 2:
            values = _strided_values(
                data,
                element_offset,
                count,
                stride,
                np.dtype(">f4"),
                available,
            )[:, :wanted].astype(np.float32)
        elif base_type == 3:
            values = _strided_values(
                data,
                element_offset,
                count,
                stride,
                np.dtype(">f2"),
                available,
            )[:, :wanted].astype(np.float32)
        elif base_type in (1, 5):
            values = _strided_values(
                data,
                element_offset,
                count,
                stride,
                np.dtype(">i2"),
                available,
            )[:, :wanted].astype(np.float32)
            if base_type == 1:
                values /= 32767.0
        elif base_type in (4, 7):
            values = _strided_values(
                data,
                element_offset,
                count,
                stride,
                np.dtype("u1"),
                available,
            )[:, :wanted]
            if base_type == 4:
                values = values.astype(np.float32) / 255.0
        else:
            raise FormatError(f"Unsupported PS3 vertex type {element.data_type}")

        result[element.semantic] = _finish_attribute(
            element.semantic, values, palette
        )
    return result


def _decode_vita_vertices(
    data: bytes,
    base_offset: int,
    count: int,
    stride: int,
    layout: list[VertexElement],
    palette: np.ndarray,
) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for element in layout:
        if element.end == -1:
            continue
        components = _attribute_components(element.semantic)
        if components == 0:
            continue
        available = element.component_count
        wanted = min(components, available)
        if element.data_type == 2:
            dtype = np.dtype("<f4")
        elif element.data_type == 3:
            dtype = np.dtype("<f2")
        elif element.data_type in (4, 7, 20):
            dtype = np.dtype("u1")
        else:
            raise FormatError(
                f"Unsupported Vita vertex type {element.data_type}"
            )
        require(
            element.offset + available * dtype.itemsize <= stride,
            f"Vertex element {element.semantic} exceeds its stride",
        )
        values = _strided_values(
            data,
            base_offset + element.offset,
            count,
            stride,
            dtype,
            available,
        )[:, :wanted]
        if element.data_type in (4, 20):
            values = values.astype(np.float32) / 255.0
        result[element.semantic] = _finish_attribute(
            element.semantic, values, palette
        )
    return result


def _finish_attribute(
    semantic: int, values: np.ndarray, palette: np.ndarray
) -> np.ndarray:
    if semantic == 7:
        local = values.astype(np.int32)
        if palette.size:
            require(
                np.all(local < palette.size),
                "Vertex bone index exceeds its mesh palette",
            )
            return palette[local]
        return local
    values = values.astype(np.float32)
    if semantic == 1:
        if values.size and np.max(values) > 1.0:
            values /= 255.0
        sums = values.sum(axis=1, keepdims=True)
        np.divide(values, sums, out=values, where=sums > 0)
    elif semantic == 3 and values.size and np.max(values) > 1.0:
        values /= 255.0
    return values


def _make_submesh(
    mesh_index: int,
    part_index: int,
    mesh: MeshInfo,
    material_range: MaterialRange,
    positions: np.ndarray,
    local_indices: np.ndarray,
    attributes: dict[int, np.ndarray],
) -> SubMesh:
    return SubMesh(
        name=f"mesh_{mesh_index:03d}_{part_index:02d}",
        material_index=mesh.material_index,
        lod=material_range.lod,
        positions=positions,
        faces=local_indices.reshape(-1, 3),
        normals=attributes.get(2),
        tangents=attributes.get(14, attributes.get(15)),
        uvs=attributes.get(8),
        colors=attributes.get(3),
        bone_indices=attributes.get(7),
        bone_weights=attributes.get(1),
    )


def _parse_vita_geometry(
    data: bytes,
    header: ModelHeader,
    bone_map_start: int,
    material_ranges: list[MaterialRange],
    mesh_infos: list[MeshInfo],
    vertex_base: int,
    face_base: int,
    layouts: list[list[VertexElement]],
) -> list[SubMesh]:
    submeshes = []
    for mesh_index, (mesh, layout) in enumerate(zip(mesh_infos, layouts)):
        require(mesh.vertex_stride > 0, f"Mesh {mesh_index} has zero stride")
        palette = _bone_palette(data, header, bone_map_start, mesh)
        index_cursor = face_base + mesh.index_buffer_start
        vertex_buffer = vertex_base + mesh.vertex_buffer_start
        ranges = _mesh_ranges(mesh, material_ranges, mesh_index)
        index_dtype = np.dtype("<u4" if mesh.index_type == 2 else "<u2")

        for part_index, material_range in enumerate(ranges):
            require(
                material_range.index_count % 3 == 0,
                f"Mesh {mesh_index} index count is not divisible by three",
            )
            index_bytes = material_range.index_count * index_dtype.itemsize
            require(
                index_cursor + index_bytes <= len(data),
                f"Mesh {mesh_index} index data is truncated",
            )
            raw_indices = np.frombuffer(
                data,
                dtype=index_dtype,
                count=material_range.index_count,
                offset=index_cursor,
            ).astype(np.int64)
            index_cursor += index_bytes
            base_candidates = (
                material_range.base_vertex,
                material_range.index_start,
                0,
            )
            base_vertex = next(
                (
                    base
                    for base in base_candidates
                    if raw_indices.size == 0
                    or (
                        np.all(raw_indices >= base)
                        and np.all(
                            raw_indices - base < material_range.vertex_count
                        )
                    )
                ),
                None,
            )
            require(
                base_vertex is not None,
                f"Mesh {mesh_index} indices do not fit their vertex range",
            )
            vertex_offset = vertex_buffer + base_vertex * mesh.vertex_stride
            attributes = _decode_vita_vertices(
                data,
                vertex_offset,
                material_range.vertex_count,
                mesh.vertex_stride,
                layout,
                palette,
            )
            positions = attributes.get(0)
            require(positions is not None, f"Mesh {mesh_index} has no positions")
            submeshes.append(
                _make_submesh(
                    mesh_index,
                    part_index,
                    mesh,
                    material_range,
                    positions,
                    (raw_indices - base_vertex).astype(np.int32),
                    attributes,
                )
            )
    return submeshes


def _align(value: int, alignment: int) -> int:
    return (value + alignment - 1) & -alignment


def _find_ps3_index_buffer(
    data: bytes,
    search_start: int,
    face_limit: int,
    vertex_count: int,
    index_count: int,
) -> tuple[int, np.ndarray]:
    require(
        vertex_count <= 0xFFFF and index_count <= 0xFFFF,
        "PS3 mesh exceeds the supported 16-bit draw header",
    )
    signature = struct.pack(">HHI", vertex_count, index_count, 0xFFFFFFFF)
    position = search_start
    while True:
        match = data.find(signature, position, face_limit)
        require(match >= 8, "PS3 mesh index header was not found")
        header = match - 8
        index_base = match + 8
        index_end = index_base + index_count * 2
        if header >= search_start and index_end <= face_limit:
            indices = np.frombuffer(
                data, dtype=np.dtype(">u2"), count=index_count, offset=index_base
            ).astype(np.int64)
            if indices.size == 0 or np.all(indices < vertex_count):
                return index_end, indices
        position = match + 1


def _ps3_buffer_allocation(
    data: bytes, buffer_index: int, used_size: int
) -> int:
    allocation_offset = 0x70 + buffer_index * 0x7C
    require(
        allocation_offset + 4 <= len(data),
        f"PS3 draw descriptor {buffer_index} is truncated",
    )
    allocation = struct.unpack_from(">I", data, allocation_offset)[0]
    require(
        used_size <= allocation <= _align(used_size, 16),
        (
            f"PS3 draw descriptor {buffer_index} allocates "
            f"0x{allocation:X} bytes for 0x{used_size:X} bytes of vertices"
        ),
    )
    return allocation


def _parse_ps3_geometry(
    data: bytes,
    header: ModelHeader,
    bone_map_start: int,
    material_ranges: list[MaterialRange],
    mesh_infos: list[MeshInfo],
    vertex_base: int,
    face_base: int,
    layouts: list[list[VertexElement]],
) -> list[SubMesh]:
    submeshes = []
    vertex_cursor = vertex_base
    face_cursor = face_base
    face_limit = face_base + header.face_size

    grouped_meshes = [
        (
            mesh_index,
            mesh,
            layout,
            _range_groups(
                _mesh_ranges(mesh, material_ranges, mesh_index)
            ),
        )
        for mesh_index, (mesh, layout) in enumerate(zip(mesh_infos, layouts))
    ]
    total_groups = sum(len(item[3]) for item in grouped_meshes)
    require(
        total_groups == header.buffer_count,
        (
            f"MDL declares {header.buffer_count} PS3 buffers but material "
            f"ranges describe {total_groups}"
        ),
    )
    group_number = 0

    for mesh_index, mesh, layout, groups in grouped_meshes:
        stride = _layout_stride(layout, ps3=True)
        require(stride > 0, f"Mesh {mesh_index} has zero stride")
        palette = _bone_palette(data, header, bone_map_start, mesh)
        part_index = 0

        for ranges in groups:
            vertex_count = max(
                (item.base_vertex + item.vertex_count for item in ranges),
                default=0,
            )
            index_count = max(
                (item.index_start + item.index_count for item in ranges),
                default=0,
            )
            face_cursor, mesh_indices = _find_ps3_index_buffer(
                data,
                face_cursor,
                face_limit,
                vertex_count,
                index_count,
            )

            for material_range in ranges:
                require(
                    material_range.index_count % 3 == 0,
                    (
                        f"Mesh {mesh_index} index count is not "
                        "divisible by three"
                    ),
                )
                index_end = (
                    material_range.index_start
                    + material_range.index_count
                )
                require(
                    index_end <= len(mesh_indices),
                    f"Mesh {mesh_index} material index range is invalid",
                )
                raw_indices = mesh_indices[
                    material_range.index_start:index_end
                ]
                base_vertex = material_range.base_vertex
                require(
                    raw_indices.size == 0
                    or (
                        np.all(raw_indices >= base_vertex)
                        and np.all(
                            raw_indices - base_vertex
                            < material_range.vertex_count
                        )
                    ),
                    (
                        f"Mesh {mesh_index} indices do not fit "
                        "their vertex range"
                    ),
                )
                attributes = _decode_ps3_vertices(
                    data,
                    vertex_cursor + base_vertex * stride,
                    material_range.vertex_count,
                    stride,
                    layout,
                    palette,
                )
                positions = attributes.get(0)
                require(
                    positions is not None,
                    f"Mesh {mesh_index} has no positions",
                )
                submeshes.append(
                    _make_submesh(
                        mesh_index,
                        part_index,
                        mesh,
                        material_range,
                        positions,
                        (raw_indices - base_vertex).astype(np.int32),
                        attributes,
                    )
                )
                part_index += 1

            buffer_size = vertex_count * stride
            require(
                vertex_cursor + buffer_size
                <= vertex_base + header.vertex_size,
                "PS3 vertex buffers exceed their declared section",
            )
            allocation = _ps3_buffer_allocation(
                data, group_number, buffer_size
            )
            group_number += 1
            vertex_cursor += allocation

    require(
        vertex_cursor <= vertex_base + header.vertex_size,
        "PS3 vertex buffers exceed their declared section",
    )
    return submeshes


def parse_model(data: bytes, endian_hint: str | None = None) -> Model:
    header = _select_header(data, endian_hint)
    (
        bone_map_start,
        material_records_start,
        mesh_records_start,
        vertex_base,
        face_base,
        fvf_base,
    ) = _section_offsets(data, header)
    bones = _parse_bones(data, header, bone_map_start)
    material_ranges = _parse_material_ranges(
        data, header, material_records_start
    )
    mesh_infos = _parse_mesh_infos(data, header, mesh_records_start)

    if header.layout == "ps3":
        layouts = _parse_ps3_layouts(data, header, fvf_base)
        submeshes = _parse_ps3_geometry(
            data,
            header,
            bone_map_start,
            material_ranges,
            mesh_infos,
            vertex_base,
            face_base,
            layouts,
        )
    else:
        layouts = _parse_vita_layouts(data, header, fvf_base)
        submeshes = _parse_vita_geometry(
            data,
            header,
            bone_map_start,
            material_ranges,
            mesh_infos,
            vertex_base,
            face_base,
            layouts,
        )
    return Model(header=header, bones=bones, submeshes=submeshes)


def _compact_mdc_layout(
    data: bytes, endian: str
) -> tuple[tuple[int, int, int, int], int, int] | None:
    if len(data) < 48:
        return None
    header = struct.unpack_from(f"{endian}4I", data)
    table_skip, count, material_data_size, _string_data_size = header
    table_start = 48 + table_skip
    entry_size = 36 if endian == ">" else 40
    entries_end = table_start + count * entry_size
    string_base = entries_end + material_data_size
    if (
        count > 100_000
        or table_start > len(data)
        or entries_end > len(data)
        or not 0 <= string_base < len(data)
    ):
        return None
    try:
        for index in range(count):
            name_relative = struct.unpack_from(
                f"{endian}I", data, table_start + index * entry_size
            )[0]
            _read_c_string(data, 0x30 + name_relative)
    except (FormatError, struct.error):
        return None
    return header, table_start, entry_size


def _find_parameter_base(
    data: bytes,
    endian: str,
    entries_end: int,
    string_base: int,
    parameter_count: int,
) -> int:
    search_end = min(string_base, entries_end + 0x200)
    for candidate in range(entries_end, search_end, 4):
        table_end = candidate + parameter_count * 8
        if table_end > string_base:
            break
        valid = True
        for index in range(parameter_count):
            kind, value_offset, name_relative = struct.unpack_from(
                f"{endian}HHI", data, candidate + index * 8
            )
            if kind not in (0x0F, 0x20):
                valid = False
                break
            if candidate + value_offset >= string_base:
                valid = False
                break
            name_offset = string_base + name_relative
            if not 0 <= name_offset < len(data):
                valid = False
                break
            try:
                _read_c_string(data, name_offset)
            except FormatError:
                valid = False
                break
        if valid:
            return candidate
    raise FormatError("MDC parameter table was not found")


def parse_materials(
    data: bytes, endian_hint: str | None = None
) -> list[MaterialDefinition]:
    endians = [endian_hint] if endian_hint in ("<", ">") else []
    endians += [endian for endian in ("<", ">") if endian not in endians]
    chosen = None
    for endian in endians:
        layout = _compact_mdc_layout(data, endian)
        if layout is not None:
            chosen = endian, layout
            break
    require(chosen is not None, "No supported console MDC layout matched the file")
    endian, (header, table_start, entry_size) = chosen
    count = header[1]
    entries_end = table_start + count * entry_size
    string_base = entries_end + header[2]
    first_count = (
        struct.unpack_from(f"{endian}H", data, table_start + 26)[0]
        if count
        else 0
    )
    first_parameter_base = (
        _find_parameter_base(
            data, endian, entries_end, string_base, first_count
        )
        if count and first_count
        else entries_end
    )
    definitions = []

    for index in range(count):
        entry = table_start + index * entry_size
        name_relative = struct.unpack_from(f"{endian}I", data, entry)[0]
        name = _read_c_string(data, 0x30 + name_relative)
        parameter_count = struct.unpack_from(f"{endian}H", data, entry + 26)[0]
        parameter_relative = struct.unpack_from(
            f"{endian}I", data, entry + 28
        )[0]
        parameter_base = first_parameter_base + parameter_relative
        require(
            parameter_base + parameter_count * 8 <= string_base,
            f"MDC material {index} parameter table is truncated",
        )
        definition = MaterialDefinition(name=name)

        for parameter_index in range(parameter_count):
            kind, value_relative, slot_relative = struct.unpack_from(
                f"{endian}HHI",
                data,
                parameter_base + parameter_index * 8,
            )
            slot_name = _read_c_string(data, string_base + slot_relative)
            value_position = parameter_base + value_relative
            if kind == 0x20:
                require(
                    value_position + 12 <= string_base,
                    f"MDC texture parameter {slot_name} is truncated",
                )
                texture_relative = struct.unpack_from(
                    f"{endian}i", data, value_position
                )[0]
                if texture_relative != -1:
                    definition.textures[slot_name] = _read_c_string(
                        data, string_base + texture_relative
                    )
            elif kind == 0x0F:
                require(
                    value_position + 16 <= string_base,
                    f"MDC vector parameter {slot_name} is truncated",
                )
                definition.parameters[slot_name] = struct.unpack_from(
                    f"{endian}4f", data, value_position
                )
            else:
                raise FormatError(
                    f"Unsupported MDC parameter type 0x{kind:02X}"
                )
        definitions.append(definition)
    return definitions
