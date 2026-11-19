# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import numpy as np

from gundam_breaker_importer.animation import parse_motion_package
from gundam_breaker_importer.archive import (
    BILZ_MAGIC,
    GDatArchive,
    IsoImage,
    PAC_TYPE_HASH,
    PS3_PAC_MAGIC,
    FormatError,
    orochi_hash,
    unwrap_bilz,
)
from gundam_breaker_importer.model import parse_materials, parse_model
from gundam_breaker_importer.texture import parse_texture


def _directory_record(
    name: bytes, lba: int, size: int, flags: int
) -> bytes:
    length = 33 + len(name) + (len(name) % 2 == 0)
    record = bytearray(length)
    record[0] = length
    struct.pack_into("<I", record, 2, lba)
    struct.pack_into(">I", record, 6, lba)
    struct.pack_into("<I", record, 10, size)
    struct.pack_into(">I", record, 14, size)
    record[25] = flags
    struct.pack_into("<H", record, 28, 1)
    struct.pack_into(">H", record, 30, 1)
    record[32] = len(name)
    record[33 : 33 + len(name)] = name
    return bytes(record)


def _minimal_iso() -> bytes:
    sector_size = 2048
    image = bytearray(24 * sector_size)
    volume = memoryview(image)[16 * sector_size : 17 * sector_size]
    volume[0] = 1
    volume[1:6] = b"CD001"
    volume[6] = 1
    struct.pack_into("<H", volume, 128, sector_size)
    struct.pack_into(">H", volume, 130, sector_size)
    root = _directory_record(b"\x00", 20, sector_size, 0x02)
    volume[156 : 156 + len(root)] = root

    directory = memoryview(image)[20 * sector_size : 21 * sector_size]
    first = _directory_record(b"FILE.DAT;1", 21, 4, 0x80)
    second = _directory_record(b"FILE.DAT;1", 22, 4, 0)
    directory[: len(first)] = first
    directory[len(first) : len(first) + len(second)] = second
    image[21 * sector_size : 21 * sector_size + 4] = b"ABCD"
    image[22 * sector_size : 22 * sector_size + 4] = b"EFGH"
    return bytes(image)


def _bc1_block(rgb565: int) -> bytes:
    return struct.pack("<HHI", rgb565, 0, 0)


def _minimal_vita_model() -> bytes:
    material_size = 112
    mesh_size = 56
    vertices = struct.pack(
        "<9f",
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    )
    indices = struct.pack("<3H", 0, 1, 2)
    layout = struct.pack("<hhBBh", 0, 0, 3, 2, 0)
    layout += struct.pack("<hhBBh", -1, 0, 0, 0, 0)
    header = b"\x00" * 4 + struct.pack(
        "<12I",
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        len(vertices),
        len(indices),
        len(layout),
        0,
    )
    material = bytearray(material_size)
    struct.pack_into("<IhhIIII", material, 0, 0, 0, 0, 3, 0, 3, 0)
    mesh = struct.pack(
        "<14I",
        0,
        1,
        0,
        len(vertices),
        12,
        0,
        len(indices),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    return header + material + mesh + vertices + indices + layout


def _minimal_motion_package(
    *,
    frame_count: int = 3,
    curve_kind: int = 6,
    interior_frames: tuple[int, ...] = (),
    curve_values: tuple[float, ...] = (0.0, 1.0),
) -> bytes:
    base = bytearray((2, 0, 0, 0))
    base.append(0)
    while len(base) % 4:
        base.append(0)
    base.extend(
        struct.pack(
            ">14f",
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    )

    kinds = (0, 0, 1, 1, curve_kind, 0, 0, 1, 1)
    clip = bytearray(struct.pack(">HH", len(kinds), frame_count))
    for group_start in range(0, len(kinds), 8):
        packed = sum(
            kind << (3 * index)
            for index, kind in enumerate(
                kinds[group_start : group_start + 8]
            )
        )
        clip.extend(packed.to_bytes(3, "big"))
    frame_width = 1 if frame_count <= 255 else 2
    if frame_width == 2 and len(clip) % 2:
        clip.append(0)
    if curve_kind >= 6:
        clip.extend(len(interior_frames).to_bytes(frame_width, "big"))
        for frame in interior_frames:
            clip.extend(frame.to_bytes(frame_width, "big"))
    while len(clip) % 4:
        clip.append(0)
    clip.extend(struct.pack(f">{len(curve_values)}f", *curve_values))

    table_end = 16
    clip_offset = table_end + len(base)
    end_offset = clip_offset + len(clip)
    return (
        struct.pack(">4I", 2, table_end, clip_offset, end_offset)
        + base
        + clip
    )


def _absolute_reference_motion_package() -> bytes:
    base = bytearray((3, 0, 0, 0, 1, 4))
    while len(base) % 4:
        base.append(0)
    base.extend((5, 6))
    while len(base) % 4:
        base.append(0)

    half_sqrt = 2.0**-0.5
    transforms = (
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        (1.0, 0.0, 0.0, 0.0, 0.0, half_sqrt, half_sqrt),
        (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    )
    for transform in transforms:
        base.extend(struct.pack(">7f", *transform))

    kinds = (0, 0, 1, 1, 1, 1, 1, 1, 1, 1)
    clip = bytearray(struct.pack(">HH", len(kinds), 1))
    for group_start in range(0, len(kinds), 8):
        packed = sum(
            kind << (3 * index)
            for index, kind in enumerate(
                kinds[group_start : group_start + 8]
            )
        )
        clip.extend(packed.to_bytes(3, "big"))
    while len(clip) % 4:
        clip.append(0)

    table_end = 16
    clip_offset = table_end + len(base)
    end_offset = clip_offset + len(clip)
    return (
        struct.pack(">4I", 2, table_end, clip_offset, end_offset)
        + base
        + clip
    )


class ArchiveTests(unittest.TestCase):
    def test_bilz_stored_and_zlib(self) -> None:
        payload = b"resource data"
        stored = (
            BILZ_MAGIC
            + struct.pack(">III", len(payload), len(payload), 0)
            + payload
        )
        compressed_payload = zlib.compress(payload)
        compressed = (
            BILZ_MAGIC
            + struct.pack(
                ">III", len(payload), len(compressed_payload), 7
            )
            + compressed_payload
        )
        self.assertEqual(unwrap_bilz(stored), payload)
        self.assertEqual(unwrap_bilz(compressed), payload)

    def test_segmented_iso_member(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "segmented.iso")
            path.write_bytes(_minimal_iso())
            image = IsoImage(path)
            try:
                member = image.find("file.dat")
                self.assertIsNotNone(member)
                self.assertEqual(member.read(0, 8), b"ABCDEFGH")
                self.assertEqual(member.read(2, 4), b"CDEF")
            finally:
                image.close()

    def test_ps3_gdat_and_pac(self) -> None:
        nested_name = "folder/model"
        pac = bytearray(0x80)
        pac[:4] = PS3_PAC_MAGIC
        pac.extend(struct.pack(">II", 1, 0))
        entry = bytearray(0x108)
        entry[: len(nested_name)] = nested_name.encode()
        struct.pack_into(">II", entry, 0x100, 0x12345678, 4)
        pac.extend(entry)
        while len(pac) % 8:
            pac.append(0)
        pac.extend(b"DATA")

        table_offset = 24
        key = (PAC_TYPE_HASH << 32) | 0x89ABCDEF
        gdat = (
            b"GDAT"
            + struct.pack(">I", 1)
            + struct.pack(">QII", key, table_offset, len(pac))
            + pac
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "archive.dat")
            path.write_bytes(gdat)
            archive = GDatArchive(path, endian=">")
            try:
                self.assertEqual(len(archive.records), 1)
                nested = archive.expand_pac(archive.records[0])
                self.assertEqual(len(nested), 1)
                self.assertEqual(nested[0].resource_hash, orochi_hash(nested_name))
                self.assertEqual(nested[0].read_raw(), b"DATA")
            finally:
                archive.close()


class ModelTests(unittest.TestCase):
    def test_minimal_vita_triangle(self) -> None:
        model = parse_model(_minimal_vita_model(), "<")
        self.assertEqual(model.header.layout, "vita")
        self.assertEqual(len(model.submeshes), 1)
        np.testing.assert_allclose(
            model.submeshes[0].positions,
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        )
        np.testing.assert_array_equal(
            model.submeshes[0].faces, ((0, 1, 2),)
        )

    def test_minimal_compact_material(self) -> None:
        data = bytearray(88 + len("material") + 1)
        struct.pack_into("<4I", data, 0, 0, 1, 0, len("material") + 1)
        struct.pack_into("<I", data, 48, 40)
        data[88:] = b"material\x00"
        materials = parse_materials(bytes(data), "<")
        self.assertEqual(len(materials), 1)
        self.assertEqual(materials[0].name, "material")

    def test_malformed_model_is_rejected(self) -> None:
        with self.assertRaises(FormatError):
            parse_model(b"\x00" * 64)


class AnimationTests(unittest.TestCase):
    def test_base_parent_and_node_type_records(self) -> None:
        package = parse_motion_package(
            _absolute_reference_motion_package()
        )
        self.assertEqual(package.base.parents, (-1, 0, 1))
        self.assertEqual(package.base.node_types, (0, 0, 4))
        self.assertEqual(package.base.joint_ids, (6, 5))

    def test_absolute_reference_quaternions_are_localized(self) -> None:
        package = parse_motion_package(
            _absolute_reference_motion_package()
        )
        expected = (0.0, 0.0, 2.0**-0.5, 2.0**-0.5)
        np.testing.assert_allclose(
            package.base.transforms[1].rotation,
            expected,
            atol=1.0e-6,
        )
        np.testing.assert_allclose(
            package.base.transforms[2].rotation,
            expected,
            atol=1.0e-6,
        )

    def test_sparse_smoothstep_motion_clip(self) -> None:
        package = parse_motion_package(_minimal_motion_package())
        self.assertEqual(package.base.node_count, 2)
        self.assertEqual(package.base.node_types, (0, 0))
        self.assertEqual(len(package.clips), 1)
        clip = package.clips[0]
        self.assertTrue(clip.global_tracks.translation.is_rest)
        self.assertTrue(clip.global_tracks.rotation.is_rest)
        self.assertFalse(clip.bone_tracks[0].translation.is_rest)
        self.assertAlmostEqual(
            clip.bone_tracks[0].translation.curves[0].evaluate(
                1.0, clip.frame_count
            ),
            0.5,
        )
        transform = clip.sample_node(package.base, 1, 2.0)
        np.testing.assert_allclose(transform.translation, (1.0, 0.0, 0.0))

    def test_hermite_tangents_use_source_fps(self) -> None:
        package = parse_motion_package(
            _minimal_motion_package(
                curve_kind=7,
                curve_values=(0.0, 30.0, 1.0, 0.0),
            )
        )
        curve = package.clips[0].bone_tracks[0].translation.curves[0]
        self.assertAlmostEqual(curve.evaluate(1.0, 3), 0.75)

    def test_two_byte_key_frames(self) -> None:
        package = parse_motion_package(
            _minimal_motion_package(
                frame_count=300,
                interior_frames=(100,),
                curve_values=(0.0, 0.25, 1.0),
            )
        )
        curve = package.clips[0].bone_tracks[0].translation.curves[0]
        self.assertEqual(curve.interior_frames, (100,))
        self.assertAlmostEqual(curve.evaluate(100.0, 300), 0.25)

    def test_truncated_motion_is_rejected(self) -> None:
        data = _minimal_motion_package()
        with self.assertRaises(FormatError):
            parse_motion_package(data[:-1])


class TextureTests(unittest.TestCase):
    def test_ps3_padded_bc1_rows(self) -> None:
        header = struct.pack(">HH", 5, 4) + bytes((0x21, 0x10, 0x4D, 0x01))
        header += b"\x00" * 4
        texture = parse_texture(
            header + _bc1_block(0xF800) + _bc1_block(0x07E0), ">"
        )
        self.assertEqual(texture.format, "BC1")
        self.assertEqual(texture.pixels.shape, (4, 5, 4))
        np.testing.assert_allclose(texture.pixels[0, 0, :3], (1.0, 0.0, 0.0))
        np.testing.assert_allclose(texture.pixels[0, 4, :3], (0.0, 1.0, 0.0))

    def test_ps3_bc1_cube_atlas(self) -> None:
        header = struct.pack(">HH", 4, 4) + bytes((0x46, 0x11, 0x4D, 0x01))
        header += b"\x00" * 4
        face = _bc1_block(0x001F)
        payload = b"".join(face + b"\x00" * (256 - len(face)) for _ in range(6))
        texture = parse_texture(header + payload, ">")
        self.assertEqual(texture.format, "BC1_CUBE")
        self.assertEqual((texture.width, texture.height), (12, 8))
        self.assertEqual(texture.pixels.shape, (8, 12, 4))
        np.testing.assert_allclose(texture.pixels[0, 0, :3], (0.0, 0.0, 1.0))

    def test_truncated_texture_is_rejected(self) -> None:
        header = struct.pack(">HH", 4, 4) + bytes((0x21, 0x11, 0x4D, 0x21))
        header += b"\x00" * 4
        with self.assertRaises(FormatError):
            parse_texture(header, ">")


if __name__ == "__main__":
    unittest.main()
