# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np

from .archive import FormatError, require


@dataclass(slots=True)
class Texture:
    width: int
    height: int
    format: str
    pixels: np.ndarray | None
    encoded: bytes
    flags: tuple[int, int, int, int]


def parse_texture(data: bytes, endian_hint: str | None = None) -> Texture:
    require(len(data) >= 12, "TEX payload is truncated")
    candidates = []
    endians = [endian_hint] if endian_hint in ("<", ">") else []
    endians += [endian for endian in ("<", ">") if endian not in endians]

    for endian in endians:
        for offset in (4, 0):
            if offset + 12 > len(data):
                continue
            width, height = struct.unpack_from(f"{endian}HH", data, offset)
            if 0 < width <= 16384 and 0 < height <= 16384:
                flags = tuple(data[offset + 4 : offset + 8])
                score = int(endian == endian_hint) + int(offset == 4 and data[:4] == b"\x00" * 4)
                candidates.append((score, endian, offset, width, height, flags))
    require(bool(candidates), "No supported TEX header layout matched the file")
    _score, endian, offset, width, height, flags = max(candidates)
    payload = data[offset + 12 :]
    flags1, flags2, flags3, flags4 = flags
    decoded_width = width
    decoded_height = height

    if endian == ">" and offset == 0 and flags1 == 0x21:
        storage_width = (
            _next_power_of_two(width) if flags2 == 0x10 else width
        )
        if flags4 < 0x10:
            texture_format = "BC1"
            pixels = _decode_bc1(
                payload, width, height, storage_width=storage_width
            )
        else:
            texture_format = "BC3"
            pixels = _decode_bc3(
                payload, width, height, storage_width=storage_width
            )
    elif endian == ">" and offset == 0 and flags1 == 0x46:
        texture_format = "BC1_CUBE"
        pixels = _decode_bc1_cube(
            payload, width, height, max(flags4, 1)
        )
        decoded_width = width * 3
        decoded_height = height * 2
    elif flags4 == 0x11 and flags3 in (0x00, 0x10):
        texture_format = "BGRA8"
        pixels = _decode_bgra8(payload, width, height)
    elif flags4 == 0x11 and flags3 == 0x30:
        texture_format = "A8"
        pixels = _decode_a8(payload, width, height)
    elif flags4 != 0x11 and flags3 == 0x10:
        texture_format = "BC2"
        pixels = _decode_bc2(payload, width, height)
    elif flags2 != 0x11 and flags3 in (0x20, 0x70):
        texture_format = "BC3"
        pixels = _decode_bc3(payload, width, height)
    elif flags2 != 0x11 and flags3 == 0x40:
        texture_format = "BC7"
        pixels = None
    else:
        texture_format = "BC1"
        pixels = _decode_bc1(payload, width, height)

    return Texture(
        width=decoded_width,
        height=decoded_height,
        format=texture_format,
        pixels=pixels,
        encoded=payload,
        flags=flags,
    )


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, (value - 1).bit_length())


def _storage_dimensions(payload_size: int, width: int, height: int, bpp: int) -> tuple[int, int]:
    expected = width * height * bpp
    if payload_size == expected:
        return width, height
    storage_width = _next_power_of_two(width)
    storage_height = _next_power_of_two(height)
    if payload_size >= storage_width * storage_height * bpp:
        return storage_width, storage_height
    require(payload_size >= expected, "TEX top mip is truncated")
    return width, height


def _decode_bgra8(payload: bytes, width: int, height: int) -> np.ndarray:
    storage_width, storage_height = _storage_dimensions(
        len(payload), width, height, 4
    )
    count = storage_width * storage_height * 4
    values = np.frombuffer(payload, dtype=np.uint8, count=count).reshape(
        storage_height, storage_width, 4
    )
    rgba = values[:height, :width, [2, 1, 0, 3]].astype(np.float32) / 255.0
    return rgba


def _decode_a8(payload: bytes, width: int, height: int) -> np.ndarray:
    storage_width, storage_height = _storage_dimensions(
        len(payload), width, height, 1
    )
    alpha = (
        np.frombuffer(
            payload, dtype=np.uint8, count=storage_width * storage_height
        )
        .reshape(storage_height, storage_width)[:height, :width]
        .astype(np.float32)
        / 255.0
    )
    rgba = np.ones((height, width, 4), dtype=np.float32)
    rgba[:, :, 3] = alpha
    return rgba


def _rgb565(values: np.ndarray) -> np.ndarray:
    red = ((values >> 11) & 0x1F).astype(np.float32) * (255.0 / 31.0)
    green = ((values >> 5) & 0x3F).astype(np.float32) * (255.0 / 63.0)
    blue = (values & 0x1F).astype(np.float32) * (255.0 / 31.0)
    return np.stack((red, green, blue), axis=-1)


def _decode_color_blocks(blocks: np.ndarray, force_four_color: bool) -> np.ndarray:
    color0 = blocks[:, 0].astype(np.uint16) | blocks[:, 1].astype(np.uint16) << 8
    color1 = blocks[:, 2].astype(np.uint16) | blocks[:, 3].astype(np.uint16) << 8
    rgb0 = _rgb565(color0)
    rgb1 = _rgb565(color1)
    palette = np.empty((len(blocks), 4, 4), dtype=np.float32)
    palette[:, 0, :3] = rgb0
    palette[:, 1, :3] = rgb1
    palette[:, :2, 3] = 255.0

    four_color = np.ones(len(blocks), dtype=bool) if force_four_color else color0 > color1
    palette[four_color, 2, :3] = (
        2.0 * rgb0[four_color] + rgb1[four_color]
    ) / 3.0
    palette[four_color, 3, :3] = (
        rgb0[four_color] + 2.0 * rgb1[four_color]
    ) / 3.0
    palette[four_color, 2:, 3] = 255.0

    three_color = ~four_color
    palette[three_color, 2, :3] = (
        rgb0[three_color] + rgb1[three_color]
    ) / 2.0
    palette[three_color, 2, 3] = 255.0
    palette[three_color, 3] = 0.0

    bits = (
        blocks[:, 4].astype(np.uint32)
        | blocks[:, 5].astype(np.uint32) << 8
        | blocks[:, 6].astype(np.uint32) << 16
        | blocks[:, 7].astype(np.uint32) << 24
    )
    indices = ((bits[:, None] >> (2 * np.arange(16, dtype=np.uint32))) & 3).astype(
        np.intp
    )
    return np.take_along_axis(palette, indices[:, :, None], axis=1)


def _blocks_to_image(
    pixels: np.ndarray,
    width: int,
    height: int,
    storage_width: int | None = None,
    storage_height: int | None = None,
) -> np.ndarray:
    stored_width = storage_width or width
    stored_height = storage_height or height
    blocks_x = math.ceil(stored_width / 4)
    blocks_y = math.ceil(stored_height / 4)
    require(
        len(pixels) == blocks_x * blocks_y,
        "Compressed TEX block count is inconsistent",
    )
    image = (
        pixels.reshape(blocks_y, blocks_x, 4, 4, 4)
        .transpose(0, 2, 1, 3, 4)
        .reshape(blocks_y * 4, blocks_x * 4, 4)
    )
    return (image[:height, :width] / 255.0).astype(np.float32)


def _decode_bc1(
    payload: bytes,
    width: int,
    height: int,
    storage_width: int | None = None,
    storage_height: int | None = None,
) -> np.ndarray:
    stored_width = storage_width or width
    stored_height = storage_height or height
    block_count = math.ceil(stored_width / 4) * math.ceil(stored_height / 4)
    required = block_count * 8
    require(len(payload) >= required, "BC1 top mip is truncated")
    blocks = np.frombuffer(payload, dtype=np.uint8, count=required).reshape(-1, 8)
    return _blocks_to_image(
        _decode_color_blocks(blocks, force_four_color=False),
        width,
        height,
        stored_width,
        stored_height,
    )


def _decode_bc1_cube(
    payload: bytes, width: int, height: int, mip_count: int
) -> np.ndarray:
    require(width == height, "PS3 cube texture faces must be square")
    maximum_mips = max(width, height).bit_length()
    require(
        1 <= mip_count <= maximum_mips,
        f"PS3 cube texture has invalid mip count {mip_count}",
    )
    chain_size = 0
    for level in range(mip_count):
        mip_width = max(1, width >> level)
        mip_height = max(1, height >> level)
        chain_size += (
            math.ceil(mip_width / 4)
            * math.ceil(mip_height / 4)
            * 8
        )
    face_stride = (chain_size + 0xFF) & ~0xFF
    top_size = math.ceil(width / 4) * math.ceil(height / 4) * 8
    require(
        len(payload) >= face_stride * 6,
        "PS3 BC1 cube texture is truncated",
    )
    atlas = np.zeros((height * 2, width * 3, 4), dtype=np.float32)
    for face in range(6):
        face_pixels = _decode_bc1(
            payload[
                face * face_stride : face * face_stride + top_size
            ],
            width,
            height,
        )
        y = (face // 3) * height
        x = (face % 3) * width
        atlas[y : y + height, x : x + width] = face_pixels
    return atlas


def _decode_bc2(
    payload: bytes,
    width: int,
    height: int,
    storage_width: int | None = None,
    storage_height: int | None = None,
) -> np.ndarray:
    stored_width = storage_width or width
    stored_height = storage_height or height
    block_count = math.ceil(stored_width / 4) * math.ceil(stored_height / 4)
    required = block_count * 16
    require(len(payload) >= required, "BC2 top mip is truncated")
    blocks = np.frombuffer(payload, dtype=np.uint8, count=required).reshape(-1, 16)
    colors = _decode_color_blocks(blocks[:, 8:], force_four_color=True)
    alpha_bits = np.zeros(len(blocks), dtype=np.uint64)
    for byte_index in range(8):
        alpha_bits |= blocks[:, byte_index].astype(np.uint64) << (8 * byte_index)
    alpha = (
        (alpha_bits[:, None] >> (4 * np.arange(16, dtype=np.uint64))) & 0xF
    ).astype(np.float32) * 17.0
    colors[:, :, 3] = alpha
    return _blocks_to_image(
        colors, width, height, stored_width, stored_height
    )


def _decode_bc3(
    payload: bytes,
    width: int,
    height: int,
    storage_width: int | None = None,
    storage_height: int | None = None,
) -> np.ndarray:
    stored_width = storage_width or width
    stored_height = storage_height or height
    block_count = math.ceil(stored_width / 4) * math.ceil(stored_height / 4)
    required = block_count * 16
    require(len(payload) >= required, "BC3 top mip is truncated")
    blocks = np.frombuffer(payload, dtype=np.uint8, count=required).reshape(-1, 16)
    colors = _decode_color_blocks(blocks[:, 8:], force_four_color=True)
    alpha0 = blocks[:, 0].astype(np.float32)
    alpha1 = blocks[:, 1].astype(np.float32)
    palette = np.empty((len(blocks), 8), dtype=np.float32)
    palette[:, 0] = alpha0
    palette[:, 1] = alpha1
    greater = alpha0 > alpha1
    for index in range(1, 7):
        palette[greater, index + 1] = (
            (7 - index) * alpha0[greater] + index * alpha1[greater]
        ) / 7.0
    for index in range(1, 5):
        palette[~greater, index + 1] = (
            (5 - index) * alpha0[~greater] + index * alpha1[~greater]
        ) / 5.0
    palette[~greater, 6] = 0.0
    palette[~greater, 7] = 255.0

    alpha_bits = np.zeros(len(blocks), dtype=np.uint64)
    for byte_index in range(6):
        alpha_bits |= blocks[:, byte_index + 2].astype(np.uint64) << (8 * byte_index)
    alpha_indices = (
        (alpha_bits[:, None] >> (3 * np.arange(16, dtype=np.uint64))) & 7
    ).astype(np.intp)
    colors[:, :, 3] = np.take_along_axis(palette, alpha_indices, axis=1)
    return _blocks_to_image(
        colors, width, height, stored_width, stored_height
    )


def make_dds(texture: Texture) -> bytes:
    require(texture.format == "BC7", "Native DDS fallback is only needed for BC7")
    block_size = math.ceil(texture.width / 4) * math.ceil(texture.height / 4) * 16
    require(len(texture.encoded) >= block_size, "BC7 top mip is truncated")
    header = bytearray(148)
    struct.pack_into("<4sI", header, 0, b"DDS ", 124)
    struct.pack_into(
        "<IIIIIII",
        header,
        8,
        0x000A1007,
        texture.height,
        texture.width,
        block_size,
        0,
        1,
        1,
    )
    struct.pack_into("<II4sIIIII", header, 76, 32, 4, b"DX10", 0, 0, 0, 0, 0)
    struct.pack_into("<IIIII", header, 128, 98, 3, 0, 1, 0)
    return bytes(header) + texture.encoded[:block_size]
