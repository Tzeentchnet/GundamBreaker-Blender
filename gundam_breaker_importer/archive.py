# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import mmap
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .profiles import GameProfile, detect_profile

GDAT_MAGIC = b"GDAT"
VITA_PAC_MAGIC = b"pac\x00"
PS3_PAC_MAGIC = b"\x00cap"
BILZ_MAGIC = b"BILZ"

TEXTURE_TYPE_HASH = 0xFC397BB3
MODEL_TYPE_HASH = 0xFA61C1D7
MATERIAL_TYPE_HASH = 0x5E12DE0D
PAC_TYPE_HASH = 0x2A79F65E
MOTION_TYPE_HASH = 0x3A4305EC

TYPE_EXTENSIONS = {
    TEXTURE_TYPE_HASH: "tex",
    0x2C1CF0E2: "dt2",
    MODEL_TYPE_HASH: "mdl",
    MATERIAL_TYPE_HASH: "mdc",
    MOTION_TYPE_HASH: "mpk",
    0x637A3623: "mse",
    0xDDE6C1B6: "msq2",
    0x598A8375: "mtsq",
    0x2DCB5FBB: "mss",
}


class FormatError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FormatError(message)


def orochi_hash(value: str) -> int:
    result = 37
    for byte in value.encode("utf-8"):
        result = (byte + 37 * result) & 0xFFFFFFFF
    return result


@dataclass(frozen=True, slots=True)
class IsoExtent:
    offset: int
    size: int


@dataclass(frozen=True, slots=True)
class IsoMember:
    image: "IsoImage"
    name: str
    extents: tuple[IsoExtent, ...]

    @property
    def size(self) -> int:
        return sum(extent.size for extent in self.extents)

    def read(self, offset: int, size: int) -> bytes:
        require(offset >= 0 and size >= 0, "Negative ISO member range")
        require(
            offset <= self.size and size <= self.size - offset,
            f"ISO range 0x{offset:X}+0x{size:X} exceeds {self.name}",
        )
        if size == 0:
            return b""

        output = bytearray(size)
        output_offset = 0
        logical_offset = 0
        remaining = size
        for extent in self.extents:
            extent_end = logical_offset + extent.size
            if offset >= extent_end:
                logical_offset = extent_end
                continue
            local_offset = max(offset - logical_offset, 0)
            count = min(extent.size - local_offset, remaining)
            output[output_offset : output_offset + count] = self.image.read(
                extent.offset + local_offset, count
            )
            output_offset += count
            remaining -= count
            offset += count
            logical_offset = extent_end
            if remaining == 0:
                break
        require(remaining == 0, f"ISO member {self.name} has incomplete extents")
        return bytes(output)


class IsoImage:
    def __init__(self, path: Path):
        self.path = path
        self._file = path.open("rb")
        self._mapping = mmap.mmap(
            self._file.fileno(), 0, access=mmap.ACCESS_READ
        )
        try:
            self.block_size, root_extents = self._parse_volume()
            self.members: dict[str, IsoMember] = {}
            self._walk_directory(root_extents, "")
        except Exception:
            self.close()
            raise

    @property
    def size(self) -> int:
        return len(self._mapping)

    def close(self) -> None:
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def read(self, offset: int, size: int) -> bytes:
        require(offset >= 0 and size >= 0, "Negative ISO range")
        require(
            offset <= len(self._mapping)
            and size <= len(self._mapping) - offset,
            f"ISO range 0x{offset:X}+0x{size:X} exceeds {self.path.name}",
        )
        return self._mapping[offset : offset + size]

    def find(self, name: str) -> IsoMember | None:
        normalized = name.replace("\\", "/").strip("/").casefold()
        return self.members.get(normalized)

    def _parse_volume(self) -> tuple[int, tuple[IsoExtent, ...]]:
        sector_size = 2048
        for sector in range(16, 256):
            offset = sector * sector_size
            require(
                offset + sector_size <= len(self._mapping),
                "ISO volume descriptor is truncated",
            )
            descriptor = self._mapping[offset : offset + sector_size]
            require(
                descriptor[1:6] == b"CD001",
                "Source is not an ISO9660 image",
            )
            if descriptor[0] == 255:
                break
            if descriptor[0] != 1:
                continue

            block_size = struct.unpack_from("<H", descriptor, 128)[0]
            require(
                block_size >= 512 and block_size <= 32768,
                f"Implausible ISO block size {block_size}",
            )
            root = self._parse_directory_record(
                descriptor, 156, block_size
            )
            require(root is not None, "ISO root directory record is invalid")
            _name, extent, _flags = root
            return block_size, (extent,)
        raise FormatError("ISO9660 primary volume descriptor was not found")

    def _parse_directory_record(
        self, data: bytes, offset: int, block_size: int | None = None
    ) -> tuple[str, IsoExtent, int] | None:
        if offset >= len(data) or data[offset] == 0:
            return None
        length = data[offset]
        require(offset + length <= len(data), "ISO directory record is truncated")
        require(length >= 34, "ISO directory record is too small")
        extent_lba = struct.unpack_from("<I", data, offset + 2)[0]
        extent_size = struct.unpack_from("<I", data, offset + 10)[0]
        flags = data[offset + 25]
        name_size = data[offset + 32]
        require(
            33 + name_size <= length,
            "ISO directory record name is truncated",
        )
        raw_name = data[offset + 33 : offset + 33 + name_size]
        name = raw_name.decode("ascii", "replace")
        unit = block_size or self.block_size
        extent = IsoExtent(extent_lba * unit, extent_size)
        require(
            extent.offset <= len(self._mapping)
            and extent.size <= len(self._mapping) - extent.offset,
            f"ISO extent for {name!r} is out of range",
        )
        return name, extent, flags

    def _read_extents(self, extents: tuple[IsoExtent, ...]) -> bytes:
        return b"".join(
            self.read(extent.offset, extent.size) for extent in extents
        )

    def _walk_directory(
        self, extents: tuple[IsoExtent, ...], prefix: str
    ) -> None:
        data = self._read_extents(extents)
        cursor = 0
        pending: dict[str, list[IsoExtent]] = {}
        while cursor < len(data):
            if data[cursor] == 0:
                cursor = ((cursor // self.block_size) + 1) * self.block_size
                continue
            record = self._parse_directory_record(data, cursor)
            require(record is not None, "Invalid ISO directory entry")
            raw_name, extent, flags = record
            cursor += data[cursor]
            if raw_name in ("\x00", "\x01"):
                continue

            name = raw_name.split(";", 1)[0]
            full_name = f"{prefix}/{name}" if prefix else name
            if flags & 0x02:
                self._walk_directory((extent,), full_name)
                continue

            key = full_name.casefold()
            parts = pending.setdefault(key, [])
            parts.append(extent)
            if flags & 0x80:
                continue
            self.members[key] = IsoMember(
                image=self,
                name=full_name,
                extents=tuple(parts),
            )
            del pending[key]
        require(
            not pending,
            f"ISO directory {prefix or '/'} has incomplete multi-extent files",
        )


def unwrap_bilz(data: bytes | memoryview) -> bytes:
    view = memoryview(data)
    signature_offset = 0
    if len(view) >= 0x90 and view[0x80:0x84] == BILZ_MAGIC:
        signature_offset = 0x80
    elif len(view) < 0x10 or view[:4] != BILZ_MAGIC:
        return bytes(view)

    require(len(view) >= signature_offset + 0x10, "BILZ header is truncated")
    output_size, stored_size, method = struct.unpack_from(
        ">III", view, signature_offset + 4
    )
    payload_offset = signature_offset + 0x10
    require(
        stored_size <= len(view) - payload_offset,
        "BILZ stored payload exceeds its resource",
    )
    payload = bytes(view[payload_offset : payload_offset + stored_size])

    if method == 0 or stored_size == output_size:
        output = payload
    elif method == 7:
        try:
            output = zlib.decompress(payload)
        except zlib.error as error:
            raise FormatError(f"BILZ zlib stream is invalid: {error}") from error
    else:
        raise FormatError(f"Unsupported BILZ compression method {method}")

    require(
        len(output) == output_size,
        f"BILZ decoded {len(output)} bytes, expected {output_size}",
    )
    return output


@dataclass(frozen=True, slots=True)
class ResourceRecord:
    archive: "GDatArchive"
    index: int
    resource_hash: int
    type_hash: int
    offset: int
    size: int
    name: str | None = None
    container_index: int | None = None

    @property
    def extension(self) -> str:
        return TYPE_EXTENSIONS.get(self.type_hash, f"{self.type_hash:08X}")

    @property
    def display_name(self) -> str:
        return self.name or f"{self.resource_hash:08X}.{self.extension}"

    def read_raw(self) -> bytes:
        return self.archive.read(self.offset, self.size)

    def read_decoded(self) -> bytes:
        return unwrap_bilz(self.read_raw())


class GDatArchive:
    def __init__(
        self,
        source: Path | IsoMember,
        csh_path: Path | None = None,
        endian: str = "<",
    ):
        self.path = source if isinstance(source, Path) else Path(source.name)
        self.csh_path = csh_path
        self.endian = endian
        self._source = source if isinstance(source, IsoMember) else None
        self._file = source.open("rb") if isinstance(source, Path) else None
        self._mapping = (
            mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
            if self._file is not None
            else None
        )
        self._size = (
            len(self._mapping) if self._mapping is not None else source.size
        )
        try:
            self.records = self._parse()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._mapping is not None:
            self._mapping.close()
            self._mapping = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def read(self, offset: int, size: int) -> bytes:
        require(offset >= 0 and size >= 0, "Negative archive range")
        require(
            offset <= self._size and size <= self._size - offset,
            f"Archive range 0x{offset:X}+0x{size:X} exceeds {self.path.name}",
        )
        if self._mapping is not None:
            return self._mapping[offset : offset + size]
        require(self._source is not None, "Archive source is closed")
        return self._source.read(offset, size)

    def _parse(self) -> list[ResourceRecord]:
        require(self._size >= 8, f"{self.path.name} is too small")
        require(
            self.read(0, 4) == GDAT_MAGIC,
            (
                f"{self.path.name} does not start with GDAT. If this is a Vita "
                "NoNpDrm dump, decrypt its PFS layer with Nativ or Vita3K first."
            ),
        )
        count = struct.unpack(f"{self.endian}I", self.read(4, 4))[0]
        require(count <= 10_000_000, f"Implausible GDAT entry count {count}")

        if self.endian == ">":
            return self._parse_ps3_table(count)
        modern = self._try_modern_table(count)
        if modern is not None:
            return modern
        if self.csh_path is None:
            raise FormatError(
                f"{self.path.name} uses a legacy GDAT table but no matching CSH was found"
            )
        return self._parse_legacy_table(count)

    def _try_modern_table(self, count: int) -> list[ResourceRecord] | None:
        table_size = count * 16
        if table_size > self._size - 8:
            return None

        rows = []
        table_end = 8 + table_size
        table = self.read(8, table_size)
        for index in range(count):
            resource_hash, type_hash, offset, size = struct.unpack_from(
                "<IIII", table, index * 16
            )
            if offset < table_end or offset > self._size:
                return None
            if size > self._size - offset:
                return None
            rows.append(
                ResourceRecord(
                    archive=self,
                    index=index,
                    resource_hash=resource_hash,
                    type_hash=type_hash,
                    offset=offset,
                    size=size,
                )
            )
        return rows

    def _parse_ps3_table(self, count: int) -> list[ResourceRecord]:
        table_size = count * 16
        require(
            table_size <= self._size - 8,
            "PS3 GDAT resource table is truncated",
        )
        table = self.read(8, table_size)
        table_end = 8 + table_size
        rows = []
        for index in range(count):
            key, offset, size = struct.unpack_from(">QII", table, index * 16)
            type_hash = key >> 32
            resource_hash = key & 0xFFFFFFFF
            require(
                offset >= table_end
                and offset <= self._size
                and size <= self._size - offset,
                f"PS3 GDAT entry {index} is out of range",
            )
            rows.append(
                ResourceRecord(
                    archive=self,
                    index=index,
                    resource_hash=resource_hash,
                    type_hash=type_hash,
                    offset=offset,
                    size=size,
                )
            )
        return rows

    def _parse_legacy_table(self, count: int) -> list[ResourceRecord]:
        require(
            count * 8 <= self._size - 8,
            "Legacy GDAT offset table is truncated",
        )
        table = self.read(8, count * 8)
        identities = _parse_csh(self.csh_path)
        require(
            len(identities) == count,
            f"CSH has {len(identities)} entries but DAT has {count}",
        )

        rows = []
        for index, (resource_hash, type_hash, _file_number) in enumerate(identities):
            offset, size = struct.unpack_from("<II", table, index * 8)
            require(
                offset <= self._size and size <= self._size - offset,
                f"Legacy GDAT entry {index} is out of range",
            )
            rows.append(
                ResourceRecord(
                    archive=self,
                    index=index,
                    resource_hash=resource_hash,
                    type_hash=type_hash,
                    offset=offset,
                    size=size,
                )
            )
        return rows

    def expand_pac(self, record: ResourceRecord) -> list[ResourceRecord]:
        if record.size < 0x88:
            return []
        header = self.read(record.offset, 0x88)
        magic = header[:4]
        if magic == VITA_PAC_MAGIC:
            endian = "<"
        elif magic == PS3_PAC_MAGIC:
            endian = ">"
        else:
            return []

        count = struct.unpack_from(f"{endian}I", header, 0x80)[0]
        require(count <= 1_000_000, f"PAC entry count {count} is implausible")
        table_size = 8 + count * 0x108
        require(
            table_size <= record.size - 0x80,
            f"PAC table in entry {record.index} is truncated",
        )
        table = self.read(record.offset + 0x80, table_size)
        relative_payload_offset = 0x80 + table_size
        if endian == ">":
            relative_payload_offset = (
                relative_payload_offset + 7
            ) & ~7
        payload_offset = record.offset + relative_payload_offset
        container_end = record.offset + record.size
        resources = []

        for index in range(count):
            entry_offset = 8 + index * 0x108
            raw_name = table[entry_offset : entry_offset + 0x100]
            name_bytes = raw_name.split(b"\x00", 1)[0]
            try:
                name = name_bytes.decode("utf-8")
            except UnicodeDecodeError:
                name = name_bytes.decode("shift_jis", "replace")
            type_hash, size = struct.unpack_from(
                f"{endian}II", table, entry_offset + 0x100
            )
            require(
                payload_offset <= container_end
                and size <= container_end - payload_offset,
                f"PAC resource {index} in entry {record.index} is out of range",
            )
            resources.append(
                ResourceRecord(
                    archive=self,
                    index=index,
                    resource_hash=orochi_hash(name),
                    type_hash=type_hash,
                    offset=payload_offset,
                    size=size,
                    name=name,
                    container_index=record.index,
                )
            )
            payload_offset += size
        return resources


def _parse_csh(path: Path) -> list[tuple[int, int, int]]:
    data = unwrap_bilz(path.read_bytes())
    require(len(data) >= 12, f"{path.name} CSH index is truncated")
    count = struct.unpack_from(">I", data, 8)[0]
    require(count <= 10_000_000, f"Implausible CSH entry count {count}")
    require(12 + count * 16 <= len(data), "CSH entry table is truncated")

    identities = []
    for index in range(count):
        hash_offset, _unknown0, file_id_offset, _unknown1 = struct.unpack_from(
            ">IIII", data, 12 + index * 16
        )
        require(hash_offset <= len(data) - 8, f"CSH hash offset {index} is invalid")
        require(
            file_id_offset <= len(data) - 4,
            f"CSH file-id offset {index} is invalid",
        )
        resource_hash, type_hash = struct.unpack_from(">II", data, hash_offset)
        file_number = struct.unpack_from(">I", data, file_id_offset)[0]
        identities.append((resource_hash, type_hash, file_number))
    return identities


@dataclass(frozen=True, slots=True)
class ModelAsset:
    resource_hash: int
    model: ResourceRecord
    material: ResourceRecord | None

    @property
    def display_name(self) -> str:
        if self.model.name:
            return self.model.name.rsplit("/", 1)[-1]
        return f"{self.resource_hash:08X}"


class GameSource:
    def __init__(
        self,
        archive_sources: Iterable[Path | IsoMember],
        profile: GameProfile | None = None,
        iso_image: IsoImage | None = None,
    ):
        sources = list(archive_sources)
        require(bool(sources), "No GxArchivedFile DAT archives were found")
        paths = [
            source if isinstance(source, Path) else Path(source.name)
            for source in sources
        ]
        self.profile = profile or detect_profile(paths)
        self._iso_image = iso_image
        self.archives: list[GDatArchive] = []
        self.resources: dict[tuple[int, int], ResourceRecord] = {}
        self.named_resources: dict[str, ResourceRecord] = {}

        ordered = sorted(
            zip(paths, sources),
            key=lambda item: self.profile.archive_priority(item[0]),
        )
        try:
            for path, source in ordered:
                csh_path = (
                    _case_insensitive_sibling(path, ".csh")
                    if isinstance(source, Path)
                    else None
                )
                archive = GDatArchive(
                    source, csh_path, endian=self.profile.endian_hint
                )
                self.archives.append(archive)
                for record in archive.records:
                    self.resources[
                        (record.resource_hash, record.type_hash)
                    ] = record
                    if record.type_hash == PAC_TYPE_HASH:
                        for nested in archive.expand_pac(record):
                            self.resources[
                                (nested.resource_hash, nested.type_hash)
                            ] = nested
                            if nested.name:
                                self.named_resources[
                                    nested.name.casefold()
                                ] = nested
        except Exception:
            self.close()
            raise

    @classmethod
    def open(cls, path: str | Path, profile: GameProfile | None = None) -> "GameSource":
        source_path = Path(path)
        require(source_path.exists(), f"Source does not exist: {source_path}")

        if source_path.is_file() and source_path.suffix.casefold() == ".iso":
            image = IsoImage(source_path)
            archive_sources = [
                member
                for member in image.members.values()
                if Path(member.name).name.casefold().startswith(
                    "gxarchivedfile"
                )
                and Path(member.name).suffix.casefold() == ".dat"
            ]
            try:
                return cls(
                    archive_sources,
                    profile,
                    iso_image=image,
                )
            except Exception:
                image.close()
                raise
        elif source_path.is_file() and source_path.suffix.casefold() == ".dat":
            archive_sources = [source_path]
        elif source_path.is_dir():
            archive_sources = _discover_archives(source_path)
        else:
            raise FormatError(
                "Choose a game folder, ISO, or GxArchivedFile DAT archive"
            )
        return cls(archive_sources, profile)

    def close(self) -> None:
        for archive in self.archives:
            archive.close()
        self.archives.clear()
        if self._iso_image is not None:
            self._iso_image.close()
            self._iso_image = None

    def find(self, resource_hash: int, type_hash: int) -> ResourceRecord | None:
        return self.resources.get((resource_hash & 0xFFFFFFFF, type_hash))

    def find_name(self, name: str, type_hash: int) -> ResourceRecord | None:
        named = self.named_resources.get(name.casefold())
        if named is not None and named.type_hash == type_hash:
            return named
        return self.find(orochi_hash(name), type_hash)

    def model_assets(self) -> list[ModelAsset]:
        assets = []
        for (resource_hash, type_hash), model in self.resources.items():
            if type_hash != MODEL_TYPE_HASH:
                continue
            assets.append(
                ModelAsset(
                    resource_hash=resource_hash,
                    model=model,
                    material=self.find(resource_hash, MATERIAL_TYPE_HASH),
                )
            )
        return sorted(
            assets,
            key=lambda item: (
                item.display_name.casefold(),
                item.resource_hash,
            ),
        )


def _discover_archives(root: Path) -> list[Path]:
    matches = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.casefold() == ".dat"
        and path.stem.casefold().startswith("gxarchivedfile")
    ]
    require(bool(matches), f"No GxArchivedFile DAT archives found under {root}")
    return matches


def _case_insensitive_sibling(path: Path, suffix: str) -> Path | None:
    expected = f"{path.stem}{suffix}".casefold()
    for sibling in path.parent.iterdir():
        if sibling.is_file() and sibling.name.casefold() == expected:
            return sibling
    return None
