# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GameProfile:
    identifier: str
    label: str
    platform: str
    endian_hint: str
    archive_path_markers: tuple[str, ...]

    def archive_priority(self, path: Path) -> tuple[int, int, str]:
        normalized = str(path).replace("\\", "/").lower()
        layer = 1 if "/patch/" in normalized or "/patch0/" in normalized else 0
        digits = "".join(character for character in path.stem if character.isdigit())
        archive_number = int(digits[-3:]) if digits else -1
        return layer, archive_number, normalized


GUNDAM_BREAKER_PS3 = GameProfile(
    identifier="GUNDAM_BREAKER_PS3",
    label="Gundam Breaker (PS3)",
    platform="PS3",
    endian_hint=">",
    archive_path_markers=("ps3_game", "usrdir"),
)

GUNDAM_BREAKER_3_VITA = GameProfile(
    identifier="GUNDAM_BREAKER_3_VITA",
    label="Gundam Breaker 3 (Vita)",
    platform="Vita",
    endian_hint="<",
    archive_path_markers=("masterpsp2", "finalizedpsp2", "pcsh00208"),
)

PROFILES = {
    GUNDAM_BREAKER_PS3.identifier: GUNDAM_BREAKER_PS3,
    GUNDAM_BREAKER_3_VITA.identifier: GUNDAM_BREAKER_3_VITA,
}


def detect_profile(paths: list[Path]) -> GameProfile:
    normalized = " ".join(str(path).replace("\\", "/").lower() for path in paths)
    scores = {
        profile: sum(marker in normalized for marker in profile.archive_path_markers)
        for profile in PROFILES.values()
    }
    return max(scores, key=scores.get) if any(scores.values()) else GUNDAM_BREAKER_PS3

