"""Constants and label helpers for the PAM WSSED pipeline."""

from __future__ import annotations

TARGET_SPECIES = [
    "DENMIN",
    "LEPLAT",
    "PHYCUV",
    "SPHSUR",
    "SCIPER",
    "BOABIS",
    "BOAFAB",
    "LEPPOD",
    "PHYALB",
]

OTHERS = "OTHERS"


def canonicalize_label(raw_label: str) -> str:
    """Strip any suffix after an underscore from a label.

    Examples
    --------
    >>> canonicalize_label("BOABIS_L")
    'BOABIS'
    >>> canonicalize_label("DENMIN_M")
    'DENMIN'
    >>> canonicalize_label("SPHSUR")
    'SPHSUR'
    """

    cleaned = raw_label.strip()
    if "_" in cleaned:
        cleaned = cleaned.split("_", 1)[0]
    return cleaned


def map_to_target(raw_label: str) -> str:
    """Map a raw label to the target space with OTHERS fallback."""

    base_label = canonicalize_label(raw_label)
    return base_label if base_label in TARGET_SPECIES else OTHERS
