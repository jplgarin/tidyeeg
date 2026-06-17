"""Channel-name normalization to canonical 10-20 / 10-10 labels.

Real EEG files label the same electrode in incompatible ways: PhysioNet pads
names to a fixed width with dots (``Fc5.``, ``Cz..``), some recorders use the
old T3/T4/T5/T6 temporal nomenclature, and casing is inconsistent across
vendors. Montage assignment downstream only works once labels match the
canonical spelling MNE uses, so normalization happens before anything else.
"""

from __future__ import annotations

import mne

# Canonical spelling is whatever the standard_1020 montage uses (Fp1, AFz, FCz,
# Cz, ...). Keying on the uppercased label lets us accept any incoming casing.
_CANONICAL: dict[str, str] = {
    name.upper(): name for name in mne.channels.make_standard_montage("standard_1020").ch_names
}

# Pre-1990s temporal labels map onto the modern 10-10 positions.
_LEGACY_ALIASES: dict[str, str] = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


def normalize_name(name: str) -> str:
    """Return the canonical label for a single channel.

    Unknown labels (non-EEG channels such as stim or EOG) are returned cleaned
    but otherwise unchanged, so we never silently drop a channel we cannot map.
    """
    cleaned = name.strip().rstrip(".").strip()
    key = cleaned.upper()
    key = _LEGACY_ALIASES.get(key, key)
    return _CANONICAL.get(key, cleaned)


def normalize_names(names: list[str]) -> list[str]:
    """Normalize a list of channel names, rejecting collisions.

    Two distinct inputs can normalize to the same label (for example ``T3`` and
    ``T7``). That would corrupt the channel-to-row mapping, so it is an error
    rather than a silent merge.
    """
    out = [normalize_name(n) for n in names]
    seen: dict[str, str] = {}
    for original, canonical in zip(names, out, strict=True):
        if canonical in seen and seen[canonical] != original:
            raise ValueError(
                f"Channel names {seen[canonical]!r} and {original!r} both normalize "
                f"to {canonical!r}; cannot disambiguate."
            )
        seen[canonical] = original
    return out
