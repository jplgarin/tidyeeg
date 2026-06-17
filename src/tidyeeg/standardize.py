"""Standardize operations: montage, reference, resample.

Each function takes a :class:`TidyEEG` and returns a new one, so they compose
freely and can be used individually. :func:`standardize` chains them in the order
that keeps later steps correct (clean names before referencing by name).
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import Any, Literal

import mne
import numpy as np

from ._channels import normalize_names
from .core import TidyEEG

OnMissing = Literal["warn", "raise", "ignore"]


def _step(eeg: TidyEEG, op: str, **changes: Any) -> TidyEEG:
    """Return a copy of ``eeg`` with ``changes`` applied and ``op`` logged to history."""
    meta = dict(eeg.meta)
    meta["history"] = [*meta.get("history", []), op]
    return replace(eeg, meta=meta, **changes)


def set_montage(
    eeg: TidyEEG,
    montage: str = "standard_1020",
    *,
    clean_names: bool = True,
    on_missing: OnMissing = "warn",
) -> TidyEEG:
    """Standardize channel names and assign a named montage.

    With ``clean_names`` (the default), labels are normalized to canonical 10-20
    spelling first, which is what lets messy real-world names match the montage.
    Channels not present in the montage are reported according to ``on_missing``;
    the default warns rather than failing, since recordings often carry a few
    non-scalp channels.
    """
    names = normalize_names(eeg.ch_names) if clean_names else list(eeg.ch_names)

    known = {n.upper() for n in mne.channels.make_standard_montage(montage).ch_names}
    missing = [n for n in names if n.upper() not in known]
    if missing:
        msg = f"{len(missing)} channel(s) not in montage {montage!r}: {missing}"
        if on_missing == "raise":
            raise ValueError(msg)
        if on_missing == "warn":
            warnings.warn(msg, stacklevel=2)

    return _step(eeg, f"set_montage({montage})", ch_names=names, montage=montage)


def set_reference(eeg: TidyEEG, reference: str | list[str] = "average") -> TidyEEG:
    """Re-reference the data.

    ``"average"`` subtracts the mean across all channels. Otherwise pass a channel
    name or a list of names to use that electrode (or their mean) as the
    reference. Re-referencing is a linear transform, so it is done directly on the
    array rather than round-tripping through MNE.
    """
    data = np.asarray(eeg.data)

    if isinstance(reference, str) and reference == "average":
        ref_signal = data.mean(axis=0, keepdims=True)
        label = "average"
    else:
        names = [reference] if isinstance(reference, str) else list(reference)
        missing = [n for n in names if n not in eeg.ch_names]
        if missing:
            raise ValueError(f"Reference channel(s) not found: {missing}")
        idx = [eeg.ch_names.index(n) for n in names]
        ref_signal = data[idx].mean(axis=0, keepdims=True)
        label = "+".join(names)

    return _step(eeg, f"set_reference({label})", data=data - ref_signal, reference=label)


def resample(eeg: TidyEEG, sfreq: float) -> TidyEEG:
    """Resample to ``sfreq`` hertz.

    Delegates to MNE so the anti-aliasing filter is applied correctly; a naive
    decimation would alias high-frequency content into the band of interest.
    """
    if sfreq <= 0:
        raise ValueError(f"sfreq must be positive, got {sfreq}")
    if sfreq == eeg.sfreq:
        return _step(eeg, f"resample({sfreq}) [no-op]")

    raw = eeg.to_mne()
    raw.resample(sfreq, verbose="ERROR")
    return _step(
        eeg,
        f"resample({sfreq})",
        data=np.asarray(raw.get_data(), dtype=np.float64),
        sfreq=float(sfreq),
    )


def standardize(
    eeg: TidyEEG,
    *,
    montage: str | None = "standard_1020",
    reference: str | list[str] | None = "average",
    sfreq: float | None = None,
) -> TidyEEG:
    """Run the standardize pipeline. Any step is skipped by passing ``None``.

    Order matters: montage (and name cleaning) runs first so a name-based
    reference resolves against canonical labels, and resampling runs last.
    """
    if montage is not None:
        eeg = set_montage(eeg, montage)
    if reference is not None:
        eeg = set_reference(eeg, reference)
    if sfreq is not None:
        eeg = resample(eeg, sfreq)
    return eeg
