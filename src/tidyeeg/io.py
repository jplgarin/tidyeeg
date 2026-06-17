"""Loaders. Every supported format funnels into the same :class:`TidyEEG`."""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import mne

from .core import TidyEEG

# Extension (without dot) -> loader, for the dispatching loader and error messages.
_SUPPORTED = ("edf", "set")


@contextmanager
def _as_extension(path: str | Path, ext: str) -> Iterator[str]:
    """Yield a path that ends in ``ext``.

    MNE's readers dispatch on the file extension, so when the caller forces a
    format that disagrees with the file name (via ``load(..., format=...)``) we
    read from a correctly-suffixed temporary copy instead of failing.
    """
    src = Path(path)
    if src.suffix.lower() == ext:
        yield str(src)
        return
    fd, tmp_path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    try:
        shutil.copyfile(src, tmp_path)
        yield tmp_path
    finally:
        os.unlink(tmp_path)


def load_edf(path: str | Path) -> TidyEEG:
    """Load a European Data Format (.edf) recording."""
    with _as_extension(path, ".edf") as p:
        raw = mne.io.read_raw_edf(p, preload=True, verbose="ERROR")
    return TidyEEG.from_mne(raw, source_path=str(path), source_format="edf")


def load_set(path: str | Path) -> TidyEEG:
    """Load an EEGLAB (.set) recording."""
    with _as_extension(path, ".set") as p:
        raw = mne.io.read_raw_eeglab(p, preload=True, verbose="ERROR")
    return TidyEEG.from_mne(raw, source_path=str(path), source_format="set")


def load(path: str | Path, *, format: str | None = None) -> TidyEEG:
    """Load EEG from a file, dispatching on the extension.

    Pass ``format`` (``"edf"`` or ``"set"``) to override detection when the file
    extension is missing or misleading.
    """
    fmt = (format or Path(path).suffix.lstrip(".")).lower()
    if fmt == "edf":
        return load_edf(path)
    if fmt == "set":
        return load_set(path)
    raise ValueError(
        f"Unsupported format {fmt!r}. tidyeeg v0.1 supports: {', '.join(_SUPPORTED)}."
    )
