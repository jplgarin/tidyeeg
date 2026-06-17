"""tidyeeg: turn messy, heterogeneous EEG into clean, consistent, model-ready data."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .core import TidyEEG
from .io import load, load_edf, load_set
from .standardize import resample, set_montage, set_reference, standardize

try:
    __version__ = version("tidyeeg")
except PackageNotFoundError:  # pragma: no cover - only hit when running from a source tree
    __version__ = "0.1.0"

__all__ = [
    "TidyEEG",
    "__version__",
    "load",
    "load_edf",
    "load_set",
    "resample",
    "set_montage",
    "set_reference",
    "standardize",
]
