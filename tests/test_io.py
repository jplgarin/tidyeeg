from __future__ import annotations

from pathlib import Path

import pytest

import tidyeeg as te
from tidyeeg import TidyEEG


def test_load_set(set_file: Path) -> None:
    eeg = te.load_set(set_file)
    assert isinstance(eeg, TidyEEG)
    assert eeg.n_channels == 6
    assert eeg.sfreq == 160.0
    assert eeg.data.shape == (6, 320)
    # Loaders preserve original labels; cleaning is a separate, explicit step.
    assert eeg.ch_names[0] == "Fc5."
    assert eeg.meta["source_format"] == "set"


def test_load_edf(edf_file: Path) -> None:
    eeg = te.load_edf(edf_file)
    assert isinstance(eeg, TidyEEG)
    assert eeg.n_channels == 6
    assert eeg.sfreq == 160.0
    assert eeg.data.shape == (6, 320)
    assert eeg.meta["source_format"] == "edf"


def test_load_dispatches_on_extension(set_file: Path, edf_file: Path) -> None:
    assert te.load(set_file).meta["source_format"] == "set"
    assert te.load(edf_file).meta["source_format"] == "edf"


def test_load_format_override(set_file: Path, tmp_path: Path) -> None:
    # File with no extension still loads when the format is given explicitly.
    renamed = tmp_path / "noext"
    renamed.write_bytes(set_file.read_bytes())
    # The companion .fdt is not used for this single-file fixture, so this is safe.
    assert te.load(renamed, format="set").n_channels == 6


def test_load_unsupported_format_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "recording.fif"
    bogus.touch()
    with pytest.raises(ValueError, match="Unsupported format"):
        te.load(bogus)
