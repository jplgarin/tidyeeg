from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pytest

import tidyeeg as te
from tidyeeg import TidyEEG


def _toy() -> TidyEEG:
    data = np.ones((2, 10), dtype=np.float64)
    return TidyEEG(data=data, ch_names=["Cz", "Pz"], sfreq=100.0)


def test_rejects_name_count_mismatch() -> None:
    with pytest.raises(ValueError, match="channels"):
        TidyEEG(data=np.zeros((3, 5)), ch_names=["a", "b"], sfreq=100.0)


def test_rejects_non_2d_data() -> None:
    with pytest.raises(ValueError, match="2D"):
        TidyEEG(data=np.zeros((3,)), ch_names=["a"], sfreq=100.0)


def test_rejects_nonpositive_sfreq() -> None:
    with pytest.raises(ValueError, match="sfreq"):
        TidyEEG(data=np.zeros((1, 5)), ch_names=["a"], sfreq=0.0)


def test_data_is_read_only() -> None:
    eeg = _toy()
    with pytest.raises(ValueError):
        eeg.data[0, 0] = 5.0


def test_derived_properties() -> None:
    eeg = _toy()
    assert eeg.n_channels == 2
    assert eeg.n_times == 10
    assert eeg.duration == pytest.approx(0.1)
    assert eeg.times[1] == pytest.approx(0.01)


def test_to_numpy_returns_writeable_copy() -> None:
    eeg = _toy()
    arr = eeg.to_numpy()
    arr[0, 0] = 99.0  # must not raise and must not touch the original
    assert eeg.data[0, 0] == 1.0


def test_to_dataframe_long_format() -> None:
    eeg = _toy()
    df = eeg.to_dataframe()
    assert list(df.columns) == ["channel", "time", "value"]
    assert len(df) == eeg.n_channels * eeg.n_times
    assert set(df["channel"].unique()) == {"Cz", "Pz"}


def test_to_mne_roundtrip_with_montage(set_file: Path) -> None:
    eeg = te.set_montage(te.load_set(set_file))
    raw = eeg.to_mne()
    assert isinstance(raw, mne.io.BaseRaw)
    assert raw.ch_names == eeg.ch_names
    assert raw.info["sfreq"] == eeg.sfreq
    # Montage assignment means real sensor positions are attached.
    assert raw.get_montage() is not None
    np.testing.assert_allclose(raw.get_data(), eeg.data)


def test_repr_is_informative() -> None:
    assert "TidyEEG(" in repr(_toy())
    assert "sfreq=100" in repr(_toy())
