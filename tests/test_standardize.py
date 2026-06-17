from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import tidyeeg as te


def test_set_montage_cleans_names(set_file: Path) -> None:
    eeg = te.set_montage(te.load_set(set_file))
    # Dots stripped, casing canonicalized, legacy T3 mapped to T7.
    assert eeg.ch_names == ["FC5", "FC3", "Cz", "Fp1", "T7", "PO3"]
    assert eeg.montage == "standard_1020"
    assert "set_montage(standard_1020)" in eeg.meta["history"]


def test_set_montage_can_keep_raw_names(set_file: Path) -> None:
    eeg = te.set_montage(te.load_set(set_file), clean_names=False, on_missing="ignore")
    assert eeg.ch_names[0] == "Fc5."


def test_set_montage_on_missing_raises(set_file: Path) -> None:
    eeg = te.load_set(set_file)
    # Without name cleaning the dotted labels are unknown to the montage.
    with pytest.raises(ValueError, match="not in montage"):
        te.set_montage(eeg, clean_names=False, on_missing="raise")


def test_average_reference_zeroes_channel_mean(set_file: Path) -> None:
    eeg = te.set_reference(te.load_set(set_file), "average")
    assert eeg.reference == "average"
    # After average referencing the mean across channels is zero at every sample.
    assert np.allclose(eeg.data.mean(axis=0), 0.0, atol=1e-12)


def test_channel_reference_zeroes_that_channel(set_file: Path) -> None:
    eeg = te.set_montage(te.load_set(set_file))
    out = te.set_reference(eeg, "Cz")
    cz = out.ch_names.index("Cz")
    assert np.allclose(out.data[cz], 0.0, atol=1e-12)
    assert out.reference == "Cz"


def test_reference_missing_channel_raises(set_file: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        te.set_reference(te.load_set(set_file), "NOPE")


def test_resample_changes_rate_and_length(set_file: Path) -> None:
    eeg = te.load_set(set_file)
    out = te.resample(eeg, 80.0)
    assert out.sfreq == 80.0
    assert out.n_times == eeg.n_times // 2
    assert out.duration == pytest.approx(eeg.duration, rel=1e-3)


def test_resample_noop_when_rate_matches(set_file: Path) -> None:
    eeg = te.load_set(set_file)
    out = te.resample(eeg, eeg.sfreq)
    assert out.n_times == eeg.n_times
    np.testing.assert_array_equal(out.data, eeg.data)


def test_standardize_composes_all_steps(set_file: Path) -> None:
    eeg = te.load_set(set_file)
    out = te.standardize(eeg, montage="standard_1020", reference="average", sfreq=80.0)
    assert out.montage == "standard_1020"
    assert out.reference == "average"
    assert out.sfreq == 80.0
    assert out.ch_names == ["FC5", "FC3", "Cz", "Fp1", "T7", "PO3"]
    assert out.meta["history"] == [
        "set_montage(standard_1020)",
        "set_reference(average)",
        "resample(80.0)",
    ]


def test_standardize_skips_none_steps(set_file: Path) -> None:
    eeg = te.load_set(set_file)
    out = te.standardize(eeg, montage=None, reference=None, sfreq=None)
    assert out.meta["history"] == []
    assert out.montage is None


def test_steps_do_not_mutate_input(set_file: Path) -> None:
    eeg = te.load_set(set_file)
    before = eeg.to_numpy()
    te.set_reference(eeg, "average")
    np.testing.assert_array_equal(eeg.to_numpy(), before)
