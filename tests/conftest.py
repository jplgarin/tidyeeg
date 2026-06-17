"""Offline, deterministic fixtures for both first-class formats.

Nothing here touches the network. The ``.set`` file is written with
``scipy.io.savemat`` in the minimal EEGLAB layout that ``mne.io.read_raw_eeglab``
accepts; the ``.edf`` file is written through MNE's exporter (backed by edfio).
Both deliberately carry PhysioNet-style messy channel names so the suite
exercises name cleaning.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import mne
import numpy as np
import numpy.typing as npt
import pytest
from scipy.io import savemat

SFREQ = 160.0
N_TIMES = 320  # 2 seconds
# Messy on purpose: trailing dots (PhysioNet padding) and legacy temporal labels.
MESSY_NAMES = ["Fc5.", "Fc3.", "Cz..", "Fp1.", "T3", "Po3."]
CLEAN_NAMES = ["FC5", "FC3", "Cz", "Fp1", "T7", "PO3"]


def _signal() -> npt.NDArray[np.float64]:
    """Deterministic per-channel sinusoids in microvolts."""
    rng = np.random.default_rng(42)
    t = np.arange(N_TIMES) / SFREQ
    freqs = np.array([5.0, 7.0, 9.0, 11.0, 13.0, 15.0])
    base = np.sin(2 * np.pi * np.outer(freqs, t)) * 20.0  # ~20 uV oscillations
    return base + rng.standard_normal((len(freqs), N_TIMES))


@pytest.fixture(scope="session")
def set_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "sample.set"
    data = _signal().astype(np.float32)  # EEGLAB stores microvolts
    chanlocs = np.array([(name,) for name in MESSY_NAMES], dtype=[("labels", "O")])
    eeg = {
        "setname": "tidyeeg-fixture",
        "nbchan": float(len(MESSY_NAMES)),
        "trials": 1.0,
        "pnts": float(N_TIMES),
        "srate": SFREQ,
        "xmin": 0.0,
        "xmax": (N_TIMES - 1) / SFREQ,
        "data": data,
        "icawinv": [],
        "icasphere": [],
        "icaweights": [],
        "icaact": [],
        "event": [],
        "epoch": [],
        "chanlocs": chanlocs,
        "ref": "common",
        "comments": "",
    }
    savemat(str(path), {"EEG": eeg})
    return path


@pytest.fixture(scope="session")
def edf_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("data") / "sample.edf"
    # EDF cannot store the trailing-dot labels (it strips/truncates them), so the
    # .edf fixture uses the legacy T3 label to still exercise name normalization.
    names = ["Fc5", "Fc3", "Cz", "Fp1", "T3", "Po3"]
    info = mne.create_info(names, SFREQ, ch_types="eeg", verbose="ERROR")
    raw = mne.io.RawArray(_signal() * 1e-6, info, verbose="ERROR")  # MNE wants volts
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mne.export.export_raw(str(path), raw, fmt="edf", verbose="ERROR")
    return path
