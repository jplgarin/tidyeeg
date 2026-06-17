"""End-to-end example on a real PhysioNet recording.

This is the only place in the project that touches the network. It fetches one
small (~1.2 MB) file from the PhysioNet EEG Motor Movement/Imagery database
(subject 1, run 1) via MNE's dataset helper, which caches it under ~/mne_data so
the download happens at most once.

Run it with:

    pip install -e ".[pandas]"
    python examples/quickstart.py
"""

from __future__ import annotations

from mne.datasets import eegbci

import tidyeeg as te


def main() -> None:
    # load_data returns local paths, downloading on first use only.
    edf_path = eegbci.load_data(subjects=1, runs=[1], update_path=True)[0]
    print(f"Loaded file: {edf_path}")

    eeg = te.load(edf_path)
    print("\nRaw, as recorded:")
    print(" ", eeg)
    print("  first channel labels:", eeg.ch_names[:5], "...")

    # The PhysioNet labels are padded with dots (Fc5., Cz..); standardize cleans
    # them, assigns the 10-20 montage, applies an average reference, and resamples.
    tidy = te.standardize(eeg, montage="standard_1020", reference="average", sfreq=128.0)
    print("\nAfter standardize:")
    print(" ", tidy)
    print("  first channel labels:", tidy.ch_names[:5], "...")
    print("  history:", tidy.meta["history"])

    df = tidy.to_dataframe()
    print("\nTidy long-format frame:")
    print(df.head(), "\n")
    print(f"  shape: {df.shape[0]} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main()
