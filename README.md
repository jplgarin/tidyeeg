# tidyeeg

EEG ETL and preprocessing. tidyeeg turns heterogeneous, messy EEG into clean,
consistent, model-ready data through a small, predictable API built on top of
[MNE-Python](https://mne.tools).

The name joins *tidy* (clean, consistent, predictable structure) with *EEG*.
Different recorders and formats label the same electrode differently, sample at
different rates, and use different references. tidyeeg loads them into one
in-memory structure with a documented schema and gives you a few explicit
operations to standardize them.

## Why

MNE is the right foundation for reading and transforming EEG, but its surface is
large and stateful. tidyeeg is a thin, opinionated layer on top: one immutable
data object, one way to load each format, and a handful of pure standardize
functions that compose. It does not hide MNE; `TidyEEG.to_mne()` hands you a full
`Raw` whenever you need it.

This is v0.1. It is intentionally tight (see [Scope](#scope)).

## Install

```bash
pip install tidyeeg            # core
pip install "tidyeeg[pandas]"  # adds TidyEEG.to_dataframe()
```

From a checkout:

```bash
pip install -e ".[dev,pandas]"
```

Requires Python 3.10+.

## Quickstart

```python
import tidyeeg as te

eeg = te.load("recording.edf")          # or .set; dispatch is by extension

tidy = te.standardize(
    eeg,
    montage="standard_1020",            # clean channel names + assign montage
    reference="average",                # re-reference
    sfreq=128.0,                        # resample
)

print(tidy)                             # TidyEEG(n_channels=64, sfreq=128 Hz, ...)
df = tidy.to_dataframe()                # long tidy format: channel, time, value
```

Each step is also usable on its own:

```python
eeg = te.set_montage(eeg)               # standardize names, assign 10-20 montage
eeg = te.set_reference(eeg, "Cz")       # or "average", or a list of channels
eeg = te.resample(eeg, 256.0)
```

### Runnable example

[`examples/quickstart.py`](examples/quickstart.py) runs the full pipeline on a
real recording. It downloads one small (~1.2 MB) file from the PhysioNet EEG
Motor Movement/Imagery database (subject 1, run 1) via MNE's dataset helper,
which caches it under `~/mne_data`, so the download happens at most once. This is
the only part of the project that uses the network.

```bash
pip install -e ".[pandas]"
python examples/quickstart.py
```

## The TidyEEG object

`TidyEEG` is an immutable container with a small, documented schema:

| Field        | Meaning                                                        |
| ------------ | ------------------------------------------------------------- |
| `data`       | `float64` array, shape `(n_channels, n_times)`, in volts      |
| `ch_names`   | channel labels, ordered to match the rows of `data`           |
| `sfreq`      | sampling rate in Hz                                            |
| `montage`    | applied montage name, or `None`                               |
| `reference`  | current reference (`"average"`, a channel name, ...), or `None` |
| `meta`       | provenance: source path/format, original rate, and `history`  |

Convenience: `n_channels`, `n_times`, `times`, `duration`; conversions
`to_numpy()`, `to_dataframe()`, `to_mne()`, and `TidyEEG.from_mne()`.

Operations never mutate in place. They return a new `TidyEEG` and append to
`meta["history"]`, so a pipeline is explicit and replayable.

## Scope

v0.1 supports:

- **Load:** `.edf` and `.set`, both into the same `TidyEEG` via `load()`.
- **Standardize:** channel names + montage, re-referencing, and resampling,
  individually or composed via `standardize()`.

Not in v0.1: filtering, artifact handling, epoching, BIDS, `.mat`, and a CLI.

## Development

```bash
pip install -e ".[dev,pandas]"
ruff check .
mypy
pytest
```

The test suite is fully offline and deterministic: fixtures for both formats are
generated programmatically, so no network or bundled binaries are required.

## License

MIT
