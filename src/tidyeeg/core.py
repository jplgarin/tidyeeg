"""The :class:`TidyEEG` container, the contract the rest of the library grows around."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import mne
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class TidyEEG:
    """An immutable, tidy view of continuous EEG.

    The schema is deliberately small and predictable so downstream code can rely
    on it: ``data`` is a ``(n_channels, n_times)`` float array in volts (the MNE
    convention), ``ch_names`` labels its rows in order, and ``sfreq`` is the
    sampling rate in hertz. ``montage`` and ``reference`` record what has been
    applied rather than re-deriving it from the data. ``meta`` carries provenance
    including a ``history`` list of the operations applied so far.

    Instances are immutable. Standardize operations return new instances instead
    of mutating in place, which keeps transformations explicit and replayable.
    """

    data: NDArray[np.float64]
    ch_names: list[str]
    sfreq: float
    montage: str | None = None
    reference: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(f"data must be 2D (n_channels, n_times), got shape {self.data.shape}")
        if self.data.shape[0] != len(self.ch_names):
            raise ValueError(
                f"data has {self.data.shape[0]} channels but {len(self.ch_names)} names were given"
            )
        if self.sfreq <= 0:
            raise ValueError(f"sfreq must be positive, got {self.sfreq}")
        # Enforce the immutability promise at the array level; rebinding fields is
        # already blocked by the frozen dataclass, but the buffer itself is shared.
        self.data.flags.writeable = False

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_times(self) -> int:
        return int(self.data.shape[1])

    @property
    def times(self) -> NDArray[np.float64]:
        """Sample times in seconds, starting at zero."""
        return np.arange(self.n_times, dtype=np.float64) / self.sfreq

    @property
    def duration(self) -> float:
        """Recording length in seconds."""
        return self.n_times / self.sfreq

    @classmethod
    def from_mne(
        cls,
        raw: mne.io.BaseRaw,
        *,
        source_path: str | None = None,
        source_format: str | None = None,
        montage: str | None = None,
        reference: str | None = None,
    ) -> TidyEEG:
        """Snapshot an MNE ``Raw`` into a tidy container.

        This is the single entry point the loaders use, so format-specific quirks
        stay in MNE and everything past this point sees the same structure.
        """
        meta: dict[str, Any] = {
            "source_path": str(source_path) if source_path is not None else None,
            "source_format": source_format,
            "orig_sfreq": float(raw.info["sfreq"]),
            "history": [],
        }
        return cls(
            data=np.asarray(raw.get_data(), dtype=np.float64),
            ch_names=list(raw.ch_names),
            sfreq=float(raw.info["sfreq"]),
            montage=montage,
            reference=reference,
            meta=meta,
        )

    def to_mne(self) -> mne.io.RawArray:
        """Build an MNE ``Raw`` from this container, the escape hatch into full MNE.

        Sensor positions are not stored in the tidy schema; they are reconstructed
        here from the recorded montage name so the result is fully usable for
        plotting, source analysis, or anything else MNE offers.
        """
        info = mne.create_info(list(self.ch_names), self.sfreq, ch_types="eeg", verbose="ERROR")
        raw = mne.io.RawArray(np.array(self.data), info, verbose="ERROR")
        if self.montage is not None:
            montage = mne.channels.make_standard_montage(self.montage)
            raw.set_montage(montage, on_missing="ignore", match_case=False, verbose="ERROR")
        return raw

    def to_numpy(self) -> NDArray[np.float64]:
        """Return a writeable copy of the underlying ``(n_channels, n_times)`` array."""
        return np.array(self.data)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the data in long tidy format with ``channel``, ``time``, ``value`` columns.

        Requires the optional pandas dependency (``pip install tidyeeg[pandas]``).
        """
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - exercised via the message, not import
            raise ImportError(
                "to_dataframe requires pandas. Install it with: pip install tidyeeg[pandas]"
            ) from exc

        return pd.DataFrame(
            {
                "channel": np.repeat(self.ch_names, self.n_times),
                "time": np.tile(self.times, self.n_channels),
                "value": np.asarray(self.data).reshape(-1),
            }
        )

    def __repr__(self) -> str:
        return (
            f"TidyEEG(n_channels={self.n_channels}, n_times={self.n_times}, "
            f"sfreq={self.sfreq:g} Hz, duration={self.duration:g} s, "
            f"montage={self.montage!r}, reference={self.reference!r})"
        )
