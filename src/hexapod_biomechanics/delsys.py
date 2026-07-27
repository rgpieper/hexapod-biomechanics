"""Delsys signal containers and per-trial NPZ I/O.

Defines the dataclasses used to carry Delsys data through the per-trial
pipeline and the loader/saver for per-trial NPZ files (typically located
at `subjects/<subject>/<session>/raw_data/delsys/`).

The block-CSV parser and segmenting logic live in
`scripts/segment_delsys.py`. Both layers use the same `DelsysSignals`
dataclass — only the channel keys differ:

  - block-level (segmenter): raw Delsys labels (e.g. `LTA`, `RANK Analog 1`)
  - trial-level (this module): semantic keys for goniometers
    (`gonio_left_df_pf`, `gonio_left_ie`, `gonio_right_df_pf`,
    `gonio_right_ie`); EMG channels keep raw lab labels.

Per-channel `t` arrays are in seconds-from-trigger-pulse — the same
timebase as Vicon's full-capture timing — so trial-level signals can be
aligned with `TrialResult.stance_start_sample_capture` downstream.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DelsysChannel:
    """One Delsys channel — time + value at a native sample rate.

    Time is always in seconds from the trigger pulse (block t=0). Each
    channel keeps its native rate; resampling to a common grid is the
    consumer's job (e.g. the per-trial pipeline interpolates goniometer
    onto Vicon's analog rate).
    """
    label: str                       # raw or semantic key, see module docstring
    t: npt.NDArray                   # (N,) seconds from trigger pulse
    v: npt.NDArray                   # (N,) signal value
    fs_hz: float                     # native sample rate
    units: str                       # e.g. "mV"
    sensor_id: str | None = None     # e.g. "74309" (Delsys serial)


@dataclass
class DelsysSignals:
    """A bundle of Delsys channels — block-level, static, or trial-level.

    `channels` is keyed by `DelsysChannel.label`. Trial-level fields
    (`trial_stem`, `block_id`, `segment_*`, `static_stem`,
    `gonio_offsets_rad`) are populated when the bundle came from
    segmenting; they are `None` for a fresh block-level bundle.

    For trials whose goniometer channels were zero-corrected against a
    static, `static_stem` records which static was used and
    `gonio_offsets_rad` records exactly what was subtracted per channel —
    so the operation is reversible / inspectable downstream.
    """
    channels: dict[str, DelsysChannel] = field(default_factory=dict)
    # Trial-level metadata (None for block-level signals)
    trial_stem: str | None = None
    block_id: str | None = None
    segment_start_s: float | None = None    # seconds from trigger pulse
    segment_end_s: float | None = None
    # Goniometer zero-offset traceability
    static_stem: str | None = None          # static used as zero reference
    gonio_offsets_rad: dict[str, float] | None = None  # per-channel offset


# ---------------------------------------------------------------------------
# Per-trial NPZ I/O
# ---------------------------------------------------------------------------
#
# NPZ schema (per file, one trial):
#
#   __channels  : (K,) U-string array of channel names (load order)
#   __metadata  : (1,) U-string array, JSON blob with all scalar metadata
#   t__<name>   : (Nk,) float64 time array for channel <name>
#   v__<name>   : (Nk,) float64 value array for channel <name>
#
# The JSON metadata blob contains trial-level fields plus a per-channel
# sub-dict of (fs_hz, units, sensor_id). Keeping scalars/strings in JSON
# avoids round-trip headaches with `np.save`'s 0-d arrays.

_CHANNEL_INDEX_KEY = "__channels"
_METADATA_KEY = "__metadata"


def save_trial_npz(file_path: Path | str, signals: DelsysSignals) -> None:
    """Write a trial-level `DelsysSignals` to NPZ.

    Args:
        file_path: Output path. Parent directory will be created if missing.
        signals: Trial-level signals (typically with semantic goniometer
            channel keys + raw EMG keys).
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    channel_names = list(signals.channels.keys())
    arrays[_CHANNEL_INDEX_KEY] = np.asarray(channel_names)

    for name, ch in signals.channels.items():
        arrays[f"t__{name}"] = ch.t.astype(np.float64, copy=False)
        arrays[f"v__{name}"] = ch.v.astype(np.float64, copy=False)

    metadata = {
        "trial_stem": signals.trial_stem,
        "block_id": signals.block_id,
        "segment_start_s": signals.segment_start_s,
        "segment_end_s": signals.segment_end_s,
        "static_stem": signals.static_stem,
        "gonio_offsets_rad": signals.gonio_offsets_rad,
        "channels": {
            name: {
                "fs_hz": ch.fs_hz,
                "units": ch.units,
                "sensor_id": ch.sensor_id,
            }
            for name, ch in signals.channels.items()
        },
    }
    arrays[_METADATA_KEY] = np.asarray([json.dumps(metadata)])

    np.savez(file_path, **arrays)


def load_trial_npz(file_path: Path | str) -> DelsysSignals:
    """Load a trial-level `DelsysSignals` from NPZ written by `save_trial_npz`.

    Args:
        file_path: Path to a trial NPZ file (e.g.
            `subjects/<subj>/<sess>/raw_data/delsys/<trial_stem>.npz`).

    Returns:
        The reconstructed `DelsysSignals` with semantic channel keys.
    """
    file_path = Path(file_path)
    with np.load(file_path, allow_pickle=False) as data:
        metadata = json.loads(str(data[_METADATA_KEY][0]))
        channel_names = [str(n) for n in data[_CHANNEL_INDEX_KEY]]

        channels: dict[str, DelsysChannel] = {}
        for name in channel_names:
            ch_meta = metadata["channels"][name]
            channels[name] = DelsysChannel(
                label=name,
                t=np.asarray(data[f"t__{name}"]),
                v=np.asarray(data[f"v__{name}"]),
                fs_hz=float(ch_meta["fs_hz"]),
                units=str(ch_meta["units"]),
                sensor_id=ch_meta.get("sensor_id"),
            )

    return DelsysSignals(
        channels=channels,
        trial_stem=metadata.get("trial_stem"),
        block_id=metadata.get("block_id"),
        segment_start_s=metadata.get("segment_start_s"),
        segment_end_s=metadata.get("segment_end_s"),
        static_stem=metadata.get("static_stem"),
        gonio_offsets_rad=metadata.get("gonio_offsets_rad"),
    )
