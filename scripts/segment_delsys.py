"""Segment Delsys block CSVs into per-trial NPZ files.

Self-contained: only `src/hexapod_biomechanics/delsys.py`'s dataclass +
`save_trial_npz` are imported. The block CSV parser, the segment-timing
CSV parser, and the CLI all live in this script. Write a config TOML to set
up a session (see `delsys_batch.template.toml` next to this script), or use
`pair` mode for ad-hoc runs.

Session configs are kept with the data rather than in this script, so no
subject identifiers are hardcoded here.

Usage:

    # Per-pair (single block):
    python scripts/segment_delsys.py pair \\
        --delsys /abs/path/DelsTrig_50pct_block_2.csv \\
        --timing /abs/path/DelsTrig_block02_log.csv \\
        --out subjects/RP041026/15102026/raw_data/delsys \\
        --df-channel 2 --ie-channel 1 \\
        [--static-delsys /abs/path/static_01_delsys.csv \\
         --static-stem static_01]

    # Batch (whole session, driven by a config TOML):
    python scripts/segment_delsys.py batch \\
        --config data/delsys_blocks/RP041026/15102026/segment_config.toml

Static (zero-offset) handling:
- The position of the goniometers during the static trial defines zero
  for any walking trial referenced to that static — mirrors the Vicon
  static calibration in `kinematics.py`.
- Each Delsys block belongs to exactly one static. Specify the
  static→blocks grouping in the config's `[statics.<label>]` tables
  (batch) or pass `--static-delsys` (per-pair). The script computes the
  per-channel
  mean goniometer angle during the static, subtracts it from every
  goniometer channel of trials in that block, and records the offsets
  + static identity in each trial's NPZ for traceability.
- The static is processed only to extract offsets; nothing about it
  lives on disk. `static_stem` is just a label (e.g. `"static_01"`)
  recorded in each zeroed trial's NPZ metadata.
- If `--static-delsys` is omitted in pair mode, no zeroing is applied
  and the NPZs record `static_stem=None`. EMG channels are never
  affected by zeroing.

Goniometer handling:
- **Channel remap (Analog 1/2 → DF/PF / IE)** varies session-to-session with
  the physical wiring. Specify via `--df-channel` / `--ie-channel`
  (per-pair) or the config's `df_channel` / `ie_channel` keys (batch).
- **V → rad conversion + sign** baked in at segmentation. Goniometer
  channels are stored in radians, on the same convention as Vicon's
  `TrialResult.alpha`. Per-channel signed scale (deg/V) controls both
  magnitude and polarity (negative flips sign for a reversed mounting).
  Per-channel CLI overrides: `--gonio-{left,right}-{df-pf,ie}-deg-per-v`;
  in batch mode use the config's `[gonio_deg_per_volt]` table. The scale
  should come from a bench characterisation of the sensor, not from
  regression against Vicon — the goniometer path exists to give a theta
  measurement independent of the marker data. Channels left unspecified
  fall back to `GONIO_DEG_PER_VOLT_DEFAULT` below and print a warning so a
  wrong scale doesn't silently ship.
- Sensors not matching `RANK` / `LANK` are passed through with raw labels
  and original units (mV); per the lab convention only goniometers
  populate Analog channels.

Other notes:
- The block CSV's row 8 ("sampling period") is unreliable for analog
  channels (Delsys docs say sampling period; observed values for goniometer
  channels are −0.0135 s, which doesn't match 1/rate). The parser uses the
  Time Series data columns directly — those timestamps are authoritative
  and already in seconds-from-trigger-pulse.
- Variable sensor sets (e.g. only one goniometer on a unilateral subject,
  or fewer EMG sensors used in some sessions) are handled naturally — the
  parser iterates whatever sensors are in the CSV header.
"""

from __future__ import annotations

import argparse
import csv
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import hexapod_biomechanics  # noqa: F401 — preflight: editable install check
except ModuleNotFoundError:
    raise SystemExit(
        "hexapod_biomechanics not importable — refresh the editable install:\n"
        "  python scripts/install.py"
    )

from hexapod_biomechanics.delsys import (
    DelsysChannel,
    DelsysSignals,
    save_trial_npz,
)


@dataclass
class BatchConfig:
    """One session's segmentation setup, loaded from a TOML file.

    Attributes:
        session_dir: Session directory; NPZs are written to its
            `raw_data/delsys/` subdirectory.
        block_dir: Directory holding the block CSVs, timing logs, and
            static CSVs. All filenames in `statics` resolve against it.
        df_channel: Goniometer Analog channel carrying DF/PF (1 or 2).
        ie_channel: Goniometer Analog channel carrying IE (1 or 2).
        gonio_deg_per_volt: Per-channel signed deg/V scale, keyed by
            semantic name. Negative flips polarity (reversed mounting).
            Channels not listed fall through to GONIO_DEG_PER_VOLT_DEFAULT.
        statics: Maps static label -> {"delsys": Path, "blocks": {block_id:
            (block CSV, timing log)}}. Each block belongs to exactly one
            static; block_id is the trial stem prefix before "_trip".
    """
    session_dir: Path
    block_dir: Path
    df_channel: int
    ie_channel: int
    gonio_deg_per_volt: dict[str, float]
    statics: dict[str, dict]


def load_batch_config(config_path: Path) -> BatchConfig:
    """Read a batch segmentation config TOML.

    See `delsys_batch.template.toml` alongside this script for the schema
    and a worked example. Block CSV names are listed explicitly per block
    rather than derived by rule, because Trigno Discover appends its own
    recording index to the filename and that index rarely lines up with
    the block number the Vicon trials use.

    Args:
        config_path: Path to the TOML config.

    Returns:
        BatchConfig with every filename resolved against `block_dir`.

    Raises:
        ValueError: If a static declares no blocks, or a block's timing log
            or CSV is missing from `block_dir`.
    """
    config_path = Path(config_path)
    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    block_dir = Path(raw["block_dir"])
    statics: dict[str, dict] = {}
    for label, entry in raw.get("statics", {}).items():
        blocks = entry.get("blocks", {})
        if not blocks:
            raise ValueError(
                f"{config_path}: static {label!r} declares no blocks."
            )
        statics[label] = {
            "delsys": block_dir / entry["delsys"],
            "blocks": {
                block_id: (
                    block_dir / csv_name,
                    block_dir / f"{block_id}_log.csv",
                )
                for block_id, csv_name in blocks.items()
            },
        }

    return BatchConfig(
        session_dir=Path(raw["session_dir"]),
        block_dir=block_dir,
        df_channel=int(raw.get("df_channel", 2)),
        ie_channel=int(raw.get("ie_channel", 1)),
        gonio_deg_per_volt=dict(raw.get("gonio_deg_per_volt", {})),
        statics=statics,
    )


# Goniometer sensor names → side. Lab convention; edit if your sensors are
# named differently OR if the sensors were physically mounted on the
# opposite ankle from their label (a common slip — verify with a Vicon vs
# goniometer comparison plot per session before trusting downstream).
GONIO_SENSOR_TO_SIDE = {"RANK": "right", "LANK": "left"}

# Placeholder scale factor for V → deg conversion. TBD per the lab's
# Delsys SG model spec. Override per-channel via CLI flags or the config's
# [gonio_deg_per_volt] table. A runtime warning fires when this default
# is used so it doesn't silently ship.
GONIO_DEG_PER_VOLT_DEFAULT = 100.0

# Semantic goniometer channel keys produced after remap. Used to validate
# CLI overrides and BATCH_GONIO_DEG_PER_VOLT keys.
GONIO_SEMANTIC_KEYS = (
    "gonio_left_df_pf",
    "gonio_left_ie",
    "gonio_right_df_pf",
    "gonio_right_ie",
)


# ---------------------------------------------------------------------------
# Block CSV parsing
# ---------------------------------------------------------------------------

@dataclass
class _BlockMetadata:
    application: str
    datetime: str
    collection_length_s: float | None


_SENSOR_LABEL_RE = re.compile(r"^(.+?)\s*\((\d+)\)\s*$")
_CHANNEL_TYPE_RE = re.compile(r"^(.+?)\s*\((.+?)\)\s*$")
_RATE_RE = re.compile(r"^([\d.]+)\s*Hz")


def parse_block_csv(csv_path: Path) -> tuple[DelsysSignals, _BlockMetadata]:
    """Parse a Delsys Trigno Discover block CSV.

    Returns:
        signals: `DelsysSignals` with raw channel labels. EMG channels keyed
            by sensor name (e.g. `LTA`); Analog channels keyed as
            `<sensor> Analog <N>` (e.g. `RANK Analog 1`).
        metadata: `_BlockMetadata` with file-level info from rows 1-3.
    """
    csv_path = Path(csv_path)
    with csv_path.open() as f:
        rows = list(csv.reader(f))

    if len(rows) < 9:
        raise ValueError(
            f"{csv_path} has only {len(rows)} rows; expected ≥9 "
            "(8 header rows + ≥1 data row)."
        )

    # Rows 1-3 (1-indexed): file-level metadata.
    application = _safe_cell(rows[0], 1)
    datetime_str = _safe_cell(rows[1], 1)
    coll_str = _safe_cell(rows[2], 1)
    try:
        collection_length_s: float | None = float(coll_str) if coll_str else None
    except ValueError:
        collection_length_s = None

    # Rows 4-7 (0-indexed: 3-6): channel layout.
    sensor_label_row = rows[3]
    column_type_row = rows[5]
    rate_row = rows[6]

    # Identify sensor boundaries: each non-empty cell in row 4 starts a
    # sensor; the empty cells that follow are part of that sensor until
    # the next non-empty cell.
    sensor_starts: list[tuple[int, str, str | None]] = []  # (col, name, sid)
    for i, cell in enumerate(sensor_label_row):
        text = cell.strip()
        if not text:
            continue
        m = _SENSOR_LABEL_RE.match(text)
        if m:
            sensor_starts.append((i, m.group(1).strip(), m.group(2)))
        else:
            sensor_starts.append((i, text, None))

    if not sensor_starts:
        raise ValueError(f"{csv_path} row 4 has no sensor labels.")

    # Determine each sensor's column span and parse its channels.
    channels: dict[str, DelsysChannel] = {}
    n_sensors = len(sensor_starts)
    n_cols_total = len(column_type_row)
    for s_idx, (start_col, sensor_name, sensor_id) in enumerate(sensor_starts):
        end_col = (
            sensor_starts[s_idx + 1][0]
            if s_idx + 1 < n_sensors
            else n_cols_total
        )
        sensor_channels = _parse_sensor_channels(
            rows=rows,
            start_col=start_col,
            end_col=end_col,
            sensor_name=sensor_name,
            sensor_id=sensor_id,
            column_type_row=column_type_row,
            rate_row=rate_row,
        )
        # Validate no key collisions across sensors (shouldn't happen, but
        # would silently corrupt data if it did).
        for label in sensor_channels:
            if label in channels:
                raise ValueError(
                    f"Duplicate channel label {label!r} in {csv_path}"
                )
        channels.update(sensor_channels)

    signals = DelsysSignals(channels=channels)
    metadata = _BlockMetadata(
        application=application,
        datetime=datetime_str,
        collection_length_s=collection_length_s,
    )
    return signals, metadata


def _parse_sensor_channels(
    *,
    rows: list[list[str]],
    start_col: int,
    end_col: int,
    sensor_name: str,
    sensor_id: str | None,
    column_type_row: list[str],
    rate_row: list[str],
) -> dict[str, DelsysChannel]:
    """Walk one sensor's columns, pairing (Time Series, value) into channels."""
    out: dict[str, DelsysChannel] = {}
    col = start_col
    while col + 1 < end_col:
        type_t = _safe_cell(column_type_row, col)
        type_v = _safe_cell(column_type_row, col + 1)
        if "Time Series" in type_t and "(s)" in type_t and type_v:
            # Channel name from value-column header, e.g. "Analog 1 (mV)".
            m = _CHANNEL_TYPE_RE.match(type_v)
            if m:
                ch_subname = m.group(1).strip()
                ch_units = m.group(2).strip()
            else:
                ch_subname = type_v
                ch_units = ""

            # Label: bare sensor name for EMG, "<sensor> <subname>" otherwise.
            if ch_subname.startswith("EMG"):
                label = sensor_name
            else:
                label = f"{sensor_name} {ch_subname}"

            # Rate from rate_row at the value column.
            rate_str = _safe_cell(rate_row, col + 1)
            r = _RATE_RE.match(rate_str)
            fs_hz = float(r.group(1)) if r else 0.0

            # Read data column pairs from rows 9+ (0-indexed: 8+).
            t_vals: list[float] = []
            v_vals: list[float] = []
            for row in rows[8:]:
                t_str = _safe_cell(row, col)
                v_str = _safe_cell(row, col + 1)
                if not t_str or not v_str:
                    continue
                try:
                    t_vals.append(float(t_str))
                    v_vals.append(float(v_str))
                except ValueError:
                    continue

            out[label] = DelsysChannel(
                label=label,
                t=np.asarray(t_vals, dtype=np.float64),
                v=np.asarray(v_vals, dtype=np.float64),
                fs_hz=fs_hz,
                units=ch_units,
                sensor_id=sensor_id,
            )
            col += 2
        else:
            # Skip unrecognized / trailing columns (e.g. junk header).
            col += 1
    return out


def _safe_cell(row: list[str], idx: int) -> str:
    """Return row[idx].strip() or '' if out of range."""
    if idx >= len(row):
        return ""
    return row[idx].strip()


# ---------------------------------------------------------------------------
# Segment-timing CSV parsing
# ---------------------------------------------------------------------------

@dataclass
class _Segment:
    name: str
    start_s: float
    end_s: float
    pert_pct: int
    direction: int


def parse_timing_csv(timing_csv: Path) -> list[_Segment]:
    """Parse a segment-timing log CSV (one row per trial in the block)."""
    timing_csv = Path(timing_csv)
    out: list[_Segment] = []
    with timing_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(
                _Segment(
                    name=row["Segment_Name"],
                    start_s=float(row["Start_Time_s"]),
                    end_s=float(row["End_Time_s"]),
                    pert_pct=int(row["Pert_Percent"]),
                    direction=int(row["Direction"]),
                )
            )
    return out


# ---------------------------------------------------------------------------
# Segmentation primitive
# ---------------------------------------------------------------------------

def compute_static_offsets(
    static_csv: Path,
    df_channel: int = 2,
    ie_channel: int = 1,
    gonio_deg_per_volt: dict[str, float] | None = None,
) -> dict[str, float]:
    """Compute per-channel goniometer zero offsets from a static Delsys CSV.

    The entire static CSV is treated as one continuous static measurement
    (no segment-timing CSV needed). Goniometer channels are V→rad
    converted with the same parameters used for block segmentation, then
    averaged across the whole capture to produce a per-channel zero
    offset (in radians). Pure helper: no filesystem side effects.

    Args:
        static_csv: Static Delsys CSV exported by Trigno Discover.
        df_channel: DF/PF Analog channel (must match the block segmenter).
        ie_channel: IE Analog channel (must match the block segmenter).
        gonio_deg_per_volt: Same scale/sign overrides as `segment_block`
            (must match block segmentation for offsets to be meaningful).

    Returns:
        Dict mapping each goniometer channel name to its mean angle (rad)
        over the static capture. Non-goniometer channels are not included.
    """
    overrides = dict(gonio_deg_per_volt or {})
    bad_keys = set(overrides) - set(GONIO_SEMANTIC_KEYS)
    if bad_keys:
        raise ValueError(
            f"Unknown gonio_deg_per_volt keys: {sorted(bad_keys)}. "
            f"Valid keys: {GONIO_SEMANTIC_KEYS}"
        )

    static_block, _meta = parse_block_csv(Path(static_csv))
    static_block = _remap_and_convert_goniometers(
        static_block, df_channel, ie_channel, overrides
    )

    # Per-channel mean over the entire static capture. Only goniometer
    # channels (semantic keys) get offsets; EMG passes through.
    return {
        label: float(np.mean(ch.v))
        for label, ch in static_block.channels.items()
        if label in GONIO_SEMANTIC_KEYS
    }


def segment_block(
    delsys_csv: Path,
    timing_csv: Path,
    output_dir: Path,
    df_channel: int = 2,
    ie_channel: int = 1,
    gonio_deg_per_volt: dict[str, float] | None = None,
    gonio_offsets_rad: dict[str, float] | None = None,
    static_stem: str | None = None,
) -> list[Path]:
    """Segment one block's Delsys data into per-trial NPZs.

    Args:
        delsys_csv: Block CSV exported by Trigno Discover.
        timing_csv: Matching segment-timing log CSV.
        output_dir: Per-trial NPZs land here. Created if missing.
        df_channel: Which goniometer Analog channel is DF/PF (1 or 2).
        ie_channel: Which goniometer Analog channel is IE (1 or 2).
        gonio_deg_per_volt: Per-channel signed deg/V scale, keyed by
            semantic name (e.g. `gonio_left_df_pf`). Negative values flip
            sign (reversed mounting). Channels not listed fall through to
            `GONIO_DEG_PER_VOLT_DEFAULT` (with a warning).
        gonio_offsets_rad: Per-channel zero offsets (radians) to subtract
            from goniometer channels — typically from `compute_static_offsets`.
            Channels not listed get no offset. If None, no zeroing applied.
        static_stem: Identity of the static used for zeroing (recorded in
            each trial's NPZ for traceability).

    Returns:
        List of output NPZ paths, one per segment.
    """
    if df_channel == ie_channel:
        raise ValueError(
            f"df_channel and ie_channel must differ (got {df_channel})"
        )
    if df_channel not in (1, 2) or ie_channel not in (1, 2):
        raise ValueError("df_channel and ie_channel must each be 1 or 2")
    overrides = dict(gonio_deg_per_volt or {})
    bad_keys = set(overrides) - set(GONIO_SEMANTIC_KEYS)
    if bad_keys:
        raise ValueError(
            f"Unknown gonio_deg_per_volt keys: {sorted(bad_keys)}. "
            f"Valid keys: {GONIO_SEMANTIC_KEYS}"
        )
    offsets = dict(gonio_offsets_rad or {})
    bad_offset_keys = set(offsets) - set(GONIO_SEMANTIC_KEYS)
    if bad_offset_keys:
        raise ValueError(
            f"Unknown gonio_offsets_rad keys: {sorted(bad_offset_keys)}. "
            f"Valid keys: {GONIO_SEMANTIC_KEYS}"
        )

    delsys_csv = Path(delsys_csv)
    timing_csv = Path(timing_csv)
    output_dir = Path(output_dir)

    block, _meta = parse_block_csv(delsys_csv)
    block = _remap_and_convert_goniometers(
        block, df_channel, ie_channel, overrides
    )
    if offsets:
        block = _apply_gonio_offsets(block, offsets)
    segments = parse_timing_csv(timing_csv)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_paths: list[Path] = []
    for seg in segments:
        trial = _slice_signals(block, seg.start_s, seg.end_s)
        trial.trial_stem = seg.name
        trial.block_id = _block_id_from_stem(seg.name)
        trial.segment_start_s = seg.start_s
        trial.segment_end_s = seg.end_s
        if offsets:
            trial.static_stem = static_stem
            # Record only offsets for channels actually present in this trial
            trial.gonio_offsets_rad = {
                k: v for k, v in offsets.items() if k in trial.channels
            }
        out_path = output_dir / f"{seg.name}.npz"
        save_trial_npz(out_path, trial)
        out_paths.append(out_path)
    return out_paths


def _apply_gonio_offsets(
    block: DelsysSignals, offsets: dict[str, float]
) -> DelsysSignals:
    """Subtract per-channel offset (rad) from goniometer channels in-place.

    Returns a new DelsysSignals with corrected goniometer channels;
    non-goniometer channels and channels without an offset pass through.
    """
    new_channels: dict[str, DelsysChannel] = {}
    for label, ch in block.channels.items():
        if label in offsets:
            new_channels[label] = DelsysChannel(
                label=label,
                t=ch.t,
                v=ch.v - offsets[label],
                fs_hz=ch.fs_hz,
                units=ch.units,
                sensor_id=ch.sensor_id,
            )
        else:
            new_channels[label] = ch
    return DelsysSignals(channels=new_channels)


def _remap_and_convert_goniometers(
    block: DelsysSignals,
    df_channel: int,
    ie_channel: int,
    deg_per_volt: dict[str, float],
) -> DelsysSignals:
    """Rename `<RANK|LANK> Analog <N>` keys to semantic names AND convert
    voltage to angle (radians).

    Conversion: `v_rad = v_volts * deg_per_volt * pi/180`. Negative
    `deg_per_volt` flips polarity. Channels missing from `deg_per_volt`
    fall through to `GONIO_DEG_PER_VOLT_DEFAULT` (with a one-time warning
    per channel so the placeholder doesn't silently ship).

    Other Analog channels (sensors not in `GONIO_SENSOR_TO_SIDE`) pass
    through unchanged with raw labels and units.
    """
    new_channels: dict[str, DelsysChannel] = {}
    warned: set[str] = set()
    for label, ch in block.channels.items():
        m = re.match(r"^(\w+)\s+Analog\s+([12])$", label)
        if m and m.group(1) in GONIO_SENSOR_TO_SIDE:
            sensor, ch_num = m.group(1), int(m.group(2))
            side = GONIO_SENSOR_TO_SIDE[sensor]
            if ch_num == df_channel:
                new_label = f"gonio_{side}_df_pf"
            elif ch_num == ie_channel:
                new_label = f"gonio_{side}_ie"
            else:
                # df_channel/ie_channel are validated to cover {1, 2}, so
                # this branch is unreachable. Keep raw label as a safety net.
                new_label = label

            if new_label in deg_per_volt:
                scale = deg_per_volt[new_label]
            else:
                scale = GONIO_DEG_PER_VOLT_DEFAULT
                if new_label not in warned:
                    print(
                        f"  WARNING: {new_label} using placeholder scale "
                        f"{GONIO_DEG_PER_VOLT_DEFAULT} deg/V. Verify per "
                        f"sensor spec and override via "
                        f"--{new_label.replace('_', '-')}-deg-per-v "
                        f"or BATCH_GONIO_DEG_PER_VOLT."
                    )
                    warned.add(new_label)

            v_rad = ch.v * scale * (np.pi / 180.0)
            new_channels[new_label] = DelsysChannel(
                label=new_label,
                t=ch.t,
                v=v_rad,
                fs_hz=ch.fs_hz,
                units="rad",
                sensor_id=ch.sensor_id,
            )
        else:
            new_channels[label] = ch
    return DelsysSignals(channels=new_channels)


def _slice_signals(
    signals: DelsysSignals, start_s: float, end_s: float
) -> DelsysSignals:
    """Slice each channel's (t, v) to [start_s, end_s] inclusive."""
    sliced: dict[str, DelsysChannel] = {}
    for label, ch in signals.channels.items():
        mask = (ch.t >= start_s) & (ch.t <= end_s)
        sliced[label] = DelsysChannel(
            label=label,
            t=ch.t[mask],
            v=ch.v[mask],
            fs_hz=ch.fs_hz,
            units=ch.units,
            sensor_id=ch.sensor_id,
        )
    return DelsysSignals(channels=sliced)


def _block_id_from_stem(trial_stem: str) -> str:
    """Trial stem prefix before `_trip`. e.g. `DelsTrig_block02_trip1_nom_fwd`
    → `DelsTrig_block02`."""
    return trial_stem.split("_trip")[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _gonio_overrides_from_args(args: argparse.Namespace) -> dict[str, float]:
    """Pull `--<key>-deg-per-v` CLI args into a `gonio_deg_per_volt` dict."""
    overrides: dict[str, float] = {}
    for key in GONIO_SEMANTIC_KEYS:
        attr = key + "_deg_per_v"
        val = getattr(args, attr, None)
        if val is not None:
            overrides[key] = val
    return overrides


def _cli_pair(args: argparse.Namespace) -> None:
    overrides = _gonio_overrides_from_args(args)

    offsets: dict[str, float] = {}
    static_stem: str | None = None
    if args.static_delsys is not None:
        static_stem = args.static_stem or args.static_delsys.stem
        offsets = compute_static_offsets(
            static_csv=args.static_delsys,
            df_channel=args.df_channel,
            ie_channel=args.ie_channel,
            gonio_deg_per_volt=overrides,
        )
        print(f"Static {static_stem!r} offsets:")
        for k, v in offsets.items():
            print(f"  {k}: {np.degrees(v):+.2f} deg")

    out_paths = segment_block(
        delsys_csv=args.delsys,
        timing_csv=args.timing,
        output_dir=args.out,
        df_channel=args.df_channel,
        ie_channel=args.ie_channel,
        gonio_deg_per_volt=overrides,
        gonio_offsets_rad=offsets or None,
        static_stem=static_stem,
    )
    print(f"Wrote {len(out_paths)} per-trial NPZ files to {args.out}")
    for p in out_paths[:3]:
        print(f"  {p.name}")
    if len(out_paths) > 3:
        print(f"  ... ({len(out_paths) - 3} more)")


def _cli_batch(args: argparse.Namespace) -> None:
    cfg = load_batch_config(args.config)

    out_dir = cfg.session_dir / "raw_data" / "delsys"
    print(f"Batch config: {args.config}")
    print(f"Session dir: {cfg.session_dir}")
    print(
        f"Goniometer mapping: df=Analog{cfg.df_channel}  "
        f"ie=Analog{cfg.ie_channel}"
    )
    print(f"Output dir: {out_dir}")

    # Validate: each block_id appears under at most one static.
    seen_blocks: dict[str, str] = {}
    for static_stem, static_cfg in cfg.statics.items():
        for block_id in static_cfg.get("blocks", {}):
            if block_id in seen_blocks:
                raise SystemExit(
                    f"Block {block_id} listed under both "
                    f"{seen_blocks[block_id]!r} and {static_stem!r}. "
                    f"Each block must reference exactly one static."
                )
            seen_blocks[block_id] = static_stem

    # Fail before writing anything if a configured CSV is missing.
    missing = [
        p
        for static_cfg in cfg.statics.values()
        for pair in static_cfg.get("blocks", {}).values()
        for p in pair
        if not p.exists()
    ] + [
        static_cfg["delsys"]
        for static_cfg in cfg.statics.values()
        if not static_cfg["delsys"].exists()
    ]
    if missing:
        raise SystemExit(
            "Missing input files:\n"
            + "\n".join(f"  {p}" for p in missing)
        )

    total_blocks = 0
    total_trials = 0
    for static_stem, static_cfg in cfg.statics.items():
        static_csv = static_cfg["delsys"]
        blocks = static_cfg.get("blocks", {})
        print(f"\n=== {static_stem} ({static_csv.name}) ===")
        offsets = compute_static_offsets(
            static_csv=static_csv,
            df_channel=cfg.df_channel,
            ie_channel=cfg.ie_channel,
            gonio_deg_per_volt=cfg.gonio_deg_per_volt,
        )
        for k, v in offsets.items():
            print(f"  {k}: {np.degrees(v):+.2f} deg")

        for block_id, (delsys_csv, timing_csv) in blocks.items():
            print(f"\n[{block_id}] {delsys_csv.name} + {timing_csv.name}")
            out_paths = segment_block(
                delsys_csv=delsys_csv,
                timing_csv=timing_csv,
                output_dir=out_dir,
                df_channel=cfg.df_channel,
                ie_channel=cfg.ie_channel,
                gonio_deg_per_volt=cfg.gonio_deg_per_volt,
                gonio_offsets_rad=offsets,
                static_stem=static_stem,
            )
            print(f"  → {len(out_paths)} trials")
            total_blocks += 1
            total_trials += len(out_paths)
    print(
        f"\nDone. {len(cfg.statics)} statics + {total_blocks} blocks → "
        f"{total_trials} per-trial NPZs in {out_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)

    p_pair = sub.add_parser(
        "pair", help="Segment one (delsys CSV, timing CSV) pair."
    )
    p_pair.add_argument("--delsys", type=Path, required=True,
                        help="Path to Delsys block CSV.")
    p_pair.add_argument("--timing", type=Path, required=True,
                        help="Path to matching segment-timing CSV.")
    p_pair.add_argument("--out", type=Path, required=True,
                        help="Output directory for per-trial NPZs.")
    p_pair.add_argument("--df-channel", type=int, default=2, choices=(1, 2),
                        help="Goniometer Analog channel for DF/PF (default 2).")
    p_pair.add_argument("--ie-channel", type=int, default=1, choices=(1, 2),
                        help="Goniometer Analog channel for IE (default 1).")
    p_pair.add_argument(
        "--static-delsys", type=Path, default=None,
        help=("Optional path to a static Delsys CSV. When provided, the "
              "static is processed first (saved as <static_stem>.npz) and "
              "its per-channel mean goniometer angle is subtracted from "
              "every trial in this block."),
    )
    p_pair.add_argument(
        "--static-stem", type=str, default=None,
        help=("NPZ filename stem for the static (default: stem of "
              "--static-delsys path)."),
    )
    # Per-channel signed deg/V overrides. Negative flips polarity.
    for key in GONIO_SEMANTIC_KEYS:
        p_pair.add_argument(
            f"--{key.replace('_', '-')}-deg-per-v",
            dest=f"{key}_deg_per_v",
            type=float,
            default=None,
            help=(
                f"Signed deg/V scale for {key} (negative flips sign). "
                f"Default: GONIO_DEG_PER_VOLT_DEFAULT "
                f"({GONIO_DEG_PER_VOLT_DEFAULT})."
            ),
        )
    p_pair.set_defaults(func=_cli_pair)

    p_batch = sub.add_parser(
        "batch",
        help="Segment every block for a session described by a config TOML.",
    )
    p_batch.add_argument(
        "--config", type=Path, required=True,
        help=("Path to a batch config TOML. See delsys_batch.template.toml "
              "next to this script for the schema."),
    )
    p_batch.set_defaults(func=_cli_batch)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
