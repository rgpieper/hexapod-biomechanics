"""Session config: per-session parameters backed by TOML with auto-detection.

The biomechanics pipeline operates on c3d files where naming conventions
(marker prefixes, Vicon device IDs, accelerometer mode) vary per session. This
module centralizes all such variability into a `SessionConfig` dataclass.

Lab-wide defaults (standard anatomical marker names, hexapod marker names) are
module constants — not duplicated per session. Per-session TOML files only
capture what varies.

Typical workflow:

    # One-time per session, pointing at the session directory (which contains
    # a raw_data/ subdir of c3d files):
    config = build_session_config(
        session_dir="subjects/C0111085/11082025",
        subject_id="C0111085",
        body_mass=70.0,
        sex="m",
    )
    save_session_config(config, "subjects/C0111085/11082025/session.toml")

    # Each analysis run:
    config = load_session_config("subjects/C0111085/11082025/session.toml")

Static blocks: `build_session_config` scans `session_dir/raw_data/` for
`static*.c3d` files and groups all non-static c3ds under the *first* static by
default. Users reassign trials to other statics by cut/pasting between the
`[static_blocks]` entries in the generated TOML.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
import tomllib

import numpy as np
import tomli_w

from hexapod_biomechanics.load_data import C3DMan

# ---------------------------------------------------------------------------
# Lab defaults — applied across all Neurobionics subjects
# ---------------------------------------------------------------------------

# Standard anatomical marker names (without subject-ID prefix), per side.
# Subjects who deviate from this set capture overrides in their session config.
STANDARD_ANKLE_MARKERS = {
    "right": {
        "medial_malleolus": "RMANK",
        "lateral_malleolus": "RANK",
        "medial_condyle": "RMKNE",
        "lateral_condyle": "RKNE",
        "calcaneus": "RHEE",
        "first_metatarsal": "RMT1",
        "fifth_metatarsal": "RMT5",
        "shank_cluster": ["RTIBA", "RTIBB", "RTIBC"],
    },
    "left": {
        "medial_malleolus": "LMANK",
        "lateral_malleolus": "LANK",
        "medial_condyle": "LMKNE",
        "lateral_condyle": "LKNE",
        "calcaneus": "LHEE",
        "first_metatarsal": "LMT1",
        "fifth_metatarsal": "LMT5",
        "shank_cluster": ["LTIBA", "LTIBB", "LTIBC"],
    },
}

# Unprefixed hexapod marker names (subject session config supplies the prefix).
HEXAPOD_MARKERS = ["Hexapod00", "Hexapod10", "Hexapod11"]

# The 8 raw Kistler channel labels (stable across all sessions).
KISTLER_RAW_CHANNELS = [
    "Raw.FX12", "Raw.FX34", "Raw.FY14", "Raw.FY23",
    "Raw.FZ1", "Raw.FZ2", "Raw.FZ3", "Raw.FZ4",
]

# Regex patterns used during auto-detection.
_KISTLER_RAW_DESC_RE = re.compile(r"^Kistler Force Plate [\d.]+::Raw \[(\d+)\]$")
_ACCEL_ACCEL_DESC_RE = re.compile(r"^Analog Accelerometer::Acceleration \[(\d+),\d\]$")
_GENERIC_VOLT_DESC_RE = re.compile(r"^Generic Analog::Electric Potential \[(\d+),\d\]$")


# ---------------------------------------------------------------------------
# SessionConfig
# ---------------------------------------------------------------------------

@dataclass
class SessionConfig:
    """Per-session parameters for loading and processing c3d files.

    Fields grouped into subject metadata, session metadata, auto-detected
    Vicon parameters, and optional marker overrides.

    Attributes:
        subject_id (str): Subject ID used as marker prefix in c3d files
            (e.g. "C0111085").
        sex (str): "m" or "f" — used for anthropometric inertial parameters.
        body_mass (float): Subject body mass in kg.
        session_id (str): Session identifier (matches data folder name — for
            this lab it's typically the session date as DDMMYYYY, but any
            unique string works).
        hexapod_prefix (str): Case-sensitive hexapod marker prefix in c3d
            (usually "hexapod", occasionally "Hexapod").
        hexapod_kistler_device_id (int): Vicon device ID of the hexapod
            Kistler's raw data block, i.e. the N in "[Kistler Force Plate
            2.0.0.0::Raw [N]]".
        accel_mode (str): "accelerometer" if accelerometers are exported as
            m/s^2, "generic_analog" if exported as raw voltage.
        accel_device_ids (list[int]): Vicon device IDs of the 4 accelerometer
            sensors (e.g. [19, 20, 21, 22] or [33, 34, 35, 36]).
        marker_overrides (dict): Optional per-side overrides when subject
            deviates from STANDARD_ANKLE_MARKERS. Structure:
            {"right": {"medial_malleolus": "RMANK2"}, "left": {...}}.
            Only carries non-default marker names; missing markers are not
            representable here (they're warned about at config build time).
        static_blocks (dict): Mapping of static trial stem → list of dynamic
            trial stems grouped under it. Each static c3d defines one static
            block; dynamic trials listed under a static are processed using
            its AnkleFrame/AnkleID solvers. Stems are c3d filenames without
            the `.c3d` extension. Default (from `build_session_config`): all
            dynamic trials grouped under the last static; user re-groups by
            editing the TOML.
        direction_side_map (dict): Mapping of trial-filename direction token
            (e.g. "fwd", "rev") → ankle side ("right" or "left") that the
            trial targets. Default `{"fwd": "right", "rev": "left"}`. Override
            per-session (e.g. amputee with all trips on the prosthesis side:
            `{"fwd": "right", "rev": "right"}`).
        pert_axis (list[float]): Expected perturbation rotation axis in the
            global frame [x, y, z]. Used by HexKistler to sign the hexapod
            angular trajectory. Default [1, 0, 0] (global x-axis).
    """

    # Subject metadata
    subject_id: str
    sex: str
    body_mass: float

    # Session metadata
    session_id: str

    # Auto-detected Vicon parameters
    hexapod_prefix: str
    hexapod_kistler_device_id: int

    # Accelerometer
    accel_mode: str
    accel_device_ids: list[int]

    # Marker overrides
    marker_overrides: dict = field(default_factory=dict)

    # Trial grouping
    static_blocks: dict[str, list[str]] = field(default_factory=dict)

    # Filename direction token → ankle side
    direction_side_map: dict[str, str] = field(
        default_factory=lambda: {"fwd": "right", "rev": "left"}
    )

    # Expected perturbation rotation axis in global frame (unit vector).
    # Used by HexKistler to sign the hexapod angular trajectory.
    # Default [1, 0, 0] = global x-axis (dorsiflexion perturbation).
    pert_axis: list[float] = field(
        default_factory=lambda: [1.0, 0.0, 0.0]
    )

    # ---- Resolved label helpers ----

    def ankle_marker_labels(self, side: str) -> dict:
        """Resolve anatomical marker names to full c3d labels for one side.

        Args:
            side (str): "right" or "left".

        Returns:
            dict: Mapping of anatomical role → full c3d label (prefixed with
                subject ID). Cluster roles (e.g. shank_cluster) map to a list
                of labels; single-marker roles map to one label.

        Raises:
            ValueError: If side is not "right" or "left".
        """
        if side not in ("right", "left"):
            raise ValueError(f"side must be 'right' or 'left', got {side!r}")

        standard = STANDARD_ANKLE_MARKERS[side]
        overrides = self.marker_overrides.get(side, {})
        resolved = {}
        for role, default_name in standard.items():
            name = overrides.get(role, default_name)
            if isinstance(name, list):
                # Cluster markers: list of names
                resolved[role] = [f"{self.subject_id}:{n}" for n in name]
            else:
                resolved[role] = f"{self.subject_id}:{name}"
        return resolved

    def hexapod_marker_labels(self) -> list[str]:
        """Return the 3 full hexapod marker labels (with prefix)."""
        return [f"{self.hexapod_prefix}:{m}" for m in HEXAPOD_MARKERS]

    def kistler_raw_analog_pairs(self) -> list[tuple[str, str]]:
        """Return the 8 (description, label) pairs for hexapod Kistler raw data.

        Suitable for passing directly to C3DMan.get_analogs.
        """
        desc = f"Kistler Force Plate 2.0.0.0::Raw [{self.hexapod_kistler_device_id}]"
        return [(desc, ch) for ch in KISTLER_RAW_CHANNELS]

    def accel_analog_pairs(self) -> list[tuple[str, str]]:
        """Return the 12 (description, label) pairs for accelerometer data.

        Ordered sensor-by-sensor, axis-by-axis (X, Y, Z).
        """
        if self.accel_mode == "accelerometer":
            desc_template = "Analog Accelerometer::Acceleration [{id},{axis}]"
            labels = ["Acceleration.X", "Acceleration.Y", "Acceleration.Z"]
        elif self.accel_mode == "generic_analog":
            desc_template = "Generic Analog::Electric Potential [{id},{axis}]"
            labels = ["Electric Potential.X", "Electric Potential.Y", "Electric Potential.Z"]
        else:
            raise ValueError(f"Unknown accel_mode: {self.accel_mode!r}")

        pairs = []
        for sensor_id in self.accel_device_ids:
            for axis_idx, label in enumerate(labels, start=1):
                pairs.append(
                    (desc_template.format(id=sensor_id, axis=axis_idx), label)
                )
        return pairs


# ---------------------------------------------------------------------------
# Auto-detection helpers
# ---------------------------------------------------------------------------

def _detect_hexapod_prefix(c3d_man: C3DMan) -> str:
    """Detect whether hexapod markers use 'hexapod' or 'Hexapod' prefix."""
    labels = c3d_man.point_map.index.get_level_values("Label")
    for lab in labels:
        if ":Hexapod00" in lab:
            prefix = lab.split(":Hexapod00")[0]
            if prefix in ("hexapod", "Hexapod"):
                return prefix
    raise ValueError(
        "No hexapod marker found in c3d — expected a label ending in "
        "':Hexapod00' (with 'hexapod' or 'Hexapod' prefix)."
    )


def _detect_hexapod_kistler_id(c3d_man: C3DMan) -> int:
    """Identify the hexapod Kistler's Vicon device ID via signal energy.

    When multiple Kistler raw data blocks are exported, only the hexapod's
    Kistler sees body-weight loading during a trip. This function computes
    the RMS of the four vertical (FZ) raw channels per candidate block and
    returns the device ID with the largest RMS.
    """
    candidate_ids = []
    descriptions = c3d_man.analog_map.index.get_level_values("Description").unique()
    for desc in descriptions:
        m = _KISTLER_RAW_DESC_RE.match(desc)
        if m:
            candidate_ids.append(int(m.group(1)))

    if not candidate_ids:
        raise ValueError(
            "No Kistler raw data block found in c3d — expected a description "
            "matching 'Kistler Force Plate ...::Raw [N]'."
        )

    if len(candidate_ids) == 1:
        return candidate_ids[0]

    # Multiple candidates — pick the one with loaded force signal.
    fz_channels = [ch for ch in KISTLER_RAW_CHANNELS if ch.startswith("Raw.FZ")]
    rms_by_id = {}
    for dev_id in candidate_ids:
        desc = f"Kistler Force Plate 2.0.0.0::Raw [{dev_id}]"
        pairs = [(desc, ch) for ch in fz_channels]
        signals = c3d_man.get_analogs(pairs).values
        # Remove DC offset (unloaded plates still have a baseline) before RMS.
        detrended = signals - signals.mean(axis=0, keepdims=True)
        rms_by_id[dev_id] = float(np.sqrt(np.mean(detrended**2)))

    best_id = max(rms_by_id, key=rms_by_id.get)
    return best_id


def _detect_accel(c3d_man: C3DMan) -> tuple[str, list[int]]:
    """Detect accelerometer export mode and the 4 sensor device IDs.

    Returns:
        (mode, device_ids): mode is "accelerometer" or "generic_analog";
        device_ids is sorted list of 4 Vicon device IDs.
    """
    descriptions = c3d_man.analog_map.index.get_level_values("Description").unique()

    # First pass: look for true Accelerometer-type devices.
    accel_ids = set()
    for desc in descriptions:
        m = _ACCEL_ACCEL_DESC_RE.match(desc)
        if m:
            accel_ids.add(int(m.group(1)))
    if accel_ids:
        return "accelerometer", sorted(accel_ids)

    # Second pass: look for Generic Analog devices with X/Y/Z channel labels.
    # Need to filter out unrelated generic-analog devices that happen to share
    # the description prefix but aren't accelerometer triads (e.g. a trigger).
    volt_ids_with_xyz = set()
    for (desc, label), _ in c3d_man.analog_map.iterrows():
        m = _GENERIC_VOLT_DESC_RE.match(desc)
        if m and label in ("Electric Potential.X", "Electric Potential.Y", "Electric Potential.Z"):
            volt_ids_with_xyz.add(int(m.group(1)))

    if len(volt_ids_with_xyz) == 4:
        return "generic_analog", sorted(volt_ids_with_xyz)

    raise ValueError(
        "Could not auto-detect accelerometers: expected either 4 'Analog "
        "Accelerometer::Acceleration' devices or 4 'Generic Analog::Electric "
        f"Potential' devices with X/Y/Z channels. Found: accelerometer_mode="
        f"{sorted(accel_ids)}, generic_analog_xyz_mode={sorted(volt_ids_with_xyz)}."
    )


def _check_marker_availability(c3d_man: C3DMan, subject_id: str) -> dict:
    """Check which standard anatomical markers are missing for this subject.

    Used only to print a warning at config build time — missing markers are
    not stored in the TOML config (TOML has no null value). If a subject is
    genuinely missing a marker, the caller must either add a marker_overrides
    entry pointing at an alternative, or accept that downstream processing
    will fail.

    Returns:
        dict: {"right": [missing_roles], "left": [missing_roles]} with entries
            only for sides that have missing roles. Empty dict means all
            standard markers present.
    """
    present_labels = set(c3d_man.point_map.index.get_level_values("Label"))
    missing = {"right": [], "left": []}
    for side, roles in STANDARD_ANKLE_MARKERS.items():
        for role, name in roles.items():
            if isinstance(name, list):
                # Cluster markers — flag if any are missing
                cluster_missing = [n for n in name if f"{subject_id}:{n}" not in present_labels]
                if cluster_missing:
                    missing[side].append(f"{role} (missing: {cluster_missing})")
            else:
                if f"{subject_id}:{name}" not in present_labels:
                    missing[side].append(f"{role} ({name})")
    return {side: m for side, m in missing.items() if m}


# ---------------------------------------------------------------------------
# Build / load / save
# ---------------------------------------------------------------------------

def _scan_session_trials(raw_data_dir: Path) -> tuple[list[str], list[str]]:
    """Scan `raw_data_dir` for c3d files and split into (statics, dynamics).

    A c3d file is treated as a static if its stem starts with 'static'; all
    other c3ds are treated as dynamic trials. Both lists are sorted by stem.
    """
    statics, dynamics = [], []
    for c3d_path in sorted(raw_data_dir.glob("*.c3d")):
        stem = c3d_path.stem
        if stem.lower().startswith("static"):
            statics.append(stem)
        else:
            dynamics.append(stem)
    return statics, dynamics


def build_session_config(
    session_dir: str | Path,
    subject_id: str,
    body_mass: float,
    sex: str,
    session_id: str | None = None,
) -> SessionConfig:
    """Build a SessionConfig by scanning a session dir and auto-detecting params.

    Finds all `*.c3d` files under `session_dir/raw_data/`, identifies statics
    by filename (stems starting with 'static'), and groups all dynamic trials
    under the *first* static by default. The first static is also used as the
    sample for Vicon parameter auto-detection.

    Args:
        session_dir (str | Path): Path to the session directory (which
            contains a `raw_data/` subdir of c3d files), e.g.
            "subjects/C0111085/11082025".
        subject_id (str): Subject ID (the marker prefix, e.g. "C0111085").
        body_mass (float): Subject body mass in kg.
        sex (str): "m" or "f".
        session_id (str | None): Session identifier. Defaults to
            `session_dir.name` (e.g. "11082025").

    Returns:
        SessionConfig: Populated config with auto-detected Vicon parameters
            and a default static-block grouping.

    Raises:
        ValueError: If no static c3d files are found in `raw_data/`.
    """
    session_dir = Path(session_dir)
    raw_data_dir = session_dir / "raw_data"
    statics, dynamics = _scan_session_trials(raw_data_dir)
    if not statics:
        raise ValueError(
            f"No static*.c3d files found in {raw_data_dir}. Expected at least "
            f"one file whose stem starts with 'static'."
        )

    # Default grouping: all dynamics under the first static, later statics empty.
    static_blocks: dict[str, list[str]] = {s: [] for s in statics}
    static_blocks[statics[0]] = list(dynamics)

    sample = raw_data_dir / f"{statics[0]}.c3d"
    c3d_man = C3DMan(str(sample))

    hexapod_prefix = _detect_hexapod_prefix(c3d_man)
    hexapod_kistler_device_id = _detect_hexapod_kistler_id(c3d_man)
    accel_mode, accel_device_ids = _detect_accel(c3d_man)

    missing = _check_marker_availability(c3d_man, subject_id)
    if missing:
        print(
            f"WARNING: subject {subject_id} is missing some standard markers:\n"
            f"  {missing}\n"
            f"  Add a [markers.<side>] override in the config, or expect "
            f"downstream processing to fail."
        )

    return SessionConfig(
        subject_id=subject_id,
        sex=sex,
        body_mass=body_mass,
        session_id=session_id or session_dir.name,
        hexapod_prefix=hexapod_prefix,
        hexapod_kistler_device_id=hexapod_kistler_device_id,
        accel_mode=accel_mode,
        accel_device_ids=accel_device_ids,
        marker_overrides={},
        static_blocks=static_blocks,
    )


def save_session_config(config: SessionConfig, path: str | Path) -> None:
    """Write a SessionConfig to a TOML file.

    The TOML structure groups fields into [subject], [session], [vicon],
    [accelerometer], and [markers] sections for readability.
    """
    doc = {
        "subject": {
            "id": config.subject_id,
            "sex": config.sex,
            "body_mass": config.body_mass,
        },
        "session": {
            "id": config.session_id,
        },
        "vicon": {
            "hexapod_prefix": config.hexapod_prefix,
            "hexapod_kistler_device_id": config.hexapod_kistler_device_id,
        },
        "accelerometer": {
            "mode": config.accel_mode,
            "device_ids": config.accel_device_ids,
        },
        "direction_side_map": config.direction_side_map,
        "hexapod": {
            "pert_axis": config.pert_axis,
        },
    }
    if config.marker_overrides:
        doc["markers"] = config.marker_overrides
    if config.static_blocks:
        doc["static_blocks"] = config.static_blocks

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(doc, f)


def load_session_config(path: str | Path) -> SessionConfig:
    """Load a SessionConfig from a TOML file."""
    with open(path, "rb") as f:
        doc = tomllib.load(f)

    return SessionConfig(
        subject_id=doc["subject"]["id"],
        sex=doc["subject"]["sex"],
        body_mass=doc["subject"]["body_mass"],
        session_id=doc["session"]["id"],
        hexapod_prefix=doc["vicon"]["hexapod_prefix"],
        hexapod_kistler_device_id=doc["vicon"]["hexapod_kistler_device_id"],
        accel_mode=doc["accelerometer"]["mode"],
        accel_device_ids=list(doc["accelerometer"]["device_ids"]),
        marker_overrides=doc.get("markers", {}),
        static_blocks={k: list(v) for k, v in doc.get("static_blocks", {}).items()},
        direction_side_map=dict(
            doc.get("direction_side_map", {"fwd": "right", "rev": "left"})
        ),
        pert_axis=list(
            doc.get("hexapod", {}).get("pert_axis", [1.0, 0.0, 0.0])
        ),
    )
