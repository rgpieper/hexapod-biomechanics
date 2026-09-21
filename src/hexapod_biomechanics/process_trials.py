"""Per-trial ankle biomechanics processing.

Loads a trial's markers + raw Kistler forces + accelerometer signals from its
c3d, subtracts MLP-predicted inertial forces from the raw Kistler channels,
then runs kinematics + dynamics. Per-condition HDF5 files contain all trials
for that condition.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

import h5py
import numpy as np
import numpy.typing as npt
from scipy.constants import g
from scipy.interpolate import interp1d

from hexapod_biomechanics.config import (
    STANCE_GAP_BRIDGE_S,
    SessionConfig,
    accel_analog_pairs,
    kistler_raw_analog_pairs,
)
from hexapod_biomechanics.delsys import DelsysChannel, DelsysSignals, load_trial_npz
from hexapod_biomechanics.force_correction import BasicMLP
from hexapod_biomechanics.forces import HexGRF, HexKistler, HexRotAxis
from hexapod_biomechanics.inverse_dynamics import AnkleDynamics, AnkleID
from hexapod_biomechanics.kinematics import AnkleFrame, AnkleKinematics
from hexapod_biomechanics.load_data import C3DMan, convert_accel
from hexapod_biomechanics.utils import (
    axis_difference,
    find_pert_thresh,
    longest_contact_slice,
)

_SIDE_SIGN = {"right": 1, "left": -1}

# Dynamic-trial filename pattern: ..._nom_<fwd|rev> or ..._pert_<N>pct_<fwd|rev>.
_TRIAL_FILENAME_RE = re.compile(
    r"_(?:(?P<pert>pert)_(?P<pct>\d+)pct|(?P<nom>nom))_(?P<dir>fwd|rev)$"
)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PerturbationInfo:
    """Perturbation metadata for a single perturbed trial.

    Sample indices are stance-relative unless suffixed `_capture`, in which case
    they index into the full Vicon capture (needed for synchronization with
    Delsys-recorded EMG/goniometer signals that share t=0 with the c3d).
    """
    onset_sample: int                       # stance-relative index of pert start
    onset_sample_capture: int               # full-capture index of pert start
    onset_stance_fraction: float            # 0..1, onset phase within stance
    window_start_sample: int                # pert window start (stance-relative)
    window_end_sample: int                  # pert window end (stance-relative)
    axis_angular_deviation: npt.NDArray     # (M,) rad, per-frame across pert window
    axis_positional_offset: npt.NDArray     # (M,) mm, per-frame across pert window


@dataclass
class TrialResult:
    """Per-trial processing output. Signal arrays cover the stance period only.

    `stance_start_sample_capture` anchors the stance-period signals to the full
    Vicon capture timebase, supporting downstream Delsys sync.
    """
    trial_name: str
    trial_type: str                # "nominal" | "perturbed"
    session_id: str
    static_block_id: str
    side: str                      # "right" | "left"
    fs_hz: float
    body_mass_kg: float
    stance_start_sample_capture: int   # full-capture index where stance begins
    # Angles come in two variants to support differentiation before and after subtraction:
    #   - raw `alpha/beta/gamma` (arccos output, UNFILTERED) — subtract raw angles,
    #     then sav-gol filter + differentiate.
    #   - `alpha_filt/beta_filt/gamma_filt` — filtered before subtraction for consistency with
    #     ankle omega, foot alpha, torque (all subtracted independently).
    time: npt.NDArray              # (N,) seconds from stance start
    alpha: npt.NDArray             # (N,) DF/PF raw [rad]
    beta: npt.NDArray              # (N,) Inv/Ev raw [rad]
    gamma: npt.NDArray             # (N,) IntR/ExtR raw [rad]
    alpha_filt: npt.NDArray        # (N,) DF/PF filtered [rad]
    beta_filt: npt.NDArray         # (N,) Inv/Ev filtered [rad]
    gamma_filt: npt.NDArray        # (N,) IntR/ExtR filtered [rad]
    omega_jcs: npt.NDArray         # (N, 3) ankle angular velocity about e1,e2,e3
    tau_full: npt.NDArray          # (N, 3) torque (full dynamics) on JCS axes [N*mm]
    tau_static: npt.NDArray        # (N, 3) torque (static dynamics) on JCS axes [N*mm]
    foot_omega_jcs: npt.NDArray    # (N, 3) foot ω projected on JCS axes [rad/s]
    foot_alpha_jcs: npt.NDArray    # (N, 3) foot α projected on JCS axes [rad/s^2]
    # Hexapod platform angular trajectory about the perturbation axis,
    # stance-sliced. Stored for all trials (nominal trials sit near zero).
    # Enables full-stance visualization of the perturbation input alongside
    # the biomechanical response.
    hex_tjct: npt.NDArray          # (N,) hexapod angle about pert axis [rad]
    # perturbation metadata (None for nominal trials)
    perturbation: Optional[PerturbationInfo] = None
    # Delsys EMG/goniometer signals — present when a per-trial NPZ from
    # `scripts/segment_delsys.py` is found at processing time. All channels
    # are interpolated onto Vicon's analog grid (`fs_hz`) and sliced to the
    # same stance period as the rest of the per-trial signals; their `.t`
    # arrays match `time` above. None for sessions / trials without Delsys.
    delsys: Optional[DelsysSignals] = None


@dataclass
class StaticBlock:
    """Solvers and body mass derived from one static trial.

    A static block serves all dynamic trials recorded under the same marker
    setup; if markers migrate or are re-placed mid-session, a new static is
    captured and a new block is built.
    """
    static_block_id: str
    session_id: str
    body_mass_kg: float
    kistler: HexKistler                      # shared across sides
    ankle_frames: dict[str, AnkleFrame]      # {"right": ..., "left": ...}
    ankle_ids: dict[str, AnkleID]            # {"right": ..., "left": ...}


@dataclass
class TrialSpec:
    """One trial's inputs for batch processing by `process_condition`."""
    trial_data: C3DMan                   # loaded c3d
    session_config: SessionConfig
    static_block: StaticBlock
    side: str                            # "right" | "left"
    trial_type: str                      # "nominal" | "perturbed"
    trial_name: str                      # filename stem, used as h5 key


@dataclass
class TrialIntermediates:
    """Full-capture intermediate arrays from one trial's processing.

    `TrialResult` is the stance-sliced, persisted subset; `TrialIntermediates`
    is the complete bundle — retained so validation and visualization code
    (e.g. `visualize.animate_*`) can inspect any step of the computation
    without re-running the pipeline.

    All array fields span the full Vicon capture length. `stance_slice`
    indexes them to produce the stance-only `TrialResult` subset.
    """
    # identity / timing
    spec: TrialSpec
    t_full: npt.NDArray                   # (F,) analog-rate timebase [s]
    fs: float
    stance_slice: slice                   # slice into full-capture arrays

    # marker trajectories (full-capture)
    CALC: npt.NDArray                     # calcaneus (F, 3)
    M1: npt.NDArray                       # first metatarsal (F, 3)
    M5: npt.NDArray                       # fifth metatarsal (F, 3)
    cluster_S: npt.NDArray                # shank cluster (F, n_shank, 3)
    hex_markers: npt.NDArray              # hexapod (F, 3, 3)

    # force processing
    raw_forces: npt.NDArray               # (F, 8) raw Kistler channels
    induced_force: npt.NDArray            # (F, 8) MLP inertial-force prediction
    corrected_forces: npt.NDArray         # (F, 8) raw_forces - induced_force
    grf: HexGRF                           # force, COP, moment, is_stance, hex_tjct

    # Kistler geometry (full-capture)
    corners: npt.NDArray                  # (F, 4, 3) plate corners in global frame
    sensors: npt.NDArray                  # (F, 4, 3) sensor locations in global frame
    kist_origin_base: npt.NDArray         # (3,) neutral Kistler origin in global frame

    # kinematics
    kin: AnkleKinematics                  # alpha/beta/gamma (RAW), e1/e2/e3, o_ajc, R_F, COM_F
    T_TG: npt.NDArray                     # (F, 4, 4) tibia/fibula frame in global
    T_CG: npt.NDArray                     # (F, 4, 4) calcaneus frame in global
    alpha_filt: npt.NDArray               # (F,) sav-gol deriv=0 filtered DF angle [rad]
    beta_filt: npt.NDArray                # (F,) sav-gol deriv=0 filtered Inv/Ev angle [rad]
    gamma_filt: npt.NDArray               # (F,) sav-gol deriv=0 filtered IntR/ExtR angle [rad]
    omega_jcs: npt.NDArray                # (F, 3) JCS angular velocity

    # dynamics
    dyn_full: AnkleDynamics               # with inertial terms
    dyn_static: AnkleDynamics             # without inertial terms

    # perturbation axis of the hexapod (perturbed trials only)
    perturbation_axis: Optional[HexRotAxis] = None


# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------

def configure_from_static(
    static_data: C3DMan,
    session_config: SessionConfig,
    static_block_id: str,
    sides: Sequence[str] = ("right", "left"),
) -> StaticBlock:
    """Build per-side AnkleFrame + AnkleID and a shared HexKistler from one static.

    Body mass is computed from the mean resultant GRF magnitude over the
    static trial and stored on the block.

    Args:
        static_data: Loaded c3d for the static trial.
        session_config: Session config with subject metadata and label mappings.
        static_block_id: Identifier for this static block (e.g. "static_01"),
            used to associate downstream trials with their reference static.
        sides: Which ankle sides to configure. Defaults to both.

    Returns:
        StaticBlock: Configured solvers, body mass, and block metadata.

    Raises:
        ValueError: If a requested side is missing any of its 10 ankle joint
            markers (with complete frames) in the static c3d.
    """
    hex_points = static_data.get_points(session_config.hexapod_marker_labels())
    hex_markers = hex_points.values
    raw_forces = static_data.get_analogs(kistler_raw_analog_pairs(static_data)).values
    nan_per_frame = np.isnan(hex_markers).any(axis=(1, 2))
    n_trim_end = 0
    while n_trim_end < len(nan_per_frame) and nan_per_frame[-(n_trim_end + 1)]:
        n_trim_end += 1
    if n_trim_end:
        hex_markers = hex_markers[:-n_trim_end]
        raw_forces = raw_forces[:-n_trim_end]
    if np.isnan(hex_markers).any():
        raise ValueError(
            f"Static has unfillable mid-capture/leading-edge gaps in hexapod "
            f"markers (raw c3d missing: {hex_points.missing_frames}). "
            f"Hexapod tracking must be complete to establish the Kistler frame."
        )

    kistler = HexKistler(
        cluster_hex_static=hex_markers,
        pert_axis=np.asarray(session_config.pert_axis, dtype=float),
    )

    # Body mass: mean resultant force magnitude during the static capture / g
    grf_static = kistler.process_forces(cluster_hex=hex_markers, raw_forces=raw_forces)
    body_mass_kg = float(np.mean(np.linalg.norm(grf_static.force, axis=-1)) / g)

    # Per-side solvers
    ankle_frames: dict[str, AnkleFrame] = {}
    ankle_ids: dict[str, AnkleID] = {}
    for side in sides:
        if side not in _SIDE_SIGN:
            raise ValueError(f"Unknown side {side!r}; expected 'right' or 'left'.")

        side_markers = _load_ankle_markers_static(static_data, session_config, side)

        ankle = AnkleFrame(
            side=_SIDE_SIGN[side],
            MM_static=side_markers["medial_malleolus"],
            LM_static=side_markers["lateral_malleolus"],
            MC_static=side_markers["medial_condyle"],
            LC_static=side_markers["lateral_condyle"],
            CALC_static=side_markers["calcaneus"],
            M1_static=side_markers["first_metatarsal"],
            M5_static=side_markers["fifth_metatarsal"],
            cluster_S_static=side_markers["shank_cluster"],
            sex=session_config.sex,
        )
        ankle_id = AnkleID(
            body_mass=body_mass_kg,
            Inorm=ankle.Inorm_C,
            sex=session_config.sex,
        )
        ankle_frames[side] = ankle
        ankle_ids[side] = ankle_id

    return StaticBlock(
        static_block_id=static_block_id,
        session_id=session_config.session_id,
        body_mass_kg=body_mass_kg,
        kistler=kistler,
        ankle_frames=ankle_frames,
        ankle_ids=ankle_ids,
    )


def _load_ankle_markers_static(
    static_data: C3DMan,
    session_config: SessionConfig,
    side: str,
) -> dict[str, npt.NDArray]:
    """Load all 10 ankle joint markers (7 single + 3 cluster) for one side.

    The returned `shank_cluster` is the TRACKING set: the 3 cluster markers
    plus the lateral malleolus (see inline comment below).
    """
    resolved = session_config.ankle_marker_labels(side)
    single_roles = (
        "medial_malleolus", "lateral_malleolus",
        "medial_condyle", "lateral_condyle",
        "calcaneus", "first_metatarsal", "fifth_metatarsal",
    )
    cluster_labels = resolved["shank_cluster"]
    flat_labels = [resolved[r] for r in single_roles] + cluster_labels

    points = static_data.get_points(flat_labels)

    markers = points.values
    nan_per_frame = np.isnan(markers).any(axis=(1, 2))
    n_trim_end = 0
    while n_trim_end < len(nan_per_frame) and nan_per_frame[-(n_trim_end + 1)]:
        n_trim_end += 1
    if n_trim_end:
        markers = markers[:-n_trim_end]
    if np.isnan(markers).any():
        nan_counts = {
            lab: int(np.isnan(markers[:, i, :]).any(axis=1).sum())
            for i, lab in enumerate(flat_labels)
        }
        nan_counts = {k: v for k, v in nan_counts.items() if v}
        raise ValueError(
            f"Static has unfillable mid-capture/leading-edge gaps in "
            f"{side}-ankle markers: {nan_counts}."
        )
    out: dict[str, npt.NDArray] = {
        role: markers[:, i, :] for i, role in enumerate(single_roles)
    }
    # The shank is TRACKED by the cluster plus the lateral malleolus. LM belongs
    # to the shank anatomically and sits ~300 mm from the cluster triad, which
    # spans only ~36 mm; orientation error scales as atan(jitter / baseline), so
    # the longer lever arm cuts jitter-driven wobble substantially. This does not
    # change LM's separate anatomical role in defining the malleolar axis.
    # PROVISIONAL: hardcoded pending a per-segment tracking-marker list in
    # SessionConfig. See isolation-method-validation.md.
    lm_idx = single_roles.index("lateral_malleolus")
    out["shank_cluster"] = np.concatenate(
        [markers[:, len(single_roles):, :], markers[:, lm_idx:lm_idx + 1, :]],
        axis=1,
    )
    return out


def _parse_trial_filename(stem: str) -> Optional[tuple[str, str, Optional[int]]]:
    """Parse a dynamic-trial filename stem into (trial_type, direction, pct).

    Expected patterns (suffix of the stem):
        `_nom_fwd`  or `_nom_rev`                 → nominal
        `_pert_<N>pct_fwd` or `_pert_<N>pct_rev`  → perturbed

    Args:
        stem: Trial filename without extension
            (e.g. "*_trip12_pert_50pct_rev").

    Returns:
        A tuple `(trial_type, direction, pct)` where `trial_type` is
        "nominal" or "perturbed", `direction` is "fwd" or "rev", and `pct` is
        the perturbation percentage (int) or None for nominal trials. Returns
        None if the stem does not match either pattern.
    """
    m = _TRIAL_FILENAME_RE.search(stem)
    if m is None:
        return None
    direction = m.group("dir")
    if m.group("nom"):
        return ("nominal", direction, None)
    return ("perturbed", direction, int(m.group("pct")))


def make_static_block_loader(
    session_config: SessionConfig,
    session_dir: str | Path,
) -> Callable[[str], StaticBlock]:
    """Build a caching loader that returns `StaticBlock`s by their stem.

    Each static c3d is loaded and passed through `configure_from_static` only
    once per sweep — repeated calls with the same stem return the cached
    block. Use when iterating many trials that share a small number of
    statics (the typical case: ~200 dynamics under 1-2 statics per session).

    Args:
        session_config: Session config providing label mappings and
            subject metadata.
        session_dir: Session directory containing `raw_data/` with
            `<stem>.c3d` files.

    Returns:
        A callable `load(static_block_id)` → `StaticBlock`. The closure
        holds an internal dict keyed by static stem; first call for a given
        stem builds the block, later calls return the cached result.
    """
    raw_data_dir = Path(session_dir) / "raw_data"
    cache: dict[str, StaticBlock] = {}

    def load(static_block_id: str) -> StaticBlock:
        if static_block_id not in cache:
            static_path = raw_data_dir / f"{static_block_id}.c3d"
            static_data = C3DMan(str(static_path))
            cache[static_block_id] = configure_from_static(
                static_data=static_data,
                session_config=session_config,
                static_block_id=static_block_id,
            )
        return cache[static_block_id]

    return load


def iter_trial_specs(
    session_config: SessionConfig,
    session_dir: str | Path,
    static_block_loader: Callable[[str], StaticBlock],
    side: Optional[str] = None,
    trial_type: Optional[str] = None,
    perturbation_pct: Optional[int] = None,
) -> Iterator[TrialSpec]:
    """Yield fully-materialized `TrialSpec`s for one condition of a session.

    Walks `session_config.static_blocks`, parses each dynamic trial's
    filename for (trial_type, direction, pct), maps direction → side via
    `session_config.direction_side_map`, applies the filter kwargs, and
    yields ready-to-process `TrialSpec` objects. Trial filenames that don't
    match the expected naming pattern are skipped with a printed warning so a
    single typo doesn't abort a batch sweep.

    Args:
        session_config: Session config supplying static-block grouping and
            direction→side mapping.
        session_dir: Session directory (contains `raw_data/`).
        static_block_loader: Callable mapping a static stem to its
            `StaticBlock`. Typically obtained via `make_static_block_loader`
            so repeated trials sharing one static reuse its solvers.
        side: If set, only yield trials on this side ("right" | "left").
        trial_type: If set, only yield trials of this type
            ("nominal" | "perturbed").
        perturbation_pct: If set, only yield perturbed trials at this stance
            percentage. Implicitly excludes nominal trials when set.

    Yields:
        `TrialSpec`: one per dynamic trial matching the filter.

    Raises:
        ValueError: If a parsed trial's direction token is not listed in
            `session_config.direction_side_map`.
    """
    raw_data_dir = Path(session_dir) / "raw_data"

    for static_stem, dynamic_stems in session_config.static_blocks.items():
        for dyn_stem in dynamic_stems:
            parsed = _parse_trial_filename(dyn_stem)
            if parsed is None:
                print(
                    f"WARNING: trial filename {dyn_stem!r} does not match the "
                    f"expected '_nom_<dir>' or '_pert_<N>pct_<dir>' pattern; "
                    f"skipping."
                )
                continue

            ttype, direction, pct = parsed

            # Map filename direction token to ankle side.
            if direction not in session_config.direction_side_map:
                raise ValueError(
                    f"Trial {dyn_stem!r} direction {direction!r} is not in "
                    f"session_config.direction_side_map "
                    f"({session_config.direction_side_map})."
                )
            trial_side = session_config.direction_side_map[direction]

            # Apply filters — None means "any".
            if side is not None and trial_side != side:
                continue
            if trial_type is not None and ttype != trial_type:
                continue
            if perturbation_pct is not None and pct != perturbation_pct:
                continue

            static_block = static_block_loader(static_stem)
            trial_path = raw_data_dir / f"{dyn_stem}.c3d"
            trial_data = C3DMan(str(trial_path))

            yield TrialSpec(
                trial_data=trial_data,
                session_config=session_config,
                static_block=static_block,
                side=trial_side,
                trial_type=ttype,
                trial_name=dyn_stem,
            )


def compute_trial_intermediates(
    spec: TrialSpec,
    force_model: BasicMLP,
    force_correction_step_size: int = 50,
    pert_threshold_deg: float = 1.0,
    terminal_window_ms: tuple[float, float] = (150.0, 500.0),
) -> TrialIntermediates:
    """Run force correction + kinematics + dynamics, returning the full bundle.

    This holds the per-trial computation that `process_trial` wraps. Breaking
    it out lets validation / visualization code inspect every array the
    pipeline touches (markers, raw + corrected forces, full GRF, kinematics
    frames, dynamics) without re-running the pipeline or re-reading the c3d.

    Args:
        spec: Trial inputs (c3d, side, static block, etc.).
        force_model: Pre-loaded MLP for inertial force correction.
        force_correction_step_size: Stride between MLP inference windows.
        pert_threshold_deg: Hexapod angle threshold for perturbation detection
            (used only for perturbed trials, when locating the rot axis).
        terminal_window_ms: (start, end) ms window past the threshold crossing
            used to sample the terminal perturbed hexapod pose for rot axis
            estimation. Ignored for nominal trials.

    Returns:
        TrialIntermediates: Full-capture arrays + stance slice + (for
            perturbed trials) the estimated hexapod rotation axis.
    """
    session_config = spec.session_config
    static_block = spec.static_block
    side = spec.side
    if side not in _SIDE_SIGN:
        raise ValueError(f"Unknown side {side!r}; expected 'right' or 'left'.")

    ankle_frame = static_block.ankle_frames[side]
    ankle_id = static_block.ankle_ids[side]
    kistler = static_block.kistler

    # Dynamic-trial markers: foot cluster (CALC, M1, M5) + shank cluster + hexapod.
    resolved = session_config.ankle_marker_labels(side)
    foot_roles = ("calcaneus", "first_metatarsal", "fifth_metatarsal")
    foot_labels = [resolved[r] for r in foot_roles]
    # Shank tracking markers = cluster + lateral malleolus; see the matching
    # comment in `_load_side_markers` for why, and keep the two in the same
    # order so the static neutral shape corresponds index-for-index.
    shank_cluster_labels = (resolved["shank_cluster"]
                            + [resolved["lateral_malleolus"]])
    hex_labels = session_config.hexapod_marker_labels()

    foot_points = spec.trial_data.get_points(foot_labels)
    shank_points = spec.trial_data.get_points(shank_cluster_labels)
    hex_points = spec.trial_data.get_points(hex_labels)

    for name, pts in (("foot", foot_points),
                      ("shank cluster", shank_points),
                      ("hexapod", hex_points)):
        if pts.missing_frames:
            raise ValueError(
                f"Trial {spec.trial_name!r} missing frames for {name} "
                f"markers: {pts.missing_frames}."
            )

    CALC = foot_points.values[:, 0, :]
    M1 = foot_points.values[:, 1, :]
    M5 = foot_points.values[:, 2, :]
    cluster_S = shank_points.values            # (F, n_shank, 3)
    hex_markers = hex_points.values            # (F, 3, 3)
    t_full = foot_points.t                     # full-capture analog timebase
    fs = float(spec.trial_data.fs)

    # Inertial force correction: subtract MLP prediction from raw Kistler.
    # The model was trained on voltage-domain accel data — if Vicon exported
    # as m/s², reverse that conversion first.
    raw_forces = spec.trial_data.get_analogs(
        kistler_raw_analog_pairs(spec.trial_data)
    ).values
    accel_mode, accel_pairs = accel_analog_pairs(spec.trial_data)
    accel = spec.trial_data.get_analogs(accel_pairs).values
    if accel_mode == "accelerometer":
        accel = convert_accel(accel)
    induced_force = force_model.predict_segments(
        [accel.astype(np.float32)],
        step_size=force_correction_step_size,
    )[0]
    corrected_forces = raw_forces - induced_force

    # Forces + kinematics + dynamics (full-capture)
    grf = kistler.process_forces(
        cluster_hex=hex_markers,
        raw_forces=corrected_forces,
    )
    # Kistler geometry — corners/sensors depend on T_KG set by process_forces.
    corners, sensors = kistler.track_plate()
    kist_origin_base = kistler.T_KG_neut[:3, 3].copy()

    kin = ankle_frame.compute_kinematics(
        CALC=CALC, M1=M1, M5=M5, cluster_S=cluster_S,
    )
    # ankle_frame.T_TG / .T_CG are instance state — copy so later trials using
    # the same (shared) AnkleFrame don't mutate this trial's stored frames.
    T_TG = ankle_frame.T_TG.copy()
    T_CG = ankle_frame.T_CG.copy()

    dyn_full = ankle_id.compute_dynamics(
        t=t_full,
        grf_force=grf.force, grf_moment_origin=grf.moment_origin,
        e1=kin.e1, e2=kin.e2, e3=kin.e3,
        o_ajc=kin.o_ajc, R_F=kin.R_F, COM_F=kin.COM_F,
    )
    dyn_static = ankle_id.compute_dynamics(
        t=t_full,
        grf_force=grf.force, grf_moment_origin=grf.moment_origin,
        e1=kin.e1, e2=kin.e2, e3=kin.e3,
        o_ajc=kin.o_ajc, R_F=kin.R_F, COM_F=None,
    )

    # Stance slice — the longest contiguous loaded run, not first-to-last
    # loaded frame. A capture often registers the plate as loaded more than
    # once (contralateral foot brushing it, platform settling, the tail of a
    # previous step), and first-to-last would swallow the unloaded frames
    # between those events along with the real step.
    if not grf.is_stance.any():
        raise ValueError(f"Trial {spec.trial_name!r}: no stance frames detected.")
    stance_slice = longest_contact_slice(
        grf.is_stance, bridge_samples=int(round(STANCE_GAP_BRIDGE_S * fs)),
    )
    i_stance_start = stance_slice.start
    i_stance_end = stance_slice.stop - 1

    # Angle handling: kin.alpha/beta/gamma are raw arccos outputs (unfiltered).
    # Apply the same Savitzky-Golay filter used by dynamics so every stored
    # kinematic signal is one-pass filtered — angles via deriv=0, ω via
    # deriv=1 of the same filter, τ via the dynamics' internal filtering.
    # Raw `alpha/beta/gamma` are also retained on TrialResult so downstream
    # code can subtract first and filter once afterwards if it prefers.
    filt = dyn_full.filter
    abg = np.column_stack([kin.alpha, kin.beta, kin.gamma])
    abg_filt = filt(abg, deriv=0)                 # (F, 3) [rad] one-pass filtered
    omega_jcs = filt(abg, deriv=1)                # (F, 3) [rad/s]
    alpha_filt = abg_filt[:, 0]
    beta_filt = abg_filt[:, 1]
    gamma_filt = abg_filt[:, 2]

    # Perturbation axis (perturbed trials only). The full PerturbationInfo
    # (window offsets, axis deviations) is derived later when producing
    # TrialResult; only the axis itself is an intermediate needed for
    # visualization.
    perturbation_axis: Optional[HexRotAxis] = None
    if spec.trial_type == "perturbed":
        perturbation_axis = _locate_perturbation_axis(
            hex_stance_tjct=grf.hex_tjct[stance_slice],
            hex_markers_full=hex_markers,
            kistler=kistler,
            fs=fs,
            i_stance_start=i_stance_start,
            pert_threshold_deg=pert_threshold_deg,
            terminal_window_ms=terminal_window_ms,
        )
    elif spec.trial_type != "nominal":
        raise ValueError(
            f"Unknown trial_type {spec.trial_type!r}; expected 'nominal' or 'perturbed'."
        )

    return TrialIntermediates(
        spec=spec,
        t_full=t_full,
        fs=fs,
        stance_slice=stance_slice,
        CALC=CALC,
        M1=M1,
        M5=M5,
        cluster_S=cluster_S,
        hex_markers=hex_markers,
        raw_forces=raw_forces,
        induced_force=induced_force,
        corrected_forces=corrected_forces,
        grf=grf,
        corners=corners,
        sensors=sensors,
        kist_origin_base=kist_origin_base,
        kin=kin,
        T_TG=T_TG,
        T_CG=T_CG,
        alpha_filt=alpha_filt,
        beta_filt=beta_filt,
        gamma_filt=gamma_filt,
        omega_jcs=omega_jcs,
        dyn_full=dyn_full,
        dyn_static=dyn_static,
        perturbation_axis=perturbation_axis,
    )


def process_trial(
    spec: TrialSpec,
    force_model: BasicMLP,
    force_correction_step_size: int = 50,
    pert_threshold_deg: float = 1.0,
    pert_pre_threshold_ms: float = 46.0,
    pert_pre_window_ms: float = 25.0,
    pert_post_window_ms: float = 100.0,
    terminal_window_ms: tuple[float, float] = (150.0, 500.0),
    delsys_path: Optional[Path] = None,
) -> TrialResult:
    """Compute intermediates for one trial and derive its stance-sliced result.

    Thin wrapper over `compute_trial_intermediates` + `_derive_trial_result`.
    Use this for the standard batch-processing path; call
    `compute_trial_intermediates` directly when you need the full-capture
    arrays (e.g. to render `visualize.animate_*` plots for validation).

    Args:
        spec: Trial inputs (c3d, side, static block, etc.).
        force_model: Pre-loaded `BasicMLP` for inertial force correction.
        force_correction_step_size: Stride between MLP inference windows.
        pert_threshold_deg: Hexapod angle threshold for perturbation detection.
        pert_pre_threshold_ms: Offset from threshold back to perturbation onset.
        pert_pre_window_ms: Pre-onset window duration.
        pert_post_window_ms: Post-onset window duration.
        terminal_window_ms: (start, end) ms window used to locate the terminal
            perturbed hexapod pose for axis-of-rotation estimation.
        delsys_path: Optional path to this trial's Delsys NPZ (typically
            `<session_dir>/raw_data/delsys/<trial_name>.npz`). When supplied
            and the file exists, every channel is cubic-interpolated onto
            Vicon's analog grid, sliced to the stance period, and attached
            to `TrialResult.delsys`. When None or missing, `delsys` is None.

    Returns:
        TrialResult: Stance-period signals + optional perturbation metadata
            + optional Delsys signals.
    """
    intermediates = compute_trial_intermediates(
        spec,
        force_model=force_model,
        force_correction_step_size=force_correction_step_size,
        pert_threshold_deg=pert_threshold_deg,
        terminal_window_ms=terminal_window_ms,
    )
    delsys: Optional[DelsysSignals] = None
    if delsys_path is not None and Path(delsys_path).exists():
        delsys = _load_and_align_delsys(
            Path(delsys_path),
            t_full=intermediates.t_full,
            fs=intermediates.fs,
            stance_slice=intermediates.stance_slice,
        )
    return _derive_trial_result(
        intermediates,
        pert_threshold_deg=pert_threshold_deg,
        pert_pre_threshold_ms=pert_pre_threshold_ms,
        pert_pre_window_ms=pert_pre_window_ms,
        pert_post_window_ms=pert_post_window_ms,
        delsys=delsys,
    )


def _load_and_align_delsys(
    delsys_path: Path,
    t_full: npt.NDArray,
    fs: float,
    stance_slice: slice,
) -> DelsysSignals:
    """Load a trial NPZ and align channels onto the stance-period Vicon grid.

    The NPZ time arrays are in seconds-from-trigger-pulse; subtract
    `segment_start_s` to get trial-local time, cubic-interpolate each
    channel onto `t_full`, then slice to the stance window.

    Returns a `DelsysSignals` whose channel `.t` arrays match
    `t_full[stance_slice] - t_full[stance_slice.start]` (stance-relative
    seconds, the same convention as `TrialResult.time`).
    """
    delsys_block = load_trial_npz(delsys_path)
    if delsys_block.segment_start_s is None:
        raise ValueError(
            f"{delsys_path} has no segment_start_s — cannot align with Vicon "
            "timebase. Was this NPZ produced by `segment_delsys.py`?"
        )
    offset_s = float(delsys_block.segment_start_s)

    t_stance = t_full[stance_slice] - t_full[stance_slice.start]

    aligned: dict[str, DelsysChannel] = {}
    for name, ch in delsys_block.channels.items():
        # NPZ time is trigger-pulse-absolute; convert to trial-local.
        t_local = ch.t - offset_s
        interp = interp1d(
            t_local,
            ch.v,
            kind="cubic",
            bounds_error=False,
            fill_value=np.nan,
            assume_sorted=True,
        )
        v_full = interp(t_full)
        aligned[name] = DelsysChannel(
            label=name,
            t=t_stance,
            v=np.asarray(v_full[stance_slice]),
            fs_hz=fs,
            units=ch.units,
            sensor_id=ch.sensor_id,
        )

    return DelsysSignals(
        channels=aligned,
        trial_stem=delsys_block.trial_stem,
        block_id=delsys_block.block_id,
        segment_start_s=delsys_block.segment_start_s,
        segment_end_s=delsys_block.segment_end_s,
        static_stem=delsys_block.static_stem,
        gonio_offsets_rad=delsys_block.gonio_offsets_rad,
    )


def _derive_trial_result(
    inter: TrialIntermediates,
    pert_threshold_deg: float,
    pert_pre_threshold_ms: float,
    pert_pre_window_ms: float,
    pert_post_window_ms: float,
    delsys: Optional[DelsysSignals] = None,
) -> TrialResult:
    """Slice intermediates to stance and build the persisted `TrialResult`.

    For perturbed trials, also computes `PerturbationInfo` (window indices and
    per-frame axis deviation) using the rot axis already attached to
    `inter.perturbation_axis`.
    """
    spec = inter.spec
    sl = inter.stance_slice
    i_stance_start = sl.start
    n_stance = sl.stop - sl.start
    static_block = spec.static_block
    kin = inter.kin

    pert_info: Optional[PerturbationInfo] = None
    if spec.trial_type == "perturbed":
        pert_info = _compute_perturbation_info(
            rot=inter.perturbation_axis,
            hex_stance_tjct=inter.grf.hex_tjct[sl],
            o_ajc_stance=kin.o_ajc[sl],
            e1_stance=kin.e1[sl],
            fs=inter.fs,
            n_stance=n_stance,
            i_stance_start=i_stance_start,
            pert_threshold_deg=pert_threshold_deg,
            pert_pre_threshold_ms=pert_pre_threshold_ms,
            pert_pre_window_ms=pert_pre_window_ms,
            pert_post_window_ms=pert_post_window_ms,
        )

    time_stance = inter.t_full[sl] - inter.t_full[i_stance_start]

    return TrialResult(
        trial_name=spec.trial_name,
        trial_type=spec.trial_type,
        session_id=spec.session_config.session_id,
        static_block_id=static_block.static_block_id,
        side=spec.side,
        fs_hz=inter.fs,
        body_mass_kg=static_block.body_mass_kg,
        stance_start_sample_capture=i_stance_start,
        time=time_stance,
        alpha=kin.alpha[sl],
        beta=kin.beta[sl],
        gamma=kin.gamma[sl],
        alpha_filt=inter.alpha_filt[sl],
        beta_filt=inter.beta_filt[sl],
        gamma_filt=inter.gamma_filt[sl],
        omega_jcs=inter.omega_jcs[sl],
        tau_full=inter.dyn_full.M_jcs[sl],
        tau_static=inter.dyn_static.M_jcs[sl],
        foot_omega_jcs=inter.dyn_full.omega_foot_jcs[sl],
        foot_alpha_jcs=inter.dyn_full.alpha_foot_jcs[sl],
        hex_tjct=inter.grf.hex_tjct[sl],
        perturbation=pert_info,
        delsys=delsys,
    )


def _locate_perturbation_axis(
    hex_stance_tjct: npt.NDArray,
    hex_markers_full: npt.NDArray,
    kistler: HexKistler,
    fs: float,
    i_stance_start: int,
    pert_threshold_deg: float,
    terminal_window_ms: tuple[float, float],
) -> HexRotAxis:
    """Estimate the hexapod rotation axis from the terminal perturbation pose."""
    threshold_rad = np.radians(pert_threshold_deg)
    i_thresh = find_pert_thresh(hex_tjct=hex_stance_tjct, thresh=threshold_rad)
    term_start_frames = round(terminal_window_ms[0] / 1000.0 * fs)
    term_end_frames = round(terminal_window_ms[1] / 1000.0 * fs)
    term_lo = i_thresh + term_start_frames
    term_hi = i_thresh + term_end_frames
    term_full = np.arange(term_lo, term_hi) + i_stance_start
    return kistler.locate_rot_axis(cluster_hex_pert=hex_markers_full[term_full])


def _compute_perturbation_info(
    rot: HexRotAxis,
    hex_stance_tjct: npt.NDArray,
    o_ajc_stance: npt.NDArray,
    e1_stance: npt.NDArray,
    fs: float,
    n_stance: int,
    i_stance_start: int,
    pert_threshold_deg: float,
    pert_pre_threshold_ms: float,
    pert_pre_window_ms: float,
    pert_post_window_ms: float,
) -> PerturbationInfo:
    """Detect perturbation onset/window and measure axis deviation from ankle."""
    threshold_rad = np.radians(pert_threshold_deg)
    i_thresh = find_pert_thresh(hex_tjct=hex_stance_tjct, thresh=threshold_rad)

    pre_thresh_frames = round(pert_pre_threshold_ms / 1000.0 * fs)
    pre_win_frames = round(pert_pre_window_ms / 1000.0 * fs)
    post_win_frames = round(pert_post_window_ms / 1000.0 * fs)

    onset = i_thresh - pre_thresh_frames             # stance-relative perturbation onset
    window_start = onset - pre_win_frames
    window_end = onset + post_win_frames

    axis_diff = axis_difference(
        target_axis=(rot.origin, rot.direction),
        actual_axis=(o_ajc_stance, e1_stance),
    )
    win_slice = slice(window_start, window_end)
    ang_dev = np.asarray(axis_diff["angle_diff"])[win_slice]
    pos_off = np.asarray(axis_diff["offset_diff"])[win_slice]

    return PerturbationInfo(
        onset_sample=int(onset),
        onset_sample_capture=int(onset + i_stance_start),
        onset_stance_fraction=float(onset / n_stance),
        window_start_sample=int(window_start),
        window_end_sample=int(window_end),
        axis_angular_deviation=ang_dev,
        axis_positional_offset=pos_off,
    )


def perturbation_window_slice(
    inter: TrialIntermediates,
    pert_threshold_deg: float = 1.0,
    pert_pre_threshold_ms: float = 46.0,
    pert_pre_window_ms: float = 25.0,
    pert_post_window_ms: float = 100.0,
) -> slice:
    """Derive the full-capture slice covering the perturbation window.

    Args:
        inter: Trial intermediates (must have `perturbation_axis`).
        pert_threshold_deg: Hexapod angle threshold for onset detection.
        pert_pre_threshold_ms: Offset from threshold crossing back to onset.
        pert_pre_window_ms: Pre-onset window duration.
        pert_post_window_ms: Post-onset window duration.

    Returns:
        A `slice` into full-capture (analog-rate) arrays covering
        `[onset - pre_win, onset + post_win]`.

    Raises:
        ValueError: If the trial has no perturbation axis (nominal trial).
    """
    if inter.perturbation_axis is None:
        raise ValueError(
            f"Trial {inter.spec.trial_name!r} is nominal — perturbation "
            f"window is undefined."
        )
    fs = inter.fs
    sl = inter.stance_slice
    i_stance_start = sl.start

    # Stance-relative onset
    threshold_rad = np.radians(pert_threshold_deg)
    i_thresh = find_pert_thresh(
        hex_tjct=inter.grf.hex_tjct[sl], thresh=threshold_rad,
    )
    pre_thresh_frames = round(pert_pre_threshold_ms / 1000.0 * fs)
    pre_win_frames = round(pert_pre_window_ms / 1000.0 * fs)
    post_win_frames = round(pert_post_window_ms / 1000.0 * fs)
    onset = i_thresh - pre_thresh_frames
    return slice(
        onset - pre_win_frames + i_stance_start,
        onset + post_win_frames + i_stance_start,
    )


def process_condition(
    trials: Sequence[TrialSpec],
    output_h5: Path,
    condition_id: str,
    side: str,
    subject_id: str,
    sex: str,
    force_model: BasicMLP,
    force_correction_step_size: int = 50,
    perturbation_pct: Optional[int] = None,
    extra_attrs: Optional[dict] = None,
    delsys_dir: Optional[Path] = None,
) -> None:
    """Process all `trials` and write a single HDF5 file for the condition.

    Caller is responsible for grouping trials into a condition and associating
    each with its static block. One file = one condition.

    Args:
        trials: Trial specs sharing the same condition.
        output_h5: Output HDF5 path.
        condition_id: File-level identifier, e.g. "RP041026_15102026_right_pert_50".
        side: "right" | "left" — the condition's ankle side.
        subject_id: Subject identifier.
        sex: "m" | "f", for downstream anthropometric reference.
        force_model: Pre-loaded MLP for inertial force correction, reused
            across every trial in the condition.
        force_correction_step_size: Stride between MLP inference windows.
        perturbation_pct: Stance percentage of the perturbation (e.g. 50, 20)
            for perturbed conditions. Pass None for nominal conditions — the
            `perturbation_pct` file attr is omitted entirely in that case.
        extra_attrs: Optional additional file-level attrs (e.g. prosthesis
            stiffness for amputee validation). Defaults to None (nothing extra).
        delsys_dir: Optional directory containing per-trial Delsys NPZ files
            (typically `<session_dir>/raw_data/delsys/`). For each trial,
            looks up `<delsys_dir>/<trial_name>.npz`; if found, the channels
            are aligned onto Vicon's analog grid + stance period and saved
            under the trial's `delsys/` h5 subgroup. Trials without a
            matching NPZ are processed normally with no Delsys data. None
            disables Delsys handling entirely.
    """
    output_h5 = Path(output_h5)
    output_h5.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_h5, "w") as f:
        f.attrs["condition_id"] = condition_id
        f.attrs["subject_id"] = subject_id
        f.attrs["sex"] = sex
        f.attrs["side"] = side
        if perturbation_pct is not None:
            f.attrs["perturbation_pct"] = perturbation_pct
        if extra_attrs:
            for key, value in extra_attrs.items():
                f.attrs[key] = value

        for spec in trials:
            delsys_path: Optional[Path] = None
            if delsys_dir is not None:
                candidate = Path(delsys_dir) / f"{spec.trial_name}.npz"
                if candidate.exists():
                    delsys_path = candidate
            result = process_trial(
                spec,
                force_model=force_model,
                force_correction_step_size=force_correction_step_size,
                delsys_path=delsys_path,
            )
            _write_trial_group(f, result)


def load_condition(h5_path: Path) -> tuple[dict, list[TrialResult]]:
    """Load a condition HDF5 back into (file_attrs, [TrialResult, ...]).

    Args:
        h5_path: Path to a condition HDF5 file produced by `process_condition`.

    Returns:
        A tuple `(file_attrs, trials)` where `file_attrs` is the dict of
        file-level attributes and `trials` is the list of per-trial results.
    """
    with h5py.File(h5_path, "r") as f:
        file_attrs = {k: _decode_attr(v) for k, v in f.attrs.items()}
        trials = [_read_trial_group(f[name]) for name in f.keys()]
    return file_attrs, trials


# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------

def _write_trial_group(f: h5py.File, result: TrialResult) -> None:
    """Write one TrialResult under its own group in the open h5 file."""
    grp = f.create_group(result.trial_name)
    grp.attrs["trial_type"] = result.trial_type
    grp.attrs["session_id"] = result.session_id
    grp.attrs["static_block_id"] = result.static_block_id
    grp.attrs["side"] = result.side
    grp.attrs["fs_hz"] = result.fs_hz
    grp.attrs["body_mass_kg"] = result.body_mass_kg
    grp.attrs["stance_start_sample_capture"] = result.stance_start_sample_capture

    grp.create_dataset("time", data=result.time)
    grp.create_dataset("alpha", data=result.alpha)
    grp.create_dataset("beta", data=result.beta)
    grp.create_dataset("gamma", data=result.gamma)
    grp.create_dataset("alpha_filt", data=result.alpha_filt)
    grp.create_dataset("beta_filt", data=result.beta_filt)
    grp.create_dataset("gamma_filt", data=result.gamma_filt)
    grp.create_dataset("omega_jcs", data=result.omega_jcs)
    grp.create_dataset("tau_full", data=result.tau_full)
    grp.create_dataset("tau_static", data=result.tau_static)
    grp.create_dataset("foot_omega_jcs", data=result.foot_omega_jcs)
    grp.create_dataset("foot_alpha_jcs", data=result.foot_alpha_jcs)
    grp.create_dataset("hex_tjct", data=result.hex_tjct)

    if result.perturbation is not None:
        pg = grp.create_group("perturbation")
        pg.attrs["onset_sample"] = result.perturbation.onset_sample
        pg.attrs["onset_sample_capture"] = result.perturbation.onset_sample_capture
        pg.attrs["onset_stance_fraction"] = result.perturbation.onset_stance_fraction
        pg.attrs["window_start_sample"] = result.perturbation.window_start_sample
        pg.attrs["window_end_sample"] = result.perturbation.window_end_sample
        pg.create_dataset(
            "axis_angular_deviation",
            data=result.perturbation.axis_angular_deviation,
        )
        pg.create_dataset(
            "axis_positional_offset",
            data=result.perturbation.axis_positional_offset,
        )

    if result.delsys is not None:
        _write_delsys_subgroup(grp, result.delsys)


def _write_delsys_subgroup(grp: h5py.Group, delsys: DelsysSignals) -> None:
    """Write `delsys/` subgroup under a trial group.

    Channel datasets are stored flat under `delsys/` keyed by channel name
    (e.g. `delsys/gonio_left_df_pf`, `delsys/LTA`). Per-channel `units` and
    `sensor_id` are dataset attrs so the round-trip preserves the
    `DelsysChannel` dataclass exactly. The shared time vector is the
    trial's existing `time` dataset (Delsys channels are aligned onto it),
    so no separate Delsys time array is stored.
    """
    dg = grp.create_group("delsys")
    if delsys.block_id is not None:
        dg.attrs["block_id"] = delsys.block_id
    if delsys.segment_start_s is not None:
        dg.attrs["segment_start_s"] = float(delsys.segment_start_s)
    if delsys.segment_end_s is not None:
        dg.attrs["segment_end_s"] = float(delsys.segment_end_s)
    if delsys.static_stem is not None:
        dg.attrs["static_stem"] = delsys.static_stem
    if delsys.gonio_offsets_rad:
        # JSON-encode dict for h5 attr compatibility.
        dg.attrs["gonio_offsets_rad"] = json.dumps(delsys.gonio_offsets_rad)
    if delsys.channels:
        # All channels share a single fs_hz (interpolated to Vicon's rate).
        first_ch = next(iter(delsys.channels.values()))
        dg.attrs["fs_hz"] = float(first_ch.fs_hz)

    for name, ch in delsys.channels.items():
        ds = dg.create_dataset(name, data=ch.v)
        ds.attrs["units"] = ch.units
        if ch.sensor_id is not None:
            ds.attrs["sensor_id"] = ch.sensor_id


def _read_trial_group(grp: h5py.Group) -> TrialResult:
    """Reconstruct a TrialResult from its h5 group."""
    perturbation: Optional[PerturbationInfo] = None
    if "perturbation" in grp:
        pg = grp["perturbation"]
        perturbation = PerturbationInfo(
            onset_sample=int(pg.attrs["onset_sample"]),
            onset_sample_capture=int(pg.attrs["onset_sample_capture"]),
            onset_stance_fraction=float(pg.attrs["onset_stance_fraction"]),
            window_start_sample=int(pg.attrs["window_start_sample"]),
            window_end_sample=int(pg.attrs["window_end_sample"]),
            axis_angular_deviation=pg["axis_angular_deviation"][...],
            axis_positional_offset=pg["axis_positional_offset"][...],
        )

    time_array = grp["time"][...]
    delsys = _read_delsys_subgroup(grp, time_array) if "delsys" in grp else None

    if "hex_tjct" not in grp:
        raise KeyError(
            f"Trial {grp.name!r} is missing the `hex_tjct` dataset — this "
            f"condition HDF5 was written by an older version of "
            f"process_session.py. Re-run process_session.py to regenerate."
        )

    return TrialResult(
        trial_name=grp.name.rsplit("/", 1)[-1],
        trial_type=_decode_attr(grp.attrs["trial_type"]),
        session_id=_decode_attr(grp.attrs["session_id"]),
        static_block_id=_decode_attr(grp.attrs["static_block_id"]),
        side=_decode_attr(grp.attrs["side"]),
        fs_hz=float(grp.attrs["fs_hz"]),
        body_mass_kg=float(grp.attrs["body_mass_kg"]),
        stance_start_sample_capture=int(grp.attrs["stance_start_sample_capture"]),
        time=time_array,
        alpha=grp["alpha"][...],
        beta=grp["beta"][...],
        gamma=grp["gamma"][...],
        alpha_filt=grp["alpha_filt"][...],
        beta_filt=grp["beta_filt"][...],
        gamma_filt=grp["gamma_filt"][...],
        omega_jcs=grp["omega_jcs"][...],
        tau_full=grp["tau_full"][...],
        tau_static=grp["tau_static"][...],
        foot_omega_jcs=grp["foot_omega_jcs"][...],
        foot_alpha_jcs=grp["foot_alpha_jcs"][...],
        hex_tjct=grp["hex_tjct"][...],
        perturbation=perturbation,
        delsys=delsys,
    )


def _read_delsys_subgroup(
    grp: h5py.Group, time_array: npt.NDArray
) -> DelsysSignals:
    """Reconstruct a `DelsysSignals` from a trial's `delsys/` subgroup."""
    dg = grp["delsys"]
    fs_hz = float(dg.attrs["fs_hz"]) if "fs_hz" in dg.attrs else float(grp.attrs["fs_hz"])

    channels: dict[str, DelsysChannel] = {}
    for ch_name in dg.keys():
        ch_ds = dg[ch_name]
        sensor_id = _decode_attr(ch_ds.attrs["sensor_id"]) if "sensor_id" in ch_ds.attrs else None
        units = _decode_attr(ch_ds.attrs["units"]) if "units" in ch_ds.attrs else ""
        channels[ch_name] = DelsysChannel(
            label=ch_name,
            t=time_array,
            v=ch_ds[...],
            fs_hz=fs_hz,
            units=units,
            sensor_id=sensor_id,
        )

    gonio_offsets_rad: Optional[dict[str, float]] = None
    if "gonio_offsets_rad" in dg.attrs:
        gonio_offsets_rad = json.loads(_decode_attr(dg.attrs["gonio_offsets_rad"]))

    return DelsysSignals(
        channels=channels,
        trial_stem=grp.name.rsplit("/", 1)[-1],
        block_id=_decode_attr(dg.attrs["block_id"]) if "block_id" in dg.attrs else None,
        segment_start_s=float(dg.attrs["segment_start_s"]) if "segment_start_s" in dg.attrs else None,
        segment_end_s=float(dg.attrs["segment_end_s"]) if "segment_end_s" in dg.attrs else None,
        static_stem=_decode_attr(dg.attrs["static_stem"]) if "static_stem" in dg.attrs else None,
        gonio_offsets_rad=gonio_offsets_rad,
    )


def _decode_attr(value):
    """Decode h5py bytes/np-string attrs back to native Python types."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.dtype.kind in ("S", "U"):
        return value.astype(str).tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
