"""Stage 2 — bootstrap isolation of the perturbation response.

The ankle-impedance experiment produces ~100 perturbed trials and ~100
nominal (unperturbed) trials per session/side. Normal walking biomechanics
vary substantially trial-to-trial, which masks the perturbation's effect
on any single comparison. We isolate the perturbation-induced deviation by
bootstrap-averaging:

    for each of `n_iterations` iterations:
        1. Randomly subsample `sample_fraction` of the perturbed trials.
        2. Stack each trial's stance-period signals over its own
           perturbation window (stance-relative, same length across trials)
           and average across the subsample.
        3. Compute the mean stance-relative onset sample across the
           subsample. Define a nominal window of the same length centered
           on that onset. Slice every nominal trial to that window and
           average across ALL of them (no subsampling on the nominal side).
        4. Subtract the nominal average from the perturbed average — per
           signal, independently — to get one `IsolatedPerturbation`.

## Angle storage

Both raw and filtered angle subtractions are always computed so the
`IsolatedPerturbation` is path-agnostic. Field naming mirrors
`TrialResult`: base name (`alpha/beta/gamma`) = raw (unfiltered),
`_filt` suffix = one-pass sav-gol deriv=0 filtered at Stage 1.

Non-angle signals (`omega_jcs`, `foot_omega_jcs`, `foot_alpha_jcs`,
`tau_full`, `tau_static`) are always from the Stage 1 one-pass-filtered
sources and are unaffected by the angle storage choice.

Stage 3 (`fit_impedance`) selects which angle fields to use based on
the isolation path:
  - **DTS** (differentiate-then-subtract): uses `*_filt` angles,
    `omega_jcs`, `foot_alpha_jcs` — all independently filtered once at
    Stage 1.
  - **STD** (subtract-then-differentiate): uses raw `alpha/beta/gamma`,
    applies a single sav-gol pass at Stage 3 to produce filtered θ, ω,
    and α from the isolated angle.

## Persistence

Outputs are returned in memory. Persistence is deferred to Stage 3, which
writes the isolated trajectories alongside their fitted impedance
parameters so every (K, B, I) triplet stays traceable to the trajectory it
was fit from.
"""

from dataclasses import dataclass
from typing import Sequence, Union

import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter

from hexapod_biomechanics.process_trials import TrialResult

RngLike = Union[int, np.random.Generator, None]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class IsolatedPerturbation:
    """One bootstrap iteration's isolated perturbation response.

    All signal arrays span the perturbation window (length M =
    pre_window + post_window samples, identical across iterations and
    trials). Time is relative to the perturbation onset: `time[0]` is
    negative (pre-onset), `time` crosses zero at the onset sample.
    """
    # identity / timing
    iteration: int                       # 0-based bootstrap iteration index
    fs_hz: float
    time: npt.NDArray                    # (M,) seconds, relative to onset
    # isolated kinematics — raw angles (unfiltered, for STD path at Stage 3)
    alpha: npt.NDArray                   # (M,) isolated DF/PF angle [rad]
    beta: npt.NDArray                    # (M,) isolated Inv/Ev angle [rad]
    gamma: npt.NDArray                   # (M,) isolated IntR/ExtR angle [rad]
    # isolated kinematics — filtered angles (sav-gol deriv=0, for DTS path)
    alpha_filt: npt.NDArray              # (M,) isolated DF/PF angle [rad]
    beta_filt: npt.NDArray               # (M,) isolated Inv/Ev angle [rad]
    gamma_filt: npt.NDArray              # (M,) isolated IntR/ExtR angle [rad]
    # isolated kinematics — velocity / acceleration (Stage 1 sav-gol derivatives)
    omega_jcs: npt.NDArray               # (M, 3) isolated ankle JCS ω [rad/s]
    foot_omega_jcs: npt.NDArray          # (M, 3) isolated foot JCS ω [rad/s]
    foot_alpha_jcs: npt.NDArray          # (M, 3) isolated foot JCS α [rad/s²]
    # isolated kinetics
    tau_full: npt.NDArray                # (M, 3) isolated full-dyn JCS torque [N*mm]
    tau_static: npt.NDArray              # (M, 3) isolated static-dyn JCS torque [N*mm]
    # bootstrap metadata
    perturbed_trial_names: list[str]     # trials in this iteration's subsample
    mean_onset_sample: int               # stance-relative mean onset (frames)
    mean_onset_stance_fraction: float    # mean onset phase across subsample
    n_nominal_averaged: int              # nominal trials averaged (all of them)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bootstrap_isolate(
    perturbed_trials: Sequence[TrialResult],
    nominal_trials: Sequence[TrialResult],
    n_iterations: int = 100,
    sample_fraction: float = 0.6,
    seed: RngLike = None,
) -> list[IsolatedPerturbation]:
    """Bootstrap-isolate the perturbation response from paired trial sets.

    Both inputs must come from the same session and ankle side (`TrialResult.side`
    must match across all trials). Perturbed trials must all have populated
    `perturbation` metadata with matching window length (enforced).

    Both raw and filtered angle subtractions are computed for every
    iteration. Stage 3 selects which to use based on the isolation path
    (DTS vs STD). See module docstring for details.

    Args:
        perturbed_trials: Perturbed-condition trial results (one side, one
            session, one perturbation percentage).
        nominal_trials: Nominal-condition trial results (same side/session).
        n_iterations: Number of bootstrap iterations.
        sample_fraction: Fraction of perturbed trials to subsample per
            iteration. `round(len * fraction)` trials are drawn without
            replacement each iteration.
        seed: Optional seed or `np.random.Generator` for reproducibility.
            None uses fresh non-deterministic entropy.

    Returns:
        One `IsolatedPerturbation` per iteration.

    Raises:
        ValueError: If inputs are empty, mixed across sides/sampling rates,
            have inconsistent perturbation windows, or if any nominal trial
            is too short to cover the mean-onset-centered window.
    """
    rng = np.random.default_rng(seed)
    fs, window_len, pre_frames = _validate_and_extract_window(
        perturbed_trials, nominal_trials
    )
    n_pert = len(perturbed_trials)
    n_sample = round(n_pert * sample_fraction)
    if n_sample < 1:
        raise ValueError(
            f"sample_fraction {sample_fraction} × {n_pert} perturbed trials "
            f"rounds to {n_sample}; need at least 1."
        )

    # Time axis relative to onset: [-pre, +post) at 1/fs spacing.
    time = (np.arange(window_len) - pre_frames) / fs

    # Pre-stack perturbed-window signals per trial for fast subsampling.
    pert_stacks = _stack_perturbed_windows(perturbed_trials)

    isolated: list[IsolatedPerturbation] = []
    for it in range(n_iterations):
        # Subsample indices (without replacement) into perturbed_trials.
        sample_idx = rng.choice(n_pert, size=n_sample, replace=False)
        sampled_names = [perturbed_trials[i].trial_name for i in sample_idx]

        # Average perturbed signals across subsample.
        pert_avg = {
            key: np.mean(arr[sample_idx], axis=0)
            for key, arr in pert_stacks.items()
        }

        # Mean onset across the subsample (rounded to int frame).
        onsets = np.array(
            [perturbed_trials[i].perturbation.onset_sample for i in sample_idx]
        )
        phases = np.array(
            [perturbed_trials[i].perturbation.onset_stance_fraction
             for i in sample_idx]
        )
        mean_onset = int(round(float(np.mean(onsets))))
        mean_phase = float(np.mean(phases))

        # Nominal window: same length, centered on mean onset.
        nom_start = mean_onset - pre_frames
        nom_end = nom_start + window_len
        nom_avg = _average_nominal_window(nominal_trials, nom_start, nom_end)

        # Isolate: independent subtraction per signal.
        isolated.append(IsolatedPerturbation(
            iteration=it,
            fs_hz=fs,
            time=time,
            # raw angles
            alpha=pert_avg["alpha"] - nom_avg["alpha"],
            beta=pert_avg["beta"] - nom_avg["beta"],
            gamma=pert_avg["gamma"] - nom_avg["gamma"],
            # filtered angles
            alpha_filt=pert_avg["alpha_filt"] - nom_avg["alpha_filt"],
            beta_filt=pert_avg["beta_filt"] - nom_avg["beta_filt"],
            gamma_filt=pert_avg["gamma_filt"] - nom_avg["gamma_filt"],
            # velocity / acceleration / torque
            omega_jcs=pert_avg["omega_jcs"] - nom_avg["omega_jcs"],
            foot_omega_jcs=pert_avg["foot_omega_jcs"]
                           - nom_avg["foot_omega_jcs"],
            foot_alpha_jcs=pert_avg["foot_alpha_jcs"]
                           - nom_avg["foot_alpha_jcs"],
            tau_full=pert_avg["tau_full"] - nom_avg["tau_full"],
            tau_static=pert_avg["tau_static"] - nom_avg["tau_static"],
            perturbed_trial_names=sampled_names,
            mean_onset_sample=mean_onset,
            mean_onset_stance_fraction=mean_phase,
            n_nominal_averaged=len(nominal_trials),
        ))

    return isolated


def differentiate_isolated_angle(
    isolated: IsolatedPerturbation,
    filt_window_duration: float = 0.05,
    filt_poly: int = 4,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Re-derive ankle angular velocity and acceleration from the isolated angle.

    Secondary path utility: differentiate-after-subtract. Applies the same
    Savitzky-Golay parameters as `AnkleID.compute_dynamics` so the
    filtering is consistent with the per-trial derivatives. Uses the raw
    (unfiltered) `alpha/beta/gamma` from the `IsolatedPerturbation` — the
    output is therefore *ankle* angular velocity / acceleration (relative
    rotation of foot w.r.t. shank), NOT foot angular velocity / acceleration
    (absolute rotation of the foot in global frame). The DTS path's
    `foot_alpha_jcs` is the latter; Stage 3 can compare the two.

    Args:
        isolated: One bootstrap iteration's isolated result.
        filt_window_duration: Savitzky-Golay window length [s].
        filt_poly: Filter polynomial order.

    Returns:
        Tuple `(omega_ankle, alpha_ankle)`, each of shape (M, 3) on the JCS
        axes (DF/PF, Inv/Ev, IntR/ExtR).
    """
    fs = isolated.fs_hz
    dt = 1.0 / fs
    win = int(fs * filt_window_duration)
    if win % 2 == 0:
        win += 1
    angles = np.column_stack([isolated.alpha, isolated.beta, isolated.gamma])
    omega = savgol_filter(
        angles, window_length=win, polyorder=filt_poly,
        deriv=1, delta=dt, axis=0,
    )
    alpha_acc = savgol_filter(
        angles, window_length=win, polyorder=filt_poly,
        deriv=2, delta=dt, axis=0,
    )
    return omega, alpha_acc


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# All TrialResult fields that are windowed and subtracted.
# Angle fields come in both raw and filtered variants; non-angle fields are
# always the Stage 1 one-pass-filtered versions.
_ANGLE_FIELDS = (
    "alpha", "beta", "gamma",
    "alpha_filt", "beta_filt", "gamma_filt",
)
_NON_ANGLE_FIELDS = (
    "omega_jcs",
    "foot_omega_jcs", "foot_alpha_jcs",
    "tau_full", "tau_static",
)
_ALL_FIELDS = (*_ANGLE_FIELDS, *_NON_ANGLE_FIELDS)


def _validate_and_extract_window(
    perturbed: Sequence[TrialResult],
    nominal: Sequence[TrialResult],
) -> tuple[float, int, int]:
    """Enforce invariants and return (fs, window_length, pre_onset_frames).

    Raises ValueError on: empty inputs, side mismatch, fs mismatch, missing
    perturbation metadata, pert window length mismatch, or nominal too
    short to support the mean-onset-centered window.

    `pre_frames` is derived from the first perturbed trial as
    `onset_sample - window_start_sample`; it's used to compute the time
    axis and to window the nominal trials symmetrically around the mean
    onset.
    """
    if not perturbed:
        raise ValueError("perturbed_trials is empty.")
    if not nominal:
        raise ValueError("nominal_trials is empty.")

    side = perturbed[0].side
    fs = perturbed[0].fs_hz
    for t in (*perturbed, *nominal):
        if t.side != side:
            raise ValueError(
                f"Side mismatch: expected {side!r}, got {t.side!r} for "
                f"trial {t.trial_name!r}."
            )
        if t.fs_hz != fs:
            raise ValueError(
                f"Sampling rate mismatch: expected {fs} Hz, got {t.fs_hz} "
                f"Hz for trial {t.trial_name!r}."
            )

    # Perturbation window consistency.
    windows = []
    for t in perturbed:
        if t.perturbation is None:
            raise ValueError(
                f"Perturbed trial {t.trial_name!r} has no perturbation "
                f"metadata."
            )
        p = t.perturbation
        windows.append((p.window_start_sample, p.window_end_sample,
                        p.onset_sample))
        if p.window_start_sample < 0:
            raise ValueError(
                f"Trial {t.trial_name!r}: pert window starts before stance "
                f"(window_start_sample={p.window_start_sample})."
            )
        if p.window_end_sample > len(t.alpha):
            raise ValueError(
                f"Trial {t.trial_name!r}: pert window ends beyond stance "
                f"({p.window_end_sample} > {len(t.alpha)})."
            )

    lengths = {w[1] - w[0] for w in windows}
    if len(lengths) != 1:
        raise ValueError(
            f"Perturbation window lengths differ across trials: {lengths}. "
            f"Re-run Stage 1 with consistent pert window kwargs."
        )
    window_len = lengths.pop()

    pre_frames_set = {w[2] - w[0] for w in windows}
    if len(pre_frames_set) != 1:
        raise ValueError(
            f"Pre-onset offset differs across perturbed trials: "
            f"{pre_frames_set}. Re-run Stage 1 with consistent kwargs."
        )
    pre_frames = pre_frames_set.pop()

    # Nominal trials must be long enough to support the worst-case window.
    # We conservatively require every nominal to cover any onset index that
    # could arise; the actual mean onset is checked per-iteration, but this
    # fails fast on obviously-too-short trials.
    max_onset = max(w[2] for w in windows)
    min_onset = min(w[2] for w in windows)
    post_frames = window_len - pre_frames
    for nt in nominal:
        if len(nt.alpha) < max_onset + post_frames:
            raise ValueError(
                f"Nominal trial {nt.trial_name!r}: stance length "
                f"{len(nt.alpha)} too short to cover window ending at "
                f"{max_onset + post_frames} (worst-case mean onset)."
            )
        if min_onset - pre_frames < 0:
            # Pre-window could extend before stance in extreme cases — but
            # we've already validated each perturbed trial's own window, so
            # this is a proxy check for nominal coverage at low onsets.
            raise ValueError(
                f"Minimum perturbed onset ({min_onset}) leaves no room for "
                f"pre-window ({pre_frames} frames) in nominal slicing."
            )

    return fs, window_len, pre_frames


def _stack_perturbed_windows(
    perturbed: Sequence[TrialResult],
) -> dict[str, npt.NDArray]:
    """Stack each perturbed trial's windowed signals into (n_trials, M, ...) arrays.

    Output dict is keyed by field name. Both raw and filtered angle variants
    are always included alongside the non-angle fields.

    Pre-stacking once avoids redoing the slicing inside the bootstrap loop.
    """
    stacks: dict[str, list[npt.NDArray]] = {k: [] for k in _ALL_FIELDS}
    for t in perturbed:
        sl = slice(t.perturbation.window_start_sample,
                   t.perturbation.window_end_sample)
        for field in _ALL_FIELDS:
            stacks[field].append(getattr(t, field)[sl])
    return {k: np.stack(v, axis=0) for k, v in stacks.items()}


def _average_nominal_window(
    nominal: Sequence[TrialResult],
    start: int,
    end: int,
) -> dict[str, npt.NDArray]:
    """Slice [start:end] from each nominal trial and average across trials.

    `start`/`end` are stance-relative indices. Bounds are re-checked here
    because the slice depends on the per-iteration mean onset, which isn't
    known during up-front validation.
    """
    avg: dict[str, list[npt.NDArray]] = {k: [] for k in _ALL_FIELDS}
    for nt in nominal:
        if start < 0 or end > len(nt.alpha):
            raise ValueError(
                f"Nominal trial {nt.trial_name!r}: requested window "
                f"[{start}:{end}] out of bounds for stance length "
                f"{len(nt.alpha)} (mean perturbed onset too near stance edges)."
            )
        sl = slice(start, end)
        for field in _ALL_FIELDS:
            avg[field].append(getattr(nt, field)[sl])
    return {k: np.mean(np.stack(v, axis=0), axis=0) for k, v in avg.items()}
