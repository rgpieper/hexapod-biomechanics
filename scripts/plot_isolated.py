"""Sanity-plot the bootstrap isolation for one (session, side, pct, path, dyn).

Requires `fit_impedance.py --save-h5` to have been run already for the
requested (path, dynamics) combination. The right column + iterations
figure are read straight from the impedance HDF5 so the plotted
trajectories are exactly what was fed to lstsq (already negated,
already rezeroed). The left column reads per-trial unsubtracted signals
from the condition HDF5 files.

Two figures:

    1. Summary (4 rows x 2 cols).
       Left col: mean +/- 1 std of unsubtracted per-trial signals
                 (perturbed sliced at each trial's own pert_window;
                 nominal sliced at a single window centered on mean
                 onset across all perturbed trials). For STD and
                 GONIO_STD paths, only angle + torque rows show on the
                 left -- omega/alpha don't exist per-trial under those
                 paths (they're derived from the isolated angle after
                 subtraction). The middle two left axes are hidden.
       Right col: per-iteration theta/omega/alpha/tau from the
                  impedance HDF5, mean +/- 1 std across iterations.
       Rows: angle, velocity, acceleration, torque (all DF axis).

    2. All bootstrap iterations (4 rows x 1 col).
       Thin translucent line per iteration + thick mean. Same source as
       the right column of the summary.

Sign convention: torque is always negated to match the sys-id fit
(impedance.py negates inverse-dynamics M_ank by Newton's 3rd law before
fitting). The right column / iterations figure pull from the impedance
HDF5 (already negated); the left column negates at extraction.

Run from the project root:

    uv run python scripts/plot_isolated.py \\
        subjects/RP041026/15102026 right 50 \\
        --path dts --dynamics static

Interactive by default. With --save DIR, writes
summary_<side>_pert_<pct>_<path>_<dyn>.png and
iterations_<side>_pert_<pct>_<path>_<dyn>.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import hexapod_biomechanics  # noqa: F401 — preflight: editable install check
except ModuleNotFoundError:
    raise SystemExit(
        "hexapod_biomechanics not importable — refresh the editable install:\n"
        "  python scripts/install.py"
    )

from hexapod_biomechanics.impedance import ImpedanceResult, load_impedance
from hexapod_biomechanics.isolation import (
    describe_nominal_selection,
    select_nominal_trials,
)
from hexapod_biomechanics.process_trials import TrialResult, load_condition

# DF axis index in the JCS decomposition.
DF_AXIS = 0

# Display unit conversions.
RAD_TO_DEG = 180.0 / np.pi
NMM_TO_NM = 1e-3


# ---------------------------------------------------------------------------
# Row spec per path
# ---------------------------------------------------------------------------
#
# Each row entry describes one display row. `left` is the per-trial source
# (or None when no per-trial equivalent exists under this path). `right` is
# the ImpedanceResult attribute holding the (n_iter, M) trajectory fed to
# lstsq. Signals are extracted in canonical SI units (rad, rad/s, rad/s^2,
# N*m); `scale` is the final display conversion.
#
# Left-source dict keys:
#   attr        : TrialResult attribute (or Delsys channel key if from_delsys)
#   is_3d       : True if the attribute is (N, 3) and needs DF-axis slice
#   from_delsys : True if attr is a Delsys channel key (t.delsys.channels[...])
#   negate      : True for torque (apply -1 at extraction time)
#   nmm_to_nm   : True for torque (also divide by 1000 at extraction time)


def _build_row_spec(path: str, dynamics: str, side: str) -> list[dict]:
    """Resolve path-dependent fields into a 4-row spec."""
    tau_attr = f"tau_{dynamics}"
    torque_left = {
        "attr": tau_attr, "is_3d": True, "from_delsys": False,
        "negate": True, "nmm_to_nm": True,
    }
    if path == "dts":
        return [
            {"label": "Ankle angle (filtered)", "unit": "deg", "scale": RAD_TO_DEG,
             "left": {"attr": "alpha_filt", "is_3d": False, "from_delsys": False},
             "right": "theta"},
            {"label": "Ankle velocity", "unit": "rad/s", "scale": 1.0,
             "left": {"attr": "omega_jcs", "is_3d": True, "from_delsys": False},
             "right": "omega"},
            {"label": "Foot acceleration", "unit": "rad/s²", "scale": 1.0,
             "left": {"attr": "foot_alpha_jcs", "is_3d": True, "from_delsys": False},
             "right": "alpha"},
            {"label": "Ankle torque", "unit": "N·m", "scale": 1.0,
             "left": torque_left, "right": "tau"},
        ]
    if path == "std":
        return [
            {"label": "Ankle angle (raw)", "unit": "deg", "scale": RAD_TO_DEG,
             "left": {"attr": "alpha", "is_3d": False, "from_delsys": False},
             "right": "theta"},
            {"label": "Ankle velocity", "unit": "rad/s", "scale": 1.0,
             "left": None, "right": "omega"},
            {"label": "Ankle acceleration", "unit": "rad/s²", "scale": 1.0,
             "left": None, "right": "alpha"},
            {"label": "Ankle torque", "unit": "N·m", "scale": 1.0,
             "left": torque_left, "right": "tau"},
        ]
    if path == "gonio_std":
        return [
            {"label": "Ankle angle (gonio)", "unit": "deg", "scale": RAD_TO_DEG,
             "left": {"attr": f"gonio_{side}_df_pf", "is_3d": False,
                      "from_delsys": True},
             "right": "theta"},
            {"label": "Ankle velocity", "unit": "rad/s", "scale": 1.0,
             "left": None, "right": "omega"},
            {"label": "Ankle acceleration", "unit": "rad/s²", "scale": 1.0,
             "left": None, "right": "alpha"},
            {"label": "Ankle torque", "unit": "N·m", "scale": 1.0,
             "left": torque_left, "right": "tau"},
        ]
    raise ValueError(f"Unknown path {path!r}.")


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------

def _extract_left_signal(t: TrialResult, src: dict) -> np.ndarray:
    """Pull a per-trial signal in canonical units (rad, rad/s, rad/s^2, N*m).

    Applies sign negation and N*mm -> N*m conversion for torque so the
    left column matches the sys-id sign convention.
    """
    if src["from_delsys"]:
        arr = t.delsys.channels[src["attr"]].v
    else:
        arr = getattr(t, src["attr"])
        if src["is_3d"]:
            arr = arr[:, DF_AXIS]
    if src.get("nmm_to_nm"):
        arr = arr * NMM_TO_NM
    if src.get("negate"):
        arr = -arr
    return arr


def _stack_perturbed_left(
    perturbed: list[TrialResult], spec: list[dict],
) -> dict[str, np.ndarray]:
    """Stack each perturbed trial's pert-window signals -> (n_trials, M)."""
    stacks: dict[str, list[np.ndarray]] = {
        r["label"]: [] for r in spec if r["left"] is not None
    }
    for t in perturbed:
        sl = slice(t.perturbation.window_start_sample,
                   t.perturbation.window_end_sample)
        for r in spec:
            if r["left"] is None:
                continue
            sig = _extract_left_signal(t, r["left"])
            stacks[r["label"]].append(sig[sl])
    return {k: np.stack(v) for k, v in stacks.items()}


def _stack_nominal_left(
    nominal: list[TrialResult], spec: list[dict],
    start: int, end: int,
) -> dict[str, np.ndarray]:
    """Slice each nominal trial at [start:end] -> (n_trials, M)."""
    stacks: dict[str, list[np.ndarray]] = {
        r["label"]: [] for r in spec if r["left"] is not None
    }
    for t in nominal:
        if start < 0 or end > len(t.alpha):
            raise ValueError(
                f"Nominal trial {t.trial_name!r}: window [{start}:{end}] "
                f"out of bounds for stance length {len(t.alpha)}."
            )
        for r in spec:
            if r["left"] is None:
                continue
            sig = _extract_left_signal(t, r["left"])
            stacks[r["label"]].append(sig[start:end])
    return {k: np.stack(v) for k, v in stacks.items()}


def _right_stack(result: ImpedanceResult, attr: str) -> np.ndarray:
    """Return the (n_iter, M) per-iteration array from the impedance result."""
    return getattr(result, attr)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_band(ax, t, mean, std, label, color):
    """Mean line + shaded std band."""
    ax.plot(t, mean, color=color, linewidth=1.5, label=label)
    ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.25)


def build_summary_figure(
    perturbed: list[TrialResult],
    nominal: list[TrialResult],
    result: ImpedanceResult,
    spec: list[dict],
    title: str,
) -> plt.Figure:
    """Build the 4x2 summary figure (unsubtracted left, isolated right)."""
    time_axis = result.time
    window_len = len(time_axis)
    dt = time_axis[1] - time_axis[0]
    pre_frames = int(round(-time_axis[0] / dt))

    # Nominal window: centered on mean onset across ALL perturbed trials.
    all_onsets = np.array([t.perturbation.onset_sample for t in perturbed])
    mean_onset = int(round(float(np.mean(all_onsets))))
    nom_start = mean_onset - pre_frames
    nom_end = nom_start + window_len

    pert_stacks = _stack_perturbed_left(perturbed, spec)
    nom_stacks = _stack_nominal_left(nominal, spec, nom_start, nom_end)

    fig, axes = plt.subplots(
        nrows=len(spec), ncols=2,
        figsize=(11, 2.6 * len(spec)), sharex=True,
    )
    fig.suptitle(title)
    t_ms = time_axis * 1000.0

    for i, r in enumerate(spec):
        ax_l, ax_r = axes[i, 0], axes[i, 1]
        label = r["label"]
        scale = r["scale"]

        # Left column: per-trial unsubtracted (if defined for this path).
        if r["left"] is not None:
            p = pert_stacks[label] * scale
            n = nom_stacks[label] * scale
            _plot_band(ax_l, t_ms, p.mean(axis=0), p.std(axis=0),
                       f"perturbed (n={len(perturbed)})", "tab:red")
            _plot_band(ax_l, t_ms, n.mean(axis=0), n.std(axis=0),
                       f"nominal (n={len(nominal)})", "tab:blue")
            ax_l.set_ylabel(f"{label}\n[{r['unit']}]")
            ax_l.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
            ax_l.grid(True, alpha=0.3)
        else:
            # No per-trial equivalent under this path; hide the axis.
            ax_l.set_visible(False)

        # Right column: per-iteration trajectories from impedance HDF5.
        right = _right_stack(result, r["right"]) * scale     # (n_iter, M)
        _plot_band(ax_r, t_ms, right.mean(axis=0), right.std(axis=0),
                   f"isolated (n={len(result.K)} iter)", "tab:purple")
        ax_r.set_ylabel(f"{label}\n[{r['unit']}]")
        ax_r.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
        ax_r.grid(True, alpha=0.3)

        if i == 0:
            if r["left"] is not None:
                ax_l.set_title("Unsubtracted (perturbed vs nominal)")
                ax_l.legend(loc="best", fontsize=8)
            ax_r.set_title("Fit trajectories (per bootstrap iter)")
            ax_r.legend(loc="best", fontsize=8)
        if i == len(spec) - 1:
            if r["left"] is not None:
                ax_l.set_xlabel("Time relative to onset [ms]")
            ax_r.set_xlabel("Time relative to onset [ms]")

    fig.tight_layout()
    return fig


def build_with_perturbation_figure(
    perturbed: list[TrialResult],
    result: ImpedanceResult,
    spec: list[dict],
    title: str,
) -> plt.Figure:
    """Isolated trajectories (mean +/- std) over the perturbation trajectory.

    5 rows x 1 col: rows 1-4 are isolated theta/omega/alpha/tau pulled from
    the impedance HDF5 (mean +/- std across bootstrap iterations); row 5 is
    the hexapod angular trajectory across the perturbation window (mean +/-
    std across all perturbed trials, from `PerturbationInfo.axis_angular_deviation`).

    The bottom row provides the "input" context for the responses above:
    each pert trial's hex angle is already windowed and time-aligned to its
    own onset, so stacking across trials and averaging is well-defined on
    the same time axis as the isolated trajectories.
    """
    time_axis = result.time
    t_ms = time_axis * 1000.0

    # Per-trial hexapod platform angle across the pert window (rad).
    # `hex_tjct` is the hexapod angular position about the perturbation axis
    # (full stance length); slice each trial at its own pert_window so the
    # band is time-aligned to onset.
    pert_traj = np.stack([
        t.hex_tjct[t.perturbation.window_start_sample
                   :t.perturbation.window_end_sample]
        for t in perturbed
    ])  # (n_trials, M)

    fig, axes = plt.subplots(
        nrows=len(spec) + 1, ncols=1,
        figsize=(8, 2.4 * (len(spec) + 1)), sharex=True,
    )
    fig.suptitle(title)

    # Rows 1..4: isolated trajectories, mean +/- std across iterations.
    for i, r in enumerate(spec):
        ax = axes[i]
        stack = _right_stack(result, r["right"]) * r["scale"]  # (n_iter, M)
        _plot_band(ax, t_ms, stack.mean(axis=0), stack.std(axis=0),
                   f"isolated (n={len(result.K)} iter)", "tab:purple")
        ax.set_ylabel(f"{r['label']}\n[{r['unit']}]")
        ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best", fontsize=8)

    # Row 5: perturbation trajectory across all perturbed trials.
    ax_pert = axes[-1]
    pert_deg = pert_traj * RAD_TO_DEG
    _plot_band(ax_pert, t_ms, pert_deg.mean(axis=0), pert_deg.std(axis=0),
               f"perturbation (n={len(perturbed)} trials)", "tab:orange")
    ax_pert.set_ylabel("Hexapod angle\n[deg]")
    ax_pert.set_xlabel("Time relative to onset [ms]")
    ax_pert.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax_pert.grid(True, alpha=0.3)
    ax_pert.legend(loc="best", fontsize=8)

    fig.tight_layout()
    return fig


def build_full_stance_figure(
    perturbed: list[TrialResult],
    nominal: list[TrialResult],
    dynamics: str,
    title: str,
) -> plt.Figure:
    """Overlay every trial's full-stance trajectory: pos/vel/acc/tau + platform.

    5 rows x 1 col, vs absolute stance time (seconds since stance start).
    Per-trial signals are plotted at each trial's own length (ragged ends);
    per-condition mean lines are computed over the common-prefix length
    (the shortest trial in each condition) so they're well-defined.

    Rows always use the canonical DTS-style signals (alpha_filt for position,
    omega_jcs[:, 0] for velocity, foot_alpha_jcs[:, 0] for acceleration),
    independent of `--path`, since STD/GONIO_STD don't have per-trial
    velocity/acceleration equivalents -- their derivatives only exist after
    the bootstrap-subtracted angle is filtered. Torque honors `--dynamics`.
    Platform angle uses the newly-stored `hex_tjct` (stance-sliced).
    """
    # (label, unit, scale, extractor) -- extractor returns a 1D (N,) array
    # in canonical SI units; `scale` then converts to display units.
    tau_attr = f"tau_{dynamics}"
    rows = [
        ("Ankle position", "deg", RAD_TO_DEG,
         lambda t: t.alpha_filt),
        ("Ankle velocity", "rad/s", 1.0,
         lambda t: t.omega_jcs[:, DF_AXIS]),
        ("Foot acceleration", "rad/s²", 1.0,
         lambda t: t.foot_alpha_jcs[:, DF_AXIS]),
        ("Ankle torque", "N·m", 1.0,
         lambda t: -getattr(t, tau_attr)[:, DF_AXIS] * NMM_TO_NM),
        ("Hexapod angle", "deg", RAD_TO_DEG,
         lambda t: t.hex_tjct),
    ]

    fig, axes = plt.subplots(
        nrows=len(rows), ncols=1,
        figsize=(9, 2.3 * len(rows)), sharex=True,
    )
    fig.suptitle(title)

    # Common-prefix lengths for the mean lines.
    n_pert_common = min(len(t.time) for t in perturbed)
    n_nom_common = min(len(t.time) for t in nominal) if nominal else 0
    # Use the first perturbed trial's time axis as the mean's x-coords (all
    # trials sample at the same fs so the prefix is shared).
    t_pert_common = perturbed[0].time[:n_pert_common]
    t_nom_common = nominal[0].time[:n_nom_common] if nominal else np.array([])

    for ax, (label, unit, scale, extract) in zip(axes, rows):
        # Per-trial thin lines.
        for t in nominal:
            ax.plot(t.time, extract(t) * scale,
                    color="tab:blue", alpha=0.15, linewidth=0.7)
        for t in perturbed:
            ax.plot(t.time, extract(t) * scale,
                    color="tab:red", alpha=0.15, linewidth=0.7)

        # Mean lines over the common prefix.
        if nominal:
            nom_stack = np.stack(
                [extract(t)[:n_nom_common] for t in nominal]
            ) * scale
            ax.plot(t_nom_common, nom_stack.mean(axis=0),
                    color="tab:blue", linewidth=2.0,
                    label=f"nominal mean (n={len(nominal)})")
        pert_stack = np.stack(
            [extract(t)[:n_pert_common] for t in perturbed]
        ) * scale
        ax.plot(t_pert_common, pert_stack.mean(axis=0),
                color="tab:red", linewidth=2.0,
                label=f"perturbed mean (n={len(perturbed)})")

        ax.set_ylabel(f"{label}\n[{unit}]")
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Time since stance start [s]")
    fig.tight_layout()
    return fig


def build_iterations_figure(
    result: ImpedanceResult,
    spec: list[dict],
    title: str,
) -> plt.Figure:
    """Build the 4x1 supplementary figure of all iterations overlaid."""
    time_axis = result.time
    fig, axes = plt.subplots(
        nrows=len(spec), ncols=1,
        figsize=(7, 2.6 * len(spec)), sharex=True,
    )
    fig.suptitle(title)
    t_ms = time_axis * 1000.0

    for i, r in enumerate(spec):
        ax = axes[i]
        stack = _right_stack(result, r["right"]) * r["scale"]    # (n_iter, M)
        for row in stack:
            ax.plot(t_ms, row, color="tab:purple", alpha=0.12, linewidth=0.7)
        ax.plot(t_ms, stack.mean(axis=0), color="black",
                linewidth=2.0, label="mean across iterations")
        ax.set_ylabel(f"{r['label']}\n[{r['unit']}]")
        ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Time relative to onset [ms]")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session_dir", type=Path,
                    help="Session directory containing conditions/*.h5.")
    ap.add_argument("side", choices=("right", "left"))
    ap.add_argument("pct", type=int,
                    help="Perturbation stance percentage, e.g. 50.")
    ap.add_argument("--path", choices=("dts", "std", "gonio_std"), default="dts",
                    help="Isolation path (default: dts).")
    ap.add_argument("--dynamics", choices=("static", "full"), default="static",
                    help="Which dynamics variant for torque (default: static).")
    ap.add_argument("--save", type=Path, default=None,
                    help="If given, write PNGs to this directory instead of "
                         "showing interactively.")
    args = ap.parse_args()

    conditions_dir = args.session_dir / "conditions"
    pert_path = conditions_dir / f"{args.side}_pert_{args.pct}.h5"
    nom_path = conditions_dir / f"{args.side}_nom.h5"
    imp_path = (conditions_dir /
                f"{args.side}_pert_{args.pct}_impedance"
                f"_{args.path}_{args.dynamics}.h5")
    for p in (pert_path, nom_path):
        if not p.exists():
            raise SystemExit(f"Missing condition file: {p}")
    if not imp_path.exists():
        raise SystemExit(
            f"Missing impedance HDF5: {imp_path}\n"
            f"Run first:\n"
            f"  uv run python scripts/fit_impedance.py {args.session_dir} "
            f"{args.side} {args.pct} "
            f"--isolation-path {args.path} --dynamics {args.dynamics} --save-h5"
        )

    _, perturbed = load_condition(pert_path)
    _, nominal = load_condition(nom_path)
    result = load_impedance(imp_path)

    # Safety check: HDF5 attrs must match requested config.
    if result.isolation_path != args.path or result.dynamics != args.dynamics:
        raise SystemExit(
            f"Impedance HDF5 attrs mismatch: file has "
            f"isolation_path={result.isolation_path!r}, "
            f"dynamics={result.dynamics!r}; "
            f"requested {args.path!r} / {args.dynamics!r}."
        )

    # Drop short nominal trials, same rule as fit_impedance.py, so the left
    # column reflects the same trial set the cached fit was built from.
    sel = select_nominal_trials(perturbed, nominal)
    print(describe_nominal_selection(sel))
    nominal = sel.kept

    spec = _build_row_spec(args.path, args.dynamics, args.side)

    path_labels = {
        "dts": "DTS (Vicon, pre-filtered)",
        "std": "STD (Vicon, post-subtraction filter)",
        "gonio_std": "GONIO_STD (Delsys goniometer)",
    }
    session_label = f"{args.session_dir.parent.name}/{args.session_dir.name}"
    title_base = (
        f"{session_label} — {args.side} pert {args.pct}% — "
        f"{args.dynamics} dynamics — {path_labels[args.path]}"
    )

    fig_summary = build_summary_figure(
        perturbed, nominal, result, spec, title=title_base,
    )
    fig_iters = build_iterations_figure(
        result, spec,
        title=(f"All {len(result.K)} bootstrap iterations — "
               f"{args.side} pert {args.pct}% — "
               f"{args.dynamics} — {args.path}"),
    )
    fig_pert = build_with_perturbation_figure(
        perturbed, result, spec,
        title=(f"Isolated trajectories over perturbation input — "
               f"{args.side} pert {args.pct}% — "
               f"{args.dynamics} — {args.path}"),
    )
    fig_full = build_full_stance_figure(
        perturbed, nominal, args.dynamics,
        title=(f"Full-stance per-trial trajectories — "
               f"{args.side} pert {args.pct}% — "
               f"{args.dynamics} dynamics"),
    )

    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        tag = f"{args.side}_pert_{args.pct}_{args.path}_{args.dynamics}"
        summary_path = args.save / f"summary_{tag}.png"
        iters_path = args.save / f"iterations_{tag}.png"
        pert_path = args.save / f"with_perturbation_{tag}.png"
        full_path = args.save / f"full_stance_{tag}.png"
        fig_summary.savefig(summary_path, dpi=150)
        fig_iters.savefig(iters_path, dpi=150)
        fig_pert.savefig(pert_path, dpi=150)
        fig_full.savefig(full_path, dpi=150)
        print(f"Wrote {summary_path}")
        print(f"Wrote {iters_path}")
        print(f"Wrote {pert_path}")
        print(f"Wrote {full_path}")
        plt.close(fig_summary)
        plt.close(fig_iters)
        plt.close(fig_pert)
        plt.close(fig_full)
    else:
        plt.show()


if __name__ == "__main__":
    main()
