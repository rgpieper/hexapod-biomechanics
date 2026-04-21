"""Sanity-plot the Stage 2 bootstrap isolation for one (session, side, pct).

Produces two figures from the condition's nominal + perturbed HDF5 files:

    1. Summary (4 rows × 2 cols).
       Left col: mean ± 1 std of UNSUBTRACTED perturbed trials and nominal
                 trials, overlaid. Perturbed trials are sliced at each
                 trial's own pert_window and time-aligned to each onset.
                 Nominal trials are sliced at a single window centered on
                 the mean onset across ALL perturbed trials.
       Right col: mean ± 1 std across the `n_iterations` bootstrap-isolated
                 trajectories (perturbed avg - nominal avg, per iteration).
       Rows (all DF axis, JCS index 0): ankle position, ankle velocity,
       foot acceleration, ankle torque.

    2. All bootstrap iterations (4 rows × 1 col).
       Thin translucent lines for every iteration + a thicker mean line.
       Same rows as above.

Run from the project root:

    uv run python scripts/plot_isolated.py \\
        subjects/RP081721/11082025 right 50

Interactive by default (pops both figures via plt.show). With --save DIR,
writes summary_<side>_pert_<pct>.png and iterations_<side>_pert_<pct>.png.
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

from hexapod_biomechanics.isolation import bootstrap_isolate
from hexapod_biomechanics.process_trials import TrialResult, load_condition

# DF axis index in the JCS decomposition (alpha / omega_jcs / tau / foot_*).
DF_AXIS = 0

# Unit conversions applied only for display.
RAD_TO_DEG = 180.0 / np.pi
NMM_TO_NM = 1e-3

# Row layout: (label, signal_key_on_trial, signal_key_on_iso, unit, scale).
# `signal_key_on_trial` is the TrialResult attribute used for the left column.
# `signal_key_on_iso` is the IsolatedPerturbation attribute for the right and
# supplementary plots. `scale` converts the raw SI value to display units.
ROW_SPEC_TEMPLATE = [
    # Ankle position — DF axis, converted to degrees.
    # iso_attr is set by _build_row_spec depending on --use-raw-angles.
    {"label": "Ankle position", "unit": "deg",
     "trial_attr": "alpha_filt", "iso_attr": "alpha_filt",
     "is_3d": False, "scale": RAD_TO_DEG},
    # Ankle velocity — DF axis of JCS ω, in rad/s.
    {"label": "Ankle velocity", "unit": "rad/s",
     "trial_attr": "omega_jcs", "iso_attr": "omega_jcs",
     "is_3d": True, "scale": 1.0},
    # Foot acceleration — DF axis of foot α, in rad/s².
    {"label": "Foot acceleration", "unit": "rad/s²",
     "trial_attr": "foot_alpha_jcs", "iso_attr": "foot_alpha_jcs",
     "is_3d": True, "scale": 1.0},
    # Ankle torque — DF axis, tau_static or tau_full depending on CLI.
    # Placeholder fields replaced by `_build_row_spec`.
    {"label": "Ankle torque", "unit": "N·m",
     "trial_attr": None, "iso_attr": None,
     "is_3d": True, "scale": NMM_TO_NM},
]


def _build_row_spec(dynamics: str, use_raw_angles: bool) -> list[dict]:
    """Fill dynamics-specific + raw/filtered-angle fields into the row spec."""
    spec = [dict(r) for r in ROW_SPEC_TEMPLATE]
    if use_raw_angles:
        # Secondary path: show raw angles for both columns.
        spec[0]["trial_attr"] = "alpha"
        spec[0]["iso_attr"] = "alpha"
    # Torque row resolves to tau_static or tau_full.
    tau_attr = f"tau_{dynamics}"
    spec[-1]["trial_attr"] = tau_attr
    spec[-1]["iso_attr"] = tau_attr
    return spec


def _extract_trial_signal(trial: TrialResult, attr: str, is_3d: bool
                          ) -> np.ndarray:
    """Pull a DF-axis signal off a TrialResult, respecting 3D vs 1D fields."""
    arr = getattr(trial, attr)
    return arr[:, DF_AXIS] if is_3d else arr


def _extract_iso_signal(iso, attr: str, is_3d: bool) -> np.ndarray:
    """Pull a DF-axis signal off an IsolatedPerturbation."""
    arr = getattr(iso, attr)
    return arr[:, DF_AXIS] if is_3d else arr


def _stack_perturbed_unsubtracted(
    perturbed: list[TrialResult], row_spec: list[dict],
) -> dict[str, np.ndarray]:
    """Stack each perturbed trial's pert-window signals into (n_trials, M) arrays.

    Each trial is sliced at its own pert_window so the resulting stack is
    time-aligned to each trial's own onset.
    """
    stacks: dict[str, list[np.ndarray]] = {r["label"]: [] for r in row_spec}
    for t in perturbed:
        sl = slice(t.perturbation.window_start_sample,
                   t.perturbation.window_end_sample)
        for r in row_spec:
            sig = _extract_trial_signal(t, r["trial_attr"], r["is_3d"])
            stacks[r["label"]].append(sig[sl])
    return {k: np.stack(v, axis=0) for k, v in stacks.items()}


def _stack_nominal_unsubtracted(
    nominal: list[TrialResult], row_spec: list[dict],
    start: int, end: int,
) -> dict[str, np.ndarray]:
    """Slice every nominal trial at [start:end] and stack into (n_trials, M) arrays."""
    stacks: dict[str, list[np.ndarray]] = {r["label"]: [] for r in row_spec}
    for t in nominal:
        if start < 0 or end > len(t.alpha):
            raise ValueError(
                f"Nominal trial {t.trial_name!r}: stance [{start}:{end}] "
                f"out of bounds for length {len(t.alpha)}."
            )
        for r in row_spec:
            sig = _extract_trial_signal(t, r["trial_attr"], r["is_3d"])
            stacks[r["label"]].append(sig[start:end])
    return {k: np.stack(v, axis=0) for k, v in stacks.items()}


def _stack_iso(
    isolated: list, row_spec: list[dict],
) -> dict[str, np.ndarray]:
    """Stack isolated trajectories into (n_iterations, M) arrays per row."""
    stacks: dict[str, list[np.ndarray]] = {r["label"]: [] for r in row_spec}
    for it in isolated:
        for r in row_spec:
            sig = _extract_iso_signal(it, r["iso_attr"], r["is_3d"])
            stacks[r["label"]].append(sig)
    return {k: np.stack(v, axis=0) for k, v in stacks.items()}


def _mean_std(stack: np.ndarray, scale: float
              ) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) along the trial/iteration axis, scaled for display."""
    return np.mean(stack, axis=0) * scale, np.std(stack, axis=0) * scale


def _plot_band(ax, t, mean, std, label, color):
    """Plot mean line + mean ± std shaded band."""
    ax.plot(t, mean, color=color, linewidth=1.5, label=label)
    ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.25)


def build_summary_figure(
    perturbed: list[TrialResult],
    nominal: list[TrialResult],
    isolated: list,
    row_spec: list[dict],
    time_axis: np.ndarray,
    title: str,
) -> plt.Figure:
    """Build the 4×2 summary figure (unsubtracted left, isolated right)."""
    # Left-column nominal window: single window centered on the mean onset
    # across ALL perturbed trials (not bootstrap-dependent). This gives a
    # stable, unbiased view of the "typical" nominal trajectory at the same
    # stance phase.
    all_onsets = np.array(
        [t.perturbation.onset_sample for t in perturbed]
    )
    mean_onset = int(round(float(np.mean(all_onsets))))
    window_len = len(time_axis)
    # Pre-frame count inferred from time_axis (time[0] = -pre / fs).
    dt = time_axis[1] - time_axis[0]
    pre_frames = int(round(-time_axis[0] / dt))
    nom_start = mean_onset - pre_frames
    nom_end = nom_start + window_len

    pert_stacks = _stack_perturbed_unsubtracted(perturbed, row_spec)
    nom_stacks = _stack_nominal_unsubtracted(
        nominal, row_spec, nom_start, nom_end,
    )
    iso_stacks = _stack_iso(isolated, row_spec)

    fig, axes = plt.subplots(
        nrows=len(row_spec), ncols=2,
        figsize=(11, 2.6 * len(row_spec)),
        sharex=True,
    )
    fig.suptitle(title)

    t_ms = time_axis * 1000.0

    for i, r in enumerate(row_spec):
        ax_l = axes[i, 0]
        ax_r = axes[i, 1]
        label = r["label"]
        unit = r["unit"]
        scale = r["scale"]

        # Left: perturbed (n_pert trials) + nominal (n_nom trials), unsubtracted.
        p_mean, p_std = _mean_std(pert_stacks[label], scale)
        n_mean, n_std = _mean_std(nom_stacks[label], scale)
        _plot_band(ax_l, t_ms, p_mean, p_std,
                   label=f"perturbed (n={len(perturbed)})", color="tab:red")
        _plot_band(ax_l, t_ms, n_mean, n_std,
                   label=f"nominal (n={len(nominal)})", color="tab:blue")

        # Right: isolated, across bootstrap iterations.
        i_mean, i_std = _mean_std(iso_stacks[label], scale)
        _plot_band(ax_r, t_ms, i_mean, i_std,
                   label=f"isolated (n={len(isolated)} iter)",
                   color="tab:purple")

        ax_l.set_ylabel(f"{label}\n[{unit}]")
        for ax in (ax_l, ax_r):
            ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
            ax.grid(True, alpha=0.3)
        if i == 0:
            ax_l.set_title("Unsubtracted (perturbed vs nominal)")
            ax_r.set_title("Isolated (bootstrap subtraction)")
            ax_l.legend(loc="best", fontsize=8)
            ax_r.legend(loc="best", fontsize=8)
        if i == len(row_spec) - 1:
            ax_l.set_xlabel("Time relative to perturbation onset [ms]")
            ax_r.set_xlabel("Time relative to perturbation onset [ms]")

    fig.tight_layout()
    return fig


def build_iterations_figure(
    isolated: list,
    row_spec: list[dict],
    time_axis: np.ndarray,
    title: str,
) -> plt.Figure:
    """Build the 4×1 supplementary figure of all iterations overlaid."""
    iso_stacks = _stack_iso(isolated, row_spec)
    fig, axes = plt.subplots(
        nrows=len(row_spec), ncols=1,
        figsize=(7, 2.6 * len(row_spec)),
        sharex=True,
    )
    fig.suptitle(title)

    t_ms = time_axis * 1000.0

    for i, r in enumerate(row_spec):
        ax = axes[i]
        label = r["label"]
        scale = r["scale"]
        stack = iso_stacks[label] * scale           # (n_iter, M)

        # Thin translucent lines for each iteration.
        for row in stack:
            ax.plot(t_ms, row, color="tab:purple",
                    alpha=0.12, linewidth=0.7)
        # Thicker mean line on top.
        ax.plot(t_ms, stack.mean(axis=0), color="black",
                linewidth=2.0, label="mean across iterations")

        ax.set_ylabel(f"{label}\n[{r['unit']}]")
        ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Time relative to perturbation onset [ms]")
    fig.tight_layout()
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session_dir", type=Path,
                    help="Session directory containing conditions/*.h5.")
    ap.add_argument("side", choices=("right", "left"))
    ap.add_argument("pct", type=int,
                    help="Perturbation stance percentage, e.g. 50.")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for reproducible bootstrap subsampling.")
    ap.add_argument("--n-iterations", type=int, default=100)
    ap.add_argument("--sample-fraction", type=float, default=0.6)
    ap.add_argument("--use-raw-angles", action="store_true",
                    help="Use the secondary path (raw angles). Default uses "
                         "filtered angles (primary path).")
    ap.add_argument("--dynamics", choices=("static", "full"), default="static",
                    help="Which dynamics variant to show for ankle torque.")
    ap.add_argument("--save", type=Path, default=None,
                    help="If given, write PNGs to this directory instead of "
                         "showing interactively.")
    args = ap.parse_args()

    conditions_dir = args.session_dir / "conditions"
    pert_path = conditions_dir / f"{args.side}_pert_{args.pct}.h5"
    nom_path = conditions_dir / f"{args.side}_nom.h5"
    for p in (pert_path, nom_path):
        if not p.exists():
            raise SystemExit(f"Missing condition file: {p}")

    _, perturbed = load_condition(pert_path)
    _, nominal = load_condition(nom_path)

    isolated = bootstrap_isolate(
        perturbed, nominal,
        n_iterations=args.n_iterations,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
    )

    row_spec = _build_row_spec(args.dynamics, args.use_raw_angles)

    path_label = "secondary (raw angles)" if args.use_raw_angles else "primary (filtered angles)"
    title_base = (
        f"{args.session_dir.parent.name}/{args.session_dir.name} — "
        f"{args.side} pert {args.pct}% — "
        f"{args.dynamics} dynamics — {path_label}"
    )

    fig_summary = build_summary_figure(
        perturbed, nominal, isolated, row_spec,
        time_axis=isolated[0].time,
        title=title_base,
    )
    fig_iters = build_iterations_figure(
        isolated, row_spec,
        time_axis=isolated[0].time,
        title=f"All {len(isolated)} bootstrap iterations — "
              f"{args.side} pert {args.pct}%",
    )

    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        summary_path = args.save / f"summary_{args.side}_pert_{args.pct}.png"
        iters_path = args.save / f"iterations_{args.side}_pert_{args.pct}.png"
        fig_summary.savefig(summary_path, dpi=150)
        fig_iters.savefig(iters_path, dpi=150)
        print(f"Wrote {summary_path}")
        print(f"Wrote {iters_path}")
        plt.close(fig_summary)
        plt.close(fig_iters)
    else:
        plt.show()


if __name__ == "__main__":
    main()
