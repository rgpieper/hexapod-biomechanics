"""Fit ankle impedance and visualise the results for one (session, side, pct).

Runs the full Stage 2 + Stage 3 pipeline on a processed session's condition
HDF5 files and produces two figures:

    1. Isolation summary (4 rows x 2 cols).
       Left col: mean +/- 1 std of unsubtracted perturbed and nominal
                 trials, overlaid. Right col: isolated perturbation
                 trajectories across bootstrap iterations (mean +/- 1 std).
       Rows: ankle angle, ankle velocity, foot/ankle acceleration, ankle
       torque (all DF axis).

    2. Impedance fit (4 rows x 1 col + parameter inset).
       Rows 1-3: the exact theta, omega, alpha trajectories fed to lstsq
       (thin translucent lines per iteration + thick mean).
       Row 4: measured tau (negated) overlaid with predicted tau from the
       fitted model, each with iteration spread + mean line.
       A text inset reports K, B, I (mean +/- std), mass-normalised values,
       and VAF.

Run from the project root:

    uv run python scripts/fit_impedance.py \\
        subjects/RP081721/11082025 right 50

Interactive by default (matplotlib windows). With --save DIR, writes PNGs.
With --save-h5, writes the ImpedanceResult to the session's conditions dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    import hexapod_biomechanics  # noqa: F401
except ModuleNotFoundError:
    raise SystemExit(
        "hexapod_biomechanics not importable -- refresh the editable install:\n"
        "  python scripts/install.py"
    )

from hexapod_biomechanics.impedance import (
    ImpedanceResult,
    fit_impedance,
    save_impedance,
)
from hexapod_biomechanics.isolation import bootstrap_isolate
from hexapod_biomechanics.process_trials import TrialResult, load_condition

# DF axis index in the JCS decomposition.
DF_AXIS = 0

# Display unit conversions.
RAD_TO_DEG = 180.0 / np.pi
NMM_TO_NM = 1e-3


# ---------------------------------------------------------------------------
# Isolation summary figure (adapted from plot_isolated.py)
# ---------------------------------------------------------------------------

# Row layout for the isolation figure.  Each entry maps a display row to the
# TrialResult attribute (unsubtracted column) and the IsolatedPerturbation
# attribute (isolated column).
_ISO_ROW_TEMPLATE = [
    {"label": "Ankle angle", "unit": "deg",
     "trial_attr": "alpha_filt", "iso_attr": "alpha_filt",
     "is_3d": False, "scale": RAD_TO_DEG},
    {"label": "Ankle velocity", "unit": "rad/s",
     "trial_attr": "omega_jcs", "iso_attr": "omega_jcs",
     "is_3d": True, "scale": 1.0},
    {"label": "Foot acceleration", "unit": "rad/s\u00b2",
     "trial_attr": "foot_alpha_jcs", "iso_attr": "foot_alpha_jcs",
     "is_3d": True, "scale": 1.0},
    {"label": "Ankle torque", "unit": "N\u00b7m",
     "trial_attr": None, "iso_attr": None,   # filled by _build_iso_spec
     "is_3d": True, "scale": NMM_TO_NM},
]


def _build_iso_spec(dynamics: str, isolation_path: str) -> list[dict]:
    """Resolve path-dependent fields in the isolation row spec."""
    spec = [dict(r) for r in _ISO_ROW_TEMPLATE]
    # Angle row: show the variant that matches the isolation path.
    if isolation_path == "std":
        spec[0]["trial_attr"] = "alpha"
        spec[0]["iso_attr"] = "alpha"
    # Torque row: static or full.
    tau_attr = f"tau_{dynamics}"
    spec[-1]["trial_attr"] = tau_attr
    spec[-1]["iso_attr"] = tau_attr
    return spec


def _extract_df(obj, attr: str, is_3d: bool) -> np.ndarray:
    """Pull the DF-axis signal from an object attribute."""
    arr = getattr(obj, attr)
    return arr[:, DF_AXIS] if is_3d else arr


def _stack_pert_unsubtracted(
    perturbed: list[TrialResult], spec: list[dict],
) -> dict[str, np.ndarray]:
    """Stack perturbed trials' pert-window signals -> (n_trials, M) per row."""
    stacks: dict[str, list[np.ndarray]] = {r["label"]: [] for r in spec}
    for t in perturbed:
        sl = slice(t.perturbation.window_start_sample,
                   t.perturbation.window_end_sample)
        for r in spec:
            sig = _extract_df(t, r["trial_attr"], r["is_3d"])
            stacks[r["label"]].append(sig[sl])
    return {k: np.stack(v) for k, v in stacks.items()}


def _stack_nom_unsubtracted(
    nominal: list[TrialResult], spec: list[dict],
    start: int, end: int,
) -> dict[str, np.ndarray]:
    """Slice nominal trials at [start:end] -> (n_trials, M) per row."""
    stacks: dict[str, list[np.ndarray]] = {r["label"]: [] for r in spec}
    for t in nominal:
        for r in spec:
            sig = _extract_df(t, r["trial_attr"], r["is_3d"])
            stacks[r["label"]].append(sig[start:end])
    return {k: np.stack(v) for k, v in stacks.items()}


def _stack_iso(isolated: list, spec: list[dict]) -> dict[str, np.ndarray]:
    """Stack isolated iterations -> (n_iter, M) per row."""
    stacks: dict[str, list[np.ndarray]] = {r["label"]: [] for r in spec}
    for it in isolated:
        for r in spec:
            sig = _extract_df(it, r["iso_attr"], r["is_3d"])
            stacks[r["label"]].append(sig)
    return {k: np.stack(v) for k, v in stacks.items()}


def _plot_band(ax, t, mean, std, label, color):
    """Mean line + shaded std band."""
    ax.plot(t, mean, color=color, linewidth=1.5, label=label)
    ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.25)


def build_isolation_figure(
    perturbed: list[TrialResult],
    nominal: list[TrialResult],
    isolated: list,
    iso_spec: list[dict],
    time_axis: np.ndarray,
    title: str,
) -> plt.Figure:
    """Build the 4x2 isolation summary figure."""
    # Nominal window: centered on mean onset across ALL perturbed trials.
    all_onsets = np.array([t.perturbation.onset_sample for t in perturbed])
    mean_onset = int(round(float(np.mean(all_onsets))))
    window_len = len(time_axis)
    dt = time_axis[1] - time_axis[0]
    pre_frames = int(round(-time_axis[0] / dt))
    nom_start = mean_onset - pre_frames
    nom_end = nom_start + window_len

    pert_stacks = _stack_pert_unsubtracted(perturbed, iso_spec)
    nom_stacks = _stack_nom_unsubtracted(nominal, iso_spec, nom_start, nom_end)
    iso_stacks = _stack_iso(isolated, iso_spec)

    fig, axes = plt.subplots(
        nrows=len(iso_spec), ncols=2,
        figsize=(11, 2.6 * len(iso_spec)), sharex=True,
    )
    fig.suptitle(title)
    t_ms = time_axis * 1000.0

    for i, r in enumerate(iso_spec):
        ax_l, ax_r = axes[i, 0], axes[i, 1]
        label = r["label"]
        scale = r["scale"]

        # Left: unsubtracted perturbed + nominal.
        p_mean = np.mean(pert_stacks[label], axis=0) * scale
        p_std = np.std(pert_stacks[label], axis=0) * scale
        n_mean = np.mean(nom_stacks[label], axis=0) * scale
        n_std = np.std(nom_stacks[label], axis=0) * scale
        _plot_band(ax_l, t_ms, p_mean, p_std,
                   f"perturbed (n={len(perturbed)})", "tab:red")
        _plot_band(ax_l, t_ms, n_mean, n_std,
                   f"nominal (n={len(nominal)})", "tab:blue")

        # Right: isolated across bootstrap iterations.
        i_mean = np.mean(iso_stacks[label], axis=0) * scale
        i_std = np.std(iso_stacks[label], axis=0) * scale
        _plot_band(ax_r, t_ms, i_mean, i_std,
                   f"isolated (n={len(isolated)} iter)", "tab:purple")

        ax_l.set_ylabel(f"{label}\n[{r['unit']}]")
        for ax in (ax_l, ax_r):
            ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
            ax.grid(True, alpha=0.3)
        if i == 0:
            ax_l.set_title("Unsubtracted (perturbed vs nominal)")
            ax_r.set_title("Isolated (bootstrap subtraction)")
            ax_l.legend(loc="best", fontsize=8)
            ax_r.legend(loc="best", fontsize=8)
        if i == len(iso_spec) - 1:
            ax_l.set_xlabel("Time relative to onset [ms]")
            ax_r.set_xlabel("Time relative to onset [ms]")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Impedance fit figure
# ---------------------------------------------------------------------------

def build_fit_figure(result: ImpedanceResult, title: str) -> plt.Figure:
    """Build the impedance-fit figure: fit trajectories + torque comparison.

    Four rows (theta, omega, alpha, tau) x 1 col.  The first three rows show
    the exact signals fed to lstsq.  The fourth overlays measured (negated)
    and model-predicted torque.  All rows show individual bootstrap iterations
    as thin translucent lines with a thick mean on top.

    A text box in the upper-right reports K, B, I stats.
    """
    t_ms = result.time * 1000.0
    n_iter = len(result.K)

    # Row definitions: (array, label, unit, scale, color).
    # Torque row handled separately for the overlay.
    rows = [
        (result.theta, "Ankle angle (\u03b8)", "deg", RAD_TO_DEG, "tab:blue"),
        (result.omega, "Ankle velocity (\u03c9)", "rad/s", 1.0, "tab:orange"),
        (result.alpha, "Ankle acceleration (\u03b1)", "rad/s\u00b2", 1.0,
         "tab:green"),
    ]

    fig, axes = plt.subplots(
        nrows=4, ncols=1,
        figsize=(8, 10.5), sharex=True,
    )
    fig.suptitle(title)

    # Rows 1-3: theta, omega, alpha.
    for ax, (arr, label, unit, scale, color) in zip(axes[:3], rows):
        scaled = arr * scale                       # (N, M)
        for row in scaled:
            ax.plot(t_ms, row, color=color, alpha=0.10, linewidth=0.7)
        ax.plot(t_ms, scaled.mean(axis=0), color="black", linewidth=2.0)
        ax.set_ylabel(f"{label}\n[{unit}]")
        ax.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
        ax.grid(True, alpha=0.3)

    # Row 4: measured tau vs predicted tau.
    ax_tau = axes[3]
    for row in result.tau:
        ax_tau.plot(t_ms, row, color="tab:red", alpha=0.10, linewidth=0.7)
    for row in result.tau_predicted:
        ax_tau.plot(t_ms, row, color="tab:cyan", alpha=0.10, linewidth=0.7)
    # Mean lines on top.
    ax_tau.plot(t_ms, result.tau.mean(axis=0), color="tab:red",
                linewidth=2.0, label="measured")
    ax_tau.plot(t_ms, result.tau_predicted.mean(axis=0), color="tab:cyan",
                linewidth=2.0, label="predicted")
    ax_tau.set_ylabel("Ankle torque (\u03c4)\n[N\u00b7m]")
    ax_tau.set_xlabel("Time relative to onset [ms]")
    ax_tau.axvline(0, color="k", lw=0.5, ls="--", alpha=0.5)
    ax_tau.grid(True, alpha=0.3)
    ax_tau.legend(loc="lower left", fontsize=9)

    # Parameter summary text box.
    text = (
        f"K = {np.mean(result.K):.1f} \u00b1 {np.std(result.K):.1f} Nm/rad"
        f"  ({np.mean(result.K_norm):.2f} \u00b1 {np.std(result.K_norm):.2f}"
        f" Nm/rad/kg)\n"
        f"B = {np.mean(result.B):.2f} \u00b1 {np.std(result.B):.2f} Nm\u00b7s/rad"
        f"  ({np.mean(result.B_norm):.4f} \u00b1 {np.std(result.B_norm):.4f}"
        f" Nm\u00b7s/rad/kg)\n"
        f"I = {np.mean(result.I):.4f} \u00b1 {np.std(result.I):.4f} kg\u00b7m\u00b2"
        f"  ({np.mean(result.I_norm):.6f} \u00b1 {np.std(result.I_norm):.6f}"
        f" kg\u00b7m\u00b2/kg)\n"
        f"VAF = {np.mean(result.vaf):.1f} \u00b1 {np.std(result.vaf):.1f} %"
        f"  |  n_iter = {n_iter}"
    )
    fig.text(
        0.5, -0.01, text,
        ha="center", va="top", fontsize=9,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8),
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(result: ImpedanceResult) -> None:
    """Print impedance parameter statistics to stdout."""
    n = len(result.K)
    print(f"\n  Impedance fit: {result.side} pert {result.perturbation_pct}%"
          f"  ({result.isolation_path.upper()}, {result.dynamics} dynamics,"
          f" rezero={result.rezero})")
    print(f"  {n} bootstrap iterations,"
          f" {result.n_nominal_averaged} nominal trials averaged")
    print()
    print(f"  K  = {np.mean(result.K):8.2f} +/- {np.std(result.K):6.2f} Nm/rad"
          f"     ({np.mean(result.K_norm):.3f} +/- {np.std(result.K_norm):.3f}"
          f" Nm/rad/kg)")
    print(f"  B  = {np.mean(result.B):8.4f} +/- {np.std(result.B):6.4f} Nm*s/rad"
          f"   ({np.mean(result.B_norm):.5f} +/- {np.std(result.B_norm):.5f}"
          f" Nm*s/rad/kg)")
    print(f"  I  = {np.mean(result.I):8.6f} +/- {np.std(result.I):6.6f} kg*m^2"
          f"   ({np.mean(result.I_norm):.7f} +/- {np.std(result.I_norm):.7f}"
          f" kg*m^2/kg)")
    print(f"  VAF = {np.mean(result.vaf):5.1f} +/- {np.std(result.vaf):4.1f} %")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fit ankle impedance and visualise results.",
    )
    ap.add_argument("session_dir", type=Path,
                    help="Session directory containing conditions/*.h5.")
    ap.add_argument("side", choices=("right", "left"))
    ap.add_argument("pct", type=int,
                    help="Perturbation stance percentage, e.g. 50.")
    # Stage 2 options.
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for bootstrap subsampling.")
    ap.add_argument("--n-iterations", type=int, default=100)
    ap.add_argument("--sample-fraction", type=float, default=0.6)
    # Stage 3 options.
    ap.add_argument("--dynamics", choices=("static", "full"), default="static",
                    help="Which dynamics variant for torque.")
    ap.add_argument("--isolation-path", choices=("dts", "std"), default="dts",
                    help="DTS (primary) or STD (secondary) isolation path.")
    ap.add_argument("--no-rezero", action="store_true",
                    help="Disable first-sample re-zeroing of theta and tau.")
    # Output options.
    ap.add_argument("--save", type=Path, default=None,
                    help="Write PNGs to this directory (instead of showing).")
    ap.add_argument("--save-h5", action="store_true",
                    help="Write the ImpedanceResult HDF5 to conditions/.")
    args = ap.parse_args()

    # Load Stage 1 condition files.
    conditions_dir = args.session_dir / "conditions"
    pert_path = conditions_dir / f"{args.side}_pert_{args.pct}.h5"
    nom_path = conditions_dir / f"{args.side}_nom.h5"
    for p in (pert_path, nom_path):
        if not p.exists():
            raise SystemExit(f"Missing condition file: {p}")

    _, perturbed = load_condition(pert_path)
    _, nominal = load_condition(nom_path)
    body_mass = perturbed[0].body_mass_kg
    print(f"Loaded {len(perturbed)} perturbed, {len(nominal)} nominal trials"
          f"  (body mass {body_mass:.1f} kg)")

    # Stage 2: bootstrap isolation.
    isolated = bootstrap_isolate(
        perturbed, nominal,
        n_iterations=args.n_iterations,
        sample_fraction=args.sample_fraction,
        seed=args.seed,
    )
    print(f"Stage 2: {len(isolated)} bootstrap iterations,"
          f" window = {len(isolated[0].time)} samples")

    # Stage 3: impedance fitting.
    result = fit_impedance(
        isolated, body_mass,
        side=args.side,
        perturbation_pct=args.pct,
        dynamics=args.dynamics,
        isolation_path=args.isolation_path,
        rezero=not args.no_rezero,
    )
    print_summary(result)

    # Optionally save the impedance HDF5.
    if args.save_h5:
        h5_name = (f"{args.side}_pert_{args.pct}_impedance"
                    f"_{args.isolation_path}.h5")
        h5_path = conditions_dir / h5_name
        save_impedance(h5_path, result)
        print(f"Wrote {h5_path}")

    # Build figures.
    session_label = (f"{args.session_dir.parent.name}/"
                     f"{args.session_dir.name}")
    path_label = ("DTS (primary)" if args.isolation_path == "dts"
                  else "STD (secondary)")

    iso_spec = _build_iso_spec(args.dynamics, args.isolation_path)
    fig_iso = build_isolation_figure(
        perturbed, nominal, isolated, iso_spec,
        time_axis=isolated[0].time,
        title=(f"{session_label} \u2014 {args.side} pert {args.pct}% \u2014 "
               f"Isolation overview ({args.dynamics} dynamics, {path_label})"),
    )
    fig_fit = build_fit_figure(
        result,
        title=(f"{session_label} \u2014 {args.side} pert {args.pct}% \u2014 "
               f"Impedance fit ({args.dynamics} dynamics, {path_label})"),
    )

    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        tag = f"{args.side}_pert_{args.pct}_{args.isolation_path}"
        iso_path = args.save / f"isolation_{tag}.png"
        fit_path = args.save / f"impedance_{tag}.png"
        fig_iso.savefig(iso_path, dpi=150, bbox_inches="tight")
        fig_fit.savefig(fit_path, dpi=150, bbox_inches="tight")
        print(f"Wrote {iso_path}")
        print(f"Wrote {fit_path}")
        plt.close(fig_iso)
        plt.close(fig_fit)
    else:
        plt.show()


if __name__ == "__main__":
    main()
