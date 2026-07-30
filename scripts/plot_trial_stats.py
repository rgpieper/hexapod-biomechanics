"""Per-trial summary-statistic histograms for outlier screening.

Reduces every trial in a session to one number per quantity (the mean, or
peak-to-peak range, over a window) and histograms those numbers separately
for each trial grouping. Trials that sit far from their group's median show
up as isolated bars in the tail, which is what makes this useful for
weeding out bad trials *before* bootstrap subtraction.

All signals are read pre-subtraction from the Stage 1 condition HDF5 files,
so this reflects the raw per-trial spread, not the isolated response.

Two window modes:

    stance (default)
        The whole stance period, each trial at its own length. Catches
        gross trial-level problems (marker dropout, mistimed step, wrong
        foot on the plate) regardless of perturbation timing.
    pert
        The perturbation analysis window only -- perturbed trials at their
        own `pert_window`, nominal trials at a window centered on the mean
        onset across that percentage's perturbed trials (the same rule
        `bootstrap_isolate` uses). Because nominal trials are re-windowed
        per percentage, nominal gets one column per percentage in this
        mode. Catches trials that are only anomalous where the fit
        actually looks.

Outlier flagging uses the modified z-score (Iglewicz & Hoaglin 1993):

    z_i = 0.6745 * (x_i - median) / MAD

which is robust to the outliers it is trying to find, unlike a mean/std
z-score. |z| > --z-thresh (default 3.5) is flagged. Flagged trials are
printed per group/quantity, plus a union list of trials flagged on any
quantity -- that union is the candidate exclusion set.

Sign conventions match the sys-id fit: torque is negated and converted
N*mm -> N*m, angles are shown in degrees.

Run from the project root:

    uv run python scripts/plot_trial_stats.py subjects/RP041026/15102026 right

    uv run python scripts/plot_trial_stats.py subjects/RP041026/15102026 right \\
        --window pert --save figures/

With --save DIR, writes the figure PNG plus a per-trial CSV of every
statistic and its modified z-score.
"""

from __future__ import annotations

import argparse
import csv
import re
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

from hexapod_biomechanics.process_trials import TrialResult, load_condition

# DF axis index in the JCS decomposition.
DF_AXIS = 0

# Display unit conversions.
RAD_TO_DEG = 180.0 / np.pi
NMM_TO_NM = 1e-3

# Iglewicz & Hoaglin's constant: makes MAD a consistent estimator of sigma
# for normally distributed data.
MAD_SCALE = 0.6745

_PERT_FILE_RE = re.compile(r"^(?P<side>right|left)_pert_(?P<pct>\d+)\.h5$")


# ---------------------------------------------------------------------------
# Quantities
# ---------------------------------------------------------------------------
#
# Each quantity reduces a TrialResult + window slice to one scalar. Most
# supply `extract`, which returns the windowed 1D signal in display units;
# `stat` then collapses it. A quantity may instead supply `scalar`, a direct
# trial -> float function, for things that aren't windowed signals (stance
# duration). Stance duration is included because a mis-segmented stance
# period dilutes every windowed mean at once, so it explains most
# correlated flags rather than being an independent outlier axis.


def _build_quantities(side: str, has_gonio: bool) -> list[dict]:
    """Return the quantity spec: label, unit, and a windowed extractor."""
    quantities = [
        {"label": "Stance duration", "unit": "s",
         "scalar": lambda t: float(t.time[-1] - t.time[0])},
        {"label": "Ankle angle (DF+)", "unit": "deg",
         "extract": lambda t, sl: t.alpha_filt[sl] * RAD_TO_DEG},
        {"label": "Ankle torque", "unit": "N·m",
         "extract": lambda t, sl: -t.tau_static[sl, DF_AXIS] * NMM_TO_NM},
        {"label": "Ankle velocity", "unit": "rad/s",
         "extract": lambda t, sl: t.omega_jcs[sl, DF_AXIS]},
        {"label": "Foot acceleration", "unit": "rad/s²",
         "extract": lambda t, sl: t.foot_alpha_jcs[sl, DF_AXIS]},
    ]
    if has_gonio:
        key = f"gonio_{side}_df_pf"
        quantities.insert(2, {
            "label": "Gonio angle (DF+)", "unit": "deg",
            "extract": (lambda t, sl, key=key:
                        t.delsys.channels[key].v[sl] * RAD_TO_DEG),
        })
    return quantities


def _reduce(sig: np.ndarray, stat: str) -> float:
    """Collapse a windowed signal to one scalar."""
    if stat == "mean":
        return float(np.mean(sig))
    if stat == "range":
        return float(np.ptp(sig))
    raise ValueError(f"Unknown stat {stat!r}.")


# ---------------------------------------------------------------------------
# Grouping / windowing
# ---------------------------------------------------------------------------

def _pert_window(t: TrialResult) -> slice:
    """The trial's own perturbation analysis window."""
    return slice(t.perturbation.window_start_sample,
                 t.perturbation.window_end_sample)


def _nominal_window(perturbed: list[TrialResult]) -> slice:
    """Nominal window centered on the mean onset across perturbed trials.

    Mirrors the alignment `bootstrap_isolate` applies to nominal trials, so
    the nominal statistics describe the same slice of stance the subtraction
    will use.
    """
    p0 = perturbed[0].perturbation
    pre_frames = p0.onset_sample - p0.window_start_sample
    window_len = p0.window_end_sample - p0.window_start_sample
    mean_onset = int(round(float(np.mean(
        [t.perturbation.onset_sample for t in perturbed]
    ))))
    start = mean_onset - pre_frames
    return slice(start, start + window_len)


def _build_groups(
    nominal: list[TrialResult],
    perturbed_by_pct: dict[int, list[TrialResult]],
    window: str,
) -> list[dict]:
    """Return the column spec: one entry per (label, trials, slicer, color)."""
    pcts = sorted(perturbed_by_pct)
    if window == "stance":
        groups = [{"label": f"nominal (n={len(nominal)})",
                   "trials": nominal,
                   "slicer": lambda t: slice(None),
                   "color": "tab:blue"}]
        for pct in pcts:
            trials = perturbed_by_pct[pct]
            groups.append({"label": f"pert {pct}% (n={len(trials)})",
                           "trials": trials,
                           "slicer": lambda t: slice(None),
                           "color": "tab:red"})
        return groups

    # window == "pert": nominal is re-windowed per percentage, so it gets one
    # column per percentage sitting next to its perturbed counterpart.
    groups = []
    for pct in pcts:
        trials = perturbed_by_pct[pct]
        nom_sl = _nominal_window(trials)
        nom_ok = [t for t in nominal
                  if nom_sl.start >= 0 and nom_sl.stop <= len(t.alpha)]
        n_dropped = len(nominal) - len(nom_ok)
        if n_dropped:
            print(f"  pert {pct}%: dropping {n_dropped} nominal trial(s) too "
                  f"short for window [{nom_sl.start}:{nom_sl.stop}]")
        groups.append({"label": f"nominal @{pct}% (n={len(nom_ok)})",
                       "trials": nom_ok,
                       "slicer": lambda t, sl=nom_sl: sl,
                       "color": "tab:blue"})
        groups.append({"label": f"pert {pct}% (n={len(trials)})",
                       "trials": trials,
                       "slicer": _pert_window,
                       "color": "tab:red"})
    return groups


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def modified_z(values: np.ndarray) -> np.ndarray:
    """Modified z-score against the median, scaled by the MAD.

    Returns zeros when the MAD is zero (a degenerate, all-identical group)
    rather than dividing by zero.
    """
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    if mad == 0:
        return np.zeros_like(values)
    return MAD_SCALE * (values - med) / mad


def compute_stats(
    groups: list[dict], quantities: list[dict], stat: str,
) -> dict[tuple[int, int], dict]:
    """Reduce every (group, quantity) pair to values + z-scores + names.

    Returns:
        {(group_index, quantity_index): {"values", "z", "names"}}.
    """
    table: dict[tuple[int, int], dict] = {}
    for gi, g in enumerate(groups):
        for qi, q in enumerate(quantities):
            if "scalar" in q:
                values = np.array([q["scalar"](t) for t in g["trials"]])
            else:
                values = np.array([
                    _reduce(q["extract"](t, g["slicer"](t)), stat)
                    for t in g["trials"]
                ])
            table[(gi, qi)] = {
                "values": values,
                "z": modified_z(values),
                "names": [t.trial_name for t in g["trials"]],
            }
    return table


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(
    groups: list[dict], quantities: list[dict],
    table: dict[tuple[int, int], dict], z_thresh: float,
) -> dict[str, set[str]]:
    """Print per-group flagged trials; return {group_label: flagged names}."""
    flagged_union: dict[str, set[str]] = {}
    for gi, g in enumerate(groups):
        print(f"\n  {g['label']}")
        union: set[str] = set()
        for qi, q in enumerate(quantities):
            entry = table[(gi, qi)]
            vals, z, names = entry["values"], entry["z"], entry["names"]
            hits = np.flatnonzero(np.abs(z) > z_thresh)
            print(f"    {q['label']:<22} median={np.median(vals):8.3f} "
                  f"MAD={np.median(np.abs(vals - np.median(vals))):7.3f} "
                  f"[{q['unit']}]  flagged={len(hits)}")
            for i in sorted(hits, key=lambda i: -abs(z[i])):
                print(f"        {names[i]:<40} {vals[i]:9.3f}  z={z[i]:+6.2f}")
                union.add(names[i])
        flagged_union[g["label"]] = union
    return flagged_union


def print_union(flagged_union: dict[str, set[str]]) -> None:
    """Print the candidate exclusion set: trials flagged on any quantity."""
    print("\n  Candidate exclusions (flagged on >=1 quantity)")
    total = 0
    for label, names in flagged_union.items():
        if not names:
            print(f"    {label}: none")
            continue
        total += len(names)
        print(f"    {label}: {len(names)}")
        for n in sorted(names):
            print(f"        {n}")
    print(f"\n  Total flagged trial-entries: {total}")


def write_csv(
    path: Path, groups: list[dict], quantities: list[dict],
    table: dict[tuple[int, int], dict], stat: str,
) -> None:
    """Write one row per (group, trial) with every statistic and z-score."""
    header = ["group", "trial"]
    for q in quantities:
        # Direct-scalar quantities (stance duration) aren't window statistics,
        # so they don't carry the `stat` suffix.
        tag = f"{q['label']} [{q['unit']}]"
        header += [tag if "scalar" in q else f"{tag} {stat}", f"{tag} z"]
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        for gi, g in enumerate(groups):
            names = table[(gi, 0)]["names"]
            for ti, name in enumerate(names):
                row = [g["label"], name]
                for qi in range(len(quantities)):
                    entry = table[(gi, qi)]
                    row += [f"{entry['values'][ti]:.6g}",
                            f"{entry['z'][ti]:.3f}"]
                writer.writerow(row)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def build_figure(
    groups: list[dict], quantities: list[dict],
    table: dict[tuple[int, int], dict],
    bins: int, z_thresh: float, stat: str, title: str,
) -> plt.Figure:
    """Grid of histograms: one row per quantity, one column per group.

    Rows share an x-axis so groups are directly comparable. The dashed line
    marks the group median; dotted lines mark the +/- z_thresh cutoffs, so
    anything outside them is a flagged trial.
    """
    n_rows, n_cols = len(quantities), len(groups)
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols,
        figsize=(3.1 * n_cols, 2.3 * n_rows),
        squeeze=False, sharey="row",
    )
    fig.suptitle(title)

    for qi, q in enumerate(quantities):
        # Common bin edges across the row so columns are visually comparable.
        row_vals = np.concatenate([table[(gi, qi)]["values"]
                                   for gi in range(n_cols)])
        edges = np.histogram_bin_edges(row_vals, bins=bins)

        for gi, g in enumerate(groups):
            ax = axes[qi][gi]
            entry = table[(gi, qi)]
            vals, z = entry["values"], entry["z"]
            ax.hist(vals, bins=edges, color=g["color"], alpha=0.75,
                    edgecolor="white", linewidth=0.4)

            med = np.median(vals)
            mad = np.median(np.abs(vals - med))
            ax.axvline(med, color="k", lw=1.0, ls="--", alpha=0.8)
            if mad > 0:
                half = z_thresh * mad / MAD_SCALE
                for x in (med - half, med + half):
                    ax.axvline(x, color="k", lw=0.8, ls=":", alpha=0.6)

            n_flag = int(np.sum(np.abs(z) > z_thresh))
            if n_flag:
                ax.text(0.97, 0.93, f"{n_flag} flagged", transform=ax.transAxes,
                        ha="right", va="top", fontsize=7, color="firebrick")

            ax.grid(True, axis="y", alpha=0.25)
            ax.tick_params(labelsize=8)
            if qi == 0:
                ax.set_title(g["label"], fontsize=9)
            if gi == 0:
                ax.set_ylabel(f"{q['label']}\n[{q['unit']}]", fontsize=9)
            if qi == n_rows - 1:
                ax.set_xlabel(f"per-trial {stat} [{q['unit']}]", fontsize=8)
            elif "scalar" in q:
                ax.set_xlabel(f"[{q['unit']}]", fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def discover_pcts(conditions_dir: Path, side: str) -> list[int]:
    """Find every perturbation percentage with a condition file for `side`."""
    pcts = []
    for p in sorted(conditions_dir.glob(f"{side}_pert_*.h5")):
        m = _PERT_FILE_RE.match(p.name)
        if m:                                    # skips *_impedance_*.h5
            pcts.append(int(m.group("pct")))
    return sorted(pcts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session_dir", type=Path,
                    help="Session directory containing conditions/*.h5.")
    ap.add_argument("side", choices=("right", "left"))
    ap.add_argument("--pct", type=int, nargs="*", default=None,
                    help="Perturbation percentages to include "
                         "(default: all found in conditions/).")
    ap.add_argument("--window", choices=("stance", "pert"), default="stance",
                    help="Averaging window: whole stance period, or the "
                         "perturbation analysis window (default: stance).")
    ap.add_argument("--stat", choices=("mean", "range"), default="mean",
                    help="Per-trial statistic: window mean, or peak-to-peak "
                         "range (default: mean).")
    ap.add_argument("--bins", type=int, default=25)
    ap.add_argument("--z-thresh", type=float, default=3.5,
                    help="Modified z-score cutoff for flagging (default: 3.5).")
    ap.add_argument("--no-gonio", action="store_true",
                    help="Skip the goniometer angle row even if available.")
    ap.add_argument("--save", type=Path, default=None,
                    help="Write PNG + CSV to this directory instead of "
                         "showing interactively.")
    args = ap.parse_args()

    conditions_dir = args.session_dir / "conditions"
    nom_path = conditions_dir / f"{args.side}_nom.h5"
    if not nom_path.exists():
        raise SystemExit(f"Missing condition file: {nom_path}")

    pcts = args.pct if args.pct else discover_pcts(conditions_dir, args.side)
    if not pcts:
        raise SystemExit(
            f"No perturbed condition files found in {conditions_dir}."
        )

    _, nominal = load_condition(nom_path)
    perturbed_by_pct: dict[int, list[TrialResult]] = {}
    for pct in pcts:
        p = conditions_dir / f"{args.side}_pert_{pct}.h5"
        if not p.exists():
            raise SystemExit(f"Missing condition file: {p}")
        _, perturbed_by_pct[pct] = load_condition(p)

    session_label = f"{args.session_dir.parent.name}/{args.session_dir.name}"
    print(f"{session_label} — {args.side}: {len(nominal)} nominal, "
          + ", ".join(f"{len(v)} pert {k}%"
                      for k, v in sorted(perturbed_by_pct.items())))

    # The gonio row needs the channel present on every trial in every group.
    gonio_key = f"gonio_{args.side}_df_pf"
    all_trials = nominal + [t for v in perturbed_by_pct.values() for t in v]
    has_gonio = (not args.no_gonio) and all(
        t.delsys is not None and gonio_key in t.delsys.channels
        for t in all_trials
    )
    if not has_gonio and not args.no_gonio:
        print(f"  ({gonio_key} not on every trial — omitting gonio row)")

    quantities = _build_quantities(args.side, has_gonio)
    groups = _build_groups(nominal, perturbed_by_pct, args.window)
    table = compute_stats(groups, quantities, args.stat)

    window_label = ("full stance" if args.window == "stance"
                    else "perturbation window")
    print(f"\n  Per-trial {args.stat} over the {window_label}; "
          f"flagging |modified z| > {args.z_thresh}")
    flagged_union = print_report(groups, quantities, table, args.z_thresh)
    print_union(flagged_union)

    fig = build_figure(
        groups, quantities, table,
        bins=args.bins, z_thresh=args.z_thresh, stat=args.stat,
        title=(f"{session_label} — {args.side} — per-trial {args.stat} "
               f"over {window_label}"),
    )

    if args.save is not None:
        args.save.mkdir(parents=True, exist_ok=True)
        tag = f"{args.side}_{args.window}_{args.stat}"
        png_path = args.save / f"trial_stats_{tag}.png"
        csv_path = args.save / f"trial_stats_{tag}.csv"
        fig.savefig(png_path, dpi=150)
        write_csv(csv_path, groups, quantities, table, args.stat)
        print(f"\nWrote {png_path}")
        print(f"Wrote {csv_path}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
