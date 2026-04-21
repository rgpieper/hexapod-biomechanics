"""Render one of the `visualize.animate_*` plots for a single trial.

Run from the project root:

    uv run python scripts/animate_trial.py \\
        subjects/RP081721/11082025 \\
        hex_20250811_trip12_pert_50pct_fwd \\
        --which perturbation

Use this for ad-hoc physical-plausibility checks after a pipeline change.
By default the animation pops up in a matplotlib window; pass `--save <path>`
(e.g. `--save /tmp/trip12.mp4` or `.gif`) to write it to disk instead.

`perturbation` and `demo` require a perturbed trial — the rotation axis
and hexapod angle trajectory aren't defined for nominal walking.
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

from hexapod_biomechanics.config import load_session_config
from hexapod_biomechanics.force_correction import BasicMLP
from hexapod_biomechanics.load_data import C3DMan
from hexapod_biomechanics.process_trials import (
    _SIDE_SIGN,
    TrialSpec,
    _parse_trial_filename,
    compute_trial_intermediates,
    make_static_block_loader,
    perturbation_window_slice,
)
from hexapod_biomechanics.visualize import (
    animate_ankle_kinematics,
    animate_demo,
    animate_grf,
    animate_perturbation,
)

# Shipped MLP geometry — keep in sync with process_session.py.
MLP_WINDOW_SIZE = 500
MLP_ACCEL_CHANNELS = 12
MLP_FORCE_CHANNELS = 8
MLP_INPUT_DIM = MLP_ACCEL_CHANNELS * MLP_WINDOW_SIZE
MLP_OUTPUT_DIM = MLP_FORCE_CHANNELS * MLP_WINDOW_SIZE

DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parent.parent / "models" / "mlp_general.pth"
)

WHICH_CHOICES = ("kinematics", "grf", "perturbation", "demo")
SCOPE_CHOICES = ("full", "stance", "pert")


def _find_static_block_id(session_config, trial_stem: str) -> str:
    """Look up which static block the given dynamic trial belongs to."""
    for static_stem, dyn_stems in session_config.static_blocks.items():
        if trial_stem in dyn_stems:
            return static_stem
    raise ValueError(
        f"Trial {trial_stem!r} is not listed under any static block in the "
        f"session config's [static_blocks] table."
    )


def _stack_tracked_markers(inter) -> np.ndarray:
    """Stack all tracked dynamic-trial markers into one (F, n_markers, 3) array."""
    foot = np.stack([inter.CALC, inter.M1, inter.M5], axis=1)  # (F, 3, 3)
    return np.concatenate(
        [foot, inter.cluster_S, inter.hex_markers], axis=1,
    )


def _load_all_markers(trial_data, session_config, n_samples: int) -> np.ndarray:
    """Load all subject + hexapod markers from the trial c3d.

    Filters `trial_data.point_map` down to labels prefixed with the subject
    ID or the hexapod prefix — this skips Vicon's unlabeled trajectories
    (stray reflections, `*N` ghosts) and any unrelated markers floating
    around the capture. Useful for broader body context in the 3D view.
    """
    sid_prefix = f"{session_config.subject_id}:"
    hex_prefix = f"{session_config.hexapod_prefix}:"
    labels = [
        lab for lab in trial_data.point_map.index
        if lab.startswith(sid_prefix) or lab.startswith(hex_prefix)
    ]
    if not labels:
        raise ValueError(
            f"No markers with subject prefix {sid_prefix!r} or hexapod "
            f"prefix {hex_prefix!r} found in trial c3d."
        )
    points = trial_data.get_points(labels, upsample=True)
    # get_points already resamples onto the analog-rate grid used elsewhere.
    assert points.values.shape[0] == n_samples, (
        f"Marker rate mismatch: got {points.values.shape[0]} samples, "
        f"expected {n_samples}."
    )
    return points.values                                # (F, n_markers, 3)


def _slice_markers(markers: np.ndarray, sl: slice) -> np.ndarray:
    return markers[sl]


def _resolve_scope_slice(inter, scope: str) -> slice:
    """Translate a --scope choice into a slice into full-capture arrays."""
    if scope == "full":
        return slice(0, inter.t_full.shape[0])
    if scope == "stance":
        return inter.stance_slice
    if scope == "pert":
        if inter.perturbation_axis is None:
            raise ValueError(
                "--scope pert requires a perturbed trial; this trial is "
                f"{inter.spec.trial_type!r}."
            )
        return perturbation_window_slice(inter)
    raise ValueError(f"Unknown --scope {scope!r}; choose from {SCOPE_CHOICES}.")


def _run_animation(
    inter,
    which: str,
    side_sign: int,
    save_path: Path | None,
    scope: str,
    all_markers: bool,
    speed: float,
) -> None:
    """Dispatch to the chosen animate_* function using the `scope`-sliced arrays."""
    sl = _resolve_scope_slice(inter, scope)
    if all_markers:
        markers = _load_all_markers(
            inter.spec.trial_data,
            inter.spec.session_config,
            inter.t_full.shape[0],
        )
    else:
        markers = _stack_tracked_markers(inter)
    t = inter.t_full[sl] - inter.t_full[sl.start]
    filename = str(save_path) if save_path is not None else None

    if which == "kinematics":
        kin = inter.kin
        anim = animate_ankle_kinematics(
            t=t,
            alpha=kin.alpha[sl],
            beta=kin.beta[sl],
            gamma=kin.gamma[sl],
            e1=kin.e1[sl],
            e2=kin.e2[sl],
            e3=kin.e3[sl],
            T_T=inter.T_TG[sl],
            T_C=inter.T_CG[sl],
            COM_F=kin.COM_F[sl],
            markers=_slice_markers(markers, sl),
            side=side_sign,
            speed=speed,
            filename=filename,
        )
    elif which == "grf":
        grf = inter.grf
        anim = animate_grf(
            t=t,
            forces=grf.force[sl],
            COP=grf.COP[sl],
            moment_free=grf.moment_free[sl],
            is_stance=grf.is_stance[sl],
            corners=inter.corners[sl],
            sensors=inter.sensors[sl],
            kist_origin_base=inter.kist_origin_base,
            side=side_sign,
            markers=_slice_markers(markers, sl),
            speed=speed,
            filename=filename,
        )
    elif which in ("perturbation", "demo"):
        if inter.perturbation_axis is None:
            raise ValueError(
                f"--which {which!r} requires a perturbed trial; this trial is "
                f"{inter.spec.trial_type!r}."
            )
        kin = inter.kin
        rot = inter.perturbation_axis
        if which == "perturbation":
            anim = animate_perturbation(
                t=t,
                alpha=kin.alpha[sl],
                M_e1=inter.dyn_full.M_jcs[sl, 0],
                o_ajc=kin.o_ajc[sl],
                e1=kin.e1[sl],
                o_rot=rot.origin,
                v_rot=rot.direction,
                pert_tjct=inter.grf.hex_tjct[sl],
                corners=inter.corners[sl],
                sensors=inter.sensors[sl],
                kist_origin_base=inter.kist_origin_base,
                side=side_sign,
                markers=_slice_markers(markers, sl),
                speed=speed,
                filename=filename,
            )
        else:  # demo
            body_mass = inter.spec.static_block.body_mass_kg
            anim = animate_demo(
                t=t,
                alpha=kin.alpha[sl],
                M_e1=-inter.dyn_full.M_jcs[sl, 0] / body_mass,
                o_ajc=kin.o_ajc[sl],
                e1=kin.e1[sl],
                e2=kin.e2[sl],
                T_T=inter.T_TG[sl],
                pert_tjct=inter.grf.hex_tjct[sl],
                corners=inter.corners[sl],
                kist_origin_base=inter.kist_origin_base,
                side=side_sign,
                markers=_slice_markers(markers, sl),
                speed=speed,
                filename=filename,
            )
    else:
        raise ValueError(f"Unknown --which {which!r}; choose from {WHICH_CHOICES}.")

    # animate_* functions call plt.close() internally when given a filename; in
    # that case there's nothing to display. When only previewing, show the
    # window interactively. Keep `anim` bound in the local scope so that the
    # FuncAnimation isn't garbage-collected before plt.show() renders it —
    # otherwise matplotlib warns "Animation was deleted without rendering".
    if save_path is None:
        plt.show()
        del anim  # explicit release after the window closes


def animate_trial(
    session_dir: Path,
    trial_stem: str,
    which: str,
    weights_path: Path,
    save_path: Path | None,
    scope: str = "stance",
    all_markers: bool = False,
    speed: float = 1.0,
) -> None:
    session_dir = Path(session_dir)
    config = load_session_config(session_dir / "session.toml")

    # Build the spec: locate the static block, parse filename for side + type,
    # load the trial c3d.
    static_stem = _find_static_block_id(config, trial_stem)
    parsed = _parse_trial_filename(trial_stem)
    if parsed is None:
        raise ValueError(
            f"Trial stem {trial_stem!r} does not match the expected "
            f"'_nom_<dir>' or '_pert_<N>pct_<dir>' pattern."
        )
    trial_type, direction, _pct = parsed
    side = config.direction_side_map[direction]

    print(f"Loading MLP weights from {weights_path}")
    force_model = BasicMLP(input_dim=MLP_INPUT_DIM, output_dim=MLP_OUTPUT_DIM)
    force_model.load_weights(str(weights_path))

    loader = make_static_block_loader(config, session_dir)
    static_block = loader(static_stem)

    trial_path = session_dir / "raw_data" / f"{trial_stem}.c3d"
    trial_data = C3DMan(str(trial_path))

    spec = TrialSpec(
        trial_data=trial_data,
        session_config=config,
        static_block=static_block,
        side=side,
        trial_type=trial_type,
        trial_name=trial_stem,
    )

    print(f"Computing intermediates for {trial_stem} ({trial_type}, {side})...")
    inter = compute_trial_intermediates(spec, force_model=force_model)

    print(
        f"Rendering '{which}' animation [scope={scope}, speed={speed}×"
        + (", all-markers" if all_markers else "")
        + "]"
        + (f" → {save_path}" if save_path else " (interactive)")
    )
    _run_animation(
        inter, which, _SIDE_SIGN[side], save_path,
        scope=scope, all_markers=all_markers, speed=speed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "session_dir",
        type=Path,
        help="Session directory (contains session.toml and raw_data/).",
    )
    parser.add_argument(
        "trial_stem",
        help="Exact dynamic trial filename stem, without .c3d extension.",
    )
    parser.add_argument(
        "--which",
        choices=WHICH_CHOICES,
        default="demo",
        help="Which animation to render (default: demo).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="If set, save the animation to this path (.mp4 or .gif) instead "
             "of displaying it interactively.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"Path to MLP weights (default: {DEFAULT_WEIGHTS}).",
    )
    parser.add_argument(
        "--scope",
        choices=SCOPE_CHOICES,
        default="stance",
        help="Time window to animate: 'full' = entire capture, "
             "'stance' = foot-on-plate interval (default), "
             "'pert' = perturbation onset window (perturbed trials only).",
    )
    parser.add_argument(
        "--all-markers",
        action="store_true",
        help="Show every marker present in the c3d (full-body context) "
             "instead of just the tracked foot/shank/hexapod clusters.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed scale (default 1.0 = real time). Values <1 "
             "slow the animation down — useful for short scopes like --scope "
             "pert where real-time is too fast to watch.",
    )
    args = parser.parse_args()
    animate_trial(
        args.session_dir, args.trial_stem, args.which,
        args.weights, args.save,
        scope=args.scope, all_markers=args.all_markers,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
