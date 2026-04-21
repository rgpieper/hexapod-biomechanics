"""Batch-process every condition in one session.

Run from the project root:

    uv run python scripts/process_session.py subjects/RP081721/11082025

Discovers all `(side, trial_type, pct)` combinations from the session's
dynamic trial filenames and writes one HDF5 per condition into
`<session_dir>/conditions/`:

    right_nom.h5       left_nom.h5
    right_pert_50.h5   left_pert_50.h5   (and/or _20, etc.)

The MLP force-correction weights are loaded once per session and reused
across every trial.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

try:
    import hexapod_biomechanics  # noqa: F401 — preflight: editable install check
except ModuleNotFoundError:
    raise SystemExit(
        "hexapod_biomechanics not importable — refresh the editable install:\n"
        "  python scripts/install.py"
    )

from hexapod_biomechanics.config import load_session_config
from hexapod_biomechanics.force_correction import BasicMLP
from hexapod_biomechanics.process_trials import (
    _parse_trial_filename,
    iter_trial_specs,
    make_static_block_loader,
    process_condition,
)

# Shipped MLP geometry — fixed by the trained weights file.
# 12 accel channels × 500 samples = 6000 input features;
# 8 Kistler channels × 500 samples = 4000 output features.
MLP_WINDOW_SIZE = 500
MLP_ACCEL_CHANNELS = 12
MLP_FORCE_CHANNELS = 8
MLP_INPUT_DIM = MLP_ACCEL_CHANNELS * MLP_WINDOW_SIZE
MLP_OUTPUT_DIM = MLP_FORCE_CHANNELS * MLP_WINDOW_SIZE

DEFAULT_WEIGHTS = (
    Path(__file__).resolve().parent.parent / "models" / "mlp_general.pth"
)


def _discover_conditions(
    session_config,
) -> list[tuple[str, str, Optional[int]]]:
    """Return the distinct `(side, trial_type, pct)` tuples in this session.

    Pure filename scan — does not load any c3d files.
    """
    found: set[tuple[str, str, Optional[int]]] = set()
    for dyn_stems in session_config.static_blocks.values():
        for stem in dyn_stems:
            parsed = _parse_trial_filename(stem)
            if parsed is None:
                continue
            ttype, direction, pct = parsed
            if direction not in session_config.direction_side_map:
                continue
            side = session_config.direction_side_map[direction]
            found.add((side, ttype, pct))
    # Stable order for logging/output predictability.
    return sorted(found, key=lambda t: (t[0], t[1], t[2] or 0))


def _condition_stem(side: str, trial_type: str, pct: Optional[int]) -> str:
    """Build the output filename stem, e.g. 'right_pert_50' or 'left_nom'."""
    if trial_type == "nominal":
        return f"{side}_nom"
    return f"{side}_pert_{pct}"


def process_session(session_dir: Path, weights_path: Path) -> None:
    session_dir = Path(session_dir)
    config_path = session_dir / "session.toml"
    conditions_dir = session_dir / "conditions"

    config = load_session_config(config_path)

    # Load MLP weights once — reused for every trial in every condition.
    print(f"Loading MLP weights from {weights_path}")
    force_model = BasicMLP(input_dim=MLP_INPUT_DIM, output_dim=MLP_OUTPUT_DIM)
    force_model.load_weights(str(weights_path))

    # Static blocks are shared across conditions — cache them across the sweep.
    static_block_loader = make_static_block_loader(config, session_dir)

    conditions = _discover_conditions(config)
    if not conditions:
        print(f"No parseable dynamic trials found in {config_path}.")
        return

    print(
        f"Session {config.session_id}: found {len(conditions)} condition(s): "
        + ", ".join(_condition_stem(s, t, p) for s, t, p in conditions)
    )

    for side, trial_type, pct in conditions:
        stem = _condition_stem(side, trial_type, pct)
        output_h5 = conditions_dir / f"{stem}.h5"
        condition_id = f"{config.subject_id}_{config.session_id}_{stem}"

        # Materialize this condition's specs (also loads their c3ds).
        specs = list(iter_trial_specs(
            session_config=config,
            session_dir=session_dir,
            static_block_loader=static_block_loader,
            side=side,
            trial_type=trial_type,
            perturbation_pct=pct,
        ))
        print(f"\n[{stem}] processing {len(specs)} trial(s) → {output_h5}")

        process_condition(
            trials=specs,
            output_h5=output_h5,
            condition_id=condition_id,
            side=side,
            subject_id=config.subject_id,
            sex=config.sex,
            force_model=force_model,
            perturbation_pct=pct,
        )

    print("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "session_dir",
        type=Path,
        help="Session directory (contains session.toml and raw_data/).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"Path to MLP weights (default: {DEFAULT_WEIGHTS}).",
    )
    args = parser.parse_args()
    process_session(args.session_dir, args.weights)


if __name__ == "__main__":
    main()
