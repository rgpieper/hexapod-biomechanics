"""Generate a session.toml for one session directory.

Run from the project root:

    uv run python scripts/build_config.py \\
        subjects/RP081721/11082025 \\
        --subject-id RP081721 --body-mass 75.2 --sex m

Scans `<session_dir>/raw_data/` for `static*.c3d` + dynamic trials,
auto-detects Vicon parameters (hexapod prefix case, Kistler device ID,
accelerometer mode + IDs), defaults to grouping all dynamic trials under
the first static, and writes the result to `<session_dir>/session.toml`.

Review the generated file before running the pipeline:
- `[static_blocks]` — if the session has multiple statics (markers re-placed
  mid-session), reassign dynamic trial stems to the correct static.
- `[direction_side_map]` — map the trip direction to associated ankle side.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import hexapod_biomechanics  # noqa: F401 — preflight: editable install check
except ModuleNotFoundError:
    raise SystemExit(
        "hexapod_biomechanics not importable — refresh the editable install:\n"
        "  python scripts/install.py"
    )

from hexapod_biomechanics.config import (
    build_session_config,
    save_session_config,
)


def build_config(
    session_dir: Path,
    subject_id: str,
    body_mass: float,
    sex: str,
    session_id: str | None,
    force: bool,
) -> None:
    session_dir = Path(session_dir)
    toml_path = session_dir / "session.toml"

    if toml_path.exists() and not force:
        raise SystemExit(
            f"{toml_path} already exists. Pass --force to overwrite."
        )

    config = build_session_config(
        session_dir=session_dir,
        subject_id=subject_id,
        body_mass=body_mass,
        sex=sex,
        session_id=session_id,
    )
    save_session_config(config, toml_path)

    # Quick post-write summary so the user can sanity-check key fields.
    n_statics = len(config.static_blocks)
    n_dynamics = sum(len(v) for v in config.static_blocks.values())
    print(f"\nWrote {toml_path}")
    print(
        f"  subject={config.subject_id}  session={config.session_id}  "
        f"body_mass={config.body_mass}kg  sex={config.sex}"
    )
    print(
        f"  hexapod_prefix={config.hexapod_prefix!r}  "
        f"kistler_id={config.hexapod_kistler_device_id}"
    )
    print(
        f"  accel_mode={config.accel_mode!r}  "
        f"accel_device_ids={config.accel_device_ids}"
    )
    print(f"  statics={n_statics}  dynamic trials grouped={n_dynamics}")
    print(
        "\nReview the generated session.toml — in particular "
        "[static_blocks] (if >1 static) and [direction_side_map] "
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "session_dir",
        type=Path,
        help="Session directory (must contain raw_data/ with c3d files).",
    )
    parser.add_argument(
        "--subject-id",
        required=True,
        help="Subject ID used as marker prefix in c3d files (e.g. RP081721).",
    )
    parser.add_argument(
        "--body-mass",
        type=float,
        required=True,
        help="Subject body mass in kg.",
    )
    parser.add_argument(
        "--sex",
        choices=("m", "f"),
        required=True,
        help="Subject sex ('m' or 'f') — used for anthropometric inertia.",
    )
    parser.add_argument(
        "--session-id",
        default=None,
        help="Session identifier (default: session_dir basename).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing session.toml instead of aborting.",
    )
    args = parser.parse_args()
    build_config(
        session_dir=args.session_dir,
        subject_id=args.subject_id,
        body_mass=args.body_mass,
        sex=args.sex,
        session_id=args.session_id,
        force=args.force,
    )


if __name__ == "__main__":
    main()
