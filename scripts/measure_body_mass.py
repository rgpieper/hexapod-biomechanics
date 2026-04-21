"""Estimate body mass from a static c3d file's Kistler measurement.

Run before `build_config.py` to get a body-mass value you can pass as
`--body-mass` — and keep consistent across sessions for the same subject:

    uv run python scripts/measure_body_mass.py \\
        subjects/RP081721/11082025/raw_data/static_01.c3d

Auto-detects the hexapod-mounted Kistler (via FZ RMS energy) so this works on
any c3d export that contains raw Kistler channels — no session.toml required.
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
    KISTLER_RAW_CHANNELS,
    _detect_hexapod_kistler_id,
)
from hexapod_biomechanics.load_data import C3DMan
from hexapod_biomechanics.utils import get_body_mass


def measure_body_mass(c3d_path: Path) -> float:
    """Load a static c3d and return body mass [kg] from its Kistler channels."""
    c3d = C3DMan(str(c3d_path))
    dev_id = _detect_hexapod_kistler_id(c3d)
    desc = f"Kistler Force Plate 2.0.0.0::Raw [{dev_id}]"
    pairs = [(desc, ch) for ch in KISTLER_RAW_CHANNELS]
    raw_forces = c3d.get_analogs(pairs).values  # (N, 8)
    return get_body_mass(raw_forces)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "c3d_path",
        type=Path,
        help="Path to a static c3d (subject standing on the hexapod plate).",
    )
    args = parser.parse_args()

    mass = measure_body_mass(args.c3d_path)
    print(f"{mass:.2f} kg  ({args.c3d_path.name})")


if __name__ == "__main__":
    main()
