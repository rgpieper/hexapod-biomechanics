"""Refresh the editable install of hexapod_biomechanics in the active env.

Run from the project root:

    python scripts/install.py

When to use:
- After cloning the repo (once your venv has the deps installed), to make
  the source importable as `hexapod_biomechanics`.
- Any time `import hexapod_biomechanics` fails with `ModuleNotFoundError`
  despite the package appearing in `pip list` / `uv pip list`.

What it does:
1. Reinstalls the project editable into the active Python environment
   (via `uv pip` if `uv` is on PATH, otherwise `python -m pip`).
2. On macOS, strips the `UF_HIDDEN` filesystem flag from `.venv/`. macOS
   inherits that flag from the leading-dot directory name, and CPython
   3.12+ silently skips hidden `.pth` files during site init — which is
   what breaks the editable install in the first place.
3. Verifies by importing the package.

Prerequisite: deps are already installed. Run `uv sync` (uv) or
`pip install -e .` (vanilla venv) first if this is a cold clone.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def refresh_editable_install() -> None:
    """Force-rewrite the editable .pth for this project (no dep reinstall)."""
    if shutil.which("uv") is not None:
        _run(["uv", "pip", "install", "-e", ".", "--no-deps", "--force-reinstall"])
        return

    pip_probe = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True,
    )
    if pip_probe.returncode == 0:
        _run([
            sys.executable, "-m", "pip",
            "install", "-e", ".", "--no-deps", "--force-reinstall",
        ])
        return

    raise SystemExit(
        "Neither `uv` nor `pip` is available. Install one (e.g. "
        "`python -m ensurepip` in a vanilla venv, or install uv) and retry."
    )


def strip_macos_hidden_flag() -> None:
    """Clear UF_HIDDEN on .venv/ so Python 3.12+ processes our .pth files.

    No-op on non-macOS platforms and when .venv/ doesn't exist (e.g. the
    user is running from a system-installed env).
    """
    if sys.platform != "darwin":
        return
    venv = ROOT / ".venv"
    if not venv.exists():
        return
    _run(["chflags", "-R", "nohidden", str(venv)])


def verify_import() -> None:
    _run([
        sys.executable, "-c",
        "import hexapod_biomechanics; print('OK:', hexapod_biomechanics.__file__)",
    ])


def main() -> None:
    refresh_editable_install()
    strip_macos_hidden_flag()
    verify_import()


if __name__ == "__main__":
    main()
