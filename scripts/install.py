"""Recovery tool: repair the editable install of hexapod_biomechanics.

Run from the project root:

    python scripts/install.py

------------------------------------------------------------------------
The underlying problem (read this first)
------------------------------------------------------------------------
This script exists to undo damage caused by **cloud-syncing the project's
parent directory** (most commonly macOS "Documents in iCloud", but Dropbox
/ OneDrive / Google Drive behave the same way). When a `.venv/` lives inside
a synced folder, the sync agent corrupts it in two ways:

1. **Duplicate files.** On a perceived write-conflict it keeps both copies
   and renames one with a trailing " 2" / " 3" (e.g.
   `__editable__.hexapod_biomechanics-0.1.0 2.pth`, or even a whole
   `.venv 2/` directory). The editable install relies on a single, exact
   `.pth` file; the duplicates leave Python unable to find the package, so
   `import hexapod_biomechanics` fails with `ModuleNotFoundError` even
   though `uv pip list` shows it installed. The failure is intermittent and
   differs between `python` and `pytest` because it depends on which `.pth`
   files happen to exist at that moment.
2. **The macOS `UF_HIDDEN` flag.** iCloud sets the `hidden` filesystem flag
   on the synced `.venv/`. (This is set by the sync agent, not by uv — a
   fresh `uv venv` created outside a synced tree has no such flag.) CPython
   3.12+ skips hidden `.pth` files during site init, which independently
   breaks the editable import.

------------------------------------------------------------------------
The real fix (do this once per machine; then you never need this script)
------------------------------------------------------------------------
Get the virtualenv OUT of the synced tree. The clean, uv-native way:

  * Keep venvs in a non-synced location, e.g. `~/.venvs/<repo-name>`, and
    point uv at it with the `UV_PROJECT_ENVIRONMENT` env var. To make that
    automatic for every repo, add a `uv` wrapper to your shell rc that sets
    `UV_PROJECT_ENVIRONMENT="$HOME/.venvs/<git-repo-name>"` before calling
    the real uv. Then `uv sync` / `uv run` build and use the external venv
    and never create a `.venv` inside the synced folder again.
  * A bare `.venv` symlink does NOT work: `uv run`'s implicit sync deletes
    the symlink and recreates a real venv in-tree.
  * Bulletproof alternative: move the whole repo out of the synced folder
    (e.g. `~/Developer/...`) and let git be your sync mechanism.

------------------------------------------------------------------------
When to use THIS script
------------------------------------------------------------------------
As a band-aid when you can't (or haven't yet) applied the real fix — e.g.
you just cloned this repo onto a machine whose parent directory is also
cloud-synced and the editable import is broken. It patches the symptom so
you can keep working; it does NOT stop the sync agent from re-corrupting
the venv on the next build.

What it does:
1. Reinstalls the project editable into the active Python environment
   (via `uv pip` if `uv` is on PATH, otherwise `python -m pip`), which
   rewrites a clean `.pth` file.
2. On macOS, strips the `UF_HIDDEN` flag from `.venv/` so CPython 3.12+
   will process its `.pth` files again.
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
