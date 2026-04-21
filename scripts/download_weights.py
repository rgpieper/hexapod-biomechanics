"""Download pre-trained model weights from GitHub Releases.

Run from the project root:

    uv run python scripts/download_weights.py

Re-running is a no-op if the file is already present and its SHA-256 matches.
To force a re-download, delete the local file first.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Bump this tag when releasing new weights. The download URL is built as
# f"{RELEASE_URL}/{tag}/{filename}" — same pattern for any file attached to
# the release.
RELEASE_URL = "https://github.com/rgpieper/hexapod-biomechanics/releases/download"
RELEASE_TAG = "weights-v1"

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@dataclass(frozen=True)
class WeightFile:
    filename: str
    sha256: str


WEIGHT_FILES: tuple[WeightFile, ...] = (
    WeightFile(
        filename="mlp_general.pth",
        sha256="baae3204b5efc300673fe3bd77e10d1d61275834b1a8a42ce134296399fc814f",
    ),
)


def sha256_of(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream-hash a file so we don't load all 141 MB into memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    """Print a simple download progress indicator."""
    downloaded = block_num * block_size
    if total_size <= 0:
        sys.stdout.write(f"\r  downloaded {downloaded / 1e6:.1f} MB")
    else:
        pct = min(100.0, 100.0 * downloaded / total_size)
        sys.stdout.write(
            f"\r  {pct:5.1f}%  "
            f"({downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB)"
        )
    sys.stdout.flush()


def download_weight(weight: WeightFile) -> None:
    """Download one weight file and verify its SHA-256."""
    dest = MODELS_DIR / weight.filename

    if dest.exists():
        actual = sha256_of(dest)
        if actual == weight.sha256:
            print(f"[skip] {weight.filename} already present and valid")
            return
        print(
            f"[warn] {weight.filename} exists but sha256 mismatch "
            f"(expected {weight.sha256[:12]}…, got {actual[:12]}…); "
            f"re-downloading."
        )
        dest.unlink()

    url = f"{RELEASE_URL}/{RELEASE_TAG}/{weight.filename}"
    print(f"[fetch] {url}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest, reporthook=_progress_hook)
    print()  # newline after progress bar

    actual = sha256_of(dest)
    if actual != weight.sha256:
        dest.unlink(missing_ok=True)
        raise RuntimeError(
            f"sha256 mismatch for {weight.filename}: "
            f"expected {weight.sha256}, got {actual}"
        )
    print(f"[ok]   {weight.filename} verified ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    print(f"Downloading weights into {MODELS_DIR}")
    for weight in WEIGHT_FILES:
        download_weight(weight)
    print("Done.")


if __name__ == "__main__":
    main()
