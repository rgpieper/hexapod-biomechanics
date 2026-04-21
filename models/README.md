# Pre-trained model weights

This directory holds pre-trained neural network weights used by the pipeline.
Weights are **not** tracked in git — they are distributed as
[GitHub Release](https://github.com/rgpieper/hexapod-biomechanics/releases)
assets.

## Available weights

| File | Purpose | Size | SHA-256 |
|---|---|---|---|
| `mlp_general.pth` | Inertial force correction for the Kistler plate — used by `hexapod_biomechanics.force_correction.BasicMLP`. Trained on the full no-load perturbation dataset across all perturbation axes/directions. | ~141 MB | `baae3204b5efc300673fe3bd77e10d1d61275834b1a8a42ce134296399fc814f` |

## How to get them

From the project root:

```bash
uv run python scripts/download_weights.py
```

This downloads every file listed above into `models/` and verifies the SHA-256
checksum. Re-running the script is a no-op if the file is already present and
valid.

Alternatively, download manually from the
[latest release](https://github.com/rgpieper/hexapod-biomechanics/releases/latest)
and drop the `.pth` file into this directory.

## Model provenance

`mlp_general.pth` was trained in the companion project
[`hexapod-inertial-force-removal`](https://github.com/rgpieper/hexapod-inertial-force-removal)
on hexapod no-load perturbation captures. Only the weights and minimal
inference code have been brought into this repository
(`src/hexapod_biomechanics/force_correction.py`). Training code, datasets, and
alternative architectures remain in the companion project.
