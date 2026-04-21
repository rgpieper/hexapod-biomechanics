# hexapod-biomechanics

Analysis pipeline for estimating **mechanical impedance of the human ankle joint** during walking.
Subjects walk across a lab stepping on an instrumented robotic platform (the "Hexapod"),
which applies a controlled angular perturbation about the ankle's dorsiflexion/plantarflexion axis mid-stance.
The pipeline processes motion capture and force plate data through three stages to identify
a second-order impedance model: **&tau; = I&alpha; + B&omega; + K&theta;**.

This code accompanies a publication describing new methods for running this experiment and
analyzing ankle impedance with full 3D kinematics and kinetics.

## Setup

**Requirements:** Python 3.12+, [uv](https://docs.astral.sh/uv/)

```bash
# Clone and install
git clone https://github.com/rgpieper/hexapod-biomechanics.git
cd hexapod-biomechanics
uv sync

# Download pre-trained MLP weights for inertial force correction (~141 MB)
uv run python scripts/download_weights.py
```

If the editable install breaks (common on macOS — see `scripts/install.py` for details):

```bash
uv run python scripts/install.py
```

## Data Organization

Each subject has one directory per session. Raw c3d captures go in `raw_data/`;
pipeline outputs land in `conditions/`.

```
subjects/
  {subject_id}/
    {session_date}/                    # DDMMYYYY
      session.toml                     # auto-generated session config
      raw_data/
        static_01.c3d                  # calibration trial (subject standing on platform)
        hex{ts}_trip{N}_nom_{dir}.c3d  # nominal (unperturbed) walking trial
        hex{ts}_trip{N}_pert_{pct}pct_{dir}.c3d  # perturbed trial
      conditions/
        {side}_nom.h5                  # Stage 1 output: nominal trials
        {side}_pert_{pct}.h5           # Stage 1 output: perturbed trials
```

- `{dir}` is `fwd` or `rev` — mapped to ankle side (`right`/`left`) via `session.toml`
- `{pct}` is the perturbation onset as percentage of stance (e.g. `50`, `20`)
- `subjects/` is gitignored — data stays local

## Pipeline

All scripts are run from the project root with `uv run python scripts/<script>.py`.

### 0. Measure body mass

Estimate the subject's body mass from a static trial's Kistler force measurement.
Run this once per subject — use the same value across all sessions.

```bash
uv run python scripts/measure_body_mass.py \
    subjects/RP081721/11082025/raw_data/static_01.c3d
```

### 1. Generate session config

Auto-detect Vicon parameters (hexapod marker prefix, Kistler device ID,
accelerometer mode) from a sample c3d and write `session.toml`.

```bash
uv run python scripts/build_config.py \
    subjects/RP081721/11082025 \
    --subject-id RP081721 --body-mass 75.2 --sex m
```

**Review the generated file** before proceeding:
- `[static_blocks]` — if markers were re-placed mid-session, reassign dynamic trials to the correct static
- `[direction_side_map]` — default is `fwd = "right"`, `rev = "left"`; override for amputee protocols

### 2. Stage 1 — Per-trial biomechanics

Process every trial in the session: load markers and forces, subtract MLP-predicted
inertial artifacts, compute ankle kinematics (ISB joint coordinate system) and
inverse dynamics (Newton-Euler), then write one HDF5 per condition.

```bash
uv run python scripts/process_session.py subjects/RP081721/11082025
```

Output: `conditions/{side}_nom.h5`, `conditions/{side}_pert_{pct}.h5`

### 3. Stages 2 + 3 — Bootstrap isolation & impedance fitting

For each `(side, pct)` condition, isolate the perturbation response via bootstrap
subtraction (Stage 2) and fit the impedance model (Stage 3).

```bash
uv run python scripts/fit_impedance.py \
    subjects/RP081721/11082025 right 50
```

This produces two figures (interactive by default) and prints impedance parameters
(K, B, I — raw and mass-normalised) to the console.

**Key options:**
- `--dynamics static|full` — static dynamics (default) omits foot inertial terms; full dynamics includes them
- `--isolation-path dts|std` — signal derivation path (DTS = primary, STD = secondary)
- `--save DIR` — write figures as PNGs instead of showing interactively
- `--save-h5` — persist the `ImpedanceResult` to `conditions/` as HDF5

### Validation & visualization

```bash
# Animate a single trial (kinematics, GRF, perturbation tracking, or demo)
uv run python scripts/animate_trial.py \
    subjects/RP081721/11082025 \
    hex_20250811_trip12_pert_50pct_fwd \
    --which demo --scope stance

# Plot Stage 2 isolation results without fitting impedance
uv run python scripts/plot_isolated.py \
    subjects/RP081721/11082025 right 50
```

## Package Modules

```
src/hexapod_biomechanics/
  config.py             Session config: lab defaults, TOML I/O, auto-detection
  load_data.py          C3D file parsing, marker/analog extraction
  force_correction.py   MLP-based inertial force removal (inference only)
  forces.py             Kistler force plate processing, GRF, hexapod tracking
  kinematics.py         ISB ankle joint coordinate system, 6-DOF kinematics
  inverse_dynamics.py   Newton-Euler inverse dynamics for the foot segment
  process_trials.py     Stage 1: per-trial processing and HDF5 I/O
  isolation.py          Stage 2: bootstrap subtraction of nominal from perturbed
  impedance.py          Stage 3: least-squares impedance fit and HDF5 I/O
  utils.py              Rigid transforms, rotation math, perturbation detection
  visualize.py          Matplotlib animations (kinematics, GRF, perturbation)
```

## Inertial Force Correction

The Hexapod's motion induces inertial forces on the Kistler plate that must be
removed before computing ground reaction forces. An MLP model
(`models/mlp_general.pth`) predicts the induced forces from accelerometer data
so they can be subtracted from the raw measurement.

The model was trained in the companion project
[`hexapod-inertial-force-removal`](https://github.com/rgpieper/hexapod-inertial-force-removal).
Only the inference code and trained weights are included here.
See `models/README.md` for weight provenance, checksums, and download instructions.

## Key References

- Wu et al. 2005 (J Biomech) — ISB recommendation for ankle joint coordinate system definitions
- Dumas et al. 2006 (J Biomech) — Adjustments to McConville et al. and Young et al. body segment inertial parameters
