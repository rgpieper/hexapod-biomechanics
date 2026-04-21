"""Stage 3 -- second-order mechanical impedance identification.

Fits the model  tau = I*alpha + B*omega + K*theta  to the bootstrap-isolated
perturbation response produced by Stage 2 (`isolation.py`).  Each bootstrap
iteration yields one (K, B, I) triplet via ordinary least squares.

Two isolation paths determine where signals get their one Savitzky-Golay pass:

  - **DTS** (differentiate-then-subtract, primary): all signals independently
    filtered at Stage 1, subtracted at Stage 2.  Stage 3 uses them as-is.
  - **STD** (subtract-then-differentiate, secondary): raw unfiltered angle is
    subtracted at Stage 2; Stage 3 applies a single sav-gol pass (deriv 0/1/2)
    to produce filtered theta, omega, and alpha from the isolated angle.

Torque always comes from Stage 1's one-pass-filtered inverse dynamics
regardless of path.

Sign convention
---------------
Inverse dynamics computes M_ank: the moment the ankle joint exerts **on the
foot segment** (Newton-Euler free body of the foot).  The impedance model
describes the ankle's passive mechanical response — the reaction moment the
ankle exerts **back on the shank** in response to imposed motion.  These are
equal and opposite (Newton's third law), so we negate the measured torque
before fitting:  ``y = -tau_measured``.  Positive K then means the ankle
resists dorsiflexion with a plantarflexion moment, which is the expected
passive stiffness behaviour.

Persistence
-----------
`save_impedance` writes one HDF5 per condition containing:
  - per-iteration impedance parameters (K, B, I, mass-normalised variants, VAF)
  - the exact trajectories (theta, omega, alpha, tau) fed to the fit, so every
    triplet is traceable to the signals it was fit from.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import json

import h5py
import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter

from hexapod_biomechanics.isolation import IsolatedPerturbation

# DF axis index in the JCS decomposition.
_DF_AXIS = 0


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ImpedanceResult:
    """Impedance identification result for one (side, pct) condition.

    All per-iteration arrays have N = n_iterations rows.  Trajectory arrays
    have shape (N, M) where M = len(time).  Trajectories store the exact
    signals fed to lstsq, after path-specific selection, filtering (STD),
    re-zeroing, unit conversion, and sign negation.

    Attributes:
        side: "right" or "left".
        perturbation_pct: Stance percentage of the perturbation (e.g. 50).
        body_mass_kg: Subject body mass used for normalisation.
        fs_hz: Sampling frequency [Hz].
        dynamics: "full" or "static" -- which torque variant was used.
        isolation_path: "dts" or "std" -- which signal-derivation path.
        rezero: Whether first-sample subtraction was applied to theta and tau.
        time: (M,) seconds relative to perturbation onset.
        K: (N,) stiffness [Nm/rad].
        B: (N,) damping [Nm*s/rad].
        I: (N,) inertia [kg*m^2].
        K_norm: (N,) mass-normalised stiffness [Nm/rad/kg].
        B_norm: (N,) mass-normalised damping [Nm*s/rad/kg].
        I_norm: (N,) mass-normalised inertia [kg*m^2/kg].
        vaf: (N,) variance accounted for [%].
        theta: (N, M) angle [rad].
        omega: (N, M) angular velocity [rad/s].
        alpha: (N, M) angular acceleration [rad/s^2].
        tau: (N, M) negated measured torque [Nm].
        tau_predicted: (N, M) model-predicted torque [Nm].
        perturbed_trial_names: Per-iteration list of trial name lists.
        mean_onset_samples: (N,) stance-relative mean onset [frames].
        mean_onset_stance_fractions: (N,) mean onset phase [0..1].
        n_nominal_averaged: Number of nominal trials averaged (constant).
    """
    # identity / metadata
    side: str
    perturbation_pct: int
    body_mass_kg: float
    fs_hz: float
    # fit configuration
    dynamics: str
    isolation_path: str
    rezero: bool
    # shared time axis
    time: npt.NDArray                          # (M,)
    # per-iteration parameters (N = n_iterations)
    K: npt.NDArray                             # (N,) [Nm/rad]
    B: npt.NDArray                             # (N,) [Nm*s/rad]
    I: npt.NDArray                             # (N,) [kg*m^2]
    K_norm: npt.NDArray                        # (N,) [Nm/rad/kg]
    B_norm: npt.NDArray                        # (N,) [Nm*s/rad/kg]
    I_norm: npt.NDArray                        # (N,) [kg*m^2/kg]
    vaf: npt.NDArray                           # (N,) [%]
    # per-iteration trajectories -- exactly as fed to lstsq
    theta: npt.NDArray                         # (N, M)
    omega: npt.NDArray                         # (N, M)
    alpha: npt.NDArray                         # (N, M)
    tau: npt.NDArray                           # (N, M) negated measured
    tau_predicted: npt.NDArray                 # (N, M)
    # bootstrap metadata
    perturbed_trial_names: list[list[str]]     # per-iteration
    mean_onset_samples: npt.NDArray            # (N,)
    mean_onset_stance_fractions: npt.NDArray   # (N,)
    n_nominal_averaged: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_impedance(
    isolated: Sequence[IsolatedPerturbation],
    body_mass_kg: float,
    side: str,
    perturbation_pct: int,
    dynamics: str = "static",
    isolation_path: str = "dts",
    rezero: bool = True,
    filt_window_duration: float = 0.05,
    filt_poly: int = 4,
) -> ImpedanceResult:
    """Fit the 2nd-order impedance model to bootstrap-isolated perturbation data.

    Args:
        isolated: Bootstrap iterations from `isolation.bootstrap_isolate`.
        body_mass_kg: Subject body mass for normalisation.
        side: "right" or "left".
        perturbation_pct: Perturbation stance percentage (e.g. 50, 20).
        dynamics: "full" or "static" -- selects which torque variant.
        isolation_path: "dts" (primary) or "std" (secondary).
        rezero: If True, subtract the first sample from theta and tau.
        filt_window_duration: STD only -- sav-gol window length [s].
        filt_poly: STD only -- sav-gol polynomial order.

    Returns:
        An ImpedanceResult containing per-iteration parameters and the exact
        trajectories fed to the least-squares fit.
    """
    if not isolated:
        raise ValueError("isolated is empty.")
    if dynamics not in ("full", "static"):
        raise ValueError(f"dynamics must be 'full' or 'static', got {dynamics!r}.")
    if isolation_path not in ("dts", "std"):
        raise ValueError(
            f"isolation_path must be 'dts' or 'std', got {isolation_path!r}."
        )

    fs = isolated[0].fs_hz
    time = isolated[0].time
    n_iter = len(isolated)
    n_samples = len(time)

    # STD: pre-compute sav-gol filter window (in samples).
    if isolation_path == "std":
        filt_win = int(fs * filt_window_duration)
        if filt_win % 2 == 0:
            filt_win += 1
        dt = 1.0 / fs

    # Pre-allocate output arrays.
    K_arr = np.empty(n_iter)
    B_arr = np.empty(n_iter)
    I_arr = np.empty(n_iter)
    vaf_arr = np.empty(n_iter)
    theta_arr = np.empty((n_iter, n_samples))
    omega_arr = np.empty((n_iter, n_samples))
    alpha_arr = np.empty((n_iter, n_samples))
    tau_arr = np.empty((n_iter, n_samples))
    tau_pred_arr = np.empty((n_iter, n_samples))
    onset_samples = np.empty(n_iter)
    onset_fractions = np.empty(n_iter)
    trial_names: list[list[str]] = []

    for i, iso in enumerate(isolated):
        # --- Select signals based on isolation path ---
        if isolation_path == "dts":
            # DTS: all signals already one-pass filtered at Stage 1.
            theta_i = iso.alpha_filt.copy()            # (M,) rad
            omega_i = iso.omega_jcs[:, _DF_AXIS].copy()  # (M,) rad/s
            alpha_i = iso.foot_alpha_jcs[:, _DF_AXIS].copy()  # (M,) rad/s^2
        else:
            # STD: apply single sav-gol pass to raw isolated angle.
            raw_angle = iso.alpha  # (M,) raw DF angle
            theta_i = savgol_filter(
                raw_angle, window_length=filt_win, polyorder=filt_poly,
                deriv=0, delta=dt, axis=0,
            )
            omega_i = savgol_filter(
                raw_angle, window_length=filt_win, polyorder=filt_poly,
                deriv=1, delta=dt, axis=0,
            )
            alpha_i = savgol_filter(
                raw_angle, window_length=filt_win, polyorder=filt_poly,
                deriv=2, delta=dt, axis=0,
            )

        # --- Torque: convert N*mm -> Nm, negate (sign convention) ---
        tau_attr = "tau_full" if dynamics == "full" else "tau_static"
        tau_raw = getattr(iso, tau_attr)[:, _DF_AXIS]  # (M,) N*mm
        tau_i = -(tau_raw / 1000.0)                    # (M,) Nm, negated

        # --- Re-zeroing (first-sample subtraction on theta and tau) ---
        if rezero:
            theta_i = theta_i - theta_i[0]
            tau_i = tau_i - tau_i[0]

        # --- Least-squares fit: tau = I*alpha + B*omega + K*theta ---
        # A matrix: (M, 3), columns = [alpha, omega, theta]
        A = np.column_stack([alpha_i, omega_i, theta_i])
        # Solve for x = [I, B, K]
        x, _, _, _ = np.linalg.lstsq(A, tau_i, rcond=None)
        I_val, B_val, K_val = x

        # Model prediction
        tau_pred = A @ x

        # VAF: variance accounted for [%]
        residual = tau_i - tau_pred
        vaf_val = (1.0 - np.var(residual) / np.var(tau_i)) * 100.0

        # Store results for this iteration
        K_arr[i] = K_val
        B_arr[i] = B_val
        I_arr[i] = I_val
        vaf_arr[i] = vaf_val
        theta_arr[i] = theta_i
        omega_arr[i] = omega_i
        alpha_arr[i] = alpha_i
        tau_arr[i] = tau_i
        tau_pred_arr[i] = tau_pred
        onset_samples[i] = iso.mean_onset_sample
        onset_fractions[i] = iso.mean_onset_stance_fraction
        trial_names.append(iso.perturbed_trial_names)

    # Mass-normalise
    K_norm = K_arr / body_mass_kg
    B_norm = B_arr / body_mass_kg
    I_norm = I_arr / body_mass_kg

    return ImpedanceResult(
        side=side,
        perturbation_pct=perturbation_pct,
        body_mass_kg=body_mass_kg,
        fs_hz=fs,
        dynamics=dynamics,
        isolation_path=isolation_path,
        rezero=rezero,
        time=time,
        K=K_arr,
        B=B_arr,
        I=I_arr,
        K_norm=K_norm,
        B_norm=B_norm,
        I_norm=I_norm,
        vaf=vaf_arr,
        theta=theta_arr,
        omega=omega_arr,
        alpha=alpha_arr,
        tau=tau_arr,
        tau_predicted=tau_pred_arr,
        perturbed_trial_names=trial_names,
        mean_onset_samples=onset_samples,
        mean_onset_stance_fractions=onset_fractions,
        n_nominal_averaged=isolated[0].n_nominal_averaged,
    )


def save_impedance(file_path: Path, result: ImpedanceResult) -> None:
    """Write an ImpedanceResult to HDF5.

    Args:
        file_path: Output path (e.g. `conditions/right_pert_50_impedance.h5`).
        result: The impedance result to persist.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(file_path, "w") as f:
        # File-level attrs
        f.attrs["side"] = result.side
        f.attrs["perturbation_pct"] = result.perturbation_pct
        f.attrs["body_mass_kg"] = result.body_mass_kg
        f.attrs["fs_hz"] = result.fs_hz
        f.attrs["dynamics"] = result.dynamics
        f.attrs["isolation_path"] = result.isolation_path
        f.attrs["rezero"] = result.rezero
        f.attrs["n_iterations"] = len(result.K)
        f.attrs["n_nominal_averaged"] = result.n_nominal_averaged

        # Datasets
        f.create_dataset("time", data=result.time)
        f.create_dataset("K", data=result.K)
        f.create_dataset("B", data=result.B)
        f.create_dataset("I", data=result.I)
        f.create_dataset("K_norm", data=result.K_norm)
        f.create_dataset("B_norm", data=result.B_norm)
        f.create_dataset("I_norm", data=result.I_norm)
        f.create_dataset("vaf", data=result.vaf)
        f.create_dataset("theta", data=result.theta)
        f.create_dataset("omega", data=result.omega)
        f.create_dataset("alpha", data=result.alpha)
        f.create_dataset("tau", data=result.tau)
        f.create_dataset("tau_predicted", data=result.tau_predicted)
        f.create_dataset("mean_onset_samples", data=result.mean_onset_samples)
        f.create_dataset(
            "mean_onset_stance_fractions",
            data=result.mean_onset_stance_fractions,
        )

        # Per-iteration trial names: variable-length string, JSON-encoded.
        # h5py special type for variable-length strings.
        vlen_str = h5py.special_dtype(vlen=str)
        names_ds = f.create_dataset(
            "perturbed_trial_names",
            shape=(len(result.perturbed_trial_names),),
            dtype=vlen_str,
        )
        for idx, names in enumerate(result.perturbed_trial_names):
            names_ds[idx] = json.dumps(names)


def load_impedance(file_path: Path) -> ImpedanceResult:
    """Load an ImpedanceResult from HDF5.

    Args:
        file_path: Path to an impedance HDF5 written by `save_impedance`.

    Returns:
        The reconstructed ImpedanceResult.
    """
    with h5py.File(file_path, "r") as f:
        # Decode variable-length string trial names from JSON.
        raw_names = f["perturbed_trial_names"][:]
        trial_names = [json.loads(s) for s in raw_names]

        return ImpedanceResult(
            side=str(f.attrs["side"]),
            perturbation_pct=int(f.attrs["perturbation_pct"]),
            body_mass_kg=float(f.attrs["body_mass_kg"]),
            fs_hz=float(f.attrs["fs_hz"]),
            dynamics=str(f.attrs["dynamics"]),
            isolation_path=str(f.attrs["isolation_path"]),
            rezero=bool(f.attrs["rezero"]),
            time=f["time"][:],
            K=f["K"][:],
            B=f["B"][:],
            I=f["I"][:],
            K_norm=f["K_norm"][:],
            B_norm=f["B_norm"][:],
            I_norm=f["I_norm"][:],
            vaf=f["vaf"][:],
            theta=f["theta"][:],
            omega=f["omega"][:],
            alpha=f["alpha"][:],
            tau=f["tau"][:],
            tau_predicted=f["tau_predicted"][:],
            perturbed_trial_names=trial_names,
            mean_onset_samples=f["mean_onset_samples"][:],
            mean_onset_stance_fractions=f["mean_onset_stance_fractions"][:],
            n_nominal_averaged=int(f.attrs["n_nominal_averaged"]),
        )
