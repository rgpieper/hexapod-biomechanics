
from typing import Tuple, Dict, Optional
import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, RotationSpline

def rigid_transform(base_points: npt.NDArray, dynamic_points: npt.NDArray) -> npt.NDArray:
    """Compute transformation of a dynamic rigid body relative to a base configuration.

    Args:
        base_points (npt.NDArray): (n_points, 3)
        dynamic_points (npt.NDArray): (n_frames, n_points, 3)

    Returns:
        npt.NDArray: transformation trajectory (n_frames, 4, 4)
    """

    centroid_base = np.mean(base_points, axis=0) # average across points, (N, 3) -> (3,)
    centroid_dynamic = np.mean(dynamic_points, axis=1) # average across points, (F, N, 3) -> (F, 3)

    H_base = base_points - centroid_base # center points to remove translation, (N, 3) - (3,) -> (N, 3)
    H_dynamic = dynamic_points - centroid_dynamic[:, np.newaxis, :] # (F, N, 3) - (F, 1, 3) -> (F, N, 3)
    H = H_dynamic.transpose(0,2,1) @ H_base # frame-wise covariance matrix, (F, 3, N) @ (N, 3) -> (F, 3, 3)

    U, _, Vh = np.linalg.svd(H) # (F, 3, 3) -> U(F, 3, 3), S(F, 3), Vh(F, 3, 3)
    R = U @ Vh # rotation matrix (F, 3, 3), via R = V @ U.T

    # check / correct for reflection
    det = np.linalg.det(R) # (F,)
    reflect_mask = det < 0
    if np.any(reflect_mask):
        U[reflect_mask, :, 2] *= -1 # multiply third column by -1
        R = U @ Vh # recompute R for all frames

    x = centroid_dynamic - (R @ centroid_base) # translation, (F, 3) - ((F, 3, 3) @ (3,)) -> (F, 3)

    n_frames = dynamic_points.shape[0]
    T = np.eye(4)[np.newaxis, :, :].repeat(n_frames, axis=0) # (F, 4, 4)
    T[:, :3, :3] = R
    T[:, :3, 3] = x
    
    return T

def normalize(v: npt.NDArray) -> npt.NDArray:
    """Normalize vector(s) across coordinate dimension.

    Args:
        v (npt.NDArray): Vector(s) to be normalized, with normalized dimension last.

    Returns:
        npt.NDArray: Normalized vector(s).
    """

    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / norms

def clamp(val: npt.ArrayLike) -> npt.ArrayLike:
    """Clamp dot products to [-1, 1] for acos stability.

    Args:
        val (npt.ArrayLike): Dot product results to be clamped.

    Returns:
        npt.ArrayLike: Clamped values.
    """

    return np.clip(val, -1.0, 1.0)

def axis_difference(target_axis: Tuple[npt.NDArray, npt.NDArray], actual_axis: Tuple[npt.NDArray, npt.NDArray]) -> Dict[str, npt.ArrayLike]:
    """Compute orientation and offset difference between two axes.

    Individual axis parameters (shape (3,)) can be passed to return single (scalar) differences.
    Frame-wise axis parameters (shape (n_frames,3)) will return frame-wise differences (shape (n_frames,)).

    Args:
        target_axis (Tuple[npt.NDArray, npt.NDArray]): Target axis parameters: (origin, direction).
        actual_axis (Tuple[npt.NDArray, npt.NDArray]): Actual axis parameters: (origin, direction). Offset will be computed between actual origin and target axis.

    Returns:
        Dict[str, npt.ArrayLike]: _description_
    """

    p_t = target_axis[0] # (n_frames, 3) or (3,)
    v_t = normalize(target_axis[1]) # (n_frames, 3) or (3,)

    p_a = actual_axis[0] # (n_frames, 3) or (3,)
    v_a = normalize(actual_axis[1]) # (n_frames, 3) or (3,)

    # orientation difference (angle between axes):
    # cos(phi) = v1 . v2
    dot_prod = np.abs(np.sum(v_t * v_a, axis=-1)) # (n_frames,) or scalar
    angle_diff = np.arccos(clamp(dot_prod)) # radians (n_frames,) or scalar

    # offset difference: distance from actual axis reference/origin to target axis
    # (Note the distinction between target and reference axis.)
    # d = norm( (p_a - p_t) x v_t ) / norm(v_t) (norm(v_t) = 1)
    target_to_actual = p_a - p_t # (n_frames,3) or (3,)
    offset_diff = np.linalg.norm(np.cross(target_to_actual, v_t)) # (n_frames,) or scalar

    return {
        'angle_diff': angle_diff,
        'offset_diff': offset_diff # difference between actual reference point/origin and target axis
    }

def inv_rodrigues(R: npt.NDArray, tol: float = 1e-6) -> Tuple[float, Optional[npt.NDArray]]:
    """Determine axis and magnitude of rotation represented by a rotation matrix.

    Args:
        R (npt.NDArray): Rotation matrix (3,3)
        tol (float, optional): Minimum theta for which axis will be computed. Defaults to 1e-6.

    Returns:
        Tuple[npt.ArrayLike, Optional[npt.NDArray]]:
            theta (float): Magnitude of rotation about axis (radians)
            n (Optional[npt.NDArray]): Axis of rotation (3,)
    """

    # trace(R) = 1 + 2*cos(theta)
    theta = np.arccos(clamp((np.trace(R) - 1) / 2.0)) # rotation angle (rad)
    if np.abs(theta) < tol: # rotation too small to compute axis
        return theta, None

    # Rodrigues' formula computes rotation matrix for rotation 'theta' about axis 'n'
    # axis of rotation encoded in skew-symmetric matrix S = [[0, -nz, ny], [nz, 0, -nx], [-ny, nx, 0]]
    # R = I + sin(theta)*S + (1 - cos(theta))*S^2
    # R.T = I + sin(theta)*(-S) + (1 - cos(theta))*S^2
    # R - R.T = 2*sin(theta)*S
    # S = [[0, -nz, ny], [nz, 0, -nx], [-ny, nx, 0]] = (R - R.T) / 2*sin(theta)
    n_skew = np.array([
        R[2, 1] - R[1, 2], # nx * sin(theta)
        R[0, 2] - R[2, 0], # ny * sin(theta)
        R[1, 0] - R[0, 1] # nz * sin(theta)
    ])
    n = n_skew / (2 * np.sin(theta))
    n = normalize(n) # rotation axis direction (3,)

    return theta, n

def differentiate_rotation(
        time: npt.NDArray,
        R: npt.NDArray,
        filt_window_duration: float = 0.05, # seconds, ~20-30Hz cutoff
        filt_poly: int = 4
    ) -> Tuple[npt.NDArray, npt.NDArray]:
    """Compute 3D angular velocity and acceleration from rotation matrix timeseries.

    Args:
        time (npt.NDArray): Time vector (n_frames,)
        R (npt.NDArray): Series of rotation matrices describing body orientation in global frame (n_frames,3,3)
        window_duration (float, optional): Time length of filter, dictating cutoff frequency (seconds). Defaults to 0.05.
        filt_poly (int, optional): Filter polynomial order. Defaults to 4.

    Returns:
        Tuple[npt.NDArray, npt.NDArray]:
            omega (npt.NDArray): 3D angular velocity represented in body frame (n_frames,3)
            alpha (npt.NDArray): 3D angular acceleration represented in body frame (n_frames,3)
    """
    
    rot = Rotation.from_matrix(R)
    spline = RotationSpline(time, rot)
    omega_raw = spline(time, order=1) # angular velocity in the body frame

    dt = np.mean(np.diff(time)) # sample period
    fs = 1 / dt # sample frequency

    # filtered frequencies is tied to time-length of filter window
    # compute stable filter window according to time length and sample frequency
    filt_window = int(fs * filt_window_duration)
    filt_window = filt_window if filt_window % 2 != 0 else filt_window + 1

    omega = savgol_filter(
        omega_raw,
        window_length=filt_window,
        polyorder=filt_poly,
        deriv=0,
        axis=0
    )
    alpha = savgol_filter(
        omega_raw,
        window_length=filt_window,
        polyorder=filt_poly,
        deriv=1,
        delta=dt,
        axis=0
    )

    return omega, alpha