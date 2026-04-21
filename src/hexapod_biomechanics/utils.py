
from typing import Tuple, Dict, Optional
import math
import numpy as np
import numpy.typing as npt
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, RotationSpline
from scipy.constants import g

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
    norms[norms < 1e-6] = 1.0
    return v / norms

def clamp(val: npt.ArrayLike) -> npt.ArrayLike:
    """Clamp dot products to [-1, 1] for acos stability.

    Args:
        val (npt.ArrayLike): Dot product results to be clamped.

    Returns:
        npt.ArrayLike: Clamped values.
    """

    return np.clip(val, -1.0, 1.0)

def axis_difference(target_axis: Tuple[npt.ArrayLike, npt.ArrayLike], actual_axis: Tuple[npt.ArrayLike, npt.ArrayLike]) -> Dict[str, npt.ArrayLike]:
    """Compute orientation and offset difference between two axes.

    Individual axis parameters (shape (3,)) can be passed to return single (scalar) differences.
    Frame-wise axis parameters (shape (n_frames,3)) will return frame-wise differences (shape (n_frames,)).

    Args:
        target_axis (Tuple[npt.ArrayLike, npt.ArrayLike]): Target axis parameters: (origin, direction).
        actual_axis (Tuple[npt.ArrayLike, npt.ArrayLike]): Actual axis parameters: (origin, direction). Offset will be computed between actual origin and target axis.

    Returns:
        Dict[str, npt.ArrayLike]:
            angle_diff (npt.ArrayLike): Angular deviation between target and actual axis [rad] (n_frames,) or scalar
            offset_diff (npt.ArrayLike): Distance between actual reference point/origin and closest point on target axis [mm] (n_frames,) or scalar
    """

    if target_axis[0] is None:
        raise ValueError("No target axis to match.")

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
    offset_diff = np.linalg.norm(np.cross(target_to_actual, v_t), axis=-1) # (n_frames,) or scalar

    return {
        'angle_diff': angle_diff,
        'offset_diff': offset_diff
    }

def inv_rodrigues(R: npt.NDArray, tol: float = 1e-6) -> Tuple[float, Optional[npt.NDArray]]:
    """Determine axis and magnitude of rotation represented by a rotation matrix.

    Args:
        R (npt.NDArray): Rotation matrix (3,3)
        tol (float, optional): Minimum theta for which axis will be computed [radians]. Defaults to 1e-6.

    Returns:
        Tuple[npt.ArrayLike, Optional[npt.NDArray]]:
            theta (float): Magnitude of rotation about axis [radians]
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
        filt_window: int = 11,
        filt_poly: int = 4
    ) -> Tuple[npt.NDArray, npt.NDArray]:
    """Compute 3D angular velocity and acceleration from rotation matrix timeseries.

    Args:
        time (npt.NDArray): Time vector [sec] (n_frames,)
        R (npt.NDArray): Series of rotation matrices describing body orientation in global frame (n_frames,3,3)
        filt_window (int, optional): Sample length of filter. Defaults to 11.
        filt_poly (int, optional): Filter polynomial order. Defaults to 4.

    Returns:
        Tuple[npt.NDArray, npt.NDArray]:
            omega (npt.NDArray): 3D angular velocity represented in body frame [rad/s] (n_frames,3)
            alpha (npt.NDArray): 3D angular acceleration represented in body frame [rad/s/s] (n_frames,3)
    """

    dt = np.mean(np.diff(time)) # sample period

    quats = Rotation.from_matrix(R).as_quat() # convert rotation matrix to quaternions for pre-filtering (continuous) (n_frames,4)

    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i-1]) < 0:
            quats[i] = -quats[i] # q and -q represent the same rotation, but must ensure the trajectory is continuous before filtering

    q_filt = savgol_filter( # orientation quaternions, unnormalized
        quats,
        window_length=filt_window,
        polyorder=filt_poly,
        deriv=0,
        axis=0
    )

    dq_raw = savgol_filter( # first derivative of quaternions
        quats,
        window_length=filt_window,
        polyorder=filt_poly,
        deriv=1,
        delta=dt,
        axis=0
    )

    ddq_raw = savgol_filter( # second derivative of quaternions
        quats,
        window_length=filt_window,
        polyorder=filt_poly,
        deriv=2,
        delta=dt,
        axis=0
    )

    # Geometric Correction: project raw derivatives back onto the unit hypersphere surface
    norms = np.linalg.norm(q_filt, axis=1, keepdims=True)
    q = q_filt/norms # re-normalize orientation quaternions
    # project first derivative: dq = (1/n) * (dq_raw - (q . dq_raw) * q)
    # (subtract component of raw quaternion derivative that is parallel to the orientation quaternion to flatten the derivative to be tangent to the hypersphere)
    q_dot_dq_raw = np.sum(q * dq_raw, axis=1, keepdims=True)
    dq = (dq_raw - q_dot_dq_raw * q) / norms
    # project second derivative: ddq = (1/n) * (ddq_raw - ((q . ddq_raw) + (dq . dq_raw)) * q - (q . dq_raw) * dq)
    # differentiation of expression to project the first derivative (w product and quotient rules) (accounts for hypersphere curvature)
    q_dot_ddq_raw = np.sum(q * ddq_raw, axis=1, keepdims=True)
    dq_dot_dq_raw = np.sum(dq * dq_raw, axis=1, keepdims=True)
    q_dot_dq_raw = np.sum(q * dq_raw, axis=1, keepdims=True)
    ddq = (ddq_raw - (q_dot_ddq_raw + dq_dot_dq_raw) * q - q_dot_dq_raw * dq) / norms

    def quat_mult(q1: npt.NDArray, q2: npt.NDArray) -> npt.NDArray:
        """Compute vectorized Hamilton product

        Args:
            q1 (npt.NDArray): first quaternion of product (n_frames,4)
            q2 (npt.NDArray): second quaternion of product (n_frames,4)

        Returns:
            npt.NDArray: vectorized Hamilton product (n_frames,4)
        """
        x1, y1, z1, w1 = q1.T
        x2, y2, z2, w2 = q2.T
        return np.column_stack((
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ))
    
    q_conj = q * np.array([-1, -1, -1, 1])
    dq_conj = dq * np.array([-1, -1, -1, 1])

    omega = 2.0 * quat_mult(q_conj, dq)[:,:3] # relationship between unit quaternion and body frame angular velocity (n_frames,3)
    alpha = 2.0 * (quat_mult(dq_conj, dq) + quat_mult(q_conj, ddq))[:,:3] # time derivative of the velocity equation (via product rule) (n_frames,3)

    return omega, alpha

def find_pert_thresh(
        hex_tjct: npt.NDArray,
        thresh: float,
    ) -> int:
    """Locate perturbation in measured hexapod orientation trajectory as threshold on that trajectory.

    Args:
        hex_tjct (npt.NDArray): Measured hexapod angle / orientation [units]
        thresh (float): Threshold for detecting perturbation [units]. Units must agree with hex_tjct.
        window_length (int): Desired perturbation window lenth [timepoints]
        pre_thresh_start (int): "Backup" period before crossing threshold to consider start of perturbation [timepoints]

    Raises:
        ValueError: No perturbation found that eclipses prescribed threshold.

    Returns:
        Tuple[npt.NDArray, int]:
            'mask' (npt.NDArray): Masking array for isolating perturbation from hex_tjct (and associated trajectories).
            'pert_start_idx' (int): Index where perturbation starts.
    """

    pert_start_idx = None
    for i, ang in enumerate(hex_tjct):
        if np.abs(ang) >= thresh:
            pert_start_idx = i
            break

    if pert_start_idx is None:
        raise ValueError("No perturbation found.")
    
    return pert_start_idx

def trapezoidal_pert(
        dt: float,
        mag: float = 2.0,
        dur: float = 0.075,
        accel: float = 1450.0
    ) -> Tuple[npt.NDArray, npt.NDArray]:
    """Built trapedoidal ramp profile, constructed in the same manner as hexapod perturbations.

    Args:
        dt (float): Trajectory time step [sec]
        mag (float, optional): Trajectory magnitude [units]. Defaults to 2.0.
        dur (float, optional): Trajectory duration [sec]. Defaults to 0.075.
        accel (float, optional): Trajectory acceleration [units/sec/sec]. Sign must agree with 'mag'. Defaults to 1450.0.

    Raises:
        ValueError: Prescribed acceleration insufficient for reaching magnitude in time."

    Returns:
        Tuple[npt.NDArray, npt.NDArray]: 
            'value' (npt.NDArray): Perturbation trajectory [units]
            't' (npt.NDArray): Associated time vector [sec]
    """
    
    
    min_accel_req = 4*mag/dur**2
    if abs(accel) < abs(min_accel_req) or accel*min_accel_req < 0:
        raise ValueError(
            "Infeasible trapezoidal profile.\n"
            f"Insufficient acceleration to re\ach value [{mag}] in time [{dur}].\n"
            f"  Desired Acceleration: {accel}\n"
            f"  Minimum Directional Acceleration: {min_accel_req}"
            )
    vel_const_candidates = (
        (accel*dur + math.sqrt((accel*dur)**2 - 4*accel*mag))/2,
        (accel*dur - math.sqrt((accel*dur)**2 - 4*accel*mag))/2
    )
    calc_dur_accel = lambda vel_const: vel_const/accel # calculate duration of acceleration (equal for deceleration)
    calc_dur_vel = lambda vel_const: dur - 2*calc_dur_accel(vel_const) # calculate duration of constant velocity
    vel_const = max(vel_const_candidates, key=calc_dur_vel) # constant (maximum) velocity
    dur_accel = calc_dur_accel(vel_const)
    dur_vel = calc_dur_vel(vel_const)

    n_steps = math.ceil(dur/dt) # number of time steps, ensuring t_final >= duration
    t = np.linspace(dt, dt*n_steps, n_steps)
    value = np.piecewise(
        t,
        [
            np.logical_and(t > 0.0, t <= dur_accel), # acceleration period
            np.logical_and(t > dur_accel, t <= dur_accel + dur_vel), # constant velocity period
            np.logical_and(t > dur_accel + dur_vel, t <= dur) # deceleration period
        ],
        [
            lambda t: 1/2*accel*t**2, # acceleration period
            lambda t: 1/2*accel*dur_accel**2 + vel_const*(t - dur_accel), # constant velocity period
            # d_dec(t>(t_a+t_v)) = 1/2 * a * t_a + v_c * t_c + v_c * (t - t_a - t_c) - 1/2 * a * (t - t_a - t_c)^2
            lambda t: mag - 1/2*accel*(dur - t)**2, # deceleration period
            dur # default value
        ]
    )

    return value, t

def find_pert_template(
        hex_tjct: npt.NDArray,
        template_tjct: npt.NDArray,
) -> int:
    """Locate perturbation in measured hexapod orientation trajectory by fitting it to a template trajectory.

    Args:
        hex_tjct (npt.NDArray): Measured hexapod angle / orientation [units]
        template_tjct (npt.NDArray): Template / expected hexapod angle / orientation [units]. Units must agree with hex_tjct.
        window_length (int): Desired perturbation window lenth [timepoints]

    Returns:
        Tuple[npt.NDArray, int]: 
            'mask' (npt.NDArray): Masking array for isolating perturbation from hex_tjct (and associated trajectories).
            'start_idx' (int): Index where perturbation starts.
    """

    N = len(hex_tjct)
    M = len(template_tjct)

    ssds = []
    for i in range(N - M + 1):
        meas_window = hex_tjct[i:i+M]
        ssds.append(np.sum((meas_window - template_tjct)**2))
    start_idx = np.argmin(ssds)

    return start_idx

def get_body_mass(raw_forces: npt.ArrayLike) -> float:
    """Estimate body mass from a static stand's raw Kistler channels.

    Sums the paired raw channels into (Fx, Fy, Fz), takes the mean resultant
    magnitude across the stand, and divides by gravity. Frame-agnostic: as long
    as the subject is at rest, the resultant is their weight regardless of the
    Kistler's orientation in the global frame.

    Args:
        raw_forces: (N, 8) array of raw Kistler channels in the standard order
            [FX12, FX34, FY14, FY23, FZ1, FZ2, FZ3, FZ4] — matches
            `config.KISTLER_RAW_CHANNELS` and the column layout produced by
            `C3DMan.get_analogs(session_config.kistler_raw_analog_pairs())`.

    Returns:
        Body mass [kg].
    """
    raw = np.asarray(raw_forces)
    fx = raw[:, 0] + raw[:, 1]
    fy = raw[:, 2] + raw[:, 3]
    fz = raw[:, 4] + raw[:, 5] + raw[:, 6] + raw[:, 7]
    F_vec = np.column_stack([fx, fy, fz])
    F_avg = float(np.mean(np.linalg.norm(F_vec, axis=-1)))
    return F_avg / g