
import numpy as np
import numpy.typing as npt

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