
import numpy as np
import numpy.typing as npt

def normalize(v: npt.NDArray) -> npt.NDArray:
    """Normalize vector(s) across coordinate dimension.

    Args:
        v (npt.NDArray): Vector(s) to be normalized, with normalized dimension last.

    Returns:
        npt.NDArray: Normalized vector(s).
    """

    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / norms