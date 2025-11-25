
from typing import Dict
from itertools import combinations
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import pdist
from src.utils import rigid_transform

class HexKistler:

    def __init__(self, cluster_hex_static: npt.NDArray) -> None:

        self.cluster_hex_base = np.nanmean(cluster_hex_static, axis=0) # (n_markers, 3)

        self.a_z0 = -41 # mm, distance from force plate surface to sensor plane / Kistler origin
        self.a = 210 # mm, distance from sensor axis to kistler y-axis
        self.b = 260 # mm, distance from sensor axis to Kistler x-axis

        h_marker = 12 # mm, marker height (distance from marker to force plate surface)

        self.sensors_K = np.array([ # sensor locations (homogeneous) in Kistler's frame
            [self.a, self.b, 0, 1],
            [-self.a, self.b, 0, 1],
            [-self.a, -self.b, 0, 1],
            [self.a, -self.b, 0, 1]
        ])

        distances = pdist(self.cluster_hex_base)
        pairs = list(combinations(range(self.cluster_hex_base.shape[0]), 2))
        origin = np.nanmean(self.cluster_hex_base[pairs[np.argmax(distances)], :], axis=0)
        origin[2] -= (h_marker + self.a_z0) # Kistler origin is below force plate surface

        # TODO: implement a Kistler calibration / localization protocol
        self.T_KG_neut = np.eye(4) # assume Kistler axes are parallel to global axes
        self.T_KG_neut[:3, 3] = origin
        self.T_KG_neut[(1,2), (1,2)] = -1 # flip y and z axes orientation

    def process_forces(
            self,
            cluster_hex: npt.NDArray,
            fx12: npt.NDArray,
            fx34: npt.NDArray,
            fy14: npt.NDArray,
            fy23: npt.NDArray,
            fz1: npt.NDArray,
            fz2: npt.NDArray,
            fz3: npt.NDArray,
            fz4: npt.NDArray,
    ) -> Dict[str, npt.NDArray]:
        
        assert cluster_hex.shape[0] == fz1.shape[0], f"Force data length ({fz1.shape[0]}) does not match marker data length ({cluster_hex.shape[0]})."
        n_frames = fz1.shape[0]

        # Kistler transformation
        T_H_move = rigid_transform(self.cluster_hex_base, cluster_hex) # hexapod movement from base to dynamic (n_frames, 4, 4)
        T_KG = T_H_move @ self.T_KG_neut # dynamic Kistler coordinate system (n_frames, 4, 4)
        R_KG = T_KG[:, :3, :3] # just rotation (n_frames, 3, 3)
        o_KG = T_KG[:, :3, 3] # translation (Kistler origin in global frame) (n_frames, 3)

        # resultant forces in Kistler frame
        Fx_K = (-fx12) + (-fx34)
        Fy_K = (-fy14) + (-fy23)
        Fz_K = (-fz1) + (-fz2) + (-fz3) + (-fz4)

        # rotate to global frame
        F_K = np.column_stack([Fx_K, Fy_K, Fz_K]) # (n_frames, 3)
        F = (R_KG @ F_K[..., np.newaxis]).squeeze() # (n_frames, 3, 3)@(n_frames, 3, 1) -> (n_frames, 3, 1) -> (n_frames, 3)

        # moment represented at Kistler origin
        Mx_K = self.b * ((-fz1) + (-fz2) - (-fz3) - (-fz4))
        My_K = self.a * (-(-fz1) + (-fz2) + (-fz3) - (-fz4))
        Mz_K = self.b * (-(-fx12) + (-fx34)) + self.a * ((-fy14) - (-fy23))

        # rotate to global frame and move to origin
        M_K = np.column_stack([Mx_K, My_K, Mz_K])
        M_G_kist = (R_KG @ M_K[..., np.newaxis]).squeeze() # (n_frames, 3, 3)@(n_frames, 3, 1) -> (n_frames, 3, 1) -> (n_frames, 3)
        M = M_G_kist * np.cross(o_KG, F) # moments represented at global origin

        # determine free moment (about Kistler z at COP)
        M_free_scalar = Mz_K - (Fy_K * COPx_K) + (Fx_K * COPy_K)
        plate_normal = (R_KG @ np.array([0, 0, 1])).squeeze()
        M_free = plate_normal * M_free_scalar[:, np.newaxis] # (n_frames, 3) @ (n_frames, 1)

        # moments represented at the center of Kistler surface (above origin)
        Mx_surf = Mx_K + (Fy_K * self.a_z0) # a_z0 defined as negative
        My_surf = My_K - (Fx_K * self.a_z0)

        # compute COP only when force plate is loaded
        Fz_load = Fz_K.copy()
        Fz_load[np.abs(Fz_load) < 18.0] = np.nan

        # compute COP in Kistler frame
        COPx_K = -My_surf / Fz_load
        COPy_K = Mx_surf / Fz_load
        COPz_K = np.full(n_frames, self.a_z0) # surface necessarily at z = a_z0 (-41mm)
        
        # transform COP to global frame
        COP_K = np.column_stack([COPx_K, COPy_K, COPz_K, np.ones(n_frames)])
        COP = (T_KG @ COP_K[..., np.newaxis]).squeeze()[:, :3] # (n_frames, 4, 4)@(n_frames, 4, 1) -> (n_frames, 4, 1) -> (n_frames, 3)

        return {
            "force": F,
            "moment_origin": M,
            "COP": COP,
            "moment_free": M_free,
            "moment_free_scalar": M_free_scalar
        }





         