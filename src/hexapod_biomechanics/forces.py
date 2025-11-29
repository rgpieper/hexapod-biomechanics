
from typing import Dict, Tuple, Optional
from itertools import combinations
import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation
from hexapod_biomechanics.utils import rigid_transform, clamp, normalize, inv_rodrigues

class HexKistler:
    """Hexapod-Kistler ground reaction force solver.
    """

    def __init__(self, cluster_hex_static: npt.NDArray) -> None:
        """Setup transformation from Kistler frame to global frame with Hexapod in the base/home configuration (static trial).

        Args:
            cluster_hex_static (npt.NDArray): Hexapod markers across a static trial [mm] (n_frames,n_markers,3)
        """

        self.cluster_hex_base = np.nanmean(cluster_hex_static, axis=0) # (n_markers, 3)

        self.a_z0 = -41 # [mm], distance from force plate surface to sensor plane / Kistler origin
        self.a = 210 # [mm], distance from sensor axis to kistler y-axis
        self.b = 260 # [mm], distance from sensor axis to Kistler x-axis

        h_marker = 12 # [mm], marker height (distance from marker to force plate surface)

        distances = pdist(self.cluster_hex_base)
        pairs = list(combinations(range(self.cluster_hex_base.shape[0]), 2))
        origin = np.nanmean(self.cluster_hex_base[pairs[np.argmax(distances)], :], axis=0)
        origin[2] -= (h_marker + (-self.a_z0)) # Kistler origin is below force plate surface (a_z0 defined negative)

        # TODO: implement a Kistler calibration / localization protocol
        self.T_KG_neut = np.eye(4) # assume Kistler axes are parallel to global axes
        self.T_KG_neut[:3, 3] = origin
        self.T_KG_neut[(1,2), (1,2)] = -1 # flip y and z axes orientation

        self.T_KG = np.eye(4)[np.newaxis, ...]

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
            stance_thresh: float = 18.0
    ) -> Dict[str, npt.NDArray]:
        """Convert raw forces to ground reaction force parameters, accounting for transformation of the Hexapod.

        Args:
            cluster_hex (npt.NDArray): Hexapod marker trajectories [mm] (n_frames,n_markers,3)
            fx12 (npt.NDArray): Raw x-direction force acting on Kistler between sensors 1 and 2 [N] (n_frames,)
            fx34 (npt.NDArray): Raw x-direction force acting on Kistler between sensros 3 and 4 [N] (n_frames,)
            fy14 (npt.NDArray): Raw y-direction force acting on Kistler between sensors 1 and 4 [N] (n_frames,)
            fy23 (npt.NDArray): Raw y-direction force acting on Kistler between sensors 2 and 3 [N] (n_frames,)
            fz1 (npt.NDArray): Raw z-direction force acting on Kistler at sensor 1 [N] (n_frames,)
            fz2 (npt.NDArray): Raw z-direction force acting on Kistler at sensor 2 [N] (n_frames,)
            fz3 (npt.NDArray): Raw z-direction force acting on Kistler at sensor 3 [N] (n_frames,)
            fz4 (npt.NDArray): Raw z-direction force acting on Kistler at sensor 4 [N] (n_frames,)
            stance_thresh (float, optional): Vertical (z) force at which subject is in stance on Hexapod, when COP will be computed [N]. Defaults to 18.0.

        Returns:
            Dict[str, npt.NDArray]: Ground reaction force parameters, including:
                'force' (npt.NDArray): Force components in global frame [N] (n_frames,3)
                'moment_origin' (npt.NDArray): Moment components in global frame represented at global origin [N*mm] (n_frames,3)
                'COP' (npt.NDArray): Center of pressure coordinates in global frame [mm] (n_frames,3)
                'moment_free' (npt.NDArray): Free moment (friction) components at COP in global frame [N*mm] (n_frames,3)
                'moment_free_scalar' (float): Magnitude of free moment [N*mm].
        """
        
        assert cluster_hex.shape[0] == fz1.shape[0], f"Force data length ({fz1.shape[0]}) does not match marker data length ({cluster_hex.shape[0]})."
        n_frames = fz1.shape[0]

        # Kistler transformation
        T_H_move = rigid_transform(self.cluster_hex_base, cluster_hex) # hexapod movement from base to dynamic (n_frames, 4, 4)
        self.T_KG = T_H_move @ self.T_KG_neut # dynamic Kistler coordinate system (n_frames, 4, 4)
        R_KG = self.T_KG[:, :3, :3] # just rotation (n_frames, 3, 3)
        o_KG = self.T_KG[:, :3, 3] # translation (Kistler origin in global frame) (n_frames, 3)

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
        M = M_G_kist + np.cross(o_KG, F) # moments represented at global origin

        # moments represented at the center of Kistler surface (above origin)
        Mx_surf = Mx_K + (Fy_K * self.a_z0) # a_z0 defined as negative
        My_surf = My_K - (Fx_K * self.a_z0)

        # compute COP only when force plate is loaded
        Fz_load = Fz_K.copy()
        Fz_load[np.abs(Fz_load) < stance_thresh] = np.nan

        # compute COP in Kistler frame
        COPx_K = -My_surf / Fz_load
        COPy_K = Mx_surf / Fz_load
        COPz_K = np.full(n_frames, self.a_z0) # surface necessarily at z = a_z0 (-41mm)
        
        # transform COP to global frame
        COP_K = np.column_stack([COPx_K, COPy_K, COPz_K, np.ones(n_frames)])
        COP = (self.T_KG @ COP_K[..., np.newaxis]).squeeze()[:, :3] # (n_frames, 4, 4)@(n_frames, 4, 1) -> (n_frames, 4, 1) -> (n_frames, 3)

        # determine free moment (about Kistler z at COP)
        M_free_scalar = Mz_K - (Fy_K * COPx_K) + (Fx_K * COPy_K)
        plate_normal = (R_KG @ np.array([0, 0, 1])).squeeze()
        M_free = plate_normal * M_free_scalar[:, np.newaxis] # (n_frames, 3) @ (n_frames, 1)

        return {
            "force": F,
            "moment_origin": M,
            "COP": COP,
            "moment_free": M_free,
            "moment_free_scalar": M_free_scalar
        }
    
    def track_plate(self) -> Tuple[npt.NDArray, npt.NDArray]:
        """Compute Kistler corner and sensor trajectories according to Hexapod marker transformations.

        NOTE: If process_forces has not been executed first, will just return base corner/sensor locations.

        Returns:
            Tuple[npt.NDArray, npt.NDArray]:
                corners (npt.NDArray): Kistler corner trajectories [mm] (n_frames,4 corners,3)
                sensors (npt.NDArray): Kistler sensor trajectories [mm] (n_frames,4 sensors,3)
        """

        corners_K = np.array([ # surface corner locations (homogeneous) in Kistler's frame
            [250, 300, self.a_z0, 1],
            [-250, 300, self.a_z0, 1],
            [-250, -300, self.a_z0, 1],
            [250, -300, self.a_z0, 1]
        ]) # (n_corners, 4)
        corners = (self.T_KG @ corners_K.T)[:, :3, :].transpose(0, 2, 1) # (n_frames, 4, 4) @ (4, n_corners) -> (n_frames, 4, n_corners) -> (n_frames, n_corners, 3)

        sensors_K = np.array([ # sensor locations (homogeneous) in Kistler's frame
            [self.a, self.b, 0, 1],
            [-self.a, self.b, 0, 1],
            [-self.a, -self.b, 0, 1],
            [self.a, -self.b, 0, 1]
        ]) # (n_sensors, 4)
        sensors = (self.T_KG @ sensors_K.T)[:, :3, :].transpose(0, 2, 1) # (n_frames, 4, 4) @ (4, n_sensors) -> (n_frames, 4, n_sensors) -> (n_frames, n_sensors, 3)

        return corners, sensors
    
    def locate_rot_axis(self, cluster_hex_pert: npt.NDArray) -> Optional[Dict[str, npt.NDArray]]:
        """Locate perturbation axis of rotation according to terminal perturbation location of hexapod markers.

        Args:
            cluster_hex_pert (npt.NDArray): Terminal perturbed hexapod marker location(s) (n_frames,n_markers,3) or (n_markers,3). Multiple provided frames will be averaged.

        Returns:
            Optional[Dict[str, npt.NDArray]]: Rotational axis parameters, including:
                'origin' (npt.NDArray): Axis reference point nearest to base/neutral Kistler origin [mm] (3,)
                'direction' (npt.NDArray): Unit vector axis of rotation (3,)
                'angle' (float): Rotation magnitude [radians]
                'angle_tjct' (npt.NDArray): Angular position trajectory of Hexapod/Kistler [radians] (n_frames,)
        """

        if len(cluster_hex_pert.shape) == 3: # frame dimension included (n_frames,n_markers,3)
            cluster_hex_pert = np.nanmean(cluster_hex_pert, axis=0, keepdims=True) # (n_markers,3)
        else: # frame dimension excluded (n_markers,3)
            cluster_hex_pert = cluster_hex_pert[np.newaxis, ...] # (1,n_markers,3)

        T = rigid_transform(self.cluster_hex_base, cluster_hex_pert).squeeze() # hexapod movement from base to perturbed position (4,4)
        R = T[:3, :3] # rotation matrix (3,3)
        x = T[:3, 3] # translation (3,)

        theta, n = inv_rodrigues(R, tol=1e-3)

        rot = Rotation.from_matrix(self.T_KG[:, :3, :3]) # rotation trajectory of Kistler
        rot_vec = rot.as_rotvec() # (n_frames,3)

        if n is None:
            return {
                "origin": None,
                "direction": n,
                "max_angle": theta,
                "angle_tjct": np.linalg.norm(rot_vec, axis=1) # if no axis found, report norm of rotation vector [rad] (n_frames,)
            }

        # tranfromation: p_end = T*p_start_hom = R*p_start + x
        # if p_start on axis of rotation: p_end = p_start + (d*n) with scalar d
        d = np.dot(x, n) # component of translation parallel to axis of rotation
        # p + (d*n) = R*p + x --> (I - R)*p = x - (d*n)
        x_ortho = x - (d * n) # translation orthogonal to axis of rotation
        # (I - R) is singular because infinite points on axis of rotation solve this equation
        # use least squares to find point closest to global origin
        p_near_orig, _, _, _ = np.linalg.lstsq(np.eye(3) - R, x_ortho, rcond=None)

        p_to_K = self.T_KG_neut[:3, 3] - p_near_orig # vector from ref point closest to origin and neutral Kistler origin
        slide = np.dot(p_to_K, n) # project vector onto rotational axis
        p_near_K = p_near_orig + (slide * n) # slide reference point along rotation axis to near neutral Kistler origin

        theta_tjct = np.sum(rot_vec * n, axis=1) # rotation about the principle Kistler rotation axis [rad] (n_frames,)

        return {
            "origin": p_near_K,
            "direction": n,
            "max_angle": theta,
            "angle_tjct": theta_tjct
        }