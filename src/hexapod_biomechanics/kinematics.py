
from typing import Dict
import numpy as np
import numpy.typing as npt
from hexapod_biomechanics.utils import normalize, rigid_transform, clamp

class AnkleFrame:
    """Ankle kinematics solver.

    Based on ankle joint coordinate system, recommended by:
    Wu G, van der Helm FC, Veeger HE, Makhsous M, Van Roy P, Anglin C, Nagels J, Karduna AR, McQuade K, Wang X, Werner FW, Buchholz B
    International Society of Biomechanics
    ISB recommendation on definitions of joint coordinate systems of various joints for the reporting of human joint motion
    Part II: shoulder, ellbow, wrist and hand
    J Biomech. 2005 May 38(5):981-992
    doi: 10.1016/j.jbiomech.2004.05.042. PMID: 15844264
    """

    def __init__(
            self,
            side: int,
            MM_static: npt.NDArray,
            LM_static: npt.NDArray,
            MC_static: npt.NDArray,
            LC_static: npt.NDArray,
            CALC_static: npt.NDArray,
            M1_static: npt.NDArray,
            M5_static: npt.NDArray,
            cluster_S_static: npt.NDArray,
            sex: str = "m"
    ) -> None:
        """Setup the base configuration of the ankle joint coordinate system with a static trial in the neutral position.

        Args:
            side (int): Ankle side (1: right, -1: left)
            MM_static (npt.NDArray): Medial malleolus across a static trial (n_frames, 3)
            LM_static (npt.NDArray): Lateral malleolus across a static trial (n_frames, 3)
            MC_static (npt.NDArray): Medial tibial condyle across a static trial (n_frames, 3)
            LC_static (npt.NDArray): Lateral tibial condyle across a static trial (n_frames, 3)
            CALC_static (npt.NDArray): Calcaneus (heel) across a static trial (n_frames,3)
            M1_static (npt.NDArray): First metatarsal across a static trial (n_frames,3)
            M5_static (npt.NDArray): Fifth metatarsal across a static trial (n_frames,3)
            cluster_S_static (npt.NDArray): Shank cluster markers across a static trial (n_frames, n_markers, 3)
            sex (npt.NDArray): Sex of subject ("m": male or "f": female) for applying anthropometric data. Defaults to "m".
        """
        
        assert side in [-1, 1], f"Invalid side: {side}. Choose '-1' or '1'."
        self.side = side

        # average across static trajectory
        MM_neut = np.nanmean(MM_static, axis=0) # shape (3,)
        LM_neut = np.nanmean(LM_static, axis=0)
        MC_neut = np.nanmean(MC_static, axis=0)
        LC_neut = np.nanmean(LC_static, axis=0)
        CALC_neut = np.nanmean(CALC_static, axis=0)
        M1_neut = np.nanmean(M1_static, axis=0)
        M5_neut = np.nanmean(M5_static, axis=0)
        self.cluster_S_neut = np.nanmean(cluster_S_static, axis=0) # shape (n_markers, 3)
        self.cluster_F_neut = np.nanmean(np.stack([CALC_static, M1_static, M5_static], axis=1), axis=0) # construct cluster from calcaneus and metatarsal markers

        IM = (MM_neut + LM_neut) / 2.0 # inter-malleolar point, midway between MM and LM
        IC = (MC_neut + LC_neut) / 2.0 # inter-condylar point, midway between MC and LC

        # base tibia/fibula coordinate system, represented in the global frame
        o_T_G = IM # origin, coincident with inter-malleolar point
        z_T_G = normalize(self.side*(LM_neut - MM_neut)) # z-axis, connecting MM and LM directed to the right
        x_T_G = normalize(self.side*np.cross(LM_neut - IC, MM_neut - IC)) # x-axis, perpendicular to the torsional plane (containing IC, MM, LM) directed anteriorly
        y_t_G = normalize(np.cross(z_T_G, x_T_G)) # y-axis, mutually perpendicular to x- and z-axes

        # transformation from base tibia/fibula coordinate system to the global frame
        self.T_TG_neut = np.eye(4)
        self.T_TG_neut[:3, 0] = x_T_G
        self.T_TG_neut[:3, 1] = y_t_G
        self.T_TG_neut[:3, 2] = z_T_G
        self.T_TG_neut[:3, 3] = o_T_G

        # base calcaneus coordinate system, represented in the global frame
        o_C_G = o_T_G # orign, coincident with tibia/fibula coordinate system in the base configuration
        y_C_G = normalize(IC - IM) # y-axis, coincident with the long-axis of the tibia/fibula in the base configuration directed cranially
        x_C_G = normalize(self.side*np.cross(MC_neut - IM, LC_neut - IM)) # x-axis, perpendicular to the frontal plane of the tibia/fibula (containing IM, MC, LC) in the base configuration directed anteriorly
        z_C_G = normalize(np.cross(x_C_G, y_C_G)) # z-axis, mutually perpendicular to the x- and y-axes

        # transformation from base calcaneus coordinate system to the global frame
        self.T_CG_neut = np.eye(4)
        self.T_CG_neut[:3, 0] = x_C_G
        self.T_CG_neut[:3, 1] = y_C_G
        self.T_CG_neut[:3, 2] = z_C_G
        self.T_CG_neut[:3, 3] = o_C_G

        # define foot segment legnth, segment coordinate system and center of mass
        # defined by Dumas et al 2006 (doi:10.1016/j.jbiomech.2006.02.013)
        o_SCS = o_T_G # origin (anatomical joint center - ankle: intermalleolar point)
        mid_met = (M1_neut + M5_neut) / 2.0 # midpoint between first and fifth metatarsals
        x_SCS = normalize(mid_met - CALC_neut) # x-axis: from calcaneus to midpoint between metatarsals
        y_SCS = normalize(self.side*np.cross(M5_neut - CALC_neut, M1_neut - CALC_neut)) # y-axis: normal to plane including calcaneus and metatarsals, pointing cranially
        z_SCS = normalize(np.cross(x_SCS, y_SCS))
        T_SCS = np.eye(4) # transformation from foot segment coordinate system to the global frame
        T_SCS[:3, 0] = x_SCS
        T_SCS[:3, 1] = y_SCS
        T_SCS[:3, 2] = z_SCS
        T_SCS[:3, 3] = o_SCS
        R_SCS = T_SCS[:3, :3] # just the rotation
        L_seg = np.linalg.norm(mid_met - o_SCS) # foot segment length (scalar): anatomical joint center to midpoint of metatarsals
        assert sex in ["m", "f"], f"Invalid sex: {sex}. Choose \"m\" or \"f\"."
        if sex == "m":
            COM_SCS = np.array([0.382, -0.151, 0.026])*L_seg
            Inorm_SCS = np.real((np.array([
                [17, 13, 8j],
                [13, 37, 0],
                [8j, 0, 36]
            ]) * L_seg)**2) # segment-mass-normalized inertia matrix: r_ij = (1/L_seg)*sqrt(I_ij/m_seg)
        elif sex == "f":
            COM_SCS = np.array([0.270, -0.218, 0.039])*L_seg
            Inorm_SCS = np.real((np.array([
                [17, 10j, 6],
                [10j, 36, 4j],
                [6, 4j, 35]
            ]) * L_seg)**2) # segment-mass-normalized inertia matrix: r_ij = (1/L_seg)*sqrt(I_ij/m_seg)
        self.COM_C = (np.linalg.inv(self.T_CG_neut) @ T_SCS @ np.append(COM_SCS, 1))[:3] # foot COM represented in calcaneus frame (fixed to calc. frame) (3,)
        self.Inorm_C = R_SCS @ Inorm_SCS @ R_SCS.T # rotate inertia matrix to calcaneus frame (3,3)

        self.T_TG = np.eye(4)[np.newaxis, ...]
        self.T_CG = np.eye(4)[np.newaxis, ...]
        self.kinematics = {
            "alpha": np.empty(0),
            "beta": np.empty(0),
            "gamma": np.empty(0),
            "q1": np.empty(0),
            "q2": np.empty(0),
            "q3": np.empty(0),
            "e1": np.empty(0),
            "e2": np.empty(0),
            "e3": np.empty(0),
            "o_ajc": np.empty(0),
            "R_F": np.empty(0),
            "COM_F": np.empty(0)
        }

    def compute_kinematics(self,
            CALC: npt.NDArray,
            M1: npt.NDArray,
            M5: npt.NDArray,
            cluster_S: npt.NDArray
    ) -> Dict[str, npt.NDArray]:
        """Compute 6-DOF ankle joint kinematics relative to neutral static configuration.

        Args:
            CALC (npt.NDArray): Calcaneus trajectory (n_frames,3)
            M1 (npt.NDArray): First metatarsal trajectory (n_frames,3)
            M5 (npt.NDArray): Fifth metatarsal trajectory (n_frames,3)
            cluster_S (npt.NDArray): Shank cluster trajectory (n_frames,n_markers,3)

        Returns:
            Dict[str, npt.NDArray]: Kinematics including:
                'alpha' (npt.NDArray): Dorsiflexion/plantarflexion angle (n_frames,)
                'beta' (npt.NDArray): Inversion/eversion angle (n_frames,)
                'gamma' (npt.NDArray): Internal/external rotation angle (n_frames,)
                'q1' (npt.NDArray): Medial/lateral shift (n_frames,)
                'q2' (npt.NDArray): Anterior/posterior shift (n_frames,)
                'q3' (npt.NDArray): Vertical shift (n_frames,)
                'e1' (npt.NDArray): Dorsiflexion/plantarflexion axis (n_frames,3)
                'e2' (npt.NDArray): Inversion/eversion axis (n_frames,3)
                'e3' (npt.NDArray): Internal/external rotation axis (n_frames,3)
                'o_ajc' (npt.NDArray): Ankle joint center: intermalleolar point (tibia/fibula frame origin) (n_frames,3)
                'R_F' (npt.NDArray): Foot orientation (rotation matrix tracking calcaneus frame) (n_frames,3,3)
                'COM_F (npt.NDArray): Foot center of mass in the global frame (n_frames,3)
        """

        T_S_move = rigid_transform(self.cluster_S_neut, cluster_S) # shank movement from base to dynamic
        T_F_move = rigid_transform(self.cluster_F_neut, np.stack([CALC, M1, M5], axis=1)) # foot movement from base to dynamic, constructing foot cluster from calcaneus and metatarsal markers

        # move anatomical frames with their rigid segment (F, 4, 4) @ (4, 4) -> (F, 4, 4)
        self.T_TG = T_S_move @ self.T_TG_neut # dynamic tibia/fibula coordinate system in the global frame
        self.T_CG = T_F_move @ self.T_CG_neut # dynamic calcaneus coordinate system in the global frame

        COM_G = (self.T_CG @ np.append(self.COM_C, 1))[:, :3] # track foot COM in global frame with calcaneus frame

        x_T_G = self.T_TG[:, :3, 0] # tibia/fibula x-axis (F, 3)
        o_T_G = self.T_TG[:, :3, 3] # tibia/fibula origin (F, 3)

        x_C_G = self.T_CG[:, :3, 0] # calcaneus x-axis (F, 3)
        o_C_G = self.T_CG[:, :3, 3] # calcaneus origin (F, 3)

        # construct ankle joint coordinate system
        e1 = self.T_TG[:, :3, 2] # tibia/fibula z-axis (fixed) (F, 3)
        e3 = self.T_CG[:, :3, 1] # calcaneus y-axis (fixed) (F, 3)
        e2 = normalize(np.cross(e3, e1)) # floating axis, perpendicular to e1 and e3 (F, 3)

        # dorsiflexion/plantarflexion:
        # rotation about e1: angle between floating axis and tibia/fibula x-axis
        dot_alpha = clamp(np.sum(x_T_G * e2, axis=1)) # dot product between tibia/fibula x-axis and floating axis, clamped to [-1, 1] for stability (F,)
        sign_alpha = np.sign(np.sum(e1 * np.cross(x_T_G, e2), axis=1)) # direction of rotation (F,)
        alpha = sign_alpha * np.arccos(dot_alpha) # (F,)

        # inversion/eversion:
        # rotation about e2: angle between tibia/fibula z-axis and calcaneus y-axis
        dot_beta = clamp(np.sum(e1 * e3, axis=1)) # (F,)
        beta = self.side * (np.pi/2 - np.arccos(dot_beta)) # assume remains witin [-pi/2, pi/2] (F,)

        # internal/external rotation
        # rotation about e3: angle between floating axis and calcaneus x-axis
        dot_gamma = clamp(np.sum(e2 * x_C_G, axis=1)) # (F,)
        sign_gamma = self.side * np.sign(np.sum(e3 * np.cross(e2, x_C_G), axis=1)) # direction of rotation (F,)
        gamma = sign_gamma * np.arccos(dot_gamma)

        # segment displacements
        v_TC = o_C_G - o_T_G # vector from tibia/fibula to calc
        q1 = self.side * np.sum(v_TC * e1, axis=1) # medial/lateral shift (F,)
        q2 = np.sum(v_TC * e2, axis=1) # anterior/posterior shift
        q3 = np.sum(v_TC * e3, axis=1)

        self.kinematics = {
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
            "q1": q1,
            "q2": q2,
            "q3": q3,
            "e1": e1,
            "e2": e2,
            "e3": e3,
            "o_ajc": o_T_G,
            "R_F": self.T_CG[:, :3, :3],
            "COM_F": COM_G
        }

        return self.kinematics