
from typing import Dict, Optional
import numpy as np
import numpy.typing as npt
from hexapod_biomechanics.utils import differentiate_rotation, normalize
from scipy.signal import savgol_filter
from scipy.constants import g

class AnkleID:

    def __init__(self, body_mass: float = 0.0, Inorm: npt.NDArray = np.eye(3), sex: str = "m"):

        # inertial properties estimated according to Dumas et al 2006 (doi:10.1016/j.jbiomech.2006.02.013)
        assert sex in ["m", "f"], f"Invalid sex: {sex}. Choose \"m\" or \"f\"."
        if sex == "m":
            self.m_F = 0.012 * body_mass # mass of foot [kg]
        elif sex == "f":
            self.m_F = 0.010 * body_mass
        
        self.I_F =  self.m_F * Inorm/1000.0 # foot inertia matrix in local foot (calcaneus) frame [kg*m*mm]

    def compute_dynamics(
            self,
            t: npt.NDArray,
            grf_force: npt.NDArray,
            grf_moment_origin: npt.NDArray,
            e1: npt.NDArray,
            e2: npt.NDArray,
            e3: npt.NDArray,
            o_ajc: npt.NDArray,
            R_F: npt.NDArray,
            COM_F: Optional[npt.NDArray] = None,
            filt_window_duration: float = 0.05, # seconds, ~20-30Hz cutoff,
            filt_poly: int = 4
    ) -> Dict[str, npt.ArrayLike]:
        
        dt = np.mean(np.diff(t)) # sample period [sec]
        fs = 1 / dt # sample frequency [Hz]

        # filtered frequencies is tied to time-length of filter window
        # compute stable filter window according to time length and sample frequency
        filt_window = int(fs * filt_window_duration)
        filt_window = filt_window if filt_window % 2 != 0 else filt_window + 1

        filt = lambda tjct, deriv : savgol_filter(
            x=tjct,
            window_length=filt_window,
            polyorder=filt_poly,
            deriv=deriv,
            delta=dt,
            axis=0
        )

        x_ajc = filt(o_ajc, deriv=0) # anatomical joint center: intermalleolar point [mm] (n_frames,3)
        omega_F, alpha_F = differentiate_rotation(t, R_F, filt_window, filt_poly) # foot rotations in local foot frame [rad/s] [rad/s/s] (n_frames,3)
        if COM_F is not None: # compute foot kinematics
            x_com = filt(COM_F, deriv=0) # center of motion position [mm] (n_frames,3)
            a_com = filt(COM_F, deriv=2) # center of motion acceleration [mm/s/s] (n_frames,3)

        F_grf = filt(grf_force, deriv=0) # [N]
        M_grf_origin = filt(grf_moment_origin, deriv=0) # [N*mm]

        # Force Balance
        # F_ank + F_grf + F_grav = m*a_com
        if COM_F is None: # simple: static with no gravity
            F_ank = -F_grf # [N]
        else:
            F_grav = np.zeros_like(F_grf) + np.array([0, 0, self.m_F * -g]) # [N] (n_frame,3)
            F_ank = (self.m_F * a_com) - F_grf - F_grav # [N] ankle interface forces (n_frames,3)


        # Moment Balance: about ankle joint center
        # interface forces (F_ank) are coincident / not contributing
        # M_ank + M_grf_ajc + M_grav_ajc = M_I_com + (m_F * ((x_com - x_ajc) x a_com))
        M_grf_ajc = M_grf_origin + np.cross(-x_ajc, F_grf) # ground reaction force moment represented at ankle joint center [N*mm] + [mm]x[N] = [N*mm] (n_frames,3)
        if COM_F is None: # simple: static with no gravity
            M_ank = -M_grf_ajc # [N*mm]
            M_I_ajc = np.zeros_like(M_ank)
            M_grav_ajc = np.zeros_like(M_ank)
        else:
            r_com_to_ajc = x_com - x_ajc # moment arm from ankle joint center to foot center of mass [mm] (n_frames,3)
            M_grav_ajc = np.cross(r_com_to_ajc, F_grav) # moment due to gravity at ankle joint center [N*mm] (n_frames,3)
            Ialpha = (self.I_F @ alpha_F[..., np.newaxis]).squeeze() # moment due to foot angular acceleration [kg*m*mm]*[rad/s/s] = [kg*m*mm/s/s] = [N*mm] (3,3) @ (n_frames,3,1) -> (n_frames,3)
            Iomega = (self.I_F @ omega_F[..., np.newaxis]).squeeze() # product of local inertia and local angular velocity [kg*m*mm]*[rad/s] = [kg*m*mm/s] (3,3) @ (n_frames,3,1) -> (n_frames,3)
            M_I_local = Ialpha + np.cross(omega_F, Iomega) # moment due to COM inertia in local foot frame [N*mm] + [kg*m*mm/s]x[rad/s] = [N*mm] (n_frames,3)
            M_I_com = (R_F @ M_I_local[..., np.newaxis]).squeeze() # moment due to COM inertia in global frame [N*mm] (n_frames,3)
            M_I_ajc = M_I_com + np.cross(r_com_to_ajc, self.m_F * a_com/1000.0) # moment due to inertia at ankle joint center [N*mm] + [mm]x[kg]*[m/s/s] = [N*mm] (n_frames,3)
            M_ank = M_I_ajc - M_grf_ajc - M_grav_ajc # ankle moments [N*mm] (n_frames,3)

        e1 = normalize(filt(e1, deriv=0)) # dorsiflexion/plantarflexion axis (n_frames,3)
        e2 = normalize(filt(e2, deriv=0)) # inversion/eversion axis (n_frames,3)
        e3 = normalize(filt(e3, deriv=0)) # internal/external rotation (n_frames,3)

        M_e1 = np.sum(M_ank * e1, axis=-1) # [N*mm] (n_frames,)
        M_e2 = np.sum(M_ank * e2, axis=-1)
        M_e3 = np.sum(M_ank * e3, axis=-1)

        F_e1 = np.sum(F_ank * e1, axis=-1) # [N*mm] (n_frames,)
        F_e2 = np.sum(F_ank * e2, axis=-1)
        F_e3 = np.sum(F_ank * e3, axis=-1)

        return {
            "M_ank": M_ank,
            "F_ank": F_ank,
            "M_e1": M_e1,
            "M_e2": M_e2,
            "M_e3": M_e3,
            "F_e1": F_e1,
            "F_e2": F_e2,
            "F_e3": F_e3,
            "omega_F": omega_F,
            "alpha_F": alpha_F,
            # Eval moment components
            "M_I": np.sum(M_I_ajc * e1, axis=-1),
            "M_grf": np.sum(M_grf_ajc * e1, axis=-1),
            "M_grav": np.sum(M_grav_ajc * e1, axis=-1)
        }