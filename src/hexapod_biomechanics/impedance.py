
from typing import Optional, Any, Tuple
import numpy as np
import numpy.typing as npt
from hexapod_biomechanics.kinematics import AnkleFrame
from hexapod_biomechanics.inverse_dynamics import AnkleID
from hexapod_biomechanics.forces import HexKistler
from hexapod_biomechanics.load_data import C3DMan
from hexapod_biomechanics.utils import find_pert_thresh, axis_difference

class AnkleImpedance:

    def __init__(
            self,
            AnkleKinematics: AnkleFrame,
            FootAnkleDynamics: AnkleID,
            PertPlatform: HexKistler,
            PlatformFilter: Optional[Any] = None
    ):

        self.AnkleKinematics = AnkleKinematics
        self.FootAnkleDynamics = FootAnkleDynamics
        self.PertPlatform = PertPlatform
        self.PlatformFilter = PlatformFilter

        self.processed_trial_names = []
        self.perturbed_trials = []
        self.unperturbed_trials = []

    def process_trial(
            self,
            t: npt.NDArray,
            fs: float,
            calc_marker: npt.NDArray,
            m1_marker: npt.NDArray,
            m5_marker: npt.NDArray,
            cluster_s_markers: npt.NDArray,
            hex_markers: npt.NDArray,
            forces: npt.NDArray,
            trial_name: str,
            trial_num: int,
            condition: str,
            side: str,
            direction: int,
            session: str,
            percent: Optional[int] = None
    ):
        
        if trial_name in self.processed_trial_names:
            print(f"WARNING: Processing duplicate trial \"{trial_name}\".")
        self.processed_trial_names.append(trial_name)
        
        # compute forces and moments
        grf = self.PertPlatform.process_forces(
            cluster_hex=hex_markers,
            raw_forces=forces
        )

        # compute kinematics
        kinematics = self.AnkleKinematics.compute_kinematics(
            CALC=calc_marker,
            M1=m1_marker,
            m5=m5_marker,
            cluster_S=cluster_s_markers
        )

        # compute dynamics
        dynamics_full = self.FootAnkleDynamics.compute_dynamics(
            t=t,
            grf_force=grf["force"],
            grf_moment_origin=grf["moment_origin"],
            e1=kinematics["e1"],
            e2=kinematics["e2"],
            e3=kinematics["e3"],
            o_ajc=kinematics["o_ajc"],
            R_F=kinematics["R_F"],
            COM_F=kinematics["COM_F"]
        )
        dynamics_static = self.FootAnkleDynamics.compute_dynamics(
            t=t,
            grf_force=grf["force"],
            grf_moment_origin=grf["moment_origin"],
            e1=kinematics["e1"],
            e2=kinematics["e2"],
            e3=kinematics["e3"],
            o_ajc=kinematics["o_ajc"],
            R_F=kinematics["R_F"],
            COM_F=None # compute without inertial parameters
        )

        # filter dorsiflexion angle and perturbation trajectory akin to dynamics
        filt = dynamics_full["filter"]
        alpha = filt(kinematics["alpha"], deriv=0)
        alpha_dot = filt(kinematics["alpha"], deriv=1) # ankle velocity (rad/s)

        # evaluate stance period
        i_stance_start, i_stance_end = np.where(grf["is_stance"])[0][[0,-1]] # start and end indices of stance
        frames_stance = i_stance_end - i_stance_start # number of indices spanning stance

        trial_data = {
            "trial_name": trial_name,
            "trial_num": trial_num,
            "condition": condition,
            "side": side,
            "direction": direction,

            "stance_numframes": frames_stance,
            "stance_duration": frames_stance / fs,

            "timepoints": t[grf["is_stance"]] - t[grf["is_stance"]][0],
            "ankle_angle": alpha[grf["is_stance"]],
            "ankle_velocity": alpha_dot[grf["is_stance"]],
            "ankle_torque_full": dynamics_full["M_jcs"][grf["is_stance"],0],
            "ankle_torque_static": dynamics_static["M_jcs"][grf["is_stance"],0],
            "foot_alpha": dynamics_full["alpha_foot_jcs"][grf["is_stance"],0],
            "foot_omega": dynamics_full["omega_foot_jcs"][grf["is_stance"],0]
        }

        if condition == "pert":

            i_pert_start, pert_window, terminal_window = get_pert_windows(
                fs=fs,
                hex_stance_tjct=grf["hex_tjct"][grf["is_stance"]]
            )
            phase_pert_start = i_pert_start / frames_stance # phase of stance when perturbation occurs (approx)

            kistler_rot = self.PertPlatform.locate_rot_axis(
                cluster_hex_pert=hex_markers[terminal_window+i_stance_start] # terminal window represents indices in stance, need to convert to full trial indices
            )
            axis_diff = axis_difference(
                target_axis=(kistler_rot["origin"], kistler_rot["direction"]),
                actual_axis=(kinematics["o_ajc"][grf["is_stance"]], kinematics["e1"][grf["is_stance"]])
            )
            axisdiff_ang = axis_diff["angle_diff"][pert_window] # radians
            axisdiff_pos = axis_diff["offset_diff"][pert_window] # mm

            trial_data.update({
                "pert_pct": percent,

                "pert_frame": i_pert_start,
                "pert_time": i_pert_start / fs,
                "pert_phase": phase_pert_start,
                "pert_window": pert_window,

                "axisdiff_ang": axisdiff_ang,
                "axisdiff_pos": axisdiff_pos
            })

            self.perturbed_trials.append(trial_data)

        elif condition == "nom":

            self.unperturbed_trials.append(trial_data)

    def process_wrapper(
            self
    ):
        
        pass

def get_pert_windows(
        fs: float,
        hex_stance_tjct: npt.NDArray,
        threshold_deg: float = 1.0,
        pre_threshold_start_ms: float = 46.0,
        post_pert_window_ms: float = 100.0,
        pre_pert_window_ms: float = 25.0,
        static_window_ms: float = [150.0, 500.0]
) -> Tuple[int, npt.NDArray, npt.NDArray]:
    
    i_pert_mid = find_pert_thresh(
        hex_tjct=hex_stance_tjct,
        thresh=np.radians(threshold_deg)
    )
    pre_threshold_start_frames = round(pre_threshold_start_ms/1000 * fs)
    post_pert_window_frames = round(post_pert_window_ms/1000 * fs)
    pre_pert_window_frames = round(pre_pert_window_ms/1000 * fs)
    i_pert_start = i_pert_mid - pre_threshold_start_frames # start index of perturbation within stance period
    pert_window = list(range(i_pert_start - pre_pert_window_frames, i_pert_start + post_pert_window_frames)) # indices of perturbation window of stance period

    static_window_frames = round(static_window_ms/1000 * fs)
    i_static_window = i_pert_mid + static_window_frames # index bounds on static period post-perturbation
    terminal_window = list(range(i_static_window[0], i_static_window[1]))

    return i_pert_start, pert_window, terminal_window
