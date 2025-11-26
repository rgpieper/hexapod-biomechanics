
from typing import Dict
import numpy as np
import numpy.typing as npt
from hexapod_biomechanics.utils import differentiate_rotation
from scipy.signal import savgol_filter

class AnkleID:

    def __init__(self):

        pass

    def compute_dynamics(
            self,
            t: npt.NDArray,
            grf_force: npt.NDArray,
            grf_moment_origin: npt.NDArray,
            e1: npt.NDArray,
            e2: npt.NDArray,
            e3: npt.NDArray,
            o_T: npt.NDArray,
            R_F: npt.NDArray,
            COM_F: npt.NDArray,
            filt_window_duration: float = 0.05, # seconds, ~20-30Hz cutoff,
            filt_poly: int = 4
    ) -> Dict[str, npt.ArrayLike]:
        
        dt = np.mean(np.diff(t)) # sample period
        fs = 1 / dt # sample frequency

        # filtered frequencies is tied to time-length of filter window
        # compute stable filter window according to time length and sample frequency
        filt_window = int(fs * filt_window_duration)
        filt_window = filt_window if filt_window % 2 != 0 else filt_window + 1
        
        ajc = o_T # anatomical joint center: intermalleolar point

        v_COM = savgol_filter(
            COM_F,
            window_length=filt_window,
            polyorder=filt_poly,
            deriv=1,
            delta=dt,
            axis=0
        )
        a_COM = savgol_filter(
            COM_F,
            window_length=filt_window,
            polyorder=filt_poly,
            deriv=2,
            delta=dt,
            axis=0
        )