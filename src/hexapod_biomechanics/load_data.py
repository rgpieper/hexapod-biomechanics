"""C3D file loading with cached in-memory frames.

`C3DMan` reads a c3d file once at construction time and caches the full point
and analog arrays. Subsequent `get_points` / `get_analogs` calls index into the
cache — no repeated file I/O.

Memory note: one `C3DMan` instance holds its entire trial in memory. When
processing many trials (e.g. a full condition), load one at a time rather than
building a list of preloaded instances.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, TypedDict
import c3d
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import interp1d
from scipy.constants import g


class GenAnalogParams(TypedDict):
    gain: float
    zero_level: float
    scaling_factor: float


class AccelerometerParams(TypedDict):
    gain: float
    zero_level: float
    sensitivity: float
    amplifier_gain: float


@dataclass
class PointData:
    """Point (marker) trajectories returned from `C3DMan.get_points`.

    Attributes:
        values: Marker coordinates (n_frames, n_markers, 3) [mm].
        t: Time vector matching `values` [sec] (n_frames,).
        missing_frames: Per-marker count of frames with missing data (residual
            < 0). Label → int. Empty dict when all markers are clean. Markers
            with any missing frames have NaN entries interpolated (or filled
            with NaN at trial ends); downstream code should inspect this field
            when missing data would corrupt the pipeline (e.g. shank cluster).
    """
    values: npt.NDArray
    t: npt.NDArray
    missing_frames: dict = field(default_factory=dict)


@dataclass
class AnalogData:
    """Analog signals returned from `C3DMan.get_analogs`.

    Attributes:
        values: Analog signals (n_samples, n_channels).
        t: Time vector matching `values` [sec] (n_samples,).
    """
    values: npt.NDArray
    t: npt.NDArray


class C3DMan:
    """C3D file reader with cached in-memory data.

    Attributes:
        c3d_path (str): Path to the c3d file.
        point_map (pd.DataFrame): Index of available marker labels.
        analog_map (pd.DataFrame): Index of available (description, label) analog channels.
        fs_point (float): Marker sample rate [Hz].
        fs_analog (float): Analog sample rate [Hz].
        fs (float): Effective rate for pipeline-consumed data (equals `fs_analog`;
            point data is upsampled to match by default).
    """

    def __init__(self, c3d_path: str):
        """Read a c3d file and cache its points and analogs in memory.

        Args:
            c3d_path: Path to the c3d file.
        """
        self.c3d_path = c3d_path

        with open(self.c3d_path, 'rb') as f:
            reader = c3d.Reader(f)

            # Metadata: label → index maps and sample rates
            desc_analog = [d.strip() for d in reader.get("ANALOG").get("DESCRIPTIONS").string_array]
            lab_analog = [lab.strip() for lab in reader.analog_labels]
            self.analog_map = pd.DataFrame({'Description': desc_analog, 'Label': lab_analog})
            self.analog_map = (
                self.analog_map.reset_index()
                .set_index(['Description', 'Label'])
                .sort_index()  # avoid lexsort-depth PerformanceWarning on .loc/.at lookups
            )

            lab_point = [lab.strip() for lab in reader.point_labels]
            self.point_map = pd.DataFrame({'Label': lab_point})
            self.point_map = self.point_map.reset_index().set_index(['Label'])

            self.fs_analog = reader.analog_rate
            self.fs_point = reader.point_rate

            # Read all frames once. Points are stored at native point rate;
            # upsampling to analog rate is deferred to get_points.
            n_frames = reader.frame_count
            n_points = len(lab_point)
            n_analogs = len(lab_analog)
            analog_per_frame = reader.analog_per_frame

            self._points_raw = np.full((n_frames, n_points, 3), np.nan)
            # Per-marker, per-frame valid mask (True where data is present).
            self._points_valid = np.zeros((n_frames, n_points), dtype=bool)
            self._analogs = np.zeros((n_frames * analog_per_frame, n_analogs))

            for frame, points, analogs in reader.read_frames():
                row_idx = frame - reader.first_frame
                valid = points[:, 3] >= 0        # residual < 0 ⇒ missing
                self._points_raw[row_idx, valid, :] = points[valid, :3]
                self._points_valid[row_idx, :] = valid

                a_start = row_idx * analog_per_frame
                a_end = a_start + analog_per_frame
                self._analogs[a_start:a_end, :] = analogs.T

            self._t_point = np.arange(n_frames) / self.fs_point
            self._t_analog = np.arange(n_frames * analog_per_frame) / self.fs_analog

    @property
    def fs(self) -> float:
        """Effective sample rate for pipeline-consumed data (analog rate).

        Point data is upsampled to this rate by default; force / analog data
        is natively at this rate.
        """
        return self.fs_analog

    def print_data_labels(self) -> None:
        """Print available analog and point labels."""
        default_rows = pd.get_option('display.max_rows')
        pd.set_option('display.max_rows', None)
        print("---------- ANALOG CHANNELS ----------")
        for idx in self.analog_map.index:
            print(idx)
        print()
        print("---------- POINT CHANNELS -----------")
        for idx in self.point_map.index:
            print(f"\'{idx}\'")
        print()
        pd.set_option('display.max_rows', default_rows)

    def get_points(self, labels: List[str], upsample: bool = True) -> PointData:
        """Extract point coordinates for the given labels in order.

        Args:
            labels: Desired point labels, in desired output order.
            upsample: Whether to upsample to analog rate via cubic interpolation.
                Default True — matches the rate of force/analog data, the shape
                the pipeline consumes.

        Returns:
            PointData with the marker values, time vector, and a
            `missing_frames` dict reporting per-label counts of frames with
            missing data (empty when all clean). Markers with any missing
            frames are cubic-interpolated across gaps; trial-boundary gaps fill
            with NaN. Inspect `missing_frames` when the pipeline requires
            complete data.

        Raises:
            KeyError: If any requested label is not present in the c3d.
        """
        unknown = [lab for lab in labels if lab not in self.point_map.index]
        if unknown:
            raise KeyError(
                f"Unknown point label(s) in c3d: {unknown}. "
                f"Use print_data_labels() to list available markers."
            )

        indices = [int(self.point_map.loc[lab, 'index']) for lab in labels]
        raw = self._points_raw[:, indices, :]         # (n_frames, n_sel, 3)
        valid = self._points_valid[:, indices]        # (n_frames, n_sel)

        missing_frames: dict[str, int] = {}
        for i, lab in enumerate(labels):
            n_missing = int((~valid[:, i]).sum())
            if n_missing:
                missing_frames[lab] = n_missing

        if not upsample:
            return PointData(values=raw.copy(), t=self._t_point.copy(),
                             missing_frames=missing_frames)

        # Upsample each marker to analog rate via cubic interpolation.
        t_target = self._t_analog
        interp = np.empty((len(t_target), len(indices), 3))
        for i in range(len(indices)):
            trajectory = raw[:, i, :]
            has_missing = missing_frames.get(labels[i], 0) > 0
            if has_missing:
                m_valid = valid[:, i]
                f = interp1d(
                    self._t_point[m_valid], trajectory[m_valid],
                    axis=0, kind='cubic',
                    bounds_error=False, fill_value=np.nan,   # NaN at gap ends
                )
            else:
                f = interp1d(
                    self._t_point, trajectory,
                    axis=0, kind='cubic',
                    bounds_error=False, fill_value='extrapolate',
                )
            interp[:, i, :] = f(t_target)

        return PointData(values=interp, t=t_target.copy(), missing_frames=missing_frames)

    def get_analogs(self, description_labels: List[Tuple[str, str]]) -> AnalogData:
        """Extract analog signals for the given (description, label) pairs.

        Args:
            description_labels: Ordered list of (description, label) pairs.

        Returns:
            AnalogData with signals (n_samples, n_channels) and time vector.

        Raises:
            KeyError: If any requested (description, label) pair is not in the c3d.
        """
        unknown = [dl for dl in description_labels if dl not in self.analog_map.index]
        if unknown:
            raise KeyError(
                f"Unknown analog (description, label) pair(s) in c3d: {unknown}. "
                f"Use print_data_labels() to list available channels."
            )

        indices = [int(np.asarray(self.analog_map.loc[dl, 'index']).flat[0])
                   for dl in description_labels]
        return AnalogData(
            values=self._analogs[:, indices].copy(),
            t=self._t_analog.copy(),
        )


def convert_accel(
        accel_data: npt.NDArray,
        gen_params: GenAnalogParams = {
            "gain": 5.00, # V
            "zero_level": 0.0,
            "scaling_factor": 1.0 # V
        },
        acc_params: AccelerometerParams = {
            "gain": 5.00, # V
            "zero_level": 2.5,
            "sensitivity": 1000.0, # mV/g
            "amplifier_gain": 1.0 # V
        }
) -> npt.NDArray:
    """Convert accelerometer data from m/s² (Vicon accelerometer export) back to
    raw electric potential (V) matching the generic analog export format.

    The MLP force correction model (force_correction.BasicMLP) was trained on
    raw voltage-domain accelerometer data exported as Generic Analog channels.
    When a session exports accelerometers in m/s² mode instead, this function
    reverses the conversion so the MLP receives the same signal domain it was
    trained on.

    The default gen_params and acc_params match our lab's accelerometer hardware
    and the Vicon export settings used to produce the MLP training data.  These
    should NOT be changed unless a new model is trained on data with different
    parameters — the training code lives in the companion repo
    (hexapod-inertial-force-removal), not here.

    Args:
        accel_data (npt.NDArray): Accelerometer data represented as acceleration (m/s)
        gen_params (GenAnalogParams, optional): Parameters defining how accelerometers are read as generic analog devices. Defaults to { "gain": 5.00, # V "zero_level": 0.0, "scaling_factor": 1.0 # V }.
        acc_params (AccelerometerParams, optional): Parameters defining how accelerometers are read as accelerometer devices. Defaults to { "gain": 5.00, # V "zero_level": 2.5, "sensitivity": 1000.0, # mV/g "amplifier_gain": 1.0 # V }.

    Returns:
        npt.NDArray: Acceleromater data converted to raw electrical potential (V)
    """

    accel_g = accel_data / g # (g), acceleration as scale of gravity
    sensor_voltage = accel_g * (acc_params["sensitivity"] / 1000.0) # (V), voltage mapped to acceleration
    amplified_voltage = sensor_voltage * acc_params["amplifier_gain"] # (V), voltage seen by ADC (without zero level offset)
    total_voltage = amplified_voltage + acc_params["zero_level"] # (V), add zero level back into signal
    raw_accel_potential = gen_params["scaling_factor"] * (total_voltage - gen_params["zero_level"]) # (V), convert to generic analog representation

    return raw_accel_potential
