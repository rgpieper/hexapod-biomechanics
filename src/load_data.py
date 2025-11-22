
from typing import List, Tuple
import c3d
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import interp1d

class C3DMan:
    """C3D Manager:
    
    Parses a c3d file to extract relevant point (marker) and analog data.
    """

    def __init__(self, c3d_path: str):
        """Initializes Manager for a c3d file, compiling relevant metadata.

        Args:
            c3d_path (str): Path to c3d file.
        """

        self.c3d_path = c3d_path

        with open(self.c3d_path, 'rb') as f:

            reader = c3d.Reader(f)

            print(f"C3D file opened from path: {c3d_path}")

            desc_analog = [desc.strip() for desc in reader.get("ANALOG").get("DESCRIPTIONS").string_array]
            lab_analog = [lab.strip() for lab in reader.analog_labels]
            self.analog_map = pd.DataFrame({'Description': desc_analog, 'Label': lab_analog})
            self.analog_map = self.analog_map.reset_index()
            self.analog_map = self.analog_map.set_index(['Description', 'Label'])
            self.fs_analog = reader.analog_rate

            lab_point = [lab.strip() for lab in reader.point_labels]
            self.point_map = pd.DataFrame({'Label': lab_point})
            self.point_map = self.point_map.reset_index()
            self.point_map = self.point_map.set_index(['Label'])
            self.fs_point = reader.point_rate

    def print_data_labels(self) -> None:
        """Displays available analog and point labels.
        """

        default_rows = pd.get_option('display.max_rows')
        pd.set_option('display.max_rows', None)
        print("---------- ANALOG CHANNELS ----------")
        for idx in self.analog_map.index:
            print(idx)
        print()
        print("---------- POINT CHANNELS -----------")
        for idx in self.point_map.index:
            print(idx)
        print()
        pd.set_option('display.max_rows', default_rows)
        
    def get_points(self, labels: List[str], upsample: bool = True) -> Tuple[npt.NDArray, npt.NDArray]:
        """Extract point coordinates with designated labels.

        Args:
            labels (List[str]): Desired point labels in desired order.
            upsample (bool, optional): Upsample point data to match analog samples. Defaults to True.

        Returns:
            Tuple[npt.NDArray, npt.NDArray]:
                points (npt.NDArray): Point coordinate trajectories with shape (n_timepoints, n_points, 3).
                t (npt.NDArray): Time vector corresponding to point coordinates.
        """

        self.point_map.sort_index(inplace=True)
        point_indices = [self.point_map.loc[label, 'index'].item() for label in labels]

        with open(self.c3d_path, 'rb') as f:
            reader = c3d.Reader(f)

            t_points = np.arange(reader.frame_count) / reader.point_rate
            raw_points = np.full((reader.frame_count, len(point_indices), 3), np.nan)

            for frame, points, _ in reader.read_frames():
                row_idx = frame - reader.first_frame

                selected = points[point_indices, :]
                coords = selected[:, :3]
                valid_mask = selected[:, 3] >= 0 # if residual is negative, marker is missing

                raw_points[row_idx, valid_mask, :] = coords[valid_mask]

            if np.isnan(raw_points).any():
                bad_markers = []
                for i, label in enumerate(labels):
                    if np.isnan(raw_points[:, i, :]).any():
                        bad_markers.append(label)
                print("WARNIGN: The following markers have missing data.")
                for marker in bad_markers:
                    print(marker)

            if upsample:
                t_analogs = np.arange(reader.frame_count*reader.analog_per_frame) / reader.analog_rate

                interp_points = np.empty((len(t_analogs), len(point_indices), 3))

                f = interp1d(t_points, raw_points, axis=0, kind='cubic', fill_value='extrapolate')
                interp_points = f(t_analogs)

                return interp_points, t_analogs
            
            else:
                return raw_points, t_points
            
    def get_analogs(self, description_labels: List[Tuple[str, str]]) -> Tuple[npt.NDArray, npt.NDArray]:
        """Extract analog data with designated description-label pairs.

        Args:
            description_labels (List[Tuple[str, str]]): Pairs of descriptions and labels of desired analogs.

        Returns:
            Tuple[npt.NDArray, npt.NDArray]:
                sel_analogs (npt.NDArray): Analog signals with shape (n_timepoints, n_analog_channels).
                t_analogs (npt.NDArray): Time vector corresponding to analog signals.
        """

        self.analog_map.sort_index(inplace=True)
        analog_indices = [self.analog_map.loc[desc_lab, 'index'].item() for desc_lab in description_labels]

        with open(self.c3d_path, 'rb') as f:
            reader = c3d.Reader(f)

            t_analogs = np.arange(reader.analog_sample_count) / reader.analog_rate
            sel_analogs = np.zeros((reader.analog_sample_count, len(analog_indices)))

            current_row = 0
            for _, _, analogs in reader.read_frames():
                end_row = current_row + reader.analog_per_frame

                selected = analogs[analog_indices, :]
                chunk = selected.T

                sel_analogs[current_row:end_row, :] = chunk

                current_row = end_row

            return sel_analogs, t_analogs


