"""MLP-based inertial force correction for the Kistler plate mounted on the
Hexapod. Migrated from the `hexapod-inertial-force-removal` project so this
repo is self-contained.

The model predicts the inertial forces induced on the Kistler by Hexapod
motion, given the 12-channel accelerometer signal. Subtracting the prediction
from the measured 8-channel raw force yields the "true" GRF.

Only the inference path is included — training code and alternate dataset
classes live in the upstream repo.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class FlattenedWindows(Dataset):
    """Slides non-overlapping windows (with optional overlap) across a list of
    time series and returns each window as a flattened 1D tensor.

    Used to feed `BasicMLP`, whose input is a flattened
    (window_size * n_channels) vector. `restructure_windowed_output` reassembles
    per-window predictions back into continuous per-series tensors by averaging
    overlapping regions.
    """

    def __init__(
            self,
            input_arrays: List[npt.ArrayLike],
            window_size: int,
            step_size: int,
    ):
        self.window_size = window_size
        self.step_size = step_size

        self.input_tensors = [
            torch.tensor(arr, dtype=torch.float32) for arr in input_arrays
        ]

        # Precompute (series_idx, start_idx) for every window in every series.
        self.index_map: List[Tuple[int, int]] = []
        for series_idx, series in enumerate(self.input_tensors):
            series_length = series.shape[0]
            max_start = series_length - self.window_size
            if max_start < 0:
                continue
            start_indices = np.arange(
                0,
                series_length - self.window_size + self.step_size,
                self.step_size,
            )
            valid = start_indices[start_indices <= max_start]
            self.index_map.extend((series_idx, int(s)) for s in valid)

        self.num_windows = len(self.index_map)

    def __len__(self) -> int:
        return self.num_windows

    def __getitem__(self, idx: int) -> torch.Tensor:
        series_idx, start_idx = self.index_map[idx]
        end_idx = start_idx + self.window_size
        window = self.input_tensors[series_idx][start_idx:end_idx, :]
        # Flatten in row-major order: frame-by-frame, channels within frame.
        return window.contiguous().flatten()

    def restructure_windowed_output(
            self,
            indexed_windows: List[Tuple[int, torch.Tensor]],
    ) -> List[torch.Tensor]:
        """Reassemble per-window predictions into per-series tensors.

        Overlapping frames (where step_size < window_size) are averaged.

        Args:
            indexed_windows: list of (dataset_index, flat_output_tensor) pairs
                produced by iterating the DataLoader and stacking outputs.

        Returns:
            One (series_length, n_output_channels) tensor per input series.
        """
        device = indexed_windows[0][1].device
        num_out_chans = indexed_windows[0][1].numel() // self.window_size

        output_tensors = [
            torch.zeros(t.shape[0], num_out_chans, device=device)
            for t in self.input_tensors
        ]
        summed = [
            torch.zeros(t.shape[0], num_out_chans, device=device)
            for t in self.input_tensors
        ]
        counts = [
            torch.zeros(t.shape[0], device=device) for t in self.input_tensors
        ]

        for index, window in indexed_windows:
            series_idx, start_idx = self.index_map[index]
            end_idx = start_idx + self.window_size
            window_2d = window.reshape(-1, num_out_chans)
            summed[series_idx][start_idx:end_idx, :] += window_2d
            counts[series_idx][start_idx:end_idx] += 1

        for series_idx, series_counts in enumerate(counts):
            mask = series_counts != 0
            output_tensors[series_idx][mask, :] = (
                summed[series_idx][mask, :]
                / series_counts[mask].unsqueeze(1)
            )

        return output_tensors


class BasicMLP(nn.Module):
    """Three-layer MLP that maps a flattened accelerometer window to a
    flattened induced-force window. Input/output standardization stats are
    stored as buffers so they travel with the weights.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_dim_1: int = 2048,
            hidden_dim_2: int = 4096,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Standardization stats — populated by load_weights.
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.zeros(input_dim))
        self.register_buffer('output_mean', torch.zeros(output_dim))
        self.register_buffer('output_std', torch.zeros(output_dim))

        self.fc1 = nn.Linear(input_dim, hidden_dim_1)
        self.fc2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.fc3 = nn.Linear(hidden_dim_2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standardize -> 2x (Linear + ReLU) -> Linear -> un-standardize.
        x_std = (x - self.input_mean) / self.input_std
        x_std = self.relu(self.fc1(x_std))
        x_std = self.relu(self.fc2(x_std))
        x_std = self.fc3(x_std)
        return x_std * self.output_std + self.output_mean

    def predict_segments(
            self,
            inputs: List[npt.NDArray],
            step_size: int,
            device: torch.device = torch.device("cpu"),
    ) -> List[npt.NDArray]:
        """Predict induced forces for one or more continuous accelerometer
        segments.

        Args:
            inputs: list of (n_samples, n_accel_channels) arrays. Channel
                count must match the one used at training time
                (`input_dim / window_size`).
            step_size: stride between consecutive windows. Smaller values
                smooth the reconstruction via overlap averaging.
            device: torch device for inference.

        Returns:
            list of (n_samples, n_force_channels) numpy arrays aligned with
            each input.
        """
        num_channels = inputs[0].shape[1]
        window_size = self.input_dim // num_channels

        dataset = FlattenedWindows(inputs, window_size, step_size)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        self.eval()
        indexed_outputs: List[Tuple[int, torch.Tensor]] = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                batch = batch.to(device)
                output = self(batch)
                indexed_outputs.append((batch_idx, output.squeeze(0)))

        output_tensors = dataset.restructure_windowed_output(indexed_outputs)
        return [t.cpu().numpy() for t in output_tensors]

    def load_weights(self, path: str) -> None:
        """Load a state_dict from disk and switch to eval mode."""
        self.load_state_dict(torch.load(path, map_location='cpu'))
        self.eval()
