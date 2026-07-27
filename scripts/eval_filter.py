"""Frequency response analysis for the lab's Savitzky-Golay filter.

Plots magnitude responses for the sav-gol filter used as a smoother
(deriv=0), a first-order differentiator (deriv=1), and a second-order
differentiator (deriv=2). Lets you see how the filter treats signal
content at different frequencies before changing
`FILT_WINDOW_DURATION_S` / `FILT_POLY` in `config.py`.

# Approach: analytical via `scipy.signal.freqz` on `savgol_coeffs`

Sav-gol is a finite impulse response (FIR) filter — its behavior is
fully described by its kernel coefficients. `savgol_coeffs(N, p,
deriv=k, delta=dt)` gives those coefficients, and `freqz` evaluates the
exact complex frequency response `H(f) = sum b_n * exp(-j*2*pi*f*n/fs)`
at any number of frequency points. No input signal needed, no edge
effects, no truncation.

A sine-sweep approach (apply the filter to `sin(2*pi*f*t)` at many test
frequencies, measure peak output amplitude, compare to the analytical
derivative `2*pi*f * cos(2*pi*f*t)` for deriv=1, etc.) gives identical
magnitude results but is slower and sensitive to windowing/edge
artifacts. Since sav-gol is FIR, the analytical path is strictly
better.

# What |H(f)| means

`|H(f)|` is the gain factor the filter applies to a sinusoid at
frequency f. Feed in `A*sin(2*pi*f*t)` and you get out
`|H(f)|*A*sin(2*pi*f*t + phase)`. So `|H|=1` is unity gain (passband),
`|H|=0.5` halves the amplitude, `|H|=0.01` keeps 1% of it (stopband).

In dB terms `|H|` becomes `20*log10(|H|)`: 0 dB = unity, -3 dB ≈ 0.707
(the "half-power" point used as the conventional cutoff), -20 dB =
1/10. On a log-y axis the curve shape is identical between `|H|` and
dB; only the tick labels differ.

For a differentiator (deriv>=1) the passband `|H|` is *larger* than 1
— a deriv=1 filter at 1 Hz shows `|H| ≈ 2*pi ≈ 6.28`. Not a defect:
the derivative of `sin(2*pi*1*t)` is `2*pi*cos(2*pi*1*t)`, amplitude
`2*pi`. That's what the filter is asked to do.

# Differentiator interpretation: absolute vs normalized

For deriv=k, the ideal operator in the frequency domain is the k-th
derivative, transfer function `(j*2*pi*f)^k`. Magnitude grows like
`(2*pi*f)^k` — linearly with frequency for k=1, quadratically for k=2.

The realized sav-gol differentiator has `|H(f)|` that grows like
`(2*pi*f)^k` at low frequency (the differentiator portion), peaks, then
rolls off (the lowpass smoothing portion). Two equivalent ways to read
this:

    Absolute response   |H(f)|
        Actual gain on each frequency component. Real noise
        amplification numbers — e.g. "a 50 Hz noise component is
        amplified by X". On log-log the passband slope = k (= 20*k
        dB/decade).

    Normalized response |H(f) / (2*pi*f)^k|
        Realized filter divided by the ideal differentiator's
        magnitude. ~1 in the passband, rolls off in the stopband.
        Reads like a standard lowpass — useful for "where does the
        filter stop acting like a clean differentiator?".

For deriv=0 the ideal operator is the identity, so the normalized
response is the same curve as the absolute response.

# Cutoff frequency

Reported in the terminal summary as the -3 dB point on the
**normalized** response: the lowest frequency at which
`|H| / (2*pi*f)^k = 1/sqrt(2)`. For deriv=0 that's the standard
smoother cutoff. For deriv=1 and 2, that's the frequency above which
the filter's gain falls more than 3 dB short of the ideal
differentiator — i.e. where it stops behaving like a clean derivative.

# Parameters in play

Window duration sets how many samples are smoothed together: longer
duration = stronger lowpass = lower cutoff. Polynomial order controls
how closely the local polynomial fit follows fast signal content:
higher order = closer to passing input through (higher cutoff, less
smoothing). At fixed window duration, raising the polynomial order
pushes the cutoff up. At fixed polyorder, lengthening the window
pushes the cutoff down.

Defaults come from `config.FILT_WINDOW_DURATION_S` and `config.FILT_POLY`
— the lab-wide constants that drive every sav-gol call in the
pipeline. Override at the CLI to explore alternatives without
modifying the config.

# Run

    uv run python scripts/eval_filter.py
    uv run python scripts/eval_filter.py --filt-window-duration 0.03
    uv run python scripts/eval_filter.py --filt-poly 6 --save eval.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import freqz, savgol_coeffs

try:
    import hexapod_biomechanics  # noqa: F401 — preflight: editable install check
except ModuleNotFoundError:
    raise SystemExit(
        "hexapod_biomechanics not importable — refresh the editable install:\n"
        "  python scripts/install.py"
    )

from hexapod_biomechanics.config import FILT_POLY, FILT_WINDOW_DURATION_S

# Decade reference frequencies marked as vertical lines on every panel.
REF_FREQS = [1.0, 10.0, 100.0]


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------

def _odd_window(fs: float, duration_s: float) -> int:
    """Return sav-gol window length in samples, forced odd (scipy requires it)."""
    n = int(fs * duration_s)
    return n if n % 2 != 0 else n + 1


def _filter_response(
    fs: float, window_len: int, polyorder: int, deriv: int, n_freq: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the analytical frequency response of one sav-gol kernel.

    Args:
        fs: Sample rate [Hz].
        window_len: Sav-gol window in samples (odd).
        polyorder: Sav-gol polynomial order (must be >= deriv).
        deriv: Derivative order (0, 1, or 2).
        n_freq: Number of frequency bins for freqz (linear from 0 to fs/2).

    Returns:
        Tuple `(f, H)` — frequency vector [Hz] (length n_freq) and complex
        response (length n_freq). `f[0] == 0` (DC); divisions by f at DC
        must be masked by the caller for deriv>=1.
    """
    dt = 1.0 / fs
    coeffs = savgol_coeffs(
        window_len, polyorder, deriv=deriv, delta=dt, use="conv",
    )
    f, H = freqz(coeffs, [1], worN=n_freq, fs=fs)
    return f, H


def _ideal_diff_magnitude(f: np.ndarray, deriv: int) -> np.ndarray:
    """|H_ideal(f)| for the ideal k-th differentiator = (2*pi*f)^k."""
    return (2.0 * np.pi * f) ** deriv


def _cutoff_freq(f: np.ndarray, mag_norm: np.ndarray) -> float | None:
    """Return the -3 dB cutoff of a normalized magnitude response, or None.

    -3 dB corresponds to `|H/H_ideal| = 1/sqrt(2) ≈ 0.7071` — the
    half-power point. Walks the response from low to high frequency and
    returns the first crossing of this threshold from above, with linear
    interpolation between adjacent freq points for sub-bin precision.
    Returns None when no crossing is found within the evaluated range
    (shouldn't happen at sensible filter parameters).
    """
    threshold = 1.0 / np.sqrt(2.0)
    below = np.where(mag_norm < threshold)[0]
    if below.size == 0 or below[0] == 0:
        return None
    i = below[0]
    f0, f1 = f[i - 1], f[i]
    m0, m1 = mag_norm[i - 1], mag_norm[i]
    fraction = (m0 - threshold) / (m0 - m1)
    return float(f0 + fraction * (f1 - f0))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_summary(
    fs: float, window_len: int, polyorder: int, duration_s: float,
) -> None:
    """Print filter facts plus the -3 dB cutoff per derivative order."""
    print(
        f"\n  Sav-gol filter\n"
        f"    fs              = {fs:.1f} Hz\n"
        f"    window duration = {duration_s*1000:.2f} ms"
        f"  ({window_len} samples)\n"
        f"    polyorder       = {polyorder}\n"
    )
    print("  -3 dB cutoff frequency (on normalized response):\n")
    for d in (0, 1, 2):
        f, H = _filter_response(fs, window_len, polyorder, d)
        mask = f > 0
        ideal = _ideal_diff_magnitude(f[mask], d)
        # For deriv=0 ideal == 1, so this just gives the absolute response.
        mag_norm = np.abs(H[mask]) / ideal
        cutoff = _cutoff_freq(f[mask], mag_norm)
        cutoff_str = f"{cutoff:7.2f} Hz" if cutoff is not None else "  (none found)"
        print(f"    deriv={d}:  {cutoff_str}")
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _add_ref_lines(ax) -> None:
    """Vertical dashed lines at the decade reference frequencies."""
    for fr in REF_FREQS:
        ax.axvline(fr, color="k", lw=0.5, ls="--", alpha=0.3)


def build_figure(fs: float, window_len: int, polyorder: int) -> plt.Figure:
    """Build the 2x3 figure: rows = (abs |H|, normalized |H|), cols = deriv.

    Normalized panels (bottom row) share a common y-axis range so the three
    derivatives can be compared directly. Range is the union of all three
    normalized responses with a small log-margin.
    """
    # Pre-compute responses for all three derivs so we can derive shared
    # y-limits before plotting the normalized row.
    responses = []
    for deriv in (0, 1, 2):
        f, H = _filter_response(fs, window_len, polyorder, deriv)
        mask = f > 0
        f_pos = f[mask]
        mag_abs = np.abs(H[mask])
        mag_norm = mag_abs / _ideal_diff_magnitude(f_pos, deriv)
        cutoff = _cutoff_freq(f_pos, mag_norm)
        responses.append({
            "deriv": deriv, "f": f_pos,
            "mag_abs": mag_abs, "mag_norm": mag_norm,
            "cutoff": cutoff,
        })

    # Shared y-range for the normalized row: union of all three curves with a
    # small log-margin so curves don't touch the axes spines.
    norm_min = min(r["mag_norm"].min() for r in responses)
    norm_max = max(r["mag_norm"].max() for r in responses)
    norm_ylim = (norm_min * 0.5, norm_max * 2.0)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    fig.suptitle(
        f"Sav-gol filter response  —  fs={fs:.0f} Hz, "
        f"N={window_len} samples ({(window_len/fs)*1000:.1f} ms), "
        f"polyorder={polyorder}"
    )

    for col, r in enumerate(responses):
        deriv = r["deriv"]
        f_pos = r["f"]

        # --- Row 0: absolute magnitude ---
        ax = axes[0, col]
        ax.set_title(f"deriv = {deriv}")
        ax.loglog(f_pos, r["mag_abs"], color="tab:purple")
        ax.set_ylabel("|H(f)|")
        _add_ref_lines(ax)
        ax.grid(True, which="both", alpha=0.3)

        # --- Row 1: normalized magnitude ---
        # For deriv=0 the ideal operator is identity, so this is literally the
        # same curve as row 0. We plot it anyway for visual consistency across
        # columns rather than blanking the cell.
        ax = axes[1, col]
        ax.loglog(f_pos, r["mag_norm"], color="tab:blue", lw=1.5)
        ax.axhline(1.0, color="black", lw=1.2, ls="--",
                   label="unity gain (|H|=1)")
        ax.axhline(1.0 / np.sqrt(2.0), color="tab:red", lw=1.5, ls="--",
                   label="-3 dB (|H|≈0.707)")

        # Apply shared y-range so the three normalized panels are directly
        # comparable (per-column cutoff/shape differences read as visual
        # differences, not as auto-scaling artifacts).
        ax.set_ylim(norm_ylim)

        # -3 dB cutoff callout, sitting just below the -3 dB threshold line.
        # `get_yaxis_transform` mixes axes-x (0..1) with data-y, so the label
        # stays anchored to the threshold value regardless of x-axis range.
        cutoff = r["cutoff"]
        cutoff_text = (
            f"$f_c$ = {cutoff:.2f} Hz" if cutoff is not None
            else "$f_c$ = (none found)"
        )
        ax.text(
            0.04, 0.5 / np.sqrt(2.0), cutoff_text,
            transform=ax.get_yaxis_transform(),
            fontsize=10, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="tab:red", alpha=0.9),
        )

        ax.set_ylabel("|H| / |ideal diff|")
        ax.set_xlabel("frequency [Hz]")
        _add_ref_lines(ax)
        ax.grid(True, which="both", alpha=0.3)
        if col == 0:
            ax.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--filt-window-duration", type=float,
                    default=FILT_WINDOW_DURATION_S,
                    help=f"Sav-gol window duration in seconds "
                         f"(default {FILT_WINDOW_DURATION_S}, from config).")
    ap.add_argument("--filt-poly", type=int, default=FILT_POLY,
                    help=f"Sav-gol polynomial order "
                         f"(default {FILT_POLY}, from config). Must be >= 2 "
                         f"so deriv=2 is supported.")
    ap.add_argument("--fs", type=float, default=2000.0,
                    help="Analog sample rate in Hz (default 2000 — the rate "
                         "the lab's Vicon data is interpolated to).")
    ap.add_argument("--save", type=Path, default=None,
                    help="If given, save PNG to this path instead of showing.")
    args = ap.parse_args()

    if args.filt_poly < 2:
        raise SystemExit(
            f"--filt-poly must be >= 2 (got {args.filt_poly}) so both deriv=1 "
            "and deriv=2 panels can be computed."
        )

    window_len = _odd_window(args.fs, args.filt_window_duration)
    if window_len <= args.filt_poly:
        raise SystemExit(
            f"Window length {window_len} samples is <= polyorder {args.filt_poly}. "
            "Either lengthen --filt-window-duration, raise --fs, or lower "
            "--filt-poly."
        )

    _print_summary(
        args.fs, window_len, args.filt_poly, args.filt_window_duration,
    )
    fig = build_figure(args.fs, window_len, args.filt_poly)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"Wrote {args.save}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
