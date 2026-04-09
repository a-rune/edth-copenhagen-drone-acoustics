"""
TDOA-based drone localisation via acoustic cross-correlation and grid search.

Pipeline:
  1. Load multi-channel audio (MicArray)
  2. Bandpass-filter to drone acoustic range
  3. Estimate TDOAs with GCC-PHAT
  4. Locate the emitter via vectorised grid search (or coarse-to-fine)
  5. Optionally refine with nonlinear least squares

See the guide doc for full theory and usage examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.optimize import minimize
from scipy.signal import butter, sosfilt


# ---------------------------------------------------------------------------
# Microphone array container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MicArray:
    """Multi-channel audio recorded from a synchronised microphone array."""

    data: np.ndarray        # shape: (n_samples, n_mics) — float32
    sample_rate: int
    mic_positions: np.ndarray  # shape: (n_mics, 3) — (x, y, z) in metres

    @classmethod
    def from_separate_files(
        cls,
        file_paths: list[Path],
        mic_positions: list[tuple[float, float, float]],
    ) -> "MicArray":
        """Load one WAV per mic and stack into a single array."""
        channels: list[np.ndarray] = []
        sr_ref: int | None = None
        for path in file_paths:
            data, sr = sf.read(path, dtype="float32", always_2d=False)
            if data.ndim > 1:
                data = data[:, 0]
            if sr_ref is None:
                sr_ref = sr
            elif sr != sr_ref:
                raise ValueError(f"Sample rate mismatch: {sr} vs {sr_ref}")
            channels.append(data)

        min_len = min(len(c) for c in channels)
        stacked = np.stack([c[:min_len] for c in channels], axis=1)
        return cls(stacked, sr_ref, np.array(mic_positions, dtype=np.float32))

    @property
    def n_mics(self) -> int:
        return self.data.shape[1]

    @property
    def n_samples(self) -> int:
        return self.data.shape[0]

    @property
    def duration_s(self) -> float:
        return self.n_samples / self.sample_rate


# ---------------------------------------------------------------------------
# Environmental helpers
# ---------------------------------------------------------------------------

def speed_of_sound(temperature_celsius: float) -> float:
    """
    Speed of sound in dry air (m/s).
    Linear approximation accurate to ~0.3% for -20 to +40 °C.
    """
    return 331.3 + 0.606 * temperature_celsius


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def bandpass(
    signal: np.ndarray,
    sr: int,
    low_hz: float = 80.0,
    high_hz: float = 4000.0,
) -> np.ndarray:
    """
    Bandpass to the drone's dominant acoustic range.
    Removes sub-bass rumble (wind, HVAC) and high-frequency hiss.
    Typical drone rotor fundamentals: 80–400 Hz; harmonics to ~4 kHz.
    """
    sos = butter(4, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
    return sosfilt(sos, signal).astype(np.float32)


def preprocess_array(
    array: MicArray,
    low_hz: float = 80.0,
    high_hz: float = 4000.0,
) -> MicArray:
    filtered = np.stack(
        [bandpass(array.data[:, i], array.sample_rate, low_hz, high_hz)
         for i in range(array.n_mics)],
        axis=1,
    )
    return MicArray(filtered, array.sample_rate, array.mic_positions)


# ---------------------------------------------------------------------------
# TDOA estimation
# ---------------------------------------------------------------------------

def gcc_phat(
    sig: np.ndarray,
    ref: np.ndarray,
    sr: int,
    max_delay_s: float | None = None,
) -> float:
    """
    GCC-PHAT time delay estimation.

    Parameters
    ----------
    sig         : signal received at microphone i
    ref         : signal received at reference microphone (mic 0)
    sr          : sample rate (Hz)
    max_delay_s : maximum expected TDOA in seconds — restricts the search
                  to physically plausible delays (baseline / speed_of_sound)

    Returns
    -------
    Estimated time delay in seconds (positive = sig arrived after ref).
    """
    n = len(sig) + len(ref) - 1
    n_fft = 1 << (n - 1).bit_length()  # next power of 2

    SIG = np.fft.rfft(sig, n=n_fft)
    REF = np.fft.rfft(ref, n=n_fft)

    # Cross-power spectrum with PHAT weighting (phase-only, magnitude → 1)
    G = SIG * np.conj(REF)
    G /= np.abs(G) + 1e-10

    cc = np.fft.irfft(G, n=n_fft)
    # Rearrange so lag 0 is at centre
    cc = np.concatenate([cc[-(n_fft // 2):], cc[:n_fft // 2]])
    lags = np.arange(-n_fft // 2, n_fft // 2) / sr

    if max_delay_s is not None:
        mask = np.abs(lags) <= max_delay_s
        cc_masked = np.where(mask, cc, -np.inf)
        peak_idx = int(np.argmax(cc_masked))
    else:
        peak_idx = int(np.argmax(cc))

    return float(lags[peak_idx])


def subpixel_peak(cc: np.ndarray, peak_idx: int) -> float:
    """Parabolic interpolation around peak_idx for sub-sample accuracy."""
    if peak_idx == 0 or peak_idx == len(cc) - 1:
        return float(peak_idx)
    y0, y1, y2 = cc[peak_idx - 1], cc[peak_idx], cc[peak_idx + 1]
    denom = 2 * y1 - y0 - y2
    if abs(denom) < 1e-12:
        return float(peak_idx)
    return float(peak_idx) + (y2 - y0) / (2 * denom)


def estimate_all_tdoas(
    array: MicArray,
    speed_of_sound_mps: float = 343.4,
) -> np.ndarray:
    """
    Compute TDOA between mic 0 (reference) and every other mic using GCC-PHAT.

    Returns
    -------
    Array of shape (n_mics - 1,) containing time delays in seconds.
    """
    ref = array.data[:, 0].astype(np.float64)

    positions = array.mic_positions
    max_baseline = float(np.max(
        np.linalg.norm(positions[1:] - positions[0], axis=1)
    ))
    max_delay = max_baseline / speed_of_sound_mps

    return np.array([
        gcc_phat(
            array.data[:, i].astype(np.float64),
            ref,
            array.sample_rate,
            max_delay_s=max_delay,
        )
        for i in range(1, array.n_mics)
    ])


def sliding_tdoa(
    array: MicArray,
    frame_s: float = 1.0,
    hop_s: float = 0.5,
    speed_of_sound_mps: float = 343.4,
) -> list[np.ndarray]:
    """
    Compute TDOA for overlapping Hanning-windowed frames.

    Returns a list of TDOA vectors (one per frame), each of shape (n_mics-1,).
    Useful for tracking a moving drone over time.
    """
    sr = array.sample_rate
    frame_len = int(frame_s * sr)
    hop_len = int(hop_s * sr)
    window = np.hanning(frame_len)

    tdoa_track: list[np.ndarray] = []
    start = 0
    while start + frame_len <= array.n_samples:
        chunk = array.data[start:start + frame_len].copy()
        chunk *= window[:, np.newaxis]
        frame_array = MicArray(chunk.astype(np.float32), sr, array.mic_positions)
        tdoas = estimate_all_tdoas(frame_array, speed_of_sound_mps)
        tdoa_track.append(tdoas)
        start += hop_len

    return tdoa_track


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def tdoa_error(
    candidate: tuple[float, float, float],
    mic_positions: np.ndarray,
    observed_tdoas: np.ndarray,
    speed_of_sound_mps: float,
) -> float:
    """
    Sum-of-squared residuals between observed TDOAs and those predicted by a
    candidate source location.

    Parameters
    ----------
    candidate          : (x, y, z) candidate emitter position
    mic_positions      : (n_mics, 3) array of mic coordinates
    observed_tdoas     : (n_mics - 1,) TDOAs relative to mic 0
    speed_of_sound_mps : c (m/s)

    Returns
    -------
    Scalar error (lower = better fit).
    """
    cx, cy, cz = candidate
    dists = np.sqrt(np.sum((mic_positions - np.array([cx, cy, cz])) ** 2, axis=1))
    predicted_tdoas = (dists[1:] - dists[0]) / speed_of_sound_mps
    residuals = observed_tdoas - predicted_tdoas
    return float(np.sum(residuals ** 2))


def grid_search_tdoa_vectorised(
    mic_positions: np.ndarray,
    observed_tdoas: np.ndarray,
    speed_of_sound_mps: float,
    x_range: tuple[float, float] = (-500, 500),
    y_range: tuple[float, float] = (-500, 500),
    z_range: tuple[float, float] = (0, 150),
    grid_size: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fully vectorised exhaustive grid search for the TDOA-minimising source
    location.  Processes 125k points in <100 ms on a modern CPU.

    Parameters
    ----------
    mic_positions      : (n_mics, 3) array
    observed_tdoas     : (n_mics - 1,) array, relative to mic 0
    speed_of_sound_mps : c (m/s)
    x_range, y_range, z_range : search bounds (metres)
    grid_size          : divisions per axis (total candidates = grid_size³)

    Returns
    -------
    best_location : (3,) array — estimated (x, y, z) in metres
    error_volume  : (grid_size, grid_size, grid_size) float32 error array
    """
    xs = np.linspace(*x_range, grid_size)
    ys = np.linspace(*y_range, grid_size)
    zs = np.linspace(*z_range, grid_size)

    # (grid_size^3, 3)
    XYZ = np.array(np.meshgrid(xs, ys, zs, indexing="ij")).reshape(3, -1).T

    # (N, n_mics, 3) → (N, n_mics)
    diffs = XYZ[:, np.newaxis, :] - mic_positions[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))

    # Predicted TDOAs: (N, n_mics-1)
    predicted_tdoas = (dists[:, 1:] - dists[:, 0:1]) / speed_of_sound_mps

    # Sum-of-squared error: (N,)
    residuals = observed_tdoas[np.newaxis, :] - predicted_tdoas
    errors = np.sum(residuals ** 2, axis=1)

    error_volume = errors.reshape(grid_size, grid_size, grid_size).astype(np.float32)
    best_idx = int(np.argmin(errors))
    return XYZ[best_idx], error_volume


def coarse_to_fine_search(
    mic_positions: np.ndarray,
    observed_tdoas: np.ndarray,
    speed_of_sound_mps: float,
    x_range: tuple[float, float] = (-600, 600),
    y_range: tuple[float, float] = (-600, 600),
    z_range: tuple[float, float] = (0, 200),
    coarse_size: int = 20,
    fine_size: int = 30,
    fine_margin: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Hierarchical (coarse-to-fine) grid search.

    Runs a coarse pass over the full volume, then a finer pass centred on
    the best coarse estimate.  Reduces evaluations from O(N³) to ~2×n³
    at coarse resolution.

    Parameters
    ----------
    fine_margin : fraction of the full x-range used as the fine-pass half-width
    """
    best_coarse, _ = grid_search_tdoa_vectorised(
        mic_positions, observed_tdoas, speed_of_sound_mps,
        x_range=x_range, y_range=y_range, z_range=z_range,
        grid_size=coarse_size,
    )

    x_span = x_range[1] - x_range[0]
    margin = fine_margin * x_span / 2

    fine_x = (best_coarse[0] - margin, best_coarse[0] + margin)
    fine_y = (best_coarse[1] - margin, best_coarse[1] + margin)
    fine_z = (max(0.0, best_coarse[2] - margin / 2), best_coarse[2] + margin / 2)

    return grid_search_tdoa_vectorised(
        mic_positions, observed_tdoas, speed_of_sound_mps,
        x_range=fine_x, y_range=fine_y, z_range=fine_z,
        grid_size=fine_size,
    )


def refine_with_nlls(
    initial_guess: np.ndarray,
    mic_positions: np.ndarray,
    observed_tdoas: np.ndarray,
    speed_of_sound_mps: float,
) -> np.ndarray:
    """
    Refine a grid-search estimate to sub-metre accuracy using L-BFGS-B.

    Parameters
    ----------
    initial_guess : (x, y, z) starting point (e.g. from grid_search_tdoa_vectorised)

    Returns
    -------
    Refined (x, y, z) estimate.
    """
    res = minimize(
        fun=tdoa_error,
        x0=initial_guess,
        args=(mic_positions, observed_tdoas, speed_of_sound_mps),
        method="L-BFGS-B",
        bounds=[(None, None), (None, None), (0.0, None)],  # z >= 0
    )
    return res.x


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def localise_drone(
    mic_wav_paths: list[str | Path],
    mic_positions_m: list[tuple[float, float, float]],
    temperature_celsius: float = 20.0,
    bandpass_low: float = 80.0,
    bandpass_high: float = 4000.0,
    search_bounds: dict | None = None,
    grid_size: int = 60,
    use_gcc_phat: bool = True,
    refine: bool = False,
    verbose: bool = True,
) -> dict:
    """
    End-to-end TDOA localisation pipeline.

    Parameters
    ----------
    mic_wav_paths      : one WAV file per microphone (synchronised, same sr)
    mic_positions_m    : (x, y, z) of each mic in metres
    temperature_celsius: ambient temperature for speed-of-sound correction
    bandpass_low/high  : Hz — pre-filter before cross-correlation
    search_bounds      : dict with keys x_range, y_range, z_range
                         (each a (min, max) tuple in metres)
    grid_size          : grid divisions per axis
    use_gcc_phat       : use GCC-PHAT (recommended) vs plain cross-correlation
    refine             : if True, apply L-BFGS-B refinement after grid search
    verbose            : print progress

    Returns
    -------
    dict with keys: location (x,y,z array), tdoas, error_volume, speed_of_sound
    """
    # 1. Load audio
    array = MicArray.from_separate_files(
        [Path(p) for p in mic_wav_paths],
        mic_positions_m,
    )
    if verbose:
        print(
            f"[load]  {array.n_mics} mics, {array.n_samples} samples, "
            f"sr={array.sample_rate} Hz, duration={array.duration_s:.2f}s"
        )

    # 2. Bandpass filter
    array = preprocess_array(array, low_hz=bandpass_low, high_hz=bandpass_high)

    # 3. Speed of sound
    c = speed_of_sound(temperature_celsius)
    if verbose:
        print(f"[sound] c = {c:.2f} m/s  (T = {temperature_celsius}°C)")

    # 4. Estimate TDOAs
    if use_gcc_phat:
        tdoas = estimate_all_tdoas(array, speed_of_sound_mps=c)
    else:
        ref = array.data[:, 0].astype(np.float64)
        tdoas_list: list[float] = []
        for i in range(1, array.n_mics):
            sig = array.data[:, i].astype(np.float64)
            corr = np.correlate(sig, ref, mode="full")
            lags = np.arange(-(len(ref) - 1), len(sig))
            peak_lag = lags[int(np.argmax(corr))]
            tdoas_list.append(-peak_lag / array.sample_rate)
        tdoas = np.array(tdoas_list)

    if verbose:
        for i, dt in enumerate(tdoas):
            print(
                f"[tdoa]  mic {i+1} vs mic 0:  "
                f"Δt = {dt*1000:.3f} ms  →  ΔD = {dt*c:.3f} m"
            )

    # 5. Grid search
    bounds = search_bounds or {
        "x_range": (-600, 600),
        "y_range": (-600, 600),
        "z_range": (0, 200),
    }

    mic_pos = np.array(mic_positions_m, dtype=np.float64)
    best_loc, error_volume = grid_search_tdoa_vectorised(
        mic_positions=mic_pos,
        observed_tdoas=tdoas,
        speed_of_sound_mps=c,
        grid_size=grid_size,
        **bounds,
    )

    # 6. Optional NLLS refinement
    if refine:
        best_loc = refine_with_nlls(best_loc, mic_pos, tdoas, c)

    if verbose:
        print(
            f"[result] Estimated drone location: "
            f"x={best_loc[0]:.1f} m, y={best_loc[1]:.1f} m, z={best_loc[2]:.1f} m"
        )

    return {
        "location": best_loc,
        "tdoas": tdoas,
        "error_volume": error_volume,
        "speed_of_sound": c,
    }


def classify_and_localise(
    mic_wav_paths: list[str],
    mic_positions: list[tuple[float, float, float]],
    model_weights: str = "base_model.pt",
    temperature_celsius: float = 20.0,
    localise_classes: set[str] | None = None,
) -> dict:
    """
    Gate TDOA localisation behind the acoustic classifier.

    Step 1: classify audio from mic 0 (single-channel, matches training).
    Step 2: run TDOA localisation only if the label is in localise_classes.

    Parameters
    ----------
    mic_wav_paths    : one WAV per mic (synchronised)
    mic_positions    : (x, y, z) in metres per mic
    model_weights    : path to FusionModel checkpoint
    localise_classes : set of class labels that trigger localisation
                       (default: {"drone"})

    Returns
    -------
    dict with keys: label, and optionally location / tdoas / error_volume
    """
    import torch
    from hs_hackathon_drone_acoustics.challenge import (
        load_model,
        decode_and_preprocess_librosa,
        predict,
    )

    if localise_classes is None:
        localise_classes = {"drone"}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_weights, device)

    with open(mic_wav_paths[0], "rb") as f:
        wav_bytes = f.read()
    mel, tab = decode_and_preprocess_librosa(wav_bytes)
    label = predict(model, mel, tab, device=device)

    print(f"[classifier] Detected: {label}")

    if label not in localise_classes:
        return {"label": label, "location": None}

    result = localise_drone(
        mic_wav_paths=mic_wav_paths,
        mic_positions_m=mic_positions,
        temperature_celsius=temperature_celsius,
        grid_size=60,
        use_gcc_phat=True,
    )
    return {"label": label, **result}
