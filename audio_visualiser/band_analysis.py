"""Stage 1: per-band onset detection on a wav file."""
import os

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks, stft

BIN_HZ = 5.5  # target frequency resolution
HOP = 512

_cache = {}


def load_spectrogram(path):
    """Returns (magnitude[freq, frame], freqs, hop_seconds, duration), cached per file."""
    key = (path, os.path.getmtime(path))
    if key in _cache:
        return _cache[key]
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    nperseg = 1 << int(np.ceil(np.log2(sr / BIN_HZ)))
    freqs, _, z = stft(mono, fs=sr, nperseg=nperseg, noverlap=nperseg - HOP, boundary=None, padded=False)
    result = (np.abs(z), freqs, HOP / sr, len(mono) / sr)
    _cache.clear()
    _cache[key] = result
    return result


def band_flux(mag, freqs, low_hz, high_hz, hop_s):
    """Normalised positive spectral flux of one band and its smoothing-window length in frames."""
    mask = (freqs >= low_hz) & (freqs < high_hz)
    if not mask.any():
        mask = np.zeros(len(freqs), dtype=bool)
        mask[np.argmin(np.abs(freqs - 0.5 * (low_hz + high_hz)))] = True
    env = np.log1p(100.0 * mag[mask].mean(axis=0))
    flux = np.maximum(np.diff(env, prepend=env[0]), 0.0)
    flux = uniform_filter1d(flux, 3)
    ref = np.percentile(flux, 99.5)
    return flux / ref if ref > 0 else flux


def detect_band(flux, hop_s, sensitivity, min_gap_s):
    """Returns (peak_indices, threshold_curve). Sensitivity 0..1, higher gives more events."""
    window = max(3, int(1.0 / hop_s))
    k = 3.0 * (1.0 - sensitivity) + 0.1
    threshold = uniform_filter1d(flux, window) + k * flux.std() + 0.02
    peaks, _ = find_peaks(flux, height=threshold, distance=max(1, int(min_gap_s / hop_s)))
    return peaks, threshold


def analyse(path, bands, global_offset, min_gap_s):
    """bands: list of dicts with name, low_hz, high_hz, sensitivity.

    Returns (per-band results, duration, hop_seconds). Each result holds the flux curve,
    threshold curve, event times and strengths.
    """
    mag, freqs, hop_s, duration = load_spectrogram(path)
    results = []
    for i, b in enumerate(bands):
        flux = band_flux(mag, freqs, b["low_hz"], b["high_hz"], hop_s)
        sens = float(np.clip(b["sensitivity"] + global_offset, 0.0, 1.0))
        peaks, threshold = detect_band(flux, hop_s, sens, min_gap_s)
        times = (peaks + 0.5 * 2 * (len(freqs) - 1) / HOP) * hop_s
        strengths = np.clip(flux[peaks], 0.0, 1.0)
        results.append({
            "index": i,
            "name": b["name"],
            "low_hz": float(b["low_hz"]),
            "high_hz": float(b["high_hz"]),
            "flux": flux,
            "threshold": threshold,
            "times": times,
            "strengths": strengths,
        })
    return results, duration, hop_s


def results_to_bands(results):
    return [
        {
            "index": r["index"],
            "name": r["name"],
            "low_hz": r["low_hz"],
            "high_hz": r["high_hz"],
            "events": [
                {"t": round(float(t), 4), "strength": round(float(s), 4)}
                for t, s in zip(r["times"], r["strengths"])
            ],
        }
        for r in results
    ]
