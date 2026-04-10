"""GRBAS voice quality analysis model.

Feature extraction uses:
  - parselmouth (Praat)  — F0, jitter, shimmer, HNR, formants, intensity
  - librosa              — MFCCs (+ Δ + ΔΔ), spectral features, chroma, ZCR
  - scipy                — CPP, GNE approximation, Teager Energy
  - python_speech_features — LPC coefficients + residual energy
  - antropy              — Sample Entropy, Approximate Entropy, Permutation Entropy
  - nolds                — DFA (Detrended Fluctuation Analysis), Hurst exponent
  - opensmile            — eGeMAPS feature set (88 standard speech features)
  - pyloudnorm           — LUFS integrated loudness & loudness range
  - webrtcvad            — VAD speech ratio, voiced/unvoiced segment statistics

ML classifiers available:
  random_forest, svm, gradient_boosting, xgboost, lightgbm, mlp, ensemble (default)
"""

import os
import warnings
import struct
import numpy as np
import parselmouth
from parselmouth.praat import call
import librosa
import joblib
import scipy.signal as spsig
import scipy.stats as spstats

import python_speech_features as psf
import antropy
import nolds
import opensmile
import pyloudnorm as pyln
import webrtcvad

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helper feature extractors
# ---------------------------------------------------------------------------

def _safe(fn, default=0.0):
    """Call fn(); return default if it raises or returns NaN/None."""
    try:
        v = fn()
        if v is None or (isinstance(v, float) and v != v):
            return default
        return float(v)
    except Exception:
        return default


def _compute_cpp(y, sr, f0_min=60, f0_max=600):
    """Cepstral Peak Prominence (CPP) — key breathiness / overall quality indicator.

    Returns (cpp_mean, cpp_std) across frames.
    """
    frame_len = int(0.040 * sr)   # 40 ms
    hop_len   = int(0.010 * sr)   # 10 ms
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)

    cpps = []
    for frame in frames.T:
        windowed = frame * np.hanning(len(frame))
        # Power cepstrum
        spectrum   = np.fft.rfft(windowed, n=4096)
        log_power  = np.log(np.abs(spectrum) ** 2 + 1e-10)
        cepstrum   = np.fft.irfft(log_power).real

        # Quefrency axis
        quefrencies = np.arange(len(cepstrum)) / sr

        # Restrict to F0 range
        q_min = 1.0 / f0_max
        q_max = 1.0 / f0_min
        mask  = (quefrencies > q_min) & (quefrencies < q_max)
        if not np.any(mask):
            continue

        idx_range = np.where(mask)[0]
        peak_idx  = idx_range[np.argmax(cepstrum[mask])]
        peak_val  = cepstrum[peak_idx]

        # Linear regression baseline over the F0-range quefrencies
        x  = quefrencies[mask]
        y_ = cepstrum[mask]
        if len(x) < 2:
            continue
        coeffs    = np.polyfit(x, y_, 1)
        baseline  = np.polyval(coeffs, quefrencies[peak_idx])
        cpps.append(peak_val - baseline)

    if not cpps:
        return 0.0, 0.0
    return float(np.mean(cpps)), float(np.std(cpps))


def _compute_cpps(y, sr, f0_min=60, f0_max=600):
    """Cepstral Peak Prominence Smoothed (CPPS) — gold-standard asthenia metric.

    CPPS is the clinically validated upgrade over raw CPP: it applies double
    smoothing (spectral domain + temporal domain) before measuring the cepstral
    peak, making it far more robust to noise and short-duration recordings.

    Clinical relevance to Asthenia (A):
        Low CPPS → poor harmonic structure → weak glottal closure → asthenic voice.
        Normative CPPS: ≥ 5 dB = healthy phonation; < 3 dB = significant weakness.

    Algorithm (following Hillenbrand & Houde 1996, Heman-Ackah 2003):
        1. Frame signal: 40 ms windows, 10 ms hop, Hanning-weighted.
        2. Compute log power spectrum for each frame.
        3. Smooth log spectrum with a 31-bin moving average (quefrency smoothing).
        4. Compute cepstrum of the smoothed log spectrum (power cepstrum).
        5. Find the peak in the F0 quefrency range [1/f0_max, 1/f0_min].
        6. Subtract a linear regression baseline from the peak value → per-frame CPP.
        7. Apply an 11-frame (~110 ms) moving average across the CPP time series
           (temporal smoothing — the "S" in CPPS).
        8. Return (mean, std, min, 10th-percentile) of the smoothed series.

    Returns:
        dict with cpps_mean, cpps_std, cpps_min, cpps_p10
    """
    frame_len  = int(0.040 * sr)   # 40 ms analysis window
    hop_len    = int(0.010 * sr)   # 10 ms hop (100 frames/s)
    n_fft      = 4096
    smooth_win = 31   # spectral smoothing window (bins) — must be odd
    temp_win   = 11   # temporal smoothing window (frames ≈ 110 ms)

    q_min = 1.0 / f0_max
    q_max = 1.0 / f0_min

    try:
        frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)
    except Exception:
        return {"cpps_mean": 0.0, "cpps_std": 0.0, "cpps_min": 0.0, "cpps_p10": 0.0}

    per_frame_cpp = []

    for frame in frames.T:
        windowed = frame * np.hanning(len(frame))

        # ── Step 2: log power spectrum in dB ───────────────────────────────
        # Clinical literature uses 10*log10 so that CPPS is in dB units.
        # Normative range: ≥ 5 dB = healthy; < 3 dB = significant weakness.
        spectrum  = np.fft.rfft(windowed, n=n_fft)
        log_power = 10.0 * np.log10(np.abs(spectrum) ** 2 + 1e-10)

        # ── Step 3: power cepstrum of the log power spectrum ────────────────
        # NOTE: The "smoothed" in CPPS refers to temporal smoothing across
        # frames (Step 7), NOT spectral smoothing here.  Spectral smoothing
        # would suppress the cepstral peak and produce artificially low values.
        cepstrum    = np.fft.irfft(log_power).real
        quefrencies = np.arange(len(cepstrum)) / sr

        # ── Step 4: find peak in F0 quefrency range ──────────────────────────
        mask = (quefrencies > q_min) & (quefrencies < q_max)
        if not np.any(mask):
            continue

        idx_range = np.where(mask)[0]
        peak_idx  = idx_range[np.argmax(cepstrum[mask])]
        peak_val  = cepstrum[peak_idx]

        # ── Step 5: linear regression baseline subtraction ──────────────────
        # Baseline accounts for the overall downward slope of the cepstrum;
        # the *prominence* of the peak above this slope is the clinical metric.
        x   = quefrencies[mask]
        yc  = cepstrum[mask]
        if len(x) < 2:
            continue
        coeffs   = np.polyfit(x, yc, 1)
        baseline = np.polyval(coeffs, quefrencies[peak_idx])
        per_frame_cpp.append(peak_val - baseline)

    if len(per_frame_cpp) < 2:
        return {"cpps_mean": 0.0, "cpps_std": 0.0, "cpps_min": 0.0, "cpps_p10": 0.0}

    # ── Step 6: temporal smoothing across frames (the "S" in CPPS) ─────────
    # Averaging CPP over ~110 ms windows reduces frame-to-frame noise without
    # destroying the clinical signal of sustained vowel weakness.
    cpp_arr = np.array(per_frame_cpp)
    kernel  = np.ones(min(temp_win, len(cpp_arr))) / min(temp_win, len(cpp_arr))
    cpps_smoothed = np.convolve(cpp_arr, kernel, mode="same")

    return {
        "cpps_mean": float(np.mean(cpps_smoothed)),
        "cpps_std":  float(np.std(cpps_smoothed)),
        "cpps_min":  float(np.min(cpps_smoothed)),
        "cpps_p10":  float(np.percentile(cpps_smoothed, 10)),
    }


def _compute_harmonic_decay(y, sr, f0_mean, n_harmonics=8):
    """Measure how fast harmonic amplitudes fall off with increasing frequency.

    Clinical relevance to Asthenia:
        In a healthy, well-supported voice the higher harmonics retain
        meaningful energy (shallow decay slope).  In a weak (asthenic) voice,
        energy collapses quickly above F0: the harmonic decay rate is steep
        (more negative slope).  This mirrors the classic 'weak harmonic
        structure' description in voice clinics.

    Method:
        - For each harmonic k = 1…n_harmonics, sum the FFT magnitude in a
          narrow band around k * F0 (±10 % of F0).
        - Fit a first-order polynomial to (harmonic_number, log_amplitude).
        - The slope is the decay rate; the R² of the fit is the regularity score.

    Returns:
        harmonic_decay_slope   — negative = energy falls with frequency
        harmonic_regularity    — R² of the linear fit (0–1; 1 = perfect decay)
    """
    if f0_mean <= 0 or len(y) < sr * 0.1:
        return {"harmonic_decay_slope": 0.0, "harmonic_regularity": 0.0}

    spectrum  = np.abs(np.fft.rfft(y, n=4096))
    freq_bins = np.fft.rfftfreq(4096, d=1.0 / sr)
    bw        = f0_mean * 0.10   # ±10 % bandwidth around each harmonic

    amp_log = []
    harm_ns = []
    for k in range(1, n_harmonics + 1):
        center = k * f0_mean
        if center + bw >= freq_bins[-1]:
            break
        mask = (freq_bins >= center - bw) & (freq_bins <= center + bw)
        if not np.any(mask):
            continue
        amp = float(np.sum(spectrum[mask]))
        if amp > 0:
            amp_log.append(np.log(amp + 1e-10))
            harm_ns.append(float(k))

    if len(harm_ns) < 3:
        return {"harmonic_decay_slope": 0.0, "harmonic_regularity": 0.0}

    x  = np.array(harm_ns)
    yy = np.array(amp_log)
    coeffs   = np.polyfit(x, yy, 1)
    slope    = float(coeffs[0])          # negative → energy falls with harmonic number
    yy_pred  = np.polyval(coeffs, x)
    ss_res   = np.sum((yy - yy_pred) ** 2)
    ss_tot   = np.sum((yy - yy.mean()) ** 2)
    r2       = float(1 - ss_res / (ss_tot + 1e-10))

    return {
        "harmonic_decay_slope":   slope,
        "harmonic_regularity":    max(0.0, min(1.0, r2)),
    }


def _compute_energy_envelope(y, sr):
    """Short-time RMS energy envelope features for Asthenia detection.

    Clinical relevance:
        Asthenic voice presents with:
          1. Low mean energy (quiet, weak phonation).
          2. Rapidly collapsing energy — the voice 'fades out' mid-utterance.
          3. Low energy variability within voiced regions (flat, monotone output).

    We characterise the RMS envelope at two time scales:
        - Fine scale  (25 ms frames) — captures rapid drops.
        - Coarse scale (200 ms blocks) — captures macro-level energy contour.

    Returns:
        energy_env_mean      — mean short-time RMS (overall loudness proxy)
        energy_env_std       — variability of short-time RMS
        energy_env_cv        — coefficient of variation (std / mean)
        energy_env_slope     — linear trend of the RMS contour
                               (negative = voice fades → strong asthenia indicator)
        energy_env_drop_rate — fraction of consecutive frame-pairs where energy
                               drops by > 20 % (captures instability / fatigue)
    """
    frame_len = int(0.025 * sr)   # 25 ms
    hop_len   = int(0.010 * sr)   # 10 ms
    try:
        rms_frames = librosa.feature.rms(
            y=y, frame_length=frame_len, hop_length=hop_len
        )[0]
    except Exception:
        rms_frames = librosa.feature.rms(y=y)[0]

    if len(rms_frames) < 3:
        return {k: 0.0 for k in (
            "energy_env_mean", "energy_env_std", "energy_env_cv",
            "energy_env_slope", "energy_env_drop_rate")}

    env_mean = float(np.mean(rms_frames))
    env_std  = float(np.std(rms_frames))
    env_cv   = float(env_std / (env_mean + 1e-10))

    # Linear trend along the utterance (time × coefficient)
    t = np.linspace(0, 1, len(rms_frames))
    slope = float(np.polyfit(t, rms_frames, 1)[0])

    # Drop rate: fraction of frame-pairs with > 20 % energy collapse
    ratios = rms_frames[1:] / (rms_frames[:-1] + 1e-10)
    drop_rate = float(np.mean(ratios < 0.80))

    return {
        "energy_env_mean":      env_mean,
        "energy_env_std":       env_std,
        "energy_env_cv":        env_cv,
        "energy_env_slope":     slope,
        "energy_env_drop_rate": drop_rate,
    }


def _compute_phonation_efficiency(features):
    """Estimate glottal phonation efficiency from existing features.

    Phonation efficiency (PE) quantifies how well the glottis converts
    subglottal air pressure into acoustic power.  A low PE indicates an
    asthenic voice: the vocal folds are not closing fully or are too flaccid
    to produce a strong acoustic output per unit of airflow.

    Proxy formula (all inputs already in `features` dict):
        PE = HNR / (HNR + 1) × (1 - shimmer_local)
             × LUFS_normalised × voiced_fraction

    This stays in [0, 1]: higher = more efficient (healthier) phonation.
    """
    hnr           = features.get("hnr", 0.0)
    shimmer       = features.get("shimmer_local", 0.0)   # 0–1 range
    lufs          = features.get("lufs_integrated", -70.0)
    voiced_frac   = features.get("voiced_fraction", 0.5)

    # Convert HNR to a 0–1 contribution (saturates around 30 dB)
    hnr_norm  = min(max(hnr, 0.0), 30.0) / 30.0
    # Normalise LUFS: −70 LUFS = 0, −14 LUFS = 1
    lufs_norm = min(max((lufs - (-70.0)) / 56.0, 0.0), 1.0)

    pe = hnr_norm * max(0.0, 1.0 - shimmer) * lufs_norm * voiced_frac
    return float(pe)


# ---------------------------------------------------------------------------
# Public Asthenia analysis function
# ---------------------------------------------------------------------------

def analyze_asthenia(audio_path: str) -> dict:
    """Analyse the Asthenia (A) dimension of the GRBAS scale for one audio file.

    Asthenia describes *weakness* of the voice: reduced intensity, poor
    harmonic structure, inadequate glottal closure, and diminished phonation
    efficiency — often associated with neurological conditions, vocal fold
    paralysis, systemic weakness, or extreme vocal fatigue.

    This function integrates four complementary measurement domains:

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Domain        │ Metric           │ Library       │ Clinical role        │
    ├─────────────────────────────────────────────────────────────────────────┤
    │ Cepstral      │ CPPS             │ numpy/scipy   │ Harmonic structure   │
    │ Glottal       │ HNR              │ parselmouth   │ Noise vs. periodicity│
    │ Spectral      │ Spectral tilt    │ librosa       │ HF energy loss       │
    │ Energy        │ RMS + LUFS       │ librosa/pln   │ Physical intensity   │
    │ Harmonic      │ Decay slope      │ numpy         │ Harmonic support     │
    │ Efficiency    │ Phonation eff.   │ composite     │ Glottal output       │
    └─────────────────────────────────────────────────────────────────────────┘

    Parameters
    ----------
    audio_path : str
        Path to any audio file readable by librosa (WAV, OGG, FLAC, etc.).

    Returns
    -------
    dict with keys:

    Primary clinical metrics
        cpps_mean               CPPS mean (dB) — primary asthenia indicator
        cpps_std                CPPS temporal variability
        cpps_p10                10th-percentile CPPS (worst frames)
        hnr                     Harmonics-to-Noise Ratio (dB)
        spectral_tilt           Slope of mean log spectrum vs. frequency (dB/Hz)
        spectral_tilt_low       Tilt in 0–2 kHz sub-band
        spectral_tilt_high      Tilt in 2–8 kHz sub-band

    Energy / intensity
        rms_mean                Mean short-time RMS amplitude
        rms_std                 RMS standard deviation
        lufs_integrated         Integrated loudness (LUFS)
        energy_env_slope        Linear trend of energy contour (negative = fading)
        energy_env_drop_rate    Fraction of frames with > 20 % energy drop

    Harmonic structure
        harmonic_decay_slope    Rate of harmonic amplitude decay (neg = steep)
        harmonic_regularity     R² of harmonic decay fit (0–1)

    Efficiency & voiced fraction
        phonation_efficiency    Composite glottal efficiency score (0–1)
        voiced_fraction         Proportion of voiced frames

    Composite
        asthenia_score          Rule-based A score on the GRBAS 0–3 scale
        asthenia_confidence     Fraction of evidence groups indicating pathology
    """
    # ── Load & normalise ────────────────────────────────────────────────────
    # Normalisation removes amplitude/volume bias so metrics reflect voice
    # *quality* rather than recording level differences between subjects.
    y, sr = librosa.load(audio_path, sr=None)
    if y.ndim > 1:
        y = y.mean(axis=0)
    y = y.astype(np.float64)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak   # peak-normalise to [-1, 1]

    results = {}

    # ── Domain 1: CPPS (Cepstral Peak Prominence Smoothed) ─────────────────
    # CPPS is the gold-standard non-invasive correlate of voice quality and
    # asthenia severity.  Low CPPS = poor harmonic energy = weak voice.
    cpps_feats = _compute_cpps(y, sr)
    results.update(cpps_feats)

    # ── Domain 2: HNR via parselmouth / Praat ──────────────────────────────
    # HNR separates the harmonic (periodic) component from the noise component.
    # Low HNR in an asthenic voice reflects incomplete glottal closure: air
    # passes through during the closed phase, adding turbulent noise.
    try:
        snd         = parselmouth.Sound(audio_path)
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        results["hnr"] = _safe(lambda: call(harmonicity, "Get mean", 0, 0))

        pitch    = call(snd, "To Pitch", 0.0, 75, 600)
        n_frames = call(pitch, "Get number of frames")
        f0_vals  = [
            call(pitch, "Get value in frame", i + 1, "Hertz")
            for i in range(n_frames)
        ]
        f0_vals  = [v for v in f0_vals if v == v]
        f0_mean  = float(np.mean(f0_vals)) if f0_vals else 0.0
        voiced_count = len(f0_vals)
        results["voiced_fraction"] = float(voiced_count / n_frames) if n_frames > 0 else 0.0
        results["f0_mean"] = f0_mean

        shimmer_feats = {}
        pp = call(snd, "To PointProcess (periodic, cc)", 75, 600)
        shimmer_feats["shimmer_local"] = _safe(
            lambda: call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))
        results.update(shimmer_feats)
    except Exception:
        results.setdefault("hnr", 0.0)
        results.setdefault("voiced_fraction", 0.0)
        results.setdefault("f0_mean", 0.0)
        results.setdefault("shimmer_local", 0.0)
        f0_mean = 0.0

    # ── Domain 3: Spectral tilt (librosa) ──────────────────────────────────
    # Spectral tilt = slope of the mean log spectrum in dB vs. frequency.
    # A steep negative tilt means most energy is concentrated in low freqs,
    # which is the acoustic signature of a weak, asthenic voice where the
    # vocal tract fails to radiate high-frequency harmonics efficiently.
    S     = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    log_S = np.mean(20 * np.log10(S + 1e-10), axis=1)

    if len(freqs) > 1:
        results["spectral_tilt"] = float(np.polyfit(freqs, log_S, 1)[0])
    else:
        results["spectral_tilt"] = 0.0

    # Sub-band tilts: separately characterise low and high spectral regions
    lo_mask = freqs < 2000
    hi_mask = (freqs >= 2000) & (freqs < 8000)
    if np.sum(lo_mask) > 2:
        results["spectral_tilt_low"] = float(
            np.polyfit(freqs[lo_mask], log_S[lo_mask], 1)[0])
    else:
        results["spectral_tilt_low"] = 0.0
    if np.sum(hi_mask) > 2:
        results["spectral_tilt_high"] = float(
            np.polyfit(freqs[hi_mask], log_S[hi_mask], 1)[0])
    else:
        results["spectral_tilt_high"] = 0.0

    # ── Domain 4: RMS energy + LUFS ────────────────────────────────────────
    # Physical intensity: the most direct measure of vocal power output.
    # Asthenic voices are clinically quieter.  We use both short-time RMS
    # (fine-grained) and LUFS (perceptually weighted, broadcast-standard) to
    # capture intensity from different perspectives.
    rms = librosa.feature.rms(y=y)[0]
    results["rms_mean"] = float(np.mean(rms))
    results["rms_std"]  = float(np.std(rms))

    try:
        meter = pyln.Meter(sr)
        lufs  = meter.integrated_loudness(y)
        results["lufs_integrated"] = float(lufs) if np.isfinite(lufs) else -70.0
    except Exception:
        results["lufs_integrated"] = -70.0

    # Energy envelope dynamics
    env_feats = _compute_energy_envelope(y, sr)
    results.update(env_feats)

    # ── Domain 5: Harmonic structure ───────────────────────────────────────
    # Healthy voices have a well-supported harmonic series up to 3–4 kHz.
    # In asthenic phonation the amplitudes collapse rapidly after F0, leaving
    # the upper harmonics buried in noise.  The decay slope and its regularity
    # (R²) together quantify this.
    harm_feats = _compute_harmonic_decay(y, sr, f0_mean)
    results.update(harm_feats)

    # ── Phonation efficiency ────────────────────────────────────────────────
    results["phonation_efficiency"] = _compute_phonation_efficiency(results)

    # ── Rule-based Asthenia score (0–3) ────────────────────────────────────
    # Five evidence groups, each producing a 0–1 score; weighted sum → 0–3.
    # The weighting order reflects clinical importance as reported in the
    # literature (Bhuta 2004, Dejonckere 1996, Bhuta & Bhuta 2004).
    cpps     = results.get("cpps_mean", 0.0)
    hnr_val  = results.get("hnr", 25.0)
    tilt     = results.get("spectral_tilt", 0.0)
    rms_m    = results.get("rms_mean", 0.05)
    pe       = results.get("phonation_efficiency", 0.5)
    decay    = results.get("harmonic_decay_slope", -1.0)
    env_slp  = results.get("energy_env_slope", 0.0)
    drop_r   = results.get("energy_env_drop_rate", 0.0)
    vf       = results.get("voiced_fraction", 1.0)

    # Group 1 — CPPS (w=0.35): primary cepstral evidence
    # Healthy: CPPS ≥ 5 dB. Severe asthenia: < 2 dB.
    g1 = min(max(5.0 - cpps, 0.0) / 5.0, 1.0)

    # Group 2 — HNR (w=0.25): harmonic vs. noise balance
    # Healthy: HNR ≥ 20 dB. Asthenic: < 12 dB.
    g2 = min(max(20.0 - hnr_val, 0.0) / 20.0, 1.0)

    # Group 3 — Energy / loudness (w=0.20): physical intensity
    # Low RMS + low LUFS + fading contour → weak phonation.
    lufs_val = results.get("lufs_integrated", -70.0)
    g3a = min(max(0.05 - rms_m, 0.0) / 0.05, 1.0)          # RMS < 0.05 = quiet
    g3b = min(max(lufs_val - (-70.0), 0.0) / 40.0, 1.0)    # LUFS closer to -70
    g3b = 1.0 - g3b                                          # invert: louder = healthier
    g3c = min(max(-env_slp, 0.0) / 0.02, 1.0)               # negative slope = fading
    g3d = min(drop_r / 0.4, 1.0)                             # many energy drops
    g3  = min(0.35 * g3a + 0.30 * g3b + 0.20 * g3c + 0.15 * g3d, 1.0)

    # Group 4 — Harmonic decay (w=0.12): upper harmonic support
    # Decay slope < -1.5 nat/harmonic → significant harmonic collapse.
    g4 = min(max(-decay - 1.5, 0.0) / 3.0, 1.0)

    # Group 5 — Phonation efficiency + voiced fraction (w=0.08)
    g5a = min(max(0.4 - pe, 0.0) / 0.4, 1.0)               # PE < 0.4 = inefficient
    g5b = min(max(0.6 - vf, 0.0) / 0.6, 1.0)               # VF < 0.6 = weak sustain
    g5  = min(0.6 * g5a + 0.4 * g5b, 1.0)

    # Weighted sum → normalise to 0–3
    a_raw = g1 * 0.35 + g2 * 0.25 + g3 * 0.20 + g4 * 0.12 + g5 * 0.08
    results["asthenia_score"]      = int(_clamp(round(a_raw / 0.30 * 3)))
    results["asthenia_confidence"] = float(
        np.mean([g1 > 0.3, g2 > 0.3, g3 > 0.3, g4 > 0.3, g5 > 0.3])
    )
    results["_group_scores"] = {
        "cpps": round(g1, 3),
        "hnr":  round(g2, 3),
        "energy": round(g3, 3),
        "harmonic_decay": round(g4, 3),
        "phonation_efficiency": round(g5, 3),
    }

    # Sanitise NaN / Inf
    for k, v in list(results.items()):
        if isinstance(v, dict):
            continue
        try:
            if not np.isfinite(float(v)):
                results[k] = 0.0
        except Exception:
            pass

    return results


def _compute_gne(y, sr, bands=None):
    """Glottal-to-Noise Excitation (GNE) approximation.

    Compares correlation of the Hilbert envelope across frequency bands:
    high correlation → voiced (glottal source), low → noise.
    Returns the mean GNE ratio (0–1, higher = cleaner voice).
    """
    if bands is None:
        bands = [(300, 900), (900, 1500), (1500, 2500), (2500, 3500)]

    envelopes = []
    for lo, hi in bands:
        b, a = spsig.butter(4, [lo / (sr / 2), hi / (sr / 2)], btype="band")
        filtered  = spsig.filtfilt(b, a, y)
        envelope  = np.abs(spsig.hilbert(filtered))
        envelopes.append(envelope)

    gne_vals = []
    ref = envelopes[0]
    for env in envelopes[1:]:
        min_len = min(len(ref), len(env))
        corr    = np.corrcoef(ref[:min_len], env[:min_len])[0, 1]
        gne_vals.append(abs(corr) if not np.isnan(corr) else 0.0)

    return float(np.mean(gne_vals)) if gne_vals else 0.0


def _compute_teo(y):
    """Teager Energy Operator features — captures amplitude+frequency modulation."""
    teo = y[1:-1] ** 2 - y[:-2] * y[2:]
    teo_abs = np.abs(teo)
    return {
        "teo_mean":     float(np.mean(teo_abs)),
        "teo_std":      float(np.std(teo_abs)),
        "teo_kurtosis": float(spstats.kurtosis(teo_abs)),
        "teo_skewness": float(spstats.skew(teo_abs)),
    }


def _compute_lpc_features(y, sr, order=12):
    """LPC coefficients and residual energy via python_speech_features."""
    frame_len = int(0.025 * sr)
    hop_len   = int(0.010 * sr)
    frames    = psf.sigproc.framesig(y, frame_len, hop_len)
    if len(frames) == 0:
        return {f"lpc_{i}": 0.0 for i in range(order)} | {"lpc_residual_energy": 0.0}

    all_lpc   = []
    residuals = []
    for frame in frames:
        windowed = frame * np.hamming(len(frame))
        try:
            a = np.array(psf.lpc(windowed, order))
            residual = np.convolve(windowed, a, mode="full")
            residuals.append(np.mean(residual ** 2))
            all_lpc.append(a[1:order + 1] if len(a) > order else np.zeros(order))
        except Exception:
            continue

    if not all_lpc:
        return {f"lpc_{i}": 0.0 for i in range(order)} | {"lpc_residual_energy": 0.0}

    mean_lpc = np.mean(all_lpc, axis=0)
    feats    = {f"lpc_{i}": float(mean_lpc[i]) for i in range(len(mean_lpc))}
    feats["lpc_residual_energy"] = float(np.mean(residuals))
    return feats


def _compute_entropy_features(y, sr, max_samples=4096):
    """Sample, Approximate and Permutation Entropy (antropy + nolds)."""
    # Downsample for speed
    step = max(1, len(y) // max_samples)
    yc   = y[::step]
    feats = {}
    feats["sample_entropy"]      = _safe(lambda: antropy.sample_entropy(yc))
    feats["app_entropy"]         = _safe(lambda: antropy.app_entropy(yc))
    feats["perm_entropy"]        = _safe(lambda: antropy.perm_entropy(yc, normalize=True))
    feats["spectral_entropy"]    = _safe(lambda: antropy.spectral_entropy(yc, sf=sr // step, method="welch", normalize=True))
    feats["svd_entropy"]         = _safe(lambda: antropy.svd_entropy(yc, normalize=True))
    return feats


def _compute_nonlinear_features(y, sr, max_samples=4096):
    """DFA and Hurst exponent via nolds — nonlinear voice dynamics."""
    step = max(1, len(y) // max_samples)
    yc   = y[::step].astype(float)
    feats = {}
    feats["dfa_alpha"]     = _safe(lambda: nolds.dfa(yc))
    feats["hurst_exp"]     = _safe(lambda: nolds.hurst_rs(yc))
    return feats


# Shared openSMILE extractor (eGeMAPS — 88 functionals, standard for voice quality)
_smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals,
)


def _compute_opensmile_features(audio_path):
    """Extract eGeMAPS feature set via openSMILE (88 standard speech features)."""
    try:
        df = _smile.process_file(audio_path)
        row = df.iloc[0]
        return {f"smile_{col}": float(row[col]) for col in df.columns}
    except Exception:
        return {}


def _compute_lufs_features(y, sr):
    """LUFS integrated loudness and loudness range via pyloudnorm."""
    feats = {}
    try:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(y.astype(float))
        feats["lufs_integrated"] = float(loudness) if np.isfinite(loudness) else -70.0
    except Exception:
        feats["lufs_integrated"] = -70.0

    try:
        # Loudness range — split into 3-second blocks if long enough
        block_size = sr * 3
        if len(y) >= block_size:
            blocks = [y[i: i + block_size] for i in range(0, len(y) - block_size, block_size)]
            block_lufs = []
            meter = pyln.Meter(sr)
            for blk in blocks:
                try:
                    lk = meter.integrated_loudness(blk.astype(float))
                    if np.isfinite(lk):
                        block_lufs.append(lk)
                except Exception:
                    continue
            if len(block_lufs) >= 2:
                feats["lufs_range"]  = float(np.max(block_lufs) - np.min(block_lufs))
                feats["lufs_std"]    = float(np.std(block_lufs))
            else:
                feats["lufs_range"] = 0.0
                feats["lufs_std"]   = 0.0
        else:
            feats["lufs_range"] = 0.0
            feats["lufs_std"]   = 0.0
    except Exception:
        feats["lufs_range"] = 0.0
        feats["lufs_std"]   = 0.0

    return feats


def _compute_vad_features(y, sr):
    """WebRTC VAD — speech ratio, segment count, mean/std voiced segment duration."""
    feats = {
        "vad_speech_ratio":        0.0,
        "vad_num_segments":        0.0,
        "vad_mean_segment_dur_ms": 0.0,
        "vad_std_segment_dur_ms":  0.0,
    }
    try:
        vad = webrtcvad.Vad(mode=2)   # mode 0–3; 2 = balanced aggressiveness

        # VAD requires 8 / 16 / 32 kHz and 10 / 20 / 30 ms frames
        target_sr    = 16000
        frame_ms     = 20
        frame_samples = int(target_sr * frame_ms / 1000)

        # Resample to 16 kHz
        y16 = librosa.resample(y.astype(float), orig_sr=sr, target_sr=target_sr)
        y16 = np.clip(y16, -1.0, 1.0)

        # Convert to 16-bit PCM bytes
        pcm = (y16 * 32767).astype(np.int16)
        raw = struct.pack(f"{len(pcm)}h", *pcm)

        is_speech = []
        for start in range(0, len(pcm) - frame_samples, frame_samples):
            frame_bytes = raw[start * 2: (start + frame_samples) * 2]
            try:
                voiced = vad.is_speech(frame_bytes, target_sr)
                is_speech.append(voiced)
            except Exception:
                is_speech.append(False)

        if not is_speech:
            return feats

        feats["vad_speech_ratio"] = float(sum(is_speech) / len(is_speech))

        # Segment analysis
        segments = []
        in_seg   = False
        seg_len  = 0
        for v in is_speech:
            if v:
                in_seg = True
                seg_len += 1
            else:
                if in_seg:
                    segments.append(seg_len * frame_ms)
                    in_seg  = False
                    seg_len = 0
        if in_seg:
            segments.append(seg_len * frame_ms)

        feats["vad_num_segments"] = float(len(segments))
        if segments:
            feats["vad_mean_segment_dur_ms"] = float(np.mean(segments))
            feats["vad_std_segment_dur_ms"]  = float(np.std(segments))
    except Exception:
        pass

    return feats


# ---------------------------------------------------------------------------
# Main feature extraction
# ---------------------------------------------------------------------------

def extract_features(audio_path):
    """Extract a comprehensive set of acoustic features for GRBAS scoring.

    Returns a flat dict of named float features.
    """
    snd  = parselmouth.Sound(audio_path)
    y, sr = librosa.load(audio_path, sr=None)

    # Ensure float64 mono
    if y.ndim > 1:
        y = y.mean(axis=0)
    y = y.astype(np.float64)

    features = {}

    # -----------------------------------------------------------------------
    # 1. F0 (fundamental frequency) — parselmouth
    # -----------------------------------------------------------------------
    pitch    = call(snd, "To Pitch", 0.0, 75, 600)
    f0_vals  = [
        call(pitch, "Get value in frame", i + 1, "Hertz")
        for i in range(call(pitch, "Get number of frames"))
    ]
    f0_vals  = [v for v in f0_vals if v == v]   # drop NaN

    if f0_vals:
        f0_arr = np.array(f0_vals)
        features["f0_mean"]    = float(np.mean(f0_arr))
        features["f0_std"]     = float(np.std(f0_arr))
        features["f0_min"]     = float(np.min(f0_arr))
        features["f0_max"]     = float(np.max(f0_arr))
        features["f0_range"]   = float(np.ptp(f0_arr))
        features["f0_median"]  = float(np.median(f0_arr))
        features["f0_iqr"]     = float(np.percentile(f0_arr, 75) - np.percentile(f0_arr, 25))
        features["f0_kurtosis"]= float(spstats.kurtosis(f0_arr))
        features["f0_skewness"]= float(spstats.skew(f0_arr))
    else:
        for k in ("f0_mean","f0_std","f0_min","f0_max","f0_range",
                  "f0_median","f0_iqr","f0_kurtosis","f0_skewness"):
            features[k] = 0.0

    n_frames    = call(pitch, "Get number of frames")
    voiced_count = sum(
        1 for i in range(n_frames)
        if call(pitch, "Get value in frame", i + 1, "Hertz") ==
           call(pitch, "Get value in frame", i + 1, "Hertz")  # not NaN
    )
    features["voiced_fraction"] = float(voiced_count / n_frames) if n_frames > 0 else 0.0

    # -----------------------------------------------------------------------
    # Parselmouth features for S (Strain) — all extracted from Pitch object
    # -----------------------------------------------------------------------
    # Voice breaks: key hyperfunctional-voice indicator
    features["n_voice_breaks"]           = _safe(lambda: call(pitch, "Get number of voice breaks"))
    features["degree_voice_breaks"]      = _safe(lambda: call(pitch, "Get degree of voice breaks"))
    # Fraction of locally unvoiced frames: high → strained phonation
    features["fraction_locally_unvoiced"] = _safe(lambda: call(pitch, "Get fraction of locally unvoiced frames", 0, 0, 0.03, 0.45, 1.5))
    # F0 coefficient of variation: high CV → unstable/effortful pitch
    if features.get("f0_mean", 0) > 0:
        features["f0_cv"] = features["f0_std"] / features["f0_mean"]
    else:
        features["f0_cv"] = 0.0

    # -----------------------------------------------------------------------
    # 2. Jitter & Shimmer — parselmouth (expanded set)
    # -----------------------------------------------------------------------
    pp = call(snd, "To PointProcess (periodic, cc)", 75, 600)

    # Number of periods and period statistics — Strain indicators
    features["n_periods"]      = _safe(lambda: call(pp, "Get number of periods",     0, 0, 0.0001, 0.02, 1.3))
    features["mean_period"]    = _safe(lambda: call(pp, "Get mean period",           0, 0, 0.0001, 0.02, 1.3))
    features["stdev_period"]   = _safe(lambda: call(pp, "Get stdev period",          0, 0, 0.0001, 0.02, 1.3))
    # Period CV: high → irregular vocal fold vibration → Strain
    if features.get("mean_period", 0) > 0:
        features["period_cv"] = features["stdev_period"] / features["mean_period"]
    else:
        features["period_cv"] = 0.0

    features["jitter_local"]  = _safe(lambda: call(pp, "Get jitter (local)",  0, 0, 0.0001, 0.02, 1.3))
    features["jitter_rap"]    = _safe(lambda: call(pp, "Get jitter (rap)",    0, 0, 0.0001, 0.02, 1.3))
    features["jitter_ppq5"]   = _safe(lambda: call(pp, "Get jitter (ppq5)",   0, 0, 0.0001, 0.02, 1.3))
    features["jitter_ddp"]    = _safe(lambda: call(pp, "Get jitter (ddp)",    0, 0, 0.0001, 0.02, 1.3))

    features["shimmer_local"]  = _safe(lambda: call([snd, pp], "Get shimmer (local)",  0, 0, 0.0001, 0.02, 1.3, 1.6))
    features["shimmer_apq3"]   = _safe(lambda: call([snd, pp], "Get shimmer (apq3)",   0, 0, 0.0001, 0.02, 1.3, 1.6))
    features["shimmer_apq5"]   = _safe(lambda: call([snd, pp], "Get shimmer (apq5)",   0, 0, 0.0001, 0.02, 1.3, 1.6))
    features["shimmer_apq11"]  = _safe(lambda: call([snd, pp], "Get shimmer (apq11)",  0, 0, 0.0001, 0.02, 1.3, 1.6))
    features["shimmer_dda"]    = _safe(lambda: call([snd, pp], "Get shimmer (dda)",    0, 0, 0.0001, 0.02, 1.3, 1.6))
    features["shimmer_db"]     = _safe(lambda: call([snd, pp], "Get shimmer (local, dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6))

    # -----------------------------------------------------------------------
    # 3. HNR — parselmouth
    # -----------------------------------------------------------------------
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    features["hnr"] = _safe(lambda: call(harmonicity, "Get mean", 0, 0))

    # -----------------------------------------------------------------------
    # 4. Formants — parselmouth
    # -----------------------------------------------------------------------
    formant = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
    for fi in range(1, 5):
        features[f"f{fi}_mean"]      = _safe(lambda fi=fi: call(formant, "Get mean",              fi, 0, 0, "Hertz"))
        features[f"f{fi}_std"]       = _safe(lambda fi=fi: call(formant, "Get standard deviation", fi, 0, 0, "Hertz"))
        features[f"f{fi}_bandwidth"] = _safe(lambda fi=fi: call(formant, "Get bandwidth at time",  fi, 0.5 * call(snd, "Get end time"), "Hertz", "Linear"))

    # -----------------------------------------------------------------------
    # 5. Intensity — parselmouth
    # -----------------------------------------------------------------------
    intensity            = call(snd, "To Intensity", 75, 0, "yes")
    features["intensity_mean"] = _safe(lambda: call(intensity, "Get mean",               0, 0, "dB"))
    features["intensity_std"]  = _safe(lambda: call(intensity, "Get standard deviation", 0, 0))
    features["intensity_min"]  = _safe(lambda: call(intensity, "Get minimum",            0, 0, "dB", "Parabolic"))
    features["intensity_max"]  = _safe(lambda: call(intensity, "Get maximum",            0, 0, "dB", "Parabolic"))

    # -----------------------------------------------------------------------
    # 6. CPP (Cepstral Peak Prominence) — scipy/numpy
    # -----------------------------------------------------------------------
    cpp_mean, cpp_std = _compute_cpp(y, sr)
    features["cpp_mean"] = cpp_mean
    features["cpp_std"]  = cpp_std

    # -----------------------------------------------------------------------
    # 6b. CPPS (Cepstral Peak Prominence Smoothed) — primary asthenia metric
    # -----------------------------------------------------------------------
    try:
        features.update(_compute_cpps(y, sr))
    except Exception:
        for k in ("cpps_mean", "cpps_std", "cpps_min", "cpps_p10"):
            features[k] = 0.0

    # -----------------------------------------------------------------------
    # 6c. Asthenia-specific features (harmonic decay + energy envelope)
    # -----------------------------------------------------------------------
    f0_mean_val = features.get("f0_mean", 0.0)
    try:
        features.update(_compute_harmonic_decay(y, sr, f0_mean_val))
    except Exception:
        features["harmonic_decay_slope"] = 0.0
        features["harmonic_regularity"]  = 0.0

    try:
        features.update(_compute_energy_envelope(y, sr))
    except Exception:
        for k in ("energy_env_mean", "energy_env_std", "energy_env_cv",
                  "energy_env_slope", "energy_env_drop_rate"):
            features[k] = 0.0

    # phonation_efficiency is computed after LUFS (step 16) so that the
    # lufs_integrated feature is available.  Placeholder set here.
    features["phonation_efficiency"] = 0.0

    # -----------------------------------------------------------------------
    # 7. GNE approximation — scipy
    # -----------------------------------------------------------------------
    features["gne"] = _safe(lambda: _compute_gne(y, sr))

    # -----------------------------------------------------------------------
    # 8. Teager Energy Operator — numpy
    # -----------------------------------------------------------------------
    if len(y) > 2:
        teo_feats = _compute_teo(y)
        features.update(teo_feats)
    else:
        for k in ("teo_mean", "teo_std", "teo_kurtosis", "teo_skewness"):
            features[k] = 0.0

    # -----------------------------------------------------------------------
    # 9. Spectral features — librosa
    # -----------------------------------------------------------------------
    S    = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid_mean"] = float(np.mean(sc))
    features["spectral_centroid_std"]  = float(np.std(sc))

    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features["spectral_bandwidth_mean"] = float(np.mean(sb))

    sr_ = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    features["spectral_rolloff_mean"] = float(np.mean(sr_))

    sf = librosa.feature.spectral_flatness(y=y)
    features["spectral_flatness_mean"] = float(np.mean(sf))
    features["spectral_flatness_std"]  = float(np.std(sf))

    scon = librosa.feature.spectral_contrast(S=S, sr=sr)
    for b in range(scon.shape[0]):
        features[f"spectral_contrast_{b}"] = float(np.mean(scon[b]))

    # Spectral tilt
    log_S = np.mean(20 * np.log10(S + 1e-10), axis=1)
    if len(freqs) > 1:
        features["spectral_tilt"] = float(np.polyfit(freqs, log_S, 1)[0])
    else:
        features["spectral_tilt"] = 0.0

    # Spectral kurtosis / skewness of log spectrum
    features["spectral_kurtosis"] = float(spstats.kurtosis(log_S))
    features["spectral_skewness"] = float(spstats.skew(log_S))

    # High-frequency energy ratio (above 4 kHz)
    hf_mask = freqs >= 4000
    if np.any(hf_mask):
        features["hf_energy_ratio"] = float(
            np.mean(S[hf_mask, :] ** 2) / (np.mean(S ** 2) + 1e-10)
        )
    else:
        features["hf_energy_ratio"] = 0.0

    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features["zcr_mean"] = float(np.mean(zcr))
    features["zcr_std"]  = float(np.std(zcr))

    # RMS energy
    rms = librosa.feature.rms(y=y)
    features["rms_mean"] = float(np.mean(rms))
    features["rms_std"]  = float(np.std(rms))

    # -----------------------------------------------------------------------
    # 10. MFCCs + Δ + ΔΔ — librosa
    # -----------------------------------------------------------------------
    mfccs   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    d_mfccs = librosa.feature.delta(mfccs)
    dd_mfccs = librosa.feature.delta(mfccs, order=2)
    for i in range(13):
        features[f"mfcc_{i}_mean"]    = float(np.mean(mfccs[i]))
        features[f"mfcc_{i}_std"]     = float(np.std(mfccs[i]))
        features[f"mfcc_d_{i}_mean"]  = float(np.mean(d_mfccs[i]))
        features[f"mfcc_dd_{i}_mean"] = float(np.mean(dd_mfccs[i]))

    # -----------------------------------------------------------------------
    # 11. Chroma — librosa
    # -----------------------------------------------------------------------
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    features["chroma_mean"]   = float(np.mean(chroma))
    features["chroma_std"]    = float(np.std(chroma))
    features["chroma_energy"] = float(np.sum(chroma ** 2))

    # -----------------------------------------------------------------------
    # 12. LPC features — python_speech_features
    # -----------------------------------------------------------------------
    try:
        lpc_feats = _compute_lpc_features(y, sr)
        features.update(lpc_feats)
    except Exception:
        for i in range(12):
            features[f"lpc_{i}"] = 0.0
        features["lpc_residual_energy"] = 0.0

    # -----------------------------------------------------------------------
    # 13. Entropy features — antropy
    # -----------------------------------------------------------------------
    try:
        features.update(_compute_entropy_features(y, sr))
    except Exception:
        for k in ("sample_entropy", "app_entropy", "perm_entropy",
                  "spectral_entropy", "svd_entropy"):
            features[k] = 0.0

    # -----------------------------------------------------------------------
    # 14. Nonlinear dynamics — nolds
    # -----------------------------------------------------------------------
    try:
        features.update(_compute_nonlinear_features(y, sr))
    except Exception:
        features["dfa_alpha"]  = 0.0
        features["hurst_exp"]  = 0.0

    # -----------------------------------------------------------------------
    # 15. openSMILE — eGeMAPS (88 standard speech & voice-quality functionals)
    # -----------------------------------------------------------------------
    try:
        features.update(_compute_opensmile_features(audio_path))
    except Exception:
        pass

    # -----------------------------------------------------------------------
    # 16. LUFS loudness (pyloudnorm)
    # -----------------------------------------------------------------------
    try:
        features.update(_compute_lufs_features(y, sr))
    except Exception:
        features["lufs_integrated"] = -70.0
        features["lufs_range"]      = 0.0
        features["lufs_std"]        = 0.0

    # Phonation efficiency — computed here so that lufs_integrated,
    # hnr, shimmer_local, and voiced_fraction are all available.
    features["phonation_efficiency"] = _compute_phonation_efficiency(features)

    # -----------------------------------------------------------------------
    # 17. VAD — speech ratio & segment statistics (webrtcvad)
    # -----------------------------------------------------------------------
    try:
        features.update(_compute_vad_features(y, sr))
    except Exception:
        features["vad_speech_ratio"]        = 0.0
        features["vad_num_segments"]        = 0.0
        features["vad_mean_segment_dur_ms"] = 0.0
        features["vad_std_segment_dur_ms"]  = 0.0

    # -----------------------------------------------------------------------
    # Sanitise: replace any remaining NaN/Inf with 0
    # -----------------------------------------------------------------------
    for k, v in list(features.items()):
        try:
            if not np.isfinite(v):
                features[k] = 0.0
        except Exception:
            features[k] = 0.0

    # Store audio path so heuristic_predict can call calculate_grbas directly
    features["_audio_path"] = audio_path

    return features


# ---------------------------------------------------------------------------
# Feature name list (fixed order for the ML models)
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # F0
    "f0_mean", "f0_std", "f0_min", "f0_max", "f0_range",
    "f0_median", "f0_iqr", "f0_kurtosis", "f0_skewness", "voiced_fraction",
    # Strain-specific Parselmouth features (Pitch + PointProcess)
    "n_voice_breaks", "degree_voice_breaks", "fraction_locally_unvoiced",
    "f0_cv", "n_periods", "mean_period", "stdev_period", "period_cv",
    # Jitter
    "jitter_local", "jitter_rap", "jitter_ppq5", "jitter_ddp",
    # Shimmer
    "shimmer_local", "shimmer_apq3", "shimmer_apq5",
    "shimmer_apq11", "shimmer_dda", "shimmer_db",
    # HNR
    "hnr",
    # Formants F1–F4
    *[f"f{i}_mean"      for i in range(1, 5)],
    *[f"f{i}_std"       for i in range(1, 5)],
    *[f"f{i}_bandwidth" for i in range(1, 5)],
    # Intensity
    "intensity_mean", "intensity_std", "intensity_min", "intensity_max",
    # CPP
    "cpp_mean", "cpp_std",
    # CPPS — smoothed CPP (gold-standard asthenia indicator)
    "cpps_mean", "cpps_std", "cpps_min", "cpps_p10",
    # Harmonic decay (asthenia)
    "harmonic_decay_slope", "harmonic_regularity",
    # Energy envelope dynamics (asthenia)
    "energy_env_mean", "energy_env_std", "energy_env_cv",
    "energy_env_slope", "energy_env_drop_rate",
    # Phonation efficiency (asthenia)
    "phonation_efficiency",
    # GNE
    "gne",
    # TEO
    "teo_mean", "teo_std", "teo_kurtosis", "teo_skewness",
    # Spectral
    "spectral_centroid_mean", "spectral_centroid_std",
    "spectral_bandwidth_mean", "spectral_rolloff_mean",
    "spectral_flatness_mean", "spectral_flatness_std",
    *[f"spectral_contrast_{b}" for b in range(7)],
    "spectral_tilt", "spectral_kurtosis", "spectral_skewness",
    "hf_energy_ratio",
    # ZCR & RMS
    "zcr_mean", "zcr_std", "rms_mean", "rms_std",
    # MFCCs + deltas
    *[f"mfcc_{i}_mean"    for i in range(13)],
    *[f"mfcc_{i}_std"     for i in range(13)],
    *[f"mfcc_d_{i}_mean"  for i in range(13)],
    *[f"mfcc_dd_{i}_mean" for i in range(13)],
    # Chroma
    "chroma_mean", "chroma_std", "chroma_energy",
    # LPC
    *[f"lpc_{i}" for i in range(12)],
    "lpc_residual_energy",
    # Entropy
    "sample_entropy", "app_entropy", "perm_entropy",
    "spectral_entropy", "svd_entropy",
    # Nonlinear
    "dfa_alpha", "hurst_exp",
    # LUFS loudness
    "lufs_integrated", "lufs_range", "lufs_std",
    # VAD
    "vad_speech_ratio", "vad_num_segments",
    "vad_mean_segment_dur_ms", "vad_std_segment_dur_ms",
    # openSMILE eGeMAPS (added dynamically — not listed here because column
    # names are discovered at runtime; the model uses only FEATURE_NAMES when
    # training so smile_ features are bonus context for the heuristic)
]


def _features_to_vector(features):
    return np.array([[features.get(n, 0.0) for n in FEATURE_NAMES]])


# ---------------------------------------------------------------------------
# Heuristic fallback (uses new features where available)
# ---------------------------------------------------------------------------

def _clamp(v, lo=0, hi=3):
    return max(lo, min(hi, v))


def _gender_blend(f0_mean):
    if f0_mean <= 0:   return 0.5
    if f0_mean < 150:  return 0.0
    if f0_mean > 200:  return 1.0
    return (f0_mean - 150) / 50.0


def calculate_grbas(audio_path):
    """Pure Praat/parselmouth GRBAS scoring from a sustained vowel /a/.

    Extracts all acoustic measures directly from the audio file using
    parselmouth (Praat) and maps them to the 0–3 GRBAS scale using
    established clinical thresholds.

    References:
        - Jitter/Shimmer norms: Teixeira & Fernandes 2014, Boersma 1993
        - HNR norms: Yumoto et al. 1982, Ferrand 2002
        - F0/Intensity norms: Baken & Orlikoff 2000
        - GRBAS scale: Hirano 1981

    Args:
        audio_path: Path to a .wav file (sustained vowel /a/).

    Returns:
        dict with keys G, R, B, A, S mapped to int scores 0–3.
    """
    # ------------------------------------------------------------------
    # 1. Load sound and create Praat objects
    # ------------------------------------------------------------------
    snd = parselmouth.Sound(audio_path)

    # Pitch object — fundamental frequency tracking
    pitch = call(snd, "To Pitch", 0.0, 75.0, 600.0)

    # PointProcess — glottal pulse detection (for jitter/shimmer)
    point_process = call(snd, "To PointProcess (periodic, cc)", 75.0, 600.0)

    # Harmonicity (HNR) object
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75.0, 0.1, 1.0)

    # Intensity object
    intensity = call(snd, "To Intensity", 75.0, 0.0, "yes")

    # ------------------------------------------------------------------
    # 2. Extract acoustic measures
    # ------------------------------------------------------------------

    # --- Jitter (local, %) ---
    # Cycle-to-cycle variation in fundamental frequency.
    # Clinical norm: < 1.04% (Teixeira 2014, Praat default threshold)
    jitter_local = call(point_process, "Get jitter (local)", 0.0, 0.0,
                        0.0001, 0.02, 1.3) * 100  # convert to %

    # --- Shimmer (local, %) ---
    # Cycle-to-cycle variation in amplitude.
    # Clinical norm: < 3.81% (Teixeira 2014, Praat default threshold)
    shimmer_local = call([snd, point_process], "Get shimmer (local)", 0.0, 0.0,
                         0.0001, 0.02, 1.3, 1.6) * 100  # convert to %

    # --- HNR (dB) ---
    # Harmonics-to-Noise Ratio: harmonic energy vs. noise.
    # Clinical norm: > 20 dB = healthy (Yumoto 1982)
    # < 20 dB = mild, < 15 dB = moderate, < 10 dB = severe dysphonia
    hnr = call(harmonicity, "Get mean", 0.0, 0.0)
    if np.isnan(hnr) or hnr < 0:
        hnr = 0.0

    # --- Pitch (F0) statistics ---
    f0_mean = call(pitch, "Get mean", 0.0, 0.0, "Hertz")
    f0_std  = call(pitch, "Get standard deviation", 0.0, 0.0, "Hertz")
    if np.isnan(f0_mean):
        f0_mean = 0.0
    if np.isnan(f0_std):
        f0_std = 0.0

    # Pitch periodicity: fraction of frames that are voiced
    n_frames = call(pitch, "Get number of frames")
    n_voiced = 0
    for i in range(1, n_frames + 1):
        val = call(pitch, "Get value in frame", i, "Hertz")
        if not np.isnan(val) and val > 0:
            n_voiced += 1
    voiced_fraction = n_voiced / max(n_frames, 1)

    # --- Intensity statistics ---
    int_mean = call(intensity, "Get mean", 0.0, 0.0, "dB")
    int_std  = call(intensity, "Get standard deviation", 0.0, 0.0)
    if np.isnan(int_mean):
        int_mean = 60.0
    if np.isnan(int_std):
        int_std = 0.0

    # ------------------------------------------------------------------
    # 3. GRBAS scoring with clinical thresholds
    # ------------------------------------------------------------------

    # === R (Roughness) ================================================
    # Primary indicator: Jitter (irregular vocal fold vibration)
    # Secondary: Shimmer (adds roughness perception)
    #
    # Jitter thresholds (Teixeira 2014, Boersma 1993):
    #   Normal:   < 1.04%
    #   Mild:     1.04% – 2.5%
    #   Moderate: 2.5%  – 4.0%
    #   Severe:   > 4.0%
    if jitter_local < 1.04:
        r_score = 0
    elif jitter_local < 2.5:
        r_score = 1
    elif jitter_local < 4.0:
        r_score = 2
    else:
        r_score = 3

    # Shimmer can push roughness up by 1 level if also abnormal
    # Shimmer thresholds: Normal < 3.81%, Moderate > 6%, Severe > 10%
    if shimmer_local > 6.0 and r_score < 3:
        r_score += 1
    elif shimmer_local > 10.0 and r_score < 3:
        r_score = min(r_score + 2, 3)

    # === B (Breathiness) ==============================================
    # Primary indicator: HNR (low HNR = more noise = more breathy)
    #
    # HNR thresholds (Yumoto 1982, Ferrand 2002):
    #   Normal:   > 20 dB
    #   Mild:     15 – 20 dB
    #   Moderate: 10 – 15 dB
    #   Severe:   < 10 dB
    if hnr >= 20:
        b_score = 0
    elif hnr >= 15:
        b_score = 1
    elif hnr >= 10:
        b_score = 2
    else:
        b_score = 3

    # Low voiced fraction indicates air leakage (breathy voice)
    # Normal voiced fraction for sustained /a/: > 90%
    if voiced_fraction < 0.75 and b_score < 3:
        b_score += 1

    # === A (Asthenia — weakness) ======================================
    # Indicators: low intensity, high pitch instability, low HNR
    # A weak voice has: quiet output, unstable F0, poor harmonic support
    #
    # Intensity thresholds (Baken & Orlikoff 2000):
    #   Normal sustained /a/: 75–85 dB SPL
    #   Weak voice: < 65 dB SPL
    a_score = 0

    # Low intensity = weak projection
    if int_mean < 55:
        a_score = 3
    elif int_mean < 60:
        a_score = 2
    elif int_mean < 65:
        a_score = 1

    # High intensity variation = unstable support
    # Normal int_std for sustained /a/: < 3 dB
    if int_std > 5.0 and a_score < 3:
        a_score += 1
    elif int_std > 3.0 and a_score < 3 and a_score > 0:
        a_score = max(a_score, 1)

    # Low HNR also contributes to weakness (incomplete closure)
    if hnr < 12 and a_score < 3:
        a_score = max(a_score, 2)

    # === S (Strain — effort/tension) ==================================
    # Indicators: pitch instability, high F0, intensity fluctuations
    # Strained voice: elevated/unstable F0, pressed phonation
    #
    # F0 standard deviation thresholds for sustained /a/:
    #   Normal:   < 3 Hz (very stable)
    #   Mild:     3 – 8 Hz
    #   Moderate: 8 – 15 Hz
    #   Severe:   > 15 Hz
    if f0_std < 3.0:
        s_score = 0
    elif f0_std < 8.0:
        s_score = 1
    elif f0_std < 15.0:
        s_score = 2
    else:
        s_score = 3

    # Elevated F0 suggests laryngeal tension
    # Male baseline ~120 Hz, Female ~210 Hz
    # If F0 is > 30% above expected range, add strain evidence
    if f0_mean > 0:
        blend = _gender_blend(f0_mean)
        baseline = 120 + blend * 90
        if f0_mean > baseline * 1.3 and s_score < 3:
            s_score += 1

    # High intensity std also suggests effortful, inconsistent phonation
    if int_std > 5.0 and s_score < 3:
        s_score = max(s_score, 1)

    # === G (Grade — overall severity) =================================
    # Composite measure based on Shimmer + HNR (as requested)
    # with consideration of all sub-scores
    #
    # G reflects the overall impression of voice abnormality.
    # Clinical approach: G >= max(R, B, A, S) typically
    # (Dejonckere 1996: G correlates with jitter + shimmer + HNR combined)
    #
    # Shimmer contribution: > 3.81% abnormal, > 10% severe
    # HNR contribution: < 20 dB abnormal, < 10 dB severe
    g_from_shimmer = 0
    if shimmer_local >= 10.0:
        g_from_shimmer = 3
    elif shimmer_local >= 6.0:
        g_from_shimmer = 2
    elif shimmer_local >= 3.81:
        g_from_shimmer = 1

    g_from_hnr = 0
    if hnr < 10:
        g_from_hnr = 3
    elif hnr < 15:
        g_from_hnr = 2
    elif hnr < 20:
        g_from_hnr = 1

    # G = max of (shimmer-based, HNR-based, and highest sub-score)
    # This ensures G is never lower than the worst dimension
    g_acoustic = max(g_from_shimmer, g_from_hnr)
    g_subscores = max(r_score, b_score, a_score, s_score)
    g_score = _clamp(max(g_acoustic, g_subscores), 0, 3)

    return {
        "G": g_score,
        "R": r_score,
        "B": b_score,
        "A": a_score,
        "S": s_score,
    }


def heuristic_predict(features):
    """Legacy wrapper — delegates to calculate_grbas if audio_path is available,
    otherwise falls back to feature-based estimation."""
    # If called with audio_path in features (set by extract_features)
    audio_path = features.get("_audio_path")
    if audio_path and os.path.exists(audio_path):
        return calculate_grbas(audio_path)

    # Fallback: simple feature-based estimation for when no audio file is available
    jitter  = features.get("jitter_local", 0) * 100
    shimmer = features.get("shimmer_local", 0) * 100
    hnr     = features.get("hnr", 25)
    f0_std  = features.get("f0_std", 0)
    int_std = features.get("intensity_std", 0)

    r = 0 if jitter < 1.04 else (1 if jitter < 2.5 else (2 if jitter < 4.0 else 3))
    if shimmer > 6.0 and r < 3: r += 1
    b = 0 if hnr >= 20 else (1 if hnr >= 15 else (2 if hnr >= 10 else 3))
    a = 0
    if int_std > 5.0: a = 2
    elif int_std > 3.0: a = 1
    if hnr < 12: a = max(a, 2)
    s = 0 if f0_std < 3 else (1 if f0_std < 8 else (2 if f0_std < 15 else 3))
    g = max(r, b, a, s)
    shimmer_g = 0 if shimmer < 3.81 else (1 if shimmer < 6 else (2 if shimmer < 10 else 3))
    hnr_g = 0 if hnr >= 20 else (1 if hnr >= 15 else (2 if hnr >= 10 else 3))
    g = _clamp(max(g, shimmer_g, hnr_g), 0, 3)

    return {"G": g, "R": r, "B": b, "A": a, "S": s}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def _build_model(model_type):
    """Return a fresh sklearn Pipeline for the given model type."""
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
        "svm": SVC(
            kernel="rbf", C=10, gamma="scale",
            class_weight="balanced", probability=True, random_state=42,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42,
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1,
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42, n_jobs=-1,
            verbose=-1,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), activation="relu",
            max_iter=500, early_stopping=True, random_state=42,
        ),
    }

    if model_type == "ensemble":
        estimators = [
            ("rf",  Pipeline([("sc", StandardScaler()), ("clf", models["random_forest"])])),
            ("gbt", Pipeline([("sc", StandardScaler()), ("clf", models["gradient_boosting"])])),
            ("xgb", Pipeline([("sc", StandardScaler()), ("clf", models["xgboost"])])),
            ("lgb", Pipeline([("sc", StandardScaler()), ("clf", models["lightgbm"])])),
        ]
        return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)

    clf = models.get(model_type)
    if clf is None:
        raise ValueError(f"Unknown model_type '{model_type}'. "
                         f"Choose from: random_forest, svm, gradient_boosting, "
                         f"xgboost, lightgbm, mlp, ensemble")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


# ---------------------------------------------------------------------------
# GRBASClassifier
# ---------------------------------------------------------------------------

class GRBASClassifier:
    """GRBAS voice quality classifier.

    Trains one model per GRBAS dimension. Falls back to heuristics when
    no trained model is available.

    Args:
        model_type: One of random_forest | svm | gradient_boosting |
                    xgboost | lightgbm | mlp | ensemble (default)
    """

    def __init__(self, model_type="ensemble"):
        self.model_type = model_type
        self.models     = {}
        self.is_trained = False

    def fit(self, X, y_dict):
        """Train one model per GRBAS dimension.

        Args:
            X:      np.ndarray of shape (n_samples, n_features)
            y_dict: dict with keys G,R,B,A,S, each a 1-D array of int labels (1–5)
        """
        self.models = {}
        for dim in "GRBAS":
            model = _build_model(self.model_type)
            model.fit(X, y_dict[dim])
            self.models[dim] = model
        self.is_trained = True

    def predict(self, features):
        """Predict GRBAS scores from a feature dict.

        Returns dict with keys G,R,B,A,S mapped to int scores 1–5.
        Falls back to heuristic scoring if not trained.
        """
        if not self.is_trained:
            return heuristic_predict(features)

        vec    = _features_to_vector(features)
        scores = {}
        for dim in "GRBAS":
            pred = self.models[dim].predict(vec)[0]
            scores[dim] = int(_clamp(pred, 0, 3))
        return scores

    def save(self, path):
        joblib.dump(
            {"models": self.models, "is_trained": self.is_trained,
             "model_type": self.model_type},
            path,
        )

    def load(self, path):
        if os.path.exists(path):
            data            = joblib.load(path)
            self.models     = data["models"]
            self.is_trained = data["is_trained"]
            self.model_type = data.get("model_type", "ensemble")
