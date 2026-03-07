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


def heuristic_predict(features):
    """Rule-based GRBAS scoring using calibrated acoustic thresholds."""
    scores = {}

    f0_mean   = features.get("f0_mean", 0)
    blend     = _gender_blend(f0_mean)
    jitter    = features.get("jitter_local", 0) * 100
    shimmer   = features.get("shimmer_local", 0) * 100
    hnr       = features.get("hnr", 25)
    tilt      = features.get("spectral_tilt", 0)
    hf_ratio  = features.get("hf_energy_ratio", 0)
    cpp       = features.get("cpp_mean", 5.0)
    gne       = features.get("gne", 0.7)
    s_entropy = features.get("sample_entropy", 0.5)

    # --- R (Roughness): jitter, shimmer, HNR, sample entropy ---
    # Scale: 0=normal, 1=slight, 2=moderate, 3=severe
    r_val = 0.0
    if jitter > 0.5:
        r_val += min((jitter - 0.5) / 3.5, 1.0)
    if shimmer > 2.0:
        r_val += min((shimmer - 2.0) / 8.0, 1.0)
    if hnr < 20:
        r_val += min((20 - hnr) / 15.0, 1.0) * 0.5
    if s_entropy > 1.5:
        r_val += min((s_entropy - 1.5) / 1.5, 1.0) * 0.4
    scores["R"] = _clamp(int(min(r_val / 0.8, 3)))

    # --- B (Breathiness): HNR, CPP, GNE, spectral tilt ---
    b_val = 0.0
    if hnr < 22:
        b_val += min((22 - hnr) / 16.0, 1.0)
    if cpp < 8.0:
        b_val += min((8.0 - cpp) / 8.0, 1.0) * 0.8   # CPP is primary indicator
    if gne < 0.5:
        b_val += min((0.5 - gne) / 0.5, 1.0) * 0.5
    if tilt < -0.01:
        b_val += min((-0.01 - tilt) / 0.04, 1.0) * 0.3
    if hf_ratio < 0.05:
        b_val += 0.15
    scores["B"] = _clamp(int(min(b_val / 0.47, 3)))

    # --- A (Asthenia): intensity std, F0 std, shimmer, HNR ---
    intensity_std = features.get("intensity_std", 5)
    f0_std        = features.get("f0_std", 0)
    a_val = 0.0
    if intensity_std < 6:
        a_val += min((6 - intensity_std) / 5.0, 1.0)
    if f0_std < 10:
        a_val += min((10 - f0_std) / 9.0, 1.0) * 0.5
    if shimmer > 3.0:
        a_val += min((shimmer - 3.0) / 7.0, 1.0) * 0.4
    if hnr < 18:
        a_val += min((18 - hnr) / 13.0, 1.0) * 0.4
    scores["A"] = _clamp(int(min(a_val / 0.67, 3)))

    # --- S (Strain): multi-library evidence accumulation ---
    #
    # Strain = hyperfunctional voice production: too much vocal effort/tension.
    # Clinical correlates: voice breaks, elevated pitch, pressed phonation,
    # irregular vibration, high subglottal pressure.
    #
    # Each group produces a 0–1 evidence score; final weighted sum → 0–3.

    # ── Group 1: Voice continuity  (Parselmouth Pitch) ──────────────── w=0.30
    n_vb  = features.get("n_voice_breaks", 0)
    dvb   = features.get("degree_voice_breaks", 0)
    flu   = features.get("fraction_locally_unvoiced", 0)
    g1  = min(n_vb / 5.0, 1.0) * 0.45
    g1 += min(dvb / 0.30, 1.0) * 0.35
    g1 += min(max(flu - 0.08, 0) / 0.40, 1.0) * 0.20
    g1  = min(g1, 1.0)

    # ── Group 2: F0 elevation & instability  (Parselmouth Pitch) ─────── w=0.25
    baseline_f0  = 120 + blend * 90
    f0_deviation = max(0, f0_mean - baseline_f0) / (baseline_f0 + 1e-6)
    f0_cv        = features.get("f0_cv", 0)
    f0_max_val   = features.get("f0_max", 0)
    f0_max_dev   = max(0, f0_max_val - (baseline_f0 * 1.5)) / (baseline_f0 + 1e-6)
    g2  = min(max(f0_deviation - 0.20, 0) / 0.80, 1.0) * 0.45
    g2 += min(max(f0_cv - 0.08, 0) / 0.35, 1.0) * 0.35
    g2 += min(f0_max_dev / 0.50, 1.0) * 0.20
    g2  = min(g2, 1.0)

    # ── Group 3: Period & amplitude irregularity  (Parselmouth PointProcess) ── w=0.20
    period_cv  = features.get("period_cv", 0)
    jit_ddp    = features.get("jitter_ddp", 0) * 100   # DDP: most sensitive jitter
    shim_db    = features.get("shimmer_db", 0)
    g3  = min(max(period_cv - 0.01, 0) / 0.20, 1.0) * 0.40
    g3 += min(max(jit_ddp - 0.8, 0) / 6.0, 1.0) * 0.35
    g3 += min(max(shim_db - 0.25, 0) / 1.20, 1.0) * 0.25
    g3  = min(g3, 1.0)

    # ── Group 4: Vocal effort energy  (scipy TEO + pyloudnorm) ──────── w=0.15
    teo_mean  = features.get("teo_mean", 0)
    teo_std   = features.get("teo_std", 0)
    lufs      = features.get("lufs_integrated", -70)
    # TEO (Teager Energy Operator): high mean + high std → pressed phonation
    teo_score = min(teo_mean / 1e-3, 1.0) * 0.50 + min(teo_std / 5e-4, 1.0) * 0.20
    # Integrated loudness: very loud (> -14 LUFS) may indicate strain
    lufs_score = min(max(lufs - (-14), 0) / 10.0, 1.0) * 0.30
    g4  = min(teo_score + lufs_score, 1.0)

    # ── Group 5: Pressed-phonation spectrum  (librosa + scipy CPP) ───── w=0.10
    # Low spectral flatness → more tonal/pressed (not breathy noise)
    sf_mean  = features.get("spectral_flatness_mean", 0.1)
    # Low CPP in strain (insufficient glottal closure efficiency)
    cpp_val  = features.get("cpp_mean", 10.0)
    # High ZCR variability → irregular closure
    zcr_std  = features.get("zcr_std", 0)
    g5  = min(max(0.15 - sf_mean, 0) / 0.15, 1.0) * 0.40
    g5 += min(max(10.0 - cpp_val, 0) / 10.0, 1.0) * 0.35
    g5 += min(zcr_std / 0.05, 1.0) * 0.25
    g5  = min(g5, 1.0)

    # ── Group 6: Formant tension  (Parselmouth Formant) ──────────────── w=0.00 bonus
    # Wide F1 bandwidth → tense/pressed phonation (bonus on top)
    f1_bw = features.get("f1_bandwidth", 0)
    g6 = min(max(f1_bw - 100, 0) / 500, 1.0)

    # ── Weighted sum → 0–3 ───────────────────────────────────────────────────
    s_raw = (
        g1 * 0.30 +
        g2 * 0.25 +
        g3 * 0.20 +
        g4 * 0.15 +
        g5 * 0.10 +
        g6 * 0.05   # small bonus
    )
    scores["S"] = _clamp(int(s_raw / 0.30 * 3))

    # --- G (Grade): composite ---
    sub       = [scores["R"], scores["B"], scores["A"], scores["S"]]
    w_avg     = 0.30*scores["R"] + 0.30*scores["B"] + 0.20*scores["A"] + 0.20*scores["S"]
    scores["G"] = _clamp(round(0.6 * max(sub) + 0.4 * w_avg))

    return scores


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
