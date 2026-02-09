"""GRBAS voice quality analysis model.

Extracts acoustic features using parselmouth (Praat) and librosa,
then maps them to GRBAS scores (0-3 scale) using either a trained
RandomForest model or calibrated heuristic thresholds.
"""

import numpy as np
import parselmouth
from parselmouth.praat import call
import librosa
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def extract_features(audio_path):
    """Extract acoustic features from an audio file.

    Returns a dict of features used for GRBAS scoring.
    """
    snd = parselmouth.Sound(audio_path)
    y, sr = librosa.load(audio_path, sr=None)

    features = {}

    # --- F0 (fundamental frequency) ---
    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    f0_values = []
    for i in range(call(pitch, "Get number of frames")):
        f0 = call(pitch, "Get value in frame", i + 1, "Hertz")
        if f0 == f0:  # not NaN
            f0_values.append(f0)

    if f0_values:
        f0_arr = np.array(f0_values)
        features["f0_mean"] = float(np.mean(f0_arr))
        features["f0_std"] = float(np.std(f0_arr))
        features["f0_min"] = float(np.min(f0_arr))
        features["f0_max"] = float(np.max(f0_arr))
        features["f0_range"] = float(np.ptp(f0_arr))
    else:
        features["f0_mean"] = 0.0
        features["f0_std"] = 0.0
        features["f0_min"] = 0.0
        features["f0_max"] = 0.0
        features["f0_range"] = 0.0

    # --- Jitter & Shimmer ---
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)

    features["jitter_local"] = call(
        point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
    )
    features["jitter_rap"] = call(
        point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3
    )
    features["jitter_ppq5"] = call(
        point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3
    )

    features["shimmer_local"] = call(
        [snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    features["shimmer_apq3"] = call(
        [snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )
    features["shimmer_apq5"] = call(
        [snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6
    )

    # --- HNR (Harmonics-to-Noise Ratio) ---
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    features["hnr"] = call(harmonicity, "Get mean", 0, 0)

    # --- Formants ---
    formant = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
    features["f1_mean"] = call(formant, "Get mean", 1, 0, 0, "Hertz")
    features["f2_mean"] = call(formant, "Get mean", 2, 0, 0, "Hertz")
    features["f3_mean"] = call(formant, "Get mean", 3, 0, 0, "Hertz")

    # --- Spectral features (librosa) ---
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
    features["spectral_centroid_std"] = float(np.std(spectral_centroid))

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))

    # Spectral tilt (slope of log spectrum)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    log_S = np.mean(20 * np.log10(S + 1e-10), axis=1)
    if len(freqs) > 1:
        coeffs = np.polyfit(freqs, log_S, 1)
        features["spectral_tilt"] = float(coeffs[0])
    else:
        features["spectral_tilt"] = 0.0

    # High-frequency energy ratio (above 4kHz vs total)
    freq_mask = freqs >= 4000
    if np.any(freq_mask):
        hf_energy = np.mean(S[freq_mask, :] ** 2)
        total_energy = np.mean(S ** 2)
        features["hf_energy_ratio"] = float(hf_energy / (total_energy + 1e-10))
    else:
        features["hf_energy_ratio"] = 0.0

    # --- MFCCs ---
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f"mfcc_{i}_mean"] = float(np.mean(mfccs[i]))
        features[f"mfcc_{i}_std"] = float(np.std(mfccs[i]))

    # --- Intensity ---
    intensity = call(snd, "To Intensity", 75, 0, "yes")
    features["intensity_mean"] = call(intensity, "Get mean", 0, 0, "dB")
    features["intensity_std"] = call(intensity, "Get standard deviation", 0, 0)

    # Replace any NaN values with 0
    for k, v in features.items():
        if v != v:  # NaN check
            features[k] = 0.0

    return features


# Feature names in fixed order for the ML model
FEATURE_NAMES = [
    "f0_mean", "f0_std", "f0_range",
    "jitter_local", "jitter_rap", "jitter_ppq5",
    "shimmer_local", "shimmer_apq3", "shimmer_apq5",
    "hnr", "spectral_centroid_mean", "spectral_tilt",
    "spectral_bandwidth_mean", "spectral_rolloff_mean",
    "hf_energy_ratio", "intensity_mean", "intensity_std",
    "f1_mean", "f2_mean", "f3_mean",
]


def _features_to_vector(features):
    """Convert feature dict to a numpy array in the expected order."""
    return np.array([[features.get(name, 0.0) for name in FEATURE_NAMES]])


def _clamp(value, lo=0, hi=3):
    return max(lo, min(hi, value))


def heuristic_predict(features):
    """Rule-based GRBAS scoring using calibrated acoustic thresholds.

    Each dimension scored 0 (normal) to 3 (severe).
    """
    scores = {}

    # --- R (Roughness): jitter, shimmer, subharmonic content ---
    jitter = features.get("jitter_local", 0) * 100  # percentage
    shimmer = features.get("shimmer_local", 0) * 100
    r_score = 0
    if jitter > 1.0 or shimmer > 3.0:
        r_score = 1
    if jitter > 2.0 or shimmer > 6.0:
        r_score = 2
    if jitter > 4.0 or shimmer > 10.0:
        r_score = 3
    scores["R"] = r_score

    # --- B (Breathiness): low HNR, steep spectral tilt ---
    hnr = features.get("hnr", 25)
    tilt = features.get("spectral_tilt", 0)
    b_score = 0
    if hnr < 20:
        b_score = 1
    if hnr < 15:
        b_score = 2
    if hnr < 10:
        b_score = 3
    # Steep negative tilt indicates more breathiness
    if tilt < -0.02:
        b_score = min(3, b_score + 1)
    scores["B"] = b_score

    # --- A (Asthenia / weakness): low intensity variation, narrow F0 range ---
    f0_range = features.get("f0_range", 0)
    intensity_std = features.get("intensity_std", 0)
    a_score = 0
    if f0_range < 20 or intensity_std < 3:
        a_score = 1
    if f0_range < 10 or intensity_std < 2:
        a_score = 2
    if f0_range < 5 and intensity_std < 1.5:
        a_score = 3
    scores["A"] = a_score

    # --- S (Strain): elevated F0, high spectral centroid, HF energy ---
    f0_mean = features.get("f0_mean", 150)
    centroid = features.get("spectral_centroid_mean", 1500)
    hf_ratio = features.get("hf_energy_ratio", 0)
    s_score = 0
    if f0_mean > 250 or centroid > 2500 or hf_ratio > 0.15:
        s_score = 1
    if f0_mean > 300 or centroid > 3000 or hf_ratio > 0.25:
        s_score = 2
    if f0_mean > 350 and centroid > 3500:
        s_score = 3
    scores["S"] = s_score

    # --- G (Grade): composite severity ---
    g_score = _clamp(round(
        0.3 * scores["R"] + 0.3 * scores["B"] + 0.2 * scores["A"] + 0.2 * scores["S"]
    ))
    # G should be at least as high as any individual score - 1
    g_score = max(g_score, max(scores["R"], scores["B"], scores["A"], scores["S"]) - 1)
    scores["G"] = _clamp(g_score)

    return scores


class GRBASClassifier:
    """GRBAS voice quality classifier.

    Uses a RandomForest pipeline when trained, or falls back
    to heuristic scoring.
    """

    def __init__(self):
        self.models = {}  # one per GRBAS dimension
        self.is_trained = False

    def fit(self, X, y_dict):
        """Train the model.

        Args:
            X: array of shape (n_samples, n_features)
            y_dict: dict with keys G,R,B,A,S each containing array of labels
        """
        for dim in "GRBAS":
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42
                )),
            ])
            pipe.fit(X, y_dict[dim])
            self.models[dim] = pipe
        self.is_trained = True

    def predict(self, features):
        """Predict GRBAS scores from a feature dict.

        Falls back to heuristic if not trained.
        """
        if not self.is_trained:
            return heuristic_predict(features)

        vec = _features_to_vector(features)
        scores = {}
        for dim in "GRBAS":
            pred = self.models[dim].predict(vec)[0]
            scores[dim] = int(_clamp(pred))
        return scores

    def save(self, path):
        joblib.dump({"models": self.models, "is_trained": self.is_trained}, path)

    def load(self, path):
        if os.path.exists(path):
            data = joblib.load(path)
            self.models = data["models"]
            self.is_trained = data["is_trained"]
