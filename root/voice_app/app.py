"""Flask backend for Voice Training App with GRBAS analysis."""

import os
import tempfile
import wave
import io
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from grbas_model import extract_features, GRBASClassifier

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

classifier = GRBASClassifier()
MODEL_PATH = os.path.join(os.path.dirname(__file__), "grbas_model.joblib")
if os.path.exists(MODEL_PATH):
    classifier.load(MODEL_PATH)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Receive audio blob, extract features, return GRBAS scores."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]

    # Save to a temp WAV file for processing
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        audio_data = audio_file.read()

        # If the browser sends raw PCM in a webm container, we need to handle it.
        # Try to write directly first; if it's already WAV, great.
        tmp.write(audio_data)

    try:
        # Try to convert if needed (handle webm/ogg from browser)
        tmp_path = _ensure_wav(tmp_path, audio_data)

        features = extract_features(tmp_path)
        scores = classifier.predict(features)

        # Select key features to send to frontend
        key_features = {
            "f0_mean": round(features.get("f0_mean", 0), 1),
            "f0_std": round(features.get("f0_std", 0), 1),
            "f0_range": round(features.get("f0_range", 0), 1),
            "jitter_percent": round(features.get("jitter_local", 0) * 100, 3),
            "shimmer_percent": round(features.get("shimmer_local", 0) * 100, 3),
            "hnr": round(features.get("hnr", 0), 1),
            "spectral_centroid": round(features.get("spectral_centroid_mean", 0), 1),
            "spectral_tilt": round(features.get("spectral_tilt", 0), 6),
            "hf_energy_ratio": round(features.get("hf_energy_ratio", 0), 4),
            "intensity_mean": round(features.get("intensity_mean", 0), 1),
            "f1": round(features.get("f1_mean", 0), 1),
            "f2": round(features.get("f2_mean", 0), 1),
            "f3": round(features.get("f3_mean", 0), 1),
        }

        return jsonify({
            "scores": scores,
            "features": key_features,
            "model_type": "trained" if classifier.is_trained else "heuristic",
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _ensure_wav(path, raw_data):
    """Convert audio to WAV if necessary using scipy/soundfile."""
    import soundfile as sf

    try:
        # Check if it's already a valid WAV/audio file soundfile can read
        data, samplerate = sf.read(path)
        # Re-write as proper WAV to ensure compatibility with parselmouth
        wav_path = path + ".converted.wav"
        sf.write(wav_path, data, samplerate, format="WAV", subtype="PCM_16")
        os.unlink(path)
        return wav_path
    except Exception:
        # If soundfile can't read it, try treating as raw PCM
        # (fallback for unusual browser formats)
        try:
            os.unlink(path)
        except OSError:
            pass
        raise ValueError(
            "Could not process audio file. Please ensure your browser "
            "records in a supported format (WAV, OGG, WebM)."
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
