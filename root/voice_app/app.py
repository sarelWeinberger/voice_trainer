"""Flask backend for Voice Training App with GRBAS analysis."""

import os
import tempfile
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from grbas_model import extract_features, GRBASClassifier
from auth_middleware import optional_token, verify_token, require_admin, db
from google.cloud.firestore import SERVER_TIMESTAMP

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

classifier = GRBASClassifier()
MODEL_PATH = os.path.join(os.path.dirname(__file__), "grbas_model.joblib")
if os.path.exists(MODEL_PATH):
    classifier.load(MODEL_PATH)


# ---- Pages ----

@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/admin")
def admin_page():
    return send_from_directory("static", "admin.html")


# ---- Voice Analysis ----

@app.route("/api/analyze", methods=["POST"])
@optional_token
def analyze():
    """Receive audio blob, extract features, return GRBAS scores."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        audio_data = audio_file.read()
        tmp.write(audio_data)

    try:
        tmp_path = _ensure_wav(tmp_path, audio_data)

        features = extract_features(tmp_path)
        scores = classifier.predict(features)

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

        response_data = {
            "scores": scores,
            "features": key_features,
            "model_type": "trained" if classifier.is_trained else "heuristic",
        }

        # Save to Firestore if user is authenticated
        if request.uid:
            try:
                db.collection("users").document(request.uid) \
                  .collection("analyses").add({
                    "timestamp": SERVER_TIMESTAMP,
                    "scores": scores,
                    "features": key_features,
                    "modelType": response_data["model_type"],
                })
            except Exception as e:
                app.logger.error(f"Firestore save failed: {e}")

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---- User Profile ----

@app.route("/api/user/profile", methods=["POST"])
@verify_token
def create_or_update_profile():
    """Create or update user profile in Firestore on login."""
    user_ref = db.collection("users").document(request.uid)
    doc = user_ref.get()

    if doc.exists:
        user_ref.update({"lastLoginAt": SERVER_TIMESTAMP})
    else:
        user_ref.set({
            "email": request.firebase_user.get("email", ""),
            "displayName": "",
            "role": "user",
            "createdAt": SERVER_TIMESTAMP,
            "lastLoginAt": SERVER_TIMESTAMP,
        })

    return jsonify({"status": "ok"})


@app.route("/api/user/role", methods=["GET"])
@verify_token
def get_user_role():
    """Return current user's role."""
    user_doc = db.collection("users").document(request.uid).get()
    if user_doc.exists:
        return jsonify({"role": user_doc.to_dict().get("role", "user")})
    return jsonify({"role": "user"})


@app.route("/api/history", methods=["GET"])
@verify_token
def get_history():
    """Return the authenticated user's analysis history."""
    analyses_ref = db.collection("users").document(request.uid) \
                     .collection("analyses") \
                     .order_by("timestamp", direction="DESCENDING") \
                     .limit(50)
    docs = analyses_ref.stream()

    history = []
    for doc in docs:
        data = doc.to_dict()
        ts = data.get("timestamp")
        history.append({
            "id": doc.id,
            "timestamp": ts.isoformat() if ts else None,
            "scores": data.get("scores", {}),
            "features": data.get("features", {}),
            "modelType": data.get("modelType", "unknown"),
        })

    return jsonify({"history": history})


# ---- Admin ----

@app.route("/api/admin/users", methods=["GET"])
@require_admin
def list_users():
    """Return all users with their latest scores."""
    users_ref = db.collection("users").stream()
    result = []

    for user_doc in users_ref:
        user_data = user_doc.to_dict()
        uid = user_doc.id

        latest = db.collection("users").document(uid) \
                   .collection("analyses") \
                   .order_by("timestamp", direction="DESCENDING") \
                   .limit(1).stream()

        latest_analysis = None
        for doc in latest:
            d = doc.to_dict()
            ts = d.get("timestamp")
            latest_analysis = {
                "timestamp": ts.isoformat() if ts else None,
                "scores": d.get("scores", {}),
            }

        result.append({
            "uid": uid,
            "email": user_data.get("email", ""),
            "role": user_data.get("role", "user"),
            "createdAt": user_data.get("createdAt").isoformat()
                         if user_data.get("createdAt") else None,
            "latestAnalysis": latest_analysis,
        })

    return jsonify({"users": result})


@app.route("/api/admin/users/<uid>/analyses", methods=["GET"])
@require_admin
def get_user_analyses(uid):
    """Return all analyses for a specific user (admin only)."""
    analyses_ref = db.collection("users").document(uid) \
                     .collection("analyses") \
                     .order_by("timestamp", direction="DESCENDING") \
                     .limit(100).stream()

    history = []
    for doc in analyses_ref:
        data = doc.to_dict()
        ts = data.get("timestamp")
        history.append({
            "id": doc.id,
            "timestamp": ts.isoformat() if ts else None,
            "scores": data.get("scores", {}),
            "features": data.get("features", {}),
        })

    return jsonify({"analyses": history})


@app.route("/api/admin/users/<uid>/role", methods=["PUT"])
@require_admin
def set_user_role(uid):
    """Set a user's role (admin only)."""
    data = request.get_json()
    role = data.get("role")
    if role not in ("user", "admin"):
        return jsonify({"error": "Invalid role"}), 400

    db.collection("users").document(uid).update({"role": role})
    return jsonify({"status": "ok"})


# ---- Helpers ----

def _ensure_wav(path, raw_data):
    """Convert audio to WAV if necessary using soundfile."""
    import soundfile as sf

    try:
        data, samplerate = sf.read(path)
        wav_path = path + ".converted.wav"
        sf.write(wav_path, data, samplerate, format="WAV", subtype="PCM_16")
        os.unlink(path)
        return wav_path
    except Exception:
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
