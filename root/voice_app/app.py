"""Flask backend for Voice Training App with GRBAS analysis.

Authentication is handled by JWT tokens (see auth_middleware.py).
Data is stored in SQLite via SQLAlchemy (see database.py).
"""

import json
import os
import tempfile
from datetime import datetime

import numpy as np
from flask import Flask, g, jsonify, request, send_from_directory
from flask_cors import CORS

from auth_middleware import (
    create_token,
    hash_password,
    log_activity,
    optional_token,
    require_admin,
    verify_password,
    verify_token,
)
from database import ActivityLog, Analysis, SessionLocal, User, init_db
from grbas_model import GRBASClassifier, extract_features
from recommendations import recommend_by_grbas, flatten_recommendations

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# Initialise database tables on startup (idempotent).
init_db()

# Load GRBAS model if a pre-trained file exists.
classifier = GRBASClassifier()
MODEL_PATH = os.path.join(os.path.dirname(__file__), "grbas_model.joblib")
if os.path.exists(MODEL_PATH):
    classifier.load(MODEL_PATH)


# ===========================================================================
# Pages
# ===========================================================================


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/admin")
def admin_page():
    return send_from_directory("static", "admin.html")


# ===========================================================================
# Auth – Registration & Login
# ===========================================================================


@app.route("/api/auth/register", methods=["POST"])
def register():
    """Register a new user account.

    Expects JSON: { "email": "...", "password": "..." }
    Passwords are hashed with bcrypt before storage; the plain text is never
    persisted.  Returns a JWT on success so the client is immediately logged in.
    """
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    db = SessionLocal()
    try:
        if db.query(User).filter(User.email == email).first():
            return jsonify({"error": "An account with that email already exists"}), 409

        user = User(
            email=email,
            password_hash=hash_password(password),
            role="user",
            created_at=datetime.utcnow(),
        )
        db.add(user)
        db.commit()
        db.refresh(user)

        log_activity(db, "register", user_id=user.id, details={"email": email})

        token = create_token(user)
        return jsonify({"token": token, "role": user.role, "email": user.email}), 201
    finally:
        db.close()


@app.route("/api/auth/login", methods=["POST"])
def login():
    """Authenticate a user and return a JWT.

    Expects JSON: { "email": "...", "password": "..." }
    bcrypt.checkpw is used for constant-time comparison, preventing timing
    attacks that could reveal whether an email address is registered.
    """
    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()

        # Intentionally return the same error message whether the email is
        # unknown or the password is wrong, to prevent user enumeration.
        if user is None or not user.is_active or not verify_password(password, user.password_hash):
            log_activity(db, "login_failed", details={"email": email})
            return jsonify({"error": "Invalid email or password"}), 401

        user.last_login_at = datetime.utcnow()
        db.commit()

        log_activity(db, "login", user_id=user.id, details={"email": email})

        token = create_token(user)
        return jsonify({"token": token, "role": user.role, "email": user.email})
    finally:
        db.close()


# ===========================================================================
# User – Profile & Role
# ===========================================================================


@app.route("/api/user/profile", methods=["GET"])
@verify_token
def get_profile():
    """Return the authenticated user's profile."""
    u = g.user
    return jsonify({
        "id": u.id,
        "email": u.email,
        "display_name": u.display_name,
        "role": u.role,
        "created_at": u.created_at.isoformat() if u.created_at else None,
        "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
    })


@app.route("/api/user/role", methods=["GET"])
@verify_token
def get_user_role():
    """Return the authenticated user's current role."""
    log_activity(g.db, "view_role", user_id=g.user.id, resource="/api/user/role")
    return jsonify({"role": g.user.role})


# ===========================================================================
# Voice Analysis
# ===========================================================================


@app.route("/api/analyze", methods=["POST"])
@optional_token
def analyze():
    """Receive an audio file, extract features, and return GRBAS scores.

    The analysis result is saved to the database when the user is authenticated.
    Anonymous requests still receive results but nothing is persisted.
    """
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

        recs = recommend_by_grbas(scores)
        recommendations = flatten_recommendations(recs)

        response_data = {
            "scores": scores,
            "features": key_features,
            "model_type": "trained" if classifier.is_trained else "heuristic",
            "recommendations": recommendations,
        }

        # Persist result and log activity if authenticated.
        if g.user:
            analysis = Analysis(
                user_id=g.user.id,
                scores=json.dumps(scores),
                features=json.dumps(key_features),
                model_type=response_data["model_type"],
                timestamp=datetime.utcnow(),
            )
            g.db.add(analysis)
            g.db.commit()
            log_activity(
                g.db, "analyze",
                user_id=g.user.id,
                details={"model_type": response_data["model_type"]},
                resource="/api/analyze",
            )

        return jsonify(response_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.route("/api/history", methods=["GET"])
@verify_token
def get_history():
    """Return the authenticated user's analysis history (most recent first)."""
    log_activity(g.db, "view_history", user_id=g.user.id, resource="/api/history")

    rows = (
        g.db.query(Analysis)
        .filter(Analysis.user_id == g.user.id)
        .order_by(Analysis.timestamp.desc())
        .limit(50)
        .all()
    )

    history = [
        {
            "id": row.id,
            "timestamp": row.timestamp.isoformat(),
            "scores": row.scores_dict(),
            "features": row.features_dict(),
            "model_type": row.model_type,
        }
        for row in rows
    ]
    return jsonify({"history": history})


# ===========================================================================
# Admin – User Management
# ===========================================================================


@app.route("/api/admin/users", methods=["GET"])
@require_admin
def list_users():
    """Return all users with their most recent analysis scores."""
    log_activity(g.db, "admin_list_users", user_id=g.user.id, resource="/api/admin/users")

    users = g.db.query(User).order_by(User.created_at.desc()).all()
    result = []

    for u in users:
        latest = (
            g.db.query(Analysis)
            .filter(Analysis.user_id == u.id)
            .order_by(Analysis.timestamp.desc())
            .first()
        )

        latest_analysis = None
        if latest:
            latest_analysis = {
                "timestamp": latest.timestamp.isoformat(),
                "scores": latest.scores_dict(),
            }

        result.append({
            "id": u.id,
            "email": u.email,
            "display_name": u.display_name,
            "role": u.role,
            "is_active": u.is_active,
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "last_login_at": u.last_login_at.isoformat() if u.last_login_at else None,
            "latest_analysis": latest_analysis,
        })

    return jsonify({"users": result})


@app.route("/api/admin/users/<int:uid>/analyses", methods=["GET"])
@require_admin
def get_user_analyses(uid):
    """Return all analyses for a specific user (admin only)."""
    log_activity(
        g.db, "admin_view_user_analyses",
        user_id=g.user.id,
        resource=f"/api/admin/users/{uid}/analyses",
        details={"target_user_id": uid},
    )

    rows = (
        g.db.query(Analysis)
        .filter(Analysis.user_id == uid)
        .order_by(Analysis.timestamp.desc())
        .limit(100)
        .all()
    )

    return jsonify({
        "analyses": [
            {
                "id": row.id,
                "timestamp": row.timestamp.isoformat(),
                "scores": row.scores_dict(),
                "features": row.features_dict(),
                "model_type": row.model_type,
            }
            for row in rows
        ]
    })


@app.route("/api/admin/users/<int:uid>/role", methods=["PUT"])
@require_admin
def set_user_role(uid):
    """Set a user's role (admin only).  Allowed values: 'user', 'admin'."""
    data = request.get_json(silent=True) or {}
    new_role = data.get("role")

    if new_role not in ("user", "admin"):
        return jsonify({"error": "Invalid role; must be 'user' or 'admin'"}), 400

    target = g.db.query(User).filter(User.id == uid).first()
    if target is None:
        return jsonify({"error": "User not found"}), 404

    old_role = target.role
    target.role = new_role
    g.db.commit()

    log_activity(
        g.db, "admin_set_role",
        user_id=g.user.id,
        resource=f"/api/admin/users/{uid}/role",
        details={"target_user_id": uid, "old_role": old_role, "new_role": new_role},
    )

    return jsonify({"status": "ok", "role": new_role})


# ===========================================================================
# Admin – Activity Log
# ===========================================================================


@app.route("/api/admin/activity-log", methods=["GET"])
@require_admin
def get_activity_log():
    """Return the system-wide activity log (admin only), newest first.

    Query params:
      limit  – max entries to return (default 100, max 500)
      offset – pagination offset (default 0)
      user_id – filter by a specific user's ID
    """
    log_activity(
        g.db, "admin_view_activity_log",
        user_id=g.user.id,
        resource="/api/admin/activity-log",
    )

    try:
        limit = min(int(request.args.get("limit", 100)), 500)
        offset = int(request.args.get("offset", 0))
    except ValueError:
        return jsonify({"error": "limit and offset must be integers"}), 400

    filter_uid = request.args.get("user_id")
    query = g.db.query(ActivityLog)
    if filter_uid:
        query = query.filter(ActivityLog.user_id == int(filter_uid))

    rows = query.order_by(ActivityLog.timestamp.desc()).offset(offset).limit(limit).all()

    return jsonify({
        "logs": [
            {
                "id": row.id,
                "user_id": row.user_id,
                "user_email": row.user.email if row.user else None,
                "action": row.action,
                "details": row.details_dict(),
                "ip_address": row.ip_address,
                "resource": row.resource,
                "timestamp": row.timestamp.isoformat(),
            }
            for row in rows
        ]
    })


# ===========================================================================
# Helpers
# ===========================================================================


def _ensure_wav(path: str, raw_data: bytes) -> str:
    """Convert audio to WAV using soundfile, replacing the original file."""
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
            "Could not process audio file. "
            "Please ensure your browser records in a supported format (WAV, OGG, WebM)."
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
