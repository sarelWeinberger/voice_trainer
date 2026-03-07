"""Firebase Auth middleware for Flask.

This module verifies Firebase ID tokens (from the web client) and provides
a Firestore client for the rest of the backend.
"""

import os
import functools

import firebase_admin
from firebase_admin import auth, credentials, firestore
from flask import request, jsonify

_firebase_init_error = None


def _init_firebase_admin():
    """Initialize Firebase Admin SDK once.

    Keeps the Flask app importable even when credentials are missing.
    """
    global _firebase_init_error
    if _firebase_init_error is not None:
        return

    # Avoid crashing in debug reloader / multiple imports.
    if firebase_admin._apps:
        return

    cred_path = os.path.join(os.path.dirname(__file__), "service-account-key.json")
    try:
        if os.path.exists(cred_path):
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
        else:
            # Fall back to GOOGLE_APPLICATION_CREDENTIALS env var or default credentials
            firebase_admin.initialize_app()
    except Exception as e:
        _firebase_init_error = e


_init_firebase_admin()

try:
    db = firestore.client() if _firebase_init_error is None else None
except Exception as e:
    _firebase_init_error = e
    db = None


def _ensure_firebase_ready():
    """Return a Flask response if Firebase isn't configured, else None."""
    if _firebase_init_error is None:
        return None
    return (
        jsonify({
            "error": "Firebase Admin SDK not configured",
            "details": str(_firebase_init_error),
        }),
        500,
    )


def verify_token(f):
    """Decorator: requires a valid Firebase ID token.

    Sets request.uid and request.firebase_user on success.
    Returns 401 if token is missing or invalid.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        not_ready = _ensure_firebase_ready()
        if not_ready is not None:
            return not_ready

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing auth token"}), 401

        token = auth_header.split("Bearer ", 1)[1]
        try:
            decoded = auth.verify_id_token(token)
            request.uid = decoded["uid"]
            request.firebase_user = decoded
        except Exception:
            return jsonify({"error": "Invalid or expired token"}), 401

        return f(*args, **kwargs)
    return wrapper


def optional_token(f):
    """Decorator: verifies token if present, but does not require it.

    Sets request.uid to the UID if authenticated, or None if not.
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        request.uid = None
        request.firebase_user = None

        not_ready = _ensure_firebase_ready()
        if not_ready is not None:
            return f(*args, **kwargs)

        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.split("Bearer ", 1)[1]
            try:
                decoded = auth.verify_id_token(token)
                request.uid = decoded["uid"]
                request.firebase_user = decoded
            except Exception:
                pass

        return f(*args, **kwargs)
    return wrapper


def require_admin(f):
    """Decorator: requires valid token AND admin role in Firestore."""
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        not_ready = _ensure_firebase_ready()
        if not_ready is not None:
            return not_ready

        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing auth token"}), 401

        token = auth_header.split("Bearer ", 1)[1]
        try:
            decoded = auth.verify_id_token(token)
            request.uid = decoded["uid"]
            request.firebase_user = decoded
        except Exception:
            return jsonify({"error": "Invalid or expired token"}), 401

        if db is None:
            return jsonify({"error": "Firestore not available"}), 500

        user_doc = db.collection("users").document(request.uid).get()
        if not user_doc.exists or user_doc.to_dict().get("role") != "admin":
            return jsonify({"error": "Admin access required"}), 403

        return f(*args, **kwargs)
    return wrapper
