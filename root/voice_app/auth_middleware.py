"""JWT + bcrypt authentication middleware for Flask.

Security design
---------------
- Passwords are hashed with bcrypt (cost factor 12).  The hash is never
  stored in plain text and is never returned in API responses.
- JWTs are signed with HMAC-SHA256 (HS256) using a secret loaded from the
  JWT_SECRET_KEY environment variable.  Set this to a strong random value in
  production (e.g. `openssl rand -hex 32`).
- The token payload includes `exp` (expiry) and `iat` (issued-at) claims,
  both verified on every request by PyJWT.
- Admin access is always double-checked against the live database `role`
  column, so a compromised or replayed token with a forged `role` field
  cannot escalate privileges.
- All significant actions are written to the ActivityLog table by callers
  via the `log_activity` helper, providing a full audit trail.
"""

import functools
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from flask import g, jsonify, request
from sqlalchemy.orm import Session

from database import ActivityLog, SessionLocal, User

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Load from environment.  The placeholder default lets the app start in
# development but should NEVER be used in production.
JWT_SECRET: str = os.environ.get("JWT_SECRET_KEY", "CHANGE_ME_IN_PRODUCTION_USE_ENV_VAR!!")
JWT_ALGORITHM: str = "HS256"
JWT_EXPIRY_HOURS: int = 24

# ---------------------------------------------------------------------------
# Password helpers
# ---------------------------------------------------------------------------


def hash_password(plain: str) -> str:
    """Return a bcrypt hash of *plain*.

    Cost factor 12 ≈ 300 ms on a modern CPU, making brute-force attacks
    expensive while still being acceptable for interactive login.
    """
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if *plain* matches the bcrypt *hashed* value."""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------


def create_token(user: User) -> str:
    """Return a signed JWT for *user* that expires in JWT_EXPIRY_HOURS."""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user.id),  # JWT spec requires sub to be a string
        "email": user.email,
        "role": user.role,
        "iat": now,
        "exp": now + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and verify a JWT.  Raises jwt.PyJWTError on any failure."""
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


# ---------------------------------------------------------------------------
# Activity logging
# ---------------------------------------------------------------------------


def log_activity(
    db: Session,
    action: str,
    user_id: Optional[int] = None,
    details: Optional[dict] = None,
    resource: Optional[str] = None,
) -> None:
    """Persist an ActivityLog entry for the current request.

    Parameters
    ----------
    db:       SQLAlchemy session (caller-owned; this function commits it).
    action:   Short verb describing the event, e.g. 'login', 'analyze'.
    user_id:  Database PK of the acting user, or None for anonymous events.
    details:  Optional dict with extra context (serialised to JSON).
    resource: Optional string identifying what was accessed/modified.
    """
    entry = ActivityLog(
        user_id=user_id,
        action=action,
        details=json.dumps(details) if details else None,
        ip_address=request.remote_addr,
        resource=resource,
        timestamp=datetime.utcnow(),
    )
    db.add(entry)
    db.commit()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_bearer() -> Optional[str]:
    """Return the raw Bearer token from the Authorization header, or None."""
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        return header.split("Bearer ", 1)[1].strip()
    return None


def _load_user(token: str) -> Optional[User]:
    """Decode *token* and return the matching active User, or None."""
    try:
        payload = decode_token(token)
        db = SessionLocal()
        try:
            return (
                db.query(User)
                .filter(User.id == int(payload["sub"]), User.is_active == True)  # noqa: E712
                .first()
            )
        finally:
            db.close()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Route decorators
# ---------------------------------------------------------------------------


def verify_token(f):
    """Require a valid JWT.

    On success, sets ``g.user`` (User ORM object) and ``g.db`` (Session).
    Returns 401 if the token is missing, expired, or invalid.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        token = _extract_bearer()
        if not token:
            return jsonify({"error": "Missing auth token"}), 401

        user = _load_user(token)
        if user is None:
            return jsonify({"error": "Invalid or expired token"}), 401

        g.user = user
        g.db = SessionLocal()
        try:
            return f(*args, **kwargs)
        finally:
            g.db.close()

    return wrapper


def optional_token(f):
    """Accept a JWT when present; set ``g.user = None`` if absent or invalid."""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        g.user = None
        token = _extract_bearer()
        if token:
            g.user = _load_user(token)

        g.db = SessionLocal()
        try:
            return f(*args, **kwargs)
        finally:
            g.db.close()

    return wrapper


def require_admin(f):
    """Require a valid JWT **and** the 'admin' role verified against the DB.

    Re-fetching the role from the database on every request means that a
    demoted admin cannot retain access by replaying an older, still-valid
    token whose payload claims ``role='admin'``.
    """

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        token = _extract_bearer()
        if not token:
            return jsonify({"error": "Missing auth token"}), 401

        user = _load_user(token)
        if user is None:
            return jsonify({"error": "Invalid or expired token"}), 401

        # Role is taken from the live database row, not from the JWT payload.
        if user.role != "admin":
            return jsonify({"error": "Admin access required"}), 403

        g.user = user
        g.db = SessionLocal()
        try:
            return f(*args, **kwargs)
        finally:
            g.db.close()

    return wrapper
