"""SQLAlchemy database models and session factory for the Voice Training app.

Tables
------
users   – registered accounts with hashed passwords
results – GRBAS analysis results linked to a user
"""

import os
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Integer, String, Text, create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# ---------------------------------------------------------------------------
# Engine / session
# ---------------------------------------------------------------------------

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_data_dir() -> str:
    """Directory that holds mutable app state (the SQLite file).

    On Hugging Face Spaces the container filesystem is rebuilt on every deploy,
    so a DB written next to the code loses every user account and every saved
    analysis. If a persistent volume is mounted (HF persistent storage mounts it
    at /data) put the DB there; otherwise fall back to the app directory, which
    keeps local development unchanged.
    """
    override = os.environ.get("APP_DATA_DIR")
    if override:
        try:
            os.makedirs(override, exist_ok=True)
            if os.access(override, os.W_OK):
                return override
        except OSError:
            pass
    # Only adopt /data when it is already mounted — creating it ourselves would
    # look persistent while living inside the disposable container layer.
    if os.path.isdir("/data") and os.access("/data", os.W_OK):
        return "/data"
    return _BASE_DIR


DATA_DIR = _resolve_data_dir()
DB_IS_PERSISTENT = bool(os.environ.get("DATABASE_URL")) or DATA_DIR != _BASE_DIR

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"sqlite:///{os.path.join(DATA_DIR, 'app.db')}",
)
# Railway provides postgres:// but SQLAlchemy requires postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


def _adopt_legacy_sqlite() -> None:
    """Move an app.db that predates DATA_DIR onto the persistent volume.

    Without this, switching a running deployment to persistent storage would
    look like every user and every saved analysis had vanished.
    """
    if DATA_DIR == _BASE_DIR or not DATABASE_URL.startswith("sqlite:///"):
        return
    target = DATABASE_URL[len("sqlite:///"):]
    legacy = os.path.join(_BASE_DIR, "app.db")
    if os.path.exists(target) or not os.path.exists(legacy):
        return
    import shutil

    try:
        shutil.copy2(legacy, target)
    except OSError:
        pass


_adopt_legacy_sqlite()

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=True)
    hashed_password = Column(String, nullable=False)
    is_admin = Column(Boolean, default=False)
    # role: 1=super_admin, 2=admin, 3=user
    role = Column(Integer, default=3, nullable=False)
    # Unique class code for admins (role 1 & 2)
    class_code = Column(String, nullable=True, unique=True, index=True)
    # Class code the user joined (for role 3 users)
    joined_class_code = Column(String, nullable=True, index=True)
    # Timestamp at which the user accepted the Terms of Service + Privacy Policy.
    # Null means consent was never given (legacy rows pre-dating the consent flow).
    terms_accepted_at = Column(DateTime, nullable=True)
    # Version of the terms text the user accepted, so a future ToS update can
    # detect users who need to re-consent.
    terms_version = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    results = relationship("Result", back_populates="user", cascade="all, delete-orphan")


class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False)
    raw_scores = Column(Text, nullable=False)   # JSON string
    recommendation = Column(Text, nullable=True)
    # Links this GRBAS analysis to the neck-effort recording captured in the same
    # session (see NeckRecording.session_id), so the two modalities can be paired
    # for model training. Null for analyses with no neck recording.
    session_id = Column(String, index=True, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="results")


class AppSetting(Base):
    """Small key/value store for server-generated settings.

    Holds the JWT signing secret when the deployment does not provide one, so
    it is stable across restarts (tokens survive) without ever falling back to
    a value committed to the repo.
    """

    __tablename__ = "app_settings"

    key = Column(String, primary_key=True)
    value = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class AllowedEmail(Base):
    """Emails an admin has invited into the app.

    The allowlist used to be a text file baked into the Docker image, so every
    email an admin added through the dashboard was silently reverted on the next
    deploy. Keeping it in the DB means an invite lives exactly as long as the
    rest of the data does.

    ``class_code`` is the inviting admin's class: an invited user is attached to
    that class on registration, so their history shows up in that admin's
    dashboard without anyone typing a class code.
    """

    __tablename__ = "allowed_emails"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)  # stored lowercase
    full_name = Column(String, nullable=True)
    class_code = Column(String, index=True, nullable=True)
    added_by = Column(String, nullable=True)     # username of the admin who invited
    created_at = Column(DateTime, default=datetime.utcnow)


class NeckRecording(Base):
    """Per-recording neck-effort feature time-series, captured for model training.

    Stores only DERIVED geometric features (no raw video frames) plus a
    ``session_id`` that links this series to the GRBAS ``Result`` and — once
    Phase 3 lands — the audio clip in object storage (``audio_uri``). The feature
    series is anonymous: a face cannot be reconstructed from it.
    """
    __tablename__ = "neck_recordings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    session_id = Column(String, index=True, nullable=False)
    feature_series = Column(Text, nullable=False)   # JSON: list of per-frame samples
    summary = Column(Text, nullable=True)           # JSON: recording_summary()
    frame_count = Column(Integer, default=0)
    # Research-data consent for this specific recording. Audio is only stored
    # (Phase 3) when this is True; the anonymous feature series is always kept.
    consent_research = Column(Boolean, default=False)
    audio_uri = Column(String, nullable=True)       # object-storage URL, filled in Phase 3
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")


# ---------------------------------------------------------------------------
# DB initialisation
# ---------------------------------------------------------------------------

def get_or_create_setting(key: str, factory) -> str:
    """Return the stored value for ``key``, creating it from ``factory`` once.

    Concurrent workers can race here; the loser of the race re-reads the row the
    winner committed, so every worker ends up with the same value.
    """
    session = SessionLocal()
    try:
        row = session.get(AppSetting, key)
        if row is not None:
            return row.value

        value = factory()
        session.add(AppSetting(key=key, value=value))
        try:
            session.commit()
        except Exception:
            session.rollback()
            row = session.get(AppSetting, key)
            if row is None:
                raise
            return row.value
        return value
    finally:
        session.close()


def init_db() -> None:
    """Create all tables if they do not already exist (idempotent)."""
    Base.metadata.create_all(bind=engine)
    _migrate_add_columns()


def _migrate_add_columns() -> None:
    """Add new columns to existing SQLite tables if they are missing."""
    from sqlalchemy import text, inspect

    insp = inspect(engine)
    existing_cols = {c["name"] for c in insp.get_columns("users")}

    migrations = []
    if "role" not in existing_cols:
        migrations.append("ALTER TABLE users ADD COLUMN role INTEGER DEFAULT 3")
    if "class_code" not in existing_cols:
        migrations.append("ALTER TABLE users ADD COLUMN class_code VARCHAR")
    if "joined_class_code" not in existing_cols:
        migrations.append("ALTER TABLE users ADD COLUMN joined_class_code VARCHAR")
    if "terms_accepted_at" not in existing_cols:
        migrations.append("ALTER TABLE users ADD COLUMN terms_accepted_at TIMESTAMP")
    if "terms_version" not in existing_cols:
        migrations.append("ALTER TABLE users ADD COLUMN terms_version VARCHAR")

    # results table: link to neck-effort recording session
    if insp.has_table("results"):
        result_cols = {c["name"] for c in insp.get_columns("results")}
        if "session_id" not in result_cols:
            migrations.append("ALTER TABLE results ADD COLUMN session_id VARCHAR")

    if migrations:
        with engine.begin() as conn:
            for stmt in migrations:
                conn.execute(text(stmt))
