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

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    f"sqlite:///{os.path.join(_BASE_DIR, 'app.db')}",
)
# Railway provides postgres:// but SQLAlchemy requires postgresql://
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

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
