"""SQLAlchemy database models and session factory for the Voice Training app.

Tables
------
users        – registered accounts with hashed passwords and roles
analyses     – GRBAS analysis results linked to a user
activity_logs – append-only audit trail of significant user actions
"""

import json
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

# Override with DATABASE_URL env var to switch to PostgreSQL in production.
DB_URL = os.environ.get(
    "DATABASE_URL",
    f"sqlite:///{os.path.join(_BASE_DIR, 'voice_trainer.db')}",
)

engine = create_engine(
    DB_URL,
    # SQLite requires this flag when used across multiple Flask threads.
    connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class User(Base):
    """Registered user account.

    Passwords are *never* stored in plain text; only the bcrypt hash is kept.
    The `role` column is the single source of truth for access control and is
    always verified from the database, never taken from the JWT alone.
    """

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    # bcrypt output is always 60 chars; 255 is future-safe.
    password_hash = Column(String(255), nullable=False)
    display_name = Column(String(100), default="")
    # Allowed values: 'user' | 'admin'
    role = Column(String(20), nullable=False, default="user")
    # Soft-delete flag; deactivated users cannot log in.
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login_at = Column(DateTime, nullable=True)

    analyses = relationship(
        "Analysis", back_populates="user", cascade="all, delete-orphan",
    )
    activity_logs = relationship(
        "ActivityLog", back_populates="user", cascade="all, delete-orphan",
    )


class Analysis(Base):
    """Persisted GRBAS analysis result for one recording session."""

    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    # Stored as JSON text so no schema migration is needed when feature sets change.
    scores = Column(Text, nullable=False)    # e.g. '{"G":1,"R":0,"B":2,"A":0,"S":1}'
    features = Column(Text, nullable=False)  # acoustic feature dict
    model_type = Column(String(50), default="heuristic")

    user = relationship("User", back_populates="analyses")

    def scores_dict(self) -> dict:
        return json.loads(self.scores)

    def features_dict(self) -> dict:
        return json.loads(self.features)


class ActivityLog(Base):
    """Append-only audit trail of significant user actions.

    user_id is nullable so that anonymous actions (e.g. failed login attempts
    from unknown email addresses) can also be recorded.
    """

    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(100), nullable=False)
    # Optional JSON payload with action-specific context.
    details = Column(Text, nullable=True)
    # IPv6 addresses can be up to 45 chars.
    ip_address = Column(String(45), nullable=True)
    resource = Column(String(255), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    user = relationship("User", back_populates="activity_logs")

    def details_dict(self) -> dict:
        return json.loads(self.details) if self.details else {}


# ---------------------------------------------------------------------------
# DB initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables if they do not already exist (idempotent)."""
    Base.metadata.create_all(bind=engine)
