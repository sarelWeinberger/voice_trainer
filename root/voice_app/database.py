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
    created_at = Column(DateTime, default=datetime.utcnow)

    results = relationship("Result", back_populates="user", cascade="all, delete-orphan")


class Result(Base):
    __tablename__ = "results"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    title = Column(String, nullable=False)
    raw_scores = Column(Text, nullable=False)   # JSON string
    recommendation = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="results")


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

    if migrations:
        with engine.begin() as conn:
            for stmt in migrations:
                conn.execute(text(stmt))
