"""FastAPI backend for Voice Training App with GRBAS analysis.

Authentication is handled via JWT tokens with OAuth2 password flow.
Data is stored via SQLAlchemy (see database.py) — SQLite locally,
Postgres in production via the DATABASE_URL environment variable.
"""

import json
import hashlib
import pathlib
import os
import random
import re
import secrets
import string
import tempfile
import uuid
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt

from database import (
    SessionLocal, User, Result, NeckRecording, AllowedEmail, init_db,
    get_or_create_setting, DATABASE_URL, DB_IS_PERSISTENT,
)
from grbas_model import GRBASClassifier, extract_features
from neck_effort_engine import NeckEffortConfig
from head_motion import analyze_payload
from neck_capture_context import parse_transport
from neck_v1_scoring import NeckAnalysis
from observable_effort import ObservableEffortEngine as NeckEffortEngine, analyze_observable_recording, analyze_video_audio, decode_live_neck_audio
from recommendations import recommend_with_context, flatten_recommendations

# =========================
# Config
# =========================
# This literal used to be the fallback signing key. It is committed to a public
# repo, so any token signed with it can be forged by anyone — it is now only
# recognised in order to be REFUSED.
_LEGACY_PLACEHOLDER_SECRET = "replace_this_with_a_long_random_secret"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


def resolve_jwt_secret() -> tuple:
    """Return (secret, source) for signing tokens; never the committed literal.

    JWT_SECRET_KEY wins. Otherwise a random secret is generated once and kept in
    the DB, so restarts do not log everyone out. Only if the DB is unreachable
    do we fall back to a per-process secret — secure, but tokens die with the
    process, so the source is reported and warned about.
    """
    env_secret = os.environ.get("JWT_SECRET_KEY", "").strip()
    if env_secret and env_secret != _LEGACY_PLACEHOLDER_SECRET:
        return env_secret, "env"
    try:
        return get_or_create_setting(
            "jwt_secret", lambda: secrets.token_urlsafe(48)), "database"
    except Exception as exc:                          # noqa: BLE001 - never fatal
        print(f"WARNING: could not persist a JWT secret ({exc}); using a "
              f"per-process one. Everyone is logged out on every restart.",
              flush=True)
        return secrets.token_urlsafe(48), "ephemeral"

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")

app = FastAPI(title="Voice Training - GRBAS Analysis")

# Initialise database tables on startup (idempotent).
init_db()

# Must come after init_db(): the fallback secret lives in a DB table.
SECRET_KEY, SECRET_SOURCE = resolve_jwt_secret()
if SECRET_SOURCE != "env":
    print(f"NOTE: JWT_SECRET_KEY is not set; using a generated secret "
          f"({SECRET_SOURCE}). Set the env var to keep sessions valid across "
          f"deploys.", flush=True)

# Load GRBAS model if a pre-trained file exists.
classifier = GRBASClassifier()
MODEL_PATH = os.path.join(os.path.dirname(__file__), "grbas_model.joblib")
if os.path.exists(MODEL_PATH):
    classifier.load(MODEL_PATH)

NECK_EFFORT_CONFIG = NeckEffortConfig(calibration_frames=10, smooth_window=12)
neck_effort_engines: Dict[int, NeckEffortEngine] = {}


# =========================
# Schemas
# =========================

SUPER_ADMIN_EMAIL = "sarelw1@gmail.com"

# ---------------------------------------------------------------------------
# Registration allowlist
# ---------------------------------------------------------------------------
# The allowlist lives in the DATABASE (``allowed_emails``). It used to be a text
# file baked into the Docker image, which meant every email an admin added
# through the dashboard was reverted by the next deploy. These paths are now
# read ONCE, to seed the table on first boot.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_ALLOWLIST_FILES = (
    "/memories/allowed_emails.txt",
    os.path.join(_APP_DIR, "memories", "allowed_emails.txt"),
)

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# Password every invited user logs in with (kept from the original flow).
DEFAULT_USER_PASSWORD = "qwertyui"


def norm_email(email: Optional[str]) -> str:
    return (email or "").strip().lower()


def email_in_allowlist(db: Session, email: Optional[str]) -> bool:
    """Case-insensitive allowlist membership check against the DB."""
    e = norm_email(email)
    if not e:
        return False
    return db.query(AllowedEmail).filter(AllowedEmail.email == e).first() is not None


def _read_legacy_allowlist_files() -> set:
    emails = set()
    for path in LEGACY_ALLOWLIST_FILES:
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        emails.add(line.lower())
        except OSError:
            continue
    return emails


def seed_allowlist(db: Session) -> None:
    """One-time import into the DB-backed allowlist.

    Seeds from the legacy text files *and* from every account that already
    exists, so switching the source of truth cannot lock anyone out.
    """
    if db.query(AllowedEmail).count() > 0:
        return

    emails = {e for e in _read_legacy_allowlist_files() if EMAIL_RE.match(e)}
    # Every existing account, so changing the source of truth locks nobody out.
    emails.update(norm_email(u.username) for u in db.query(User).all())
    emails.add(norm_email(SUPER_ADMIN_EMAIL))
    emails.discard("")
    emails.discard("admin")  # the legacy non-email admin account

    for email in sorted(emails):
        db.add(AllowedEmail(email=email, added_by="system-migration"))
    if emails:
        db.commit()


def generate_class_code() -> str:
    """Generate a random 6-char alphanumeric class code."""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))


# Version string for the current Terms of Service + Privacy Policy text.
# Bump this when the legal pages change in a way that requires re-consent.
TERMS_VERSION = "2026-06-07"


class UserCreate(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = None
    class_code: Optional[str] = None  # optional class code to join
    accepted_terms: bool = False  # must be True to register


class UserOut(BaseModel):
    id: int
    username: str
    full_name: Optional[str]
    is_admin: bool
    role: int
    class_code: Optional[str] = None
    joined_class_code: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str


class ResultOut(BaseModel):
    id: int
    title: str
    raw_scores: str
    recommendation: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


class AdminResultOut(BaseModel):
    id: int
    username: str
    full_name: Optional[str]
    title: str
    raw_scores: str
    recommendation: Optional[str]
    created_at: datetime


class RoleUpdate(BaseModel):
    is_admin: bool
    role: Optional[int] = None  # 1=super_admin, 2=admin, 3=user


class NeckEffortOut(BaseModel):
    ok: bool
    mode: str
    calibrated: bool
    calibration_progress: float
    score: Optional[int]
    score_label: Optional[str]
    confidence: float
    features: Optional[Dict[str, float]]
    reasons: List[str]
    continuous_score: Optional[float] = None
    debug: Optional[Dict[str, Any]] = None
    last_valid_score: Optional[int] = None
    model_version: Optional[str] = None
    observables: Optional[Dict[str, Any]] = None
    unavailable: Optional[Dict[str, str]] = None
    uncertainty: Optional[Dict[str, Any]] = None
    explanation: Optional[str] = None
    neckAnalysis: Optional[NeckAnalysis] = None


# =========================
# Helpers
# =========================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)


def hash_password(password: str):
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)



def authenticate_user(db: Session, username: str, password: str):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None

    allowed = email_in_allowlist(db, user.username)

    # Removing an email from the allowlist has to actually revoke access.
    # Without this a removed user could still sign in with the shared password,
    # since their password hash stays valid. Admins are exempt so an admin can
    # never lock themselves out of the dashboard.
    if user.role == 3 and not allowed:
        return None

    # Invited users log in with the shared password; admins and legacy accounts
    # keep their own. Both are accepted so seeding an admin's own address into
    # the allowlist cannot lock them out of the dashboard.
    if allowed and password == DEFAULT_USER_PASSWORD:
        return user
    if verify_password(password, user.hashed_password):
        return user
    return None


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise credentials_exception
    return user


async def get_optional_user(request: Request, db: Session = Depends(get_db)):
    """Extract user from Bearer token if present; return None otherwise."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header.split("Bearer ", 1)[1].strip()
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            return None
        return db.query(User).filter(User.username == username).first()
    except JWTError:
        return None


def get_admin_user(current_user: User = Depends(get_current_user)):
    if not current_user.is_admin and current_user.role not in (1, 2):
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


def get_super_admin_user(current_user: User = Depends(get_current_user)):
    if current_user.role != 1:
        raise HTTPException(status_code=403, detail="Super admin access required")
    return current_user


def get_neck_effort_engine(user_id: int) -> NeckEffortEngine:
    engine = neck_effort_engines.get(user_id)
    if engine is None:
        engine = NeckEffortEngine(NECK_EFFORT_CONFIG)
        neck_effort_engines[user_id] = engine
    return engine


def get_latest_neck_effort(current_user: User) -> Optional[Dict[str, Any]]:
    engine = neck_effort_engines.get(current_user.id)
    if engine is None:
        return None
    return engine.latest_result(max_age_seconds=10.0)


def get_recording_neck_effort(current_user: User) -> Optional[Dict[str, Any]]:
    engine = neck_effort_engines.get(current_user.id)
    if engine is None:
        return None
    return engine.recording_summary(max_age_seconds=120.0)


# =========================
# Seed admin on startup
# =========================
@app.on_event("startup")
def create_default_admin():
    db = SessionLocal()
    try:
        # Migrate existing users: add role/class_code columns if missing
        _migrate_roles(db)

        # Ensure super admin exists
        super_admin = db.query(User).filter(User.username == SUPER_ADMIN_EMAIL).first()
        if not super_admin:
            super_admin = User(
                username=SUPER_ADMIN_EMAIL,
                full_name="Super Admin",
                hashed_password=hash_password("admin1234"),
                is_admin=True,
                role=1,
                class_code=generate_class_code(),
            )
            db.add(super_admin)
            db.commit()
        else:
            if super_admin.role != 1:
                super_admin.role = 1
                super_admin.is_admin = True
                if not super_admin.class_code:
                    super_admin.class_code = generate_class_code()
                db.commit()

        # Keep legacy admin user
        existing = db.query(User).filter(User.username == "admin").first()
        if not existing:
            admin = User(
                username="admin",
                full_name="System Admin",
                hashed_password=hash_password("admin1234"),
                is_admin=True,
                role=2,
                class_code=generate_class_code(),
            )
            db.add(admin)
            db.commit()
        else:
            if existing.role == 3:
                existing.role = 2
                existing.is_admin = True
            if not existing.class_code:
                existing.class_code = generate_class_code()
                db.commit()

        # First boot after the allowlist moved from a text file into the DB.
        seed_allowlist(db)

        if not DB_IS_PERSISTENT:
            print(
                "WARNING: no persistent database configured — user accounts, "
                "invites and analysis history live in "
                f"{DATABASE_URL} and are LOST on the next rebuild. Set "
                "DATABASE_URL (Postgres) or mount persistent storage at /data.",
                flush=True,
            )
    finally:
        db.close()


def _migrate_roles(db: Session):
    """Backfill role/class_code for existing users after schema change."""
    users = db.query(User).all()
    changed = False
    for u in users:
        if u.role is None:
            u.role = 2 if u.is_admin else 3
            changed = True
        if u.is_admin and not u.class_code:
            u.class_code = generate_class_code()
            changed = True
        # Ensure super admin email always gets role 1
        if u.username == SUPER_ADMIN_EMAIL and u.role != 1:
            u.role = 1
            u.is_admin = True
            if not u.class_code:
                u.class_code = generate_class_code()
            changed = True
    if changed:
        db.commit()


# =========================
# Pages
# =========================
@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/admin")
async def admin_page():
    return FileResponse(os.path.join(STATIC_DIR, "admin.html"))


# =========================
# Auth – Registration & Login
# =========================
@app.post("/api/register", response_model=UserOut)

def register(user_in: UserCreate, db: Session = Depends(get_db)):
    if not user_in.username or not user_in.password:
        raise HTTPException(status_code=400, detail="Username and password are required")

    # Consent is a hard gate: refuse to create the account without it. This is
    # the audit trail the ToS relies on ("the user actively agreed at signup").
    if not user_in.accepted_terms:
        raise HTTPException(
            status_code=400,
            detail="יש לאשר את תנאי השימוש ומדיניות הפרטיות לפני הרשמה",
        )

    existing = db.query(User).filter(User.username == user_in.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    # Only allow registration if an admin invited this email
    invite = (
        db.query(AllowedEmail)
        .filter(AllowedEmail.email == norm_email(user_in.username))
        .first()
    )
    if invite is None:
        raise HTTPException(status_code=403, detail="User not allowed to register")

    # Password is always 'qwertyui' for allowlist users
    if user_in.password != DEFAULT_USER_PASSWORD:
        raise HTTPException(status_code=400, detail="Password must be 'qwertyui' for allowed users")

    # Validate class code if provided
    joined_class = None
    if user_in.class_code and user_in.class_code.strip():
        code = user_in.class_code.strip().upper()
        owner = db.query(User).filter(User.class_code == code).first()
        if not owner:
            raise HTTPException(status_code=400, detail="קוד כיתה לא תקין")
        joined_class = code
    elif invite.class_code:
        # Attach to the class of the admin who invited them, so their history
        # lands in that admin's dashboard without them typing a code.
        joined_class = invite.class_code

    user = User(
        username=user_in.username,
        full_name=user_in.full_name or invite.full_name,
        hashed_password=hash_password(DEFAULT_USER_PASSWORD),
        is_admin=False,
        role=3,
        joined_class_code=joined_class,
        terms_accepted_at=datetime.utcnow(),
        terms_version=TERMS_VERSION,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/api/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        email = norm_email(form_data.username)
        invited = db.query(AllowedEmail).filter(AllowedEmail.email == email).first()
        account = db.query(User).filter(User.username == form_data.username).first()
        if invited is not None and account is None:
            # Invited but never signed up: the account is created at
            # registration, which is where consent to the Terms is recorded.
            raise HTTPException(
                status_code=400,
                detail="החשבון עדיין לא נוצר — לחצו על 'הרשמה' ואשרו את תנאי השימוש",
            )
        if account is not None and account.role == 3 and invited is None:
            raise HTTPException(
                status_code=403,
                detail="ההרשאה לחשבון הזה הוסרה. פנו למנהל הכיתה.",
            )
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/api/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    return current_user


@app.delete("/api/me")
def delete_my_account(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Right-to-be-forgotten: irreversibly delete the user and all their
    results. The user must be authenticated; we trust the JWT as proof of
    intent. Super admins cannot delete themselves via this route — they must
    do it manually to avoid lock-out.
    """
    if current_user.role == 1:
        raise HTTPException(
            status_code=400,
            detail="Super admin accounts must be deleted manually",
        )
    db.delete(current_user)
    db.commit()
    return {"status": "deleted"}


# =========================
# Voice Analysis
# =========================
@app.post("/api/analyze")
async def analyze(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        audio_data = await audio.read()
        tmp.write(audio_data)

    try:
        tmp_path = _ensure_wav(tmp_path, audio_data)
        features = await run_in_threadpool(extract_features, tmp_path)
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

        latest_neck_effort = None
        engine = neck_effort_engines.get(current_user.id)
        if session_id and engine and engine.state.recording_session_id == session_id:
            latest_neck_effort = engine.recording_summary(max_age_seconds=120.0)
        observable_effort = await run_in_threadpool(analyze_observable_recording,
            tmp_path, (latest_neck_effort or {}).get("timeline"))
        recs = recommend_with_context(scores, neck_effort=latest_neck_effort)
        recommendations = flatten_recommendations(recs)

        response_data = {
            "scores": scores,
            "features": key_features,
            "model_type": "trained" if classifier.is_trained else "heuristic",
            "neck_effort": latest_neck_effort,
            "observable_effort": observable_effort,
            "recommendations": recommendations,
        }

        # Neck inference runs independently; attach it if it completed during audio analysis.
        if session_id and engine and engine.state.recording_session_id == session_id:
            ready = engine.recording_summary(max_age_seconds=120.0)
            if ready is not None:
                response_data["neck_effort"] = latest_neck_effort = ready

        # Persist result for authenticated user.
        result = Result(
                user_id=current_user.id,
                title="GRBAS Analysis",
                raw_scores=json.dumps({
                    "scores": scores,
                    "features": key_features,
                    "model_type": response_data["model_type"],
                    "neck_effort": latest_neck_effort,
                    "observable_effort": observable_effort,
                }),
                recommendation=json.dumps(recommendations) if recommendations else None,
                session_id=session_id,
            )
        db.add(result)
        db.commit()

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/observable-effort/recording")
def analyze_synchronized_effort(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    audio_offset_seconds: float = Form(0.0),
    current_user: User = Depends(get_current_user),
):
    """Research analysis of CFR video + WAV audio with a declared start offset."""
    if not np.isfinite(audio_offset_seconds) or abs(audio_offset_seconds) > 60:
        raise HTTPException(status_code=400, detail="Invalid audio offset")
    with tempfile.TemporaryDirectory() as directory:
        paths = []
        for upload, suffix in ((video, ".mp4"), (audio, ".wav")):
            data = upload.file.read(50 * 1024 * 1024 + 1)
            if not data or len(data) > 50 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Each recording must be between 1 byte and 50 MB")
            path = pathlib.Path(directory) / ("input" + suffix)
            path.write_bytes(data)
            paths.append(path)
        try:
            result = analyze_video_audio(*paths, audio_offset_seconds=audio_offset_seconds)
            if not result.get("ok"):
                raise HTTPException(status_code=502, detail="; ".join(result.get("reasons", ["Neck analysis failed"])))
            return result
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))


@app.post("/api/neck-effort/reset")
def reset_neck_effort(current_user: User = Depends(get_current_user)):
    try:
        engine = get_neck_effort_engine(current_user.id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    engine.reset_session()
    return {"status": "ok"}


@app.post("/api/neck-effort/recording/start")
def start_neck_effort_recording(current_user: User = Depends(get_current_user)):
    try:
        engine = get_neck_effort_engine(current_user.id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    session_id = uuid.uuid4().hex
    engine.start_recording_session(session_id)
    return {"status": "ok", "session_id": session_id}


@app.post("/api/neck-effort/recording/stop")
def stop_neck_effort_recording(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    audio: Annotated[Optional[UploadFile], File()] = None,
    audio_offset_seconds: Annotated[float, Form()] = 0.0,
    session_id: Annotated[Optional[str], Form()] = None,
    motion_trace: Annotated[Optional[str], Form()] = None,
    capture_transport: Annotated[Optional[str], Form()] = None,
):
    try:
        engine = get_neck_effort_engine(current_user.id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    motion_analysis = None
    try:
        transport_analysis=parse_transport(capture_transport)
    except (ValueError,TypeError):
        raise HTTPException(status_code=400,detail='Invalid capture receipt summary') from None
    if transport_analysis is not None and not session_id:
        raise HTTPException(status_code=400,detail='Capture summary requires its recording session')
    if motion_trace is not None:
        if not session_id:
            raise HTTPException(status_code=400,detail='Motion trace requires its recording session')
        try:
            motion_analysis = analyze_payload(motion_trace)
        except (ValueError,TypeError):
            raise HTTPException(status_code=400,detail='Invalid head motion trace') from None
    audio_context=None
    if audio is not None:
        if not session_id:
            raise HTTPException(status_code=400,detail='Audio requires its neck recording session')
        try:
            audio_context=decode_live_neck_audio(audio.file.read(5*1024*1024+1),audio_offset_seconds)
        except (ValueError,RuntimeError,OSError):
            raise HTTPException(status_code=400,detail='Invalid live neck audio') from None
    with engine.lock:
        if session_id is not None and session_id!=engine.state.recording_session_id:
            raise HTTPException(status_code=409,detail='Recording session is no longer current')
        engine.stop_recording_session()
        summary = engine.finalize(audio=audio_context,
            timing='browser_estimated_offset' if audio_context is not None else 'unverified',
            movement=motion_analysis,capture_transport=transport_analysis)
    session_id = engine.state.recording_session_id
    series = engine.export_recording()

    # Persist the anonymous neck-effort feature series for model training. This is
    # best-effort: a storage failure must never break the user's recording flow.
    if session_id and series:
        try:
            db.add(
                NeckRecording(
                    user_id=current_user.id,
                    session_id=session_id,
                    feature_series=json.dumps(series),
                    summary=json.dumps(summary) if summary else None,
                    frame_count=len(series),
                    consent_research=False,
                )
            )
            db.commit()
        except Exception:
            db.rollback()

    # If audio finished first, attach the independent neck result to its saved history.
    if session_id:
        try:
            for result in db.query(Result).filter(Result.user_id == current_user.id, Result.session_id == session_id).all():
                saved = json.loads(result.raw_scores)
                saved["neck_effort"] = summary
                result.raw_scores = json.dumps(saved)
            db.commit()
        except Exception:
            db.rollback()

    return {
        "status": "ok",
        "session_id": session_id,
        "neck_effort": summary,
    }


@app.post("/api/neck-effort", response_model=NeckEffortOut)
async def analyze_neck_effort_frame(
    frame: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    image_bytes = await frame.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Frame is required")

    np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Invalid image frame")

    try:
        engine = get_neck_effort_engine(current_user.id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    if not engine.state.recording_active:
        return {
            "ok": True,
            "mode": "no_score",
            "calibrated": engine.is_calibrated(),
            "calibration_progress": engine.calibration_progress(),
            "score": None,
            "score_label": None,
            "confidence": 0.0,
            "features": None,
            "reasons": ["recording_required"],
        }
    return engine.process_frame(image_bgr)


@app.post("/api/neck-effort/frames")
def analyze_neck_effort_frames(
    frames: List[UploadFile] = File(...),
    timestamps: str = Form(...),
    session_id: str = Form(...),
    current_user: User = Depends(get_current_user),
    batch_id: Annotated[Optional[str], Form()] = None,
):
    """Small bounded batches; client capture clock remains provisionally synchronized."""
    if not 1 <= len(frames) <= 10 or len(timestamps) > 4096:
        raise HTTPException(status_code=400, detail="Invalid neck frame batch")
    try:
        times = json.loads(timestamps)
        if not isinstance(times, list) or len(times) != len(frames):
            raise ValueError()
        if any(type(t) not in (int,float) or not np.isfinite(t) or not 0 <= t <= 60 for t in times):
            raise ValueError()
        if any(b <= a for a,b in zip(times,times[1:])):
            raise ValueError()
    except (ValueError,TypeError):
        raise HTTPException(status_code=400, detail="Invalid frame timestamps")
    if batch_id is not None and (not isinstance(batch_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',batch_id)):
        raise HTTPException(status_code=400,detail='Invalid frame batch identifier')
    encoded_images=[]
    digest=hashlib.sha256(json.dumps(times,separators=(',',':')).encode())
    for frame in frames:
        data=frame.file.read(1024*1024+1)
        if not data or len(data)>1024*1024:
            raise HTTPException(status_code=413, detail="Invalid frame size")
        digest.update(len(data).to_bytes(4,'big'));digest.update(data)
        encoded_images.append(data)
    engine=get_neck_effort_engine(current_user.id)
    with engine.lock:
        if not engine.state.recording_active or engine.state.recording_session_id != session_id:
            raise HTTPException(status_code=409, detail="Recording session is no longer active")
        if batch_id is not None:
            prior=engine.upload_batches.get(batch_id)
            if prior is not None:
                if prior['digest']!=digest.hexdigest():
                    raise HTTPException(status_code=409,detail='Batch identifier was reused with different contents')
                if prior['receipt'] is not None:return prior['receipt']
            elif len(engine.upload_batches)>=300:
                raise HTTPException(status_code=413,detail='Too many frame batches')
        else:prior=None
        if any(entry['receipt'] is None for key,entry in engine.upload_batches.items() if key!=batch_id):
            raise HTTPException(status_code=409,detail='An earlier frame batch is unfinished')
        processed=prior['processed'] if prior else 0
        if len(engine.rows)+len(frames)-processed>engine.cfg.max_frames:
            raise HTTPException(status_code=413,detail='Recording frame limit reached')
        if processed<len(times) and engine.rows and times[processed] <= engine.rows[-1]['t']:
            raise HTTPException(status_code=409, detail="Frame batch is out of order")
        images=[]
        for data in encoded_images:
            image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
            if image is None or max(image.shape[:2])>1920:
                raise HTTPException(status_code=400, detail="Invalid frame image")
            images.append(image)
        if batch_id is not None and prior is None:
            prior=dict(digest=digest.hexdigest(),processed=0,receipt=None)
            engine.upload_batches[batch_id]=prior
        result=engine.latest
        for image,t in zip(images[processed:],times[processed:]):
            before=len(engine.rows)
            try:
                result=engine.process_frame(image,timestamp=float(t))
            except (RuntimeError,ValueError,cv2.error):
                raise HTTPException(status_code=503,detail='Frame processing did not complete') from None
            if len(engine.rows)!=before+1:
                raise HTTPException(status_code=503,detail='Frame processing did not complete')
            if prior is not None:prior['processed']+=1
        receipt=dict(result,accepted_frames=len(images),batch_id=batch_id)
        if batch_id is not None:
            prior['receipt']=receipt
        return receipt


# =========================
# Results / History
# =========================
@app.get("/api/my-results", response_model=List[ResultOut])
def my_results(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return (
        db.query(Result)
        .filter(Result.user_id == current_user.id)
        .order_by(Result.created_at.desc())
        .limit(50)
        .all()
    )


@app.get("/api/history")
def get_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return analysis history in the format expected by the frontend."""
    rows = (
        db.query(Result)
        .filter(Result.user_id == current_user.id)
        .order_by(Result.created_at.desc())
        .limit(50)
        .all()
    )

    history = []
    for row in rows:
        try:
            data = json.loads(row.raw_scores)
            recs = None
            if row.recommendation:
                try:
                    recs = json.loads(row.recommendation)
                except (json.JSONDecodeError, TypeError):
                    pass
            history.append({
                "id": row.id,
                "timestamp": row.created_at.isoformat(),
                "scores": data.get("scores", {}),
                "features": data.get("features", {}),
                "model_type": data.get("model_type", "unknown"),
                "neck_effort": data.get("neck_effort"),
                "observable_effort": data.get("observable_effort"),
                "recommendations": recs,
            })
        except (json.JSONDecodeError, AttributeError):
            pass

    return {"history": history}


# =========================
# Admin – User Management
# =========================
@app.get("/api/admin/users")
def admin_users(
    db: Session = Depends(get_db),
    admin_user: User = Depends(get_admin_user),
):
    if admin_user.role == 1:
        # Super admin sees all users
        users = db.query(User).order_by(User.created_at.desc()).all()
    else:
        # Regular admin sees only users in their class
        users = db.query(User).filter(
            User.joined_class_code == admin_user.class_code
        ).order_by(User.created_at.desc()).all()

    result = []
    for u in users:
        latest = (
            db.query(Result)
            .filter(Result.user_id == u.id)
            .order_by(Result.created_at.desc())
            .first()
        )
        results_count = db.query(Result).filter(Result.user_id == u.id).count()

        latest_analysis = None
        if latest:
            try:
                data = json.loads(latest.raw_scores)
                latest_analysis = {
                    "timestamp": latest.created_at.isoformat(),
                    "scores": data.get("scores", {}),
                }
            except (json.JSONDecodeError, AttributeError):
                pass

        result.append({
            "id": u.id,
            "username": u.username,
            "full_name": u.full_name,
            "is_admin": u.is_admin,
            "role": u.role,
            "class_code": u.class_code,
            "joined_class_code": u.joined_class_code,
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "latest_analysis": latest_analysis,
            "results_count": results_count,
        })

    return {"users": result}


@app.get("/api/admin/users/{uid}/results")
def get_user_results(
    uid: int,
    db: Session = Depends(get_db),
    admin_user: User = Depends(get_admin_user),
):
    # Regular admin: verify target user is in their class
    if admin_user.role != 1:
        target = db.query(User).filter(User.id == uid).first()
        if not target or target.joined_class_code != admin_user.class_code:
            raise HTTPException(status_code=403, detail="Access denied")

    rows = (
        db.query(Result)
        .filter(Result.user_id == uid)
        .order_by(Result.created_at.desc())
        .limit(100)
        .all()
    )

    results = []
    for row in rows:
        try:
            data = json.loads(row.raw_scores)
            recs = None
            if row.recommendation:
                try:
                    recs = json.loads(row.recommendation)
                except (json.JSONDecodeError, TypeError):
                    pass
            results.append({
                "id": row.id,
                "timestamp": row.created_at.isoformat(),
                "scores": data.get("scores", {}),
                "features": data.get("features", {}),
                "model_type": data.get("model_type", "unknown"),
                "neck_effort": data.get("neck_effort"),
                "observable_effort": data.get("observable_effort"),
                "recommendations": recs,
            })
        except (json.JSONDecodeError, AttributeError):
            pass

    return {"results": results}


@app.get("/api/admin/results", response_model=List[AdminResultOut])
def admin_results(
    db: Session = Depends(get_db),
    admin_user: User = Depends(get_admin_user),
):
    rows = (
        db.query(Result, User)
        .join(User, Result.user_id == User.id)
        .order_by(Result.created_at.desc())
        .all()
    )

    output = []
    for result, user in rows:
        output.append(
            AdminResultOut(
                id=result.id,
                username=user.username,
                full_name=user.full_name,
                title=result.title,
                raw_scores=result.raw_scores,
                recommendation=result.recommendation,
                created_at=result.created_at,
            )
        )
    return output


@app.put("/api/admin/users/{uid}/role")
def set_user_role(
    uid: int,
    body: RoleUpdate,
    db: Session = Depends(get_db),
    admin_user: User = Depends(get_admin_user),
):
    """Set a user's role. Only super admin can promote to admin/super_admin."""
    target = db.query(User).filter(User.id == uid).first()
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")

    new_role = body.role if body.role is not None else (2 if body.is_admin else 3)

    # Only super admin can set roles
    if admin_user.role != 1:
        raise HTTPException(status_code=403, detail="Only super admin can change roles")

    # Cannot demote yourself
    if target.id == admin_user.id:
        raise HTTPException(status_code=400, detail="Cannot change your own role")

    target.role = new_role
    target.is_admin = new_role in (1, 2)

    # Generate class code for new admins
    if new_role in (1, 2) and not target.class_code:
        target.class_code = generate_class_code()

    # Remove class code from demoted users
    if new_role == 3:
        target.class_code = None

    db.commit()
    return {
        "status": "ok",
        "is_admin": target.is_admin,
        "role": target.role,
        "class_code": target.class_code,
    }


@app.delete("/api/admin/users/{uid}")
def admin_delete_user(
    uid: int,
    db: Session = Depends(get_db),
    admin_user: User = Depends(get_super_admin_user),
):
    """Irreversibly delete a user and all their results. Super admin only.
    Cannot delete yourself or another super admin."""
    target = db.query(User).filter(User.id == uid).first()
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")
    if target.id == admin_user.id:
        raise HTTPException(status_code=400, detail="לא ניתן למחוק את החשבון שלך")
    if target.role == 1:
        raise HTTPException(status_code=400, detail="לא ניתן למחוק סופר-אדמין")

    db.delete(target)
    db.commit()
    return {"status": "deleted", "id": uid}


# =========================
# Allowlist management (admins)
# =========================
# An admin adds an email here to give someone access to the app. Role-2 admins
# manage only their own class; the super admin sees every invite.
class AllowedEmailIn(BaseModel):
    email: str
    full_name: Optional[str] = None


def _invite_status(db: Session, row: AllowedEmail) -> Dict[str, Any]:
    """Serialise one invite together with the account/history behind it."""
    user = db.query(User).filter(User.username == row.email).first()
    results_count = 0
    last_analysis_at = None
    if user is not None:
        results_count = db.query(Result).filter(Result.user_id == user.id).count()
        last = (
            db.query(Result)
            .filter(Result.user_id == user.id)
            .order_by(Result.created_at.desc())
            .first()
        )
        if last is not None and last.created_at:
            last_analysis_at = last.created_at.isoformat()

    return {
        "email": row.email,
        "full_name": row.full_name or (user.full_name if user else None),
        "class_code": row.class_code,
        "added_by": row.added_by,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "registered": user is not None,
        "user_id": user.id if user else None,
        "results_count": results_count,
        "last_analysis_at": last_analysis_at,
    }


def _visible_invites(db: Session, admin_user: User):
    q = db.query(AllowedEmail)
    if admin_user.role != 1:
        q = q.filter(AllowedEmail.class_code == admin_user.class_code)
    return q.order_by(AllowedEmail.created_at.desc()).all()


@app.get("/api/admin/allowed-emails")
def list_allowed_emails(
    db: Session = Depends(get_db),
    admin_user: User = Depends(get_admin_user),
):
    """Return the invites this admin manages, each with its history summary."""
    invites = [_invite_status(db, row) for row in _visible_invites(db, admin_user)]
    return {
        "invites": invites,
        "emails": [i["email"] for i in invites],  # kept for older clients
        # Surfaced in the dashboard: on a Space with no persistent volume and no
        # DATABASE_URL, invites and history do not survive the next deploy.
        "db_persistent": DB_IS_PERSISTENT,
        "secret_source": SECRET_SOURCE,
    }


@app.post("/api/admin/allowed-emails")
def add_allowed_email(
    body: AllowedEmailIn,
    db: Session = Depends(get_db),
    admin_user: User = Depends(get_admin_user),
):
    """Invite an email. The account itself is created when the person registers
    (that is where consent to the Terms is recorded), and is auto-attached to
    this admin's class."""
    email = norm_email(body.email)
    if not EMAIL_RE.match(email):
        raise HTTPException(status_code=400, detail="כתובת אימייל לא תקינה")

    existing = db.query(AllowedEmail).filter(AllowedEmail.email == email).first()
    if existing is not None:
        if admin_user.role == 1 or existing.class_code == admin_user.class_code:
            raise HTTPException(status_code=400, detail="האימייל כבר ברשימת המורשים")
        raise HTTPException(
            status_code=400,
            detail="האימייל כבר משויך לכיתה אחרת",
        )

    row = AllowedEmail(
        email=email,
        full_name=(body.full_name or "").strip() or None,
        class_code=admin_user.class_code,
        added_by=admin_user.username,
    )
    db.add(row)

    # An account may already exist (legacy user, or re-invited after removal).
    # Pull it into this admin's class so their history shows in the dashboard.
    user = db.query(User).filter(User.username == email).first()
    if user is not None and user.role == 3 and not user.joined_class_code:
        user.joined_class_code = admin_user.class_code

    db.commit()
    db.refresh(row)
    return {"status": "ok", **_invite_status(db, row)}


@app.delete("/api/admin/allowed-emails")
def remove_allowed_email(
    body: dict,
    db: Session = Depends(get_db),
    admin_user: User = Depends(get_admin_user),
):
    """Revoke access for an email. Body: {"email": "..."}

    The account and its saved history are kept — deleting those is a separate,
    super-admin action (DELETE /api/admin/users/{uid})."""
    email = norm_email((body or {}).get("email"))
    row = db.query(AllowedEmail).filter(AllowedEmail.email == email).first()
    if row is None:
        raise HTTPException(status_code=404, detail="האימייל לא נמצא ברשימה")
    if admin_user.role != 1 and row.class_code != admin_user.class_code:
        raise HTTPException(status_code=403, detail="האימייל שייך לכיתה אחרת")
    if email == norm_email(SUPER_ADMIN_EMAIL):
        raise HTTPException(status_code=400, detail="לא ניתן להסיר את הסופר-אדמין")

    db.delete(row)
    db.commit()
    return {"status": "ok", "email": email}


# =========================
# User class code management
# =========================
@app.post("/api/join-class")
def join_class(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    body: dict = None,
):
    """Join a class by code. Body: {\"class_code\": \"ABC123\"}"""
    if not body or not body.get("class_code"):
        raise HTTPException(status_code=400, detail="Class code is required")

    code = body["class_code"].strip().upper()
    owner = db.query(User).filter(User.class_code == code).first()
    if not owner:
        raise HTTPException(status_code=400, detail="\u05e7\u05d5\u05d3 \u05db\u05d9\u05ea\u05d4 \u05dc\u05d0 \u05ea\u05e7\u05d9\u05df")

    current_user.joined_class_code = code
    db.commit()
    return {"status": "ok", "joined_class_code": code}


@app.post("/api/leave-class")
def leave_class(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Leave current class."""
    current_user.joined_class_code = None
    db.commit()
    return {"status": "ok"}


# =========================
# Standalone Recommendations
# =========================
class RecommendationRequest(BaseModel):
    scores: Dict[str, int]
    neck_effort: Optional[Dict[str, Any]] = None


@app.post("/api/recommendations")
def get_recommendations(
    body: RecommendationRequest,
    current_user: User = Depends(get_current_user),
):
    """Return practice recommendations for the given GRBAS scores."""
    recs = recommend_with_context(body.scores, neck_effort=body.neck_effort)
    return {"recommendations": flatten_recommendations(recs)}


@app.get("/api/results/{result_id}/recommendations")
def get_result_recommendations(
    result_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Re-generate recommendations for a stored result."""
    row = db.query(Result).filter(
        Result.id == result_id,
        Result.user_id == current_user.id,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Result not found")

    if row.recommendation:
        try:
            return {"recommendations": json.loads(row.recommendation)}
        except (json.JSONDecodeError, TypeError):
            pass

    try:
        data = json.loads(row.raw_scores)
    except (json.JSONDecodeError, TypeError):
        return {"recommendations": []}

    scores = data.get("scores", {})
    neck_effort = data.get("neck_effort")
    recs = recommend_with_context(scores, neck_effort=neck_effort)
    flat = flatten_recommendations(recs)

    row.recommendation = json.dumps(flat)
    db.commit()

    return {"recommendations": flat}


# =========================
# Helpers
# =========================
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


# Mount static files last so API routes take precedence.
app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
