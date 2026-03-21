"""FastAPI backend for Voice Training App with GRBAS analysis.

Authentication is handled via JWT tokens with OAuth2 password flow.
Data is stored in SQLite via SQLAlchemy (see database.py).
"""

import json
import os
import tempfile
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from jose import JWTError, jwt

from database import SessionLocal, User, Result, init_db
from grbas_model import GRBASClassifier, extract_features
from neck_effort_engine import NeckEffortConfig, NeckEffortEngine
from recommendations import recommend_with_context, flatten_recommendations

# =========================
# Config
# =========================
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "replace_this_with_a_long_random_secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/login")

app = FastAPI(title="Voice Training - GRBAS Analysis")

# Initialise database tables on startup (idempotent).
init_db()

# Load GRBAS model if a pre-trained file exists.
classifier = GRBASClassifier()
MODEL_PATH = os.path.join(os.path.dirname(__file__), "grbas_model.joblib")
if os.path.exists(MODEL_PATH):
    classifier.load(MODEL_PATH)

NECK_EFFORT_CONFIG = NeckEffortConfig(calibration_frames=12, smooth_window=12)
neck_effort_engines: Dict[int, NeckEffortEngine] = {}


# =========================
# Schemas
# =========================
class UserCreate(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = None


class UserOut(BaseModel):
    id: int
    username: str
    full_name: Optional[str]
    is_admin: bool
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
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


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
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
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
    return engine.state.last_valid_result


# =========================
# Seed admin on startup
# =========================
@app.on_event("startup")
def create_default_admin():
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == "admin").first()
        if not existing:
            admin = User(
                username="admin",
                full_name="System Admin",
                hashed_password=hash_password("admin1234"),
                is_admin=True,
            )
            db.add(admin)
            db.commit()
    finally:
        db.close()


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
    if len(user_in.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    existing = db.query(User).filter(User.username == user_in.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    user = User(
        username=user_in.username,
        full_name=user_in.full_name,
        hashed_password=hash_password(user_in.password),
        is_admin=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.post("/api/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/api/me", response_model=UserOut)
def me(current_user: User = Depends(get_current_user)):
    return current_user


# =========================
# Voice Analysis
# =========================
@app.post("/api/analyze")
async def analyze(
    audio: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        audio_data = await audio.read()
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

        latest_neck_effort = get_latest_neck_effort(current_user)
        recs = recommend_with_context(scores, neck_effort=latest_neck_effort)
        recommendations = flatten_recommendations(recs)

        response_data = {
            "scores": scores,
            "features": key_features,
            "model_type": "trained" if classifier.is_trained else "heuristic",
            "neck_effort": latest_neck_effort,
            "recommendations": recommendations,
        }

        # Persist result for authenticated user.
        result = Result(
                user_id=current_user.id,
                title="GRBAS Analysis",
                raw_scores=json.dumps({
                    "scores": scores,
                    "features": key_features,
                    "model_type": response_data["model_type"],
                    "neck_effort": latest_neck_effort,
                }),
                recommendation=json.dumps(recommendations) if recommendations else None,
            )
        db.add(result)
        db.commit()

        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/api/neck-effort/reset")
def reset_neck_effort(current_user: User = Depends(get_current_user)):
    engine = get_neck_effort_engine(current_user.id)
    engine.reset_session()
    return {"status": "ok"}


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

    engine = get_neck_effort_engine(current_user.id)
    return engine.process_frame(image_bgr)


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
            history.append({
                "id": row.id,
                "timestamp": row.created_at.isoformat(),
                "scores": data.get("scores", {}),
                "features": data.get("features", {}),
                "model_type": data.get("model_type", "unknown"),
                "neck_effort": data.get("neck_effort"),
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
    users = db.query(User).order_by(User.created_at.desc()).all()
    result = []

    for u in users:
        latest = (
            db.query(Result)
            .filter(Result.user_id == u.id)
            .order_by(Result.created_at.desc())
            .first()
        )

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
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "latest_analysis": latest_analysis,
        })

    return {"users": result}


@app.get("/api/admin/users/{uid}/results")
def get_user_results(
    uid: int,
    db: Session = Depends(get_db),
    admin_user: User = Depends(get_admin_user),
):
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
            results.append({
                "id": row.id,
                "timestamp": row.created_at.isoformat(),
                "scores": data.get("scores", {}),
                "features": data.get("features", {}),
                "model_type": data.get("model_type", "unknown"),
                "neck_effort": data.get("neck_effort"),
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
    """Set a user's admin status."""
    target = db.query(User).filter(User.id == uid).first()
    if target is None:
        raise HTTPException(status_code=404, detail="User not found")

    target.is_admin = body.is_admin
    db.commit()
    return {"status": "ok", "is_admin": target.is_admin}


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
