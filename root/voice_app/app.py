"""FastAPI backend for Voice Training App with GRBAS analysis.

Authentication is handled via JWT tokens with OAuth2 password flow.
Data is stored in SQLite via SQLAlchemy (see database.py).
"""

import json
import pathlib
import os
import random
import string
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

NECK_EFFORT_CONFIG = NeckEffortConfig(calibration_frames=45, smooth_window=20)
neck_effort_engines: Dict[int, NeckEffortEngine] = {}


# =========================
# Schemas
# =========================

SUPER_ADMIN_EMAIL = "sarelw1@gmail.com"

# Load allowed emails from memory file
ALLOWED_EMAILS_PATH = "/memories/allowed_emails.txt"
def load_allowed_emails():
    try:
        with open(ALLOWED_EMAILS_PATH, "r") as f:
            emails = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        return set(emails)
    except Exception:
        return set()

ALLOWED_EMAILS = load_allowed_emails()


def generate_class_code() -> str:
    """Generate a random 6-char alphanumeric class code."""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))


class UserCreate(BaseModel):
    username: str
    password: str
    full_name: Optional[str] = None
    class_code: Optional[str] = None  # optional class code to join


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
    # Allow login for all existing users (backward compatible)
    # For users in the allowlist, require password 'qwertyui'
    if user.username in ALLOWED_EMAILS:
        if password == "qwertyui":
            return user
        else:
            return None
    # For all other users, use their set password
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
    return engine.state.last_valid_result


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

    existing = db.query(User).filter(User.username == user_in.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    # Only allow registration if email is in allowlist or is a legacy user
    if user_in.username not in ALLOWED_EMAILS:
        raise HTTPException(status_code=403, detail="User not allowed to register")

    # Password is always 'qwertyui' for allowlist users
    if user_in.password != "qwertyui":
        raise HTTPException(status_code=400, detail="Password must be 'qwertyui' for allowed users")

    # Validate class code if provided
    joined_class = None
    if user_in.class_code and user_in.class_code.strip():
        code = user_in.class_code.strip().upper()
        owner = db.query(User).filter(User.class_code == code).first()
        if not owner:
            raise HTTPException(status_code=400, detail="קוד כיתה לא תקין")
        joined_class = code

    user = User(
        username=user_in.username,
        full_name=user_in.full_name,
        hashed_password=hash_password("qwertyui"),
        is_admin=False,
        role=3,
        joined_class_code=joined_class,
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
