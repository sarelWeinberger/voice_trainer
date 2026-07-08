FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libxcb1 libxcb-render0 libxcb-shm0 \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    gcc g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages system-wide (as root during build)
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true
# --no-deps is critical: without it, opencv re-resolves its numpy>=2 dep and
# pulls the latest numpy (2.5+), overriding the numpy<2.4 pin in requirements.
# That breaks `import numba` ("Numba needs NumPy 2.4 or less") in the check below.
RUN pip install --no-cache-dir --force-reinstall --no-deps 'opencv-python-headless>=4.8'
RUN python -c "import mediapipe as mp, numpy, numba; assert hasattr(mp, 'solutions'), 'MediaPipe legacy solutions API missing'; print(mp.__version__, numpy.__version__, numba.__version__)"
# Pre-download MediaPipe solution models at build time (as root, while
# site-packages is writable). At runtime the app runs as non-root 'user' and
# CANNOT write here, so without this the first Pose()/FaceMesh() call dies with
# PermissionError trying to download the .tflite model. Instantiate with the
# SAME params neck_effort_engine uses so every needed model is fetched now; the
# chmod below then makes the downloaded files world-readable.
RUN python -c "import mediapipe as mp; \
p = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2); p.close(); \
f = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True); f.close(); \
print('MediaPipe models pre-downloaded')"
# Patch webrtcvad.py: it uses pkg_resources only for __version__
RUN python -c "\
import glob; \
[open(f,'w').write(open(f).read().replace('import pkg_resources\n','').replace(\"pkg_resources.get_distribution('webrtcvad').version\",\"'2.0.10'\")) \
for f in glob.glob('/usr/local/lib/python*/site-packages/webrtcvad.py')]"

# Create non-root user required by HF Spaces
RUN useradd -m -u 1000 user

COPY . .
RUN mkdir -p /memories && cp memories/allowed_emails.txt /memories/allowed_emails.txt \
    && chown -R user:user /app /memories

RUN chmod -R a+rX /usr/local/lib/python3.12/site-packages

USER user
ENV PORT=7860 \
    HOME=/home/user \
    PYTHONPATH=/usr/local/lib/python3.12/site-packages
EXPOSE 7860
CMD /usr/local/bin/python -m uvicorn app:app --host 0.0.0.0 --port $PORT
