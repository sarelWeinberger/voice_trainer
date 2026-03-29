FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libxcb1 libxcb-render0 libxcb-shm0 \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    gcc g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip uninstall -y opencv-python opencv-contrib-python || true
RUN pip install --no-cache-dir 'opencv-python-headless>=4.8'
# Patch webrtcvad.py: it uses pkg_resources only for __version__, replace with literal
RUN python -c "\
import glob; \
[open(f,'w').write(open(f).read().replace('import pkg_resources\n','').replace(\"pkg_resources.get_distribution('webrtcvad').version\",\"'2.0.10'\")) \
for f in glob.glob('/usr/local/lib/python*/site-packages/webrtcvad.py')]"

COPY . .
ENV PORT=8000
CMD python -m uvicorn app:app --host 0.0.0.0 --port $PORT
