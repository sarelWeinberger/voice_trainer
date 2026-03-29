FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    libxcb1 libxcb-render0 libxcb-shm0 \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    gcc g++ build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir setuptools uvicorn[standard] fastapi
RUN pip install --no-cache-dir -r requirements.txt
RUN pip uninstall -y opencv-python opencv-contrib-python || true
RUN pip install --no-cache-dir 'opencv-python-headless>=4.8'

COPY . .
ENV PORT=8000
CMD python -m uvicorn app:app --host 0.0.0.0 --port $PORT
