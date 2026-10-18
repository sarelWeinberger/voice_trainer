---
title: Voice Trainer
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Voice Trainer

FastAPI application with GRBAS analysis, experimental visual neck observations,
head-motion measurements, exercises, and account administration.

## Run locally

From the repository root, run `./run.sh`, then open http://localhost:5000.
The launcher uses `root/voice_app/venv/bin/python`. For a fresh environment:

```bash
python3 -m venv root/voice_app/venv
root/voice_app/venv/bin/pip install -r root/voice_app/requirements.txt
```

Set `OPENAI_API_KEY` in the server environment for neck interpretation.
Export environment variables in the terminal before launching; the existing
launcher does not load `.env` files automatically. Never put keys in frontend
code. An optional `grbas_model.joblib` beside `app.py` is loaded when
present; otherwise the existing GRBAS fallback is used.

## Hugging Face

Use this directory as the Docker Space repository. Configure `OPENAI_API_KEY`
as a Space Secret. Set `DATABASE_URL` for a persistent PostgreSQL database and
`JWT_SECRET_KEY` for a stable signing secret. Without `DATABASE_URL`, SQLite
uses `APP_DATA_DIR`, an existing writable `/data`, or the app directory.
Data in the container's disposable filesystem does not survive rebuilds.

`./deploy_hf.sh` stages only `runtime-files.txt` plus an optional model file,
then commits and pushes to the existing Space. `HF_DIR` can select a deployment
checkout. The Docker context has a separate runtime allowlist.

Desktop uses `/`; mobile uses `/mobile.html`; administration uses `/admin`.
The existing clinician rating page remains at `/voiced_rating.html`.

## Development

Tests live at repository-level `tests/`; run `./check.sh` from the repository
root. Training tools, datasets, prior research, and detailed neck-analysis
notes live in `development/`, outside the runtime directory and Docker image.
See `development/README.md` in the source repository.

Neck grades describe experimental external observations, not direct muscle
tension or a diagnosis. Their clinical accuracy has not been validated.
