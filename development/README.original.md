---
title: Voice Trainer
emoji: 🎤
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Voice Trainer — GRBAS Analysis & Neck Effort Assessment

A voice training web app that analyzes vocal quality using the GRBAS scale
and provides real-time neck effort (posture) feedback via webcam.

## Features
- **Voice analysis**: GRBAS scoring (Grade, Roughness, Breathiness, Asthenia, Strain)
  from sustained /a/ vowel recordings
- **Neck effort assessment**: Real-time posture analysis using MediaPipe
- **Exercise recommendations**: YouTube-linked voice exercises
- **Progress tracking**: History and daily practice timer
- **Mobile-friendly**: Dedicated mobile interface

## Access
- Desktop: main URL
- Mobile: add `/mobile.html` to the URL

## Model training

The GRBAS classifier is trained in two deliberately separate stages
(`train_grbas.py`). Synthetic TTS-proxy audio is generated *from* its own
labels by a fixed DSP formula, so a model fit only on synthetic data learns to
invert that formula — not clinical voice quality. Synthetic data is therefore
used for pretraining/augmentation only, and every headline metric is measured
on real, clinician-labeled recordings.

```bash
# Stage 1 — synthetic baseline (pipeline check; not a production model)
python render_synthetic_grbas_audio.py --extract-features   # -> synthetic_training_features.csv
python train_grbas.py --mode pretrain                       # -> grbas_pretrained.joblib

# Stage 2 — production model; real data drives train/val/test
python train_grbas.py --mode finetune                       # -> grbas_model.joblib
```

`finetune` mode requires `real_training_features.csv`: one row per
clinician-labeled recording, with the columns in `grbas_model.FEATURE_NAMES`
plus `sample_id`, `speaker_id`, labels `clinician_grbas_{g,r,b,a,s}` (0–3), and
an optional `split` column. Without a `split` column the trainer makes a
speaker-grouped 70/15/15 split. Synthetic rows are mixed into the **train**
split only, down-sampled via `--synthetic-ratio` (0 = real only).

Hard-failure guards: no speaker shared across splits, no synthetic sample in
val/test, no duplicated `sample_id`. The production model is not written unless
the real test set has at least `--min-real-test` samples (default 15).

### Getting real labeled data

The Saarbrücken Voice Database (SVD, CC BY 4.0) provides real healthy and
pathological voices but no GRBAS labels. `build_svd_clinician_sheet.py` selects
a stratified cohort (default 250 speakers — ~40% healthy controls, the rest
spread across diagnoses, gender-balanced — one sustained /a/ recording each)
and writes a clinician rating workbook with 0–3 GRBAS dropdowns:

```bash
python build_svd_clinician_sheet.py --metadata svd_metadata.csv \
    --n-total 250 --audio-src /path/to/svd/wavs --audio-out svd_clinician_audio
```

A clinician fills `svd_clinician_grbas_sheet.xlsx`; feature extraction over the
rated recordings then produces `real_training_features.csv` for `finetune`.

The cohort actually sent out came from the **VOICED** database (PhysioNet,
`voice-icar-federico-ii-database`) — 208 sustained /a/ recordings, one per
speaker. `build_voiced_clinician_web.py` publishes the rating page
(`static/voiced_rating.html` + `static/voiced_audio/`), and the clinician
returns `voiced_grbas_clinician_ratings.csv`:

```bash
# QC first: how many usable ratings came back, and how they are distributed
python build_real_training_features.py --ratings voiced_grbas_clinician_ratings.csv --dry-run

# then extract features -> real_training_features.csv
python build_real_training_features.py --ratings voiced_grbas_clinician_ratings.csv
python train_grbas.py --mode finetune
```

Rows marked `usable = no`, rows with an incomplete G/R/B/A/S set and duplicate
session ids are dropped and reported. Every recording is resampled to 16 kHz
first (`--target-sr`): VOICED is 8 kHz, the app analyses 44.1/48 kHz browser
audio, and features must describe the same bands in both. The workbook route is
accepted too — pass the filled `.xlsx` to `--ratings`.

## Deployment

The app reads two required environment variables in production:

| Var | Notes |
|---|---|
| `DATABASE_URL` | Postgres connection string. `postgres://` is auto-rewritten to `postgresql://`. Without it the app falls back to SQLite under `/data` if a persistent volume is mounted, else next to the code — where the container rebuild wipes every account, invite and saved analysis. |
| `JWT_SECRET_KEY` | Long random string. Generate with `python -c "import secrets; print(secrets.token_urlsafe(48))"`. |

Any free Postgres tier works — Neon, Supabase, Railway, Render. Paste the
connection string into `DATABASE_URL`.

Without `JWT_SECRET_KEY` the app generates a random secret and stores it in the
`app_settings` table, so sessions survive restarts as long as the database does.
It never signs tokens with the placeholder literal that used to be the fallback
(that value is public, so tokens signed with it are forgeable by anyone). If the
DB cannot be reached the secret becomes per-process and everyone is logged out on
each restart; the admin dashboard shows a banner when either of these is the case.

### Hugging Face Spaces

1. Create a new Space, SDK = **Docker**.
2. Push this directory to the Space's repo.
3. Under *Settings → Variables and secrets* add `DATABASE_URL` and `JWT_SECRET_KEY` as **Secrets**. Both are needed before the Space is used for real data: without the first every deploy starts from an empty database, and setting the second keeps sessions valid across deploys.
4. After first boot, run `python seed_admin.py` once (via the Space terminal) to create the super-admin account.

### Railway

`railway.toml` already points at the Dockerfile. Add a Postgres plugin (Railway injects `DATABASE_URL` automatically) and set `JWT_SECRET_KEY` in the service variables.

### Local dev

No env vars needed — SQLite at `./app.db` is used by default.

## Disclaimer
Educational/research tool. Not a clinical diagnostic instrument.


### Observable effort research baseline

Current neck analysis uses `ObservableEffortEngine` from `observable_effort.py`. It reports posture/acoustic correlates and abstains from muscle-effort or diagnosis scores. The desktop UI accepts CFR video + WAV audio with a declared start offset and allows JSON export. The live capture timing is not certified synchronized. See [feasibility, experiments and validation limits](research/OBSERVABLE_EFFORT.md) and [measured proxy results](research/baseline_results.json).

Reproduce the public-data experiment with:

```bash
venv/bin/python research/run_baselines.py /tmp/lombardgrid_audio.zip
```

Dataset source and attribution are in the report. The proxy classifier is not deployed as an effort model.


### Neck Analysis V1 replacement

The active neck engine now uses CV evidence + structured OpenAI interpretation + deterministic five-dimensional scoring. Configure server `OPENAI_API_KEY`; see [Neck Analysis V1](NECK_ANALYSIS_V1.md) for setup, API compatibility, privacy, scoring and exact tests. Earlier observable-correlates notes above document the superseded research baseline, not the current neck runtime.
