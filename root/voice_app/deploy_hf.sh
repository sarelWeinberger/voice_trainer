#!/usr/bin/env bash
#
# Deploy the local voice_app working tree to the Hugging Face Space.
#
# Why this exists: the HF Space lives in a SEPARATE git repo whose root is the
# *contents* of root/voice_app/ (not the GitHub monorepo root), and there is no
# GitHub -> HF sync. So pushing to GitHub does NOT update the Space. This script
# mirrors the local app files into a local clone of the Space and pushes them.
#
# Source of truth = your local working tree (committed or not). Whatever is in
# root/voice_app right now is what goes live.
#
# ONE-TIME AUTH SETUP (only needed once per machine):
#   1. Create a *write* token: https://huggingface.co/settings/tokens
#   2. Run this script. On the first push, git will ask for credentials:
#        Username: <your HF username>
#        Password: <paste the write token>
#      Because credential.helper=store is set in the clone, it is remembered.
#
# Usage:
#   ./deploy_hf.sh                       # deploy with an auto commit message
#   ./deploy_hf.sh "fix login allowlist" # deploy with a custom message
#
set -euo pipefail

SRC="/home/elira/voice_trainer/root/voice_app"
HF_DIR="/home/elira/hf_voice_trainer_space"
HF_URL="https://huggingface.co/spaces/Voice-trainer/voice-trainer"
BRANCH="main"
MSG="${1:-Deploy local voice_app to HF Space}"

command -v rsync >/dev/null || { echo "ERROR: rsync not installed"; exit 1; }

# Clone the Space if we don't have a local copy yet.
if [ ! -d "$HF_DIR/.git" ]; then
  echo "No local Space clone found — cloning $HF_URL ..."
  GIT_LFS_SKIP_SMUDGE=1 git clone "$HF_URL" "$HF_DIR"
fi

# Make pushes remember the token, and give commits an identity.
git -C "$HF_DIR" config credential.helper store
git -C "$HF_DIR" config user.name  "Sarel Weinberger"
git -C "$HF_DIR" config user.email "sarelw1@gmail.com"

# Pull first so we don't diverge from any edits made directly on HF.
git -C "$HF_DIR" pull --ff-only origin "$BRANCH" || true

# Mirror local app files into the Space clone.
#   --delete keeps the Space == local (removes stale files there).
#   Excludes mirror .dockerignore + .gitignore: secrets, venv, bulky dev data.
rsync -a --delete \
  --exclude='.git/' \
  --exclude='venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' --exclude='*.pyo' --exclude='*.pyd' \
  --exclude='*.db' \
  --exclude='.env' \
  --exclude='service-account-key.json' \
  --exclude='tmp_synth/' \
  --exclude='synthetic_audio/' \
  --exclude='synthetic_training_features*.csv' \
  --exclude='clinical_review_queue.csv' \
  --exclude='publication_data_targets.csv' \
  --exclude='synthetic_grbas_manifest.csv' \
  --exclude='voiced_data/' \
  --exclude='voiced_database/' \
  --exclude='voiced_clinician_audio/' \
  --exclude='voiced_clinician_grbas_sheet.xlsx' \
  --exclude='voiced_clinician_package.zip' \
  --exclude='voiced_clinician_rating.html' \
  --exclude='*.bak' \
  --exclude='.vscode/' --exclude='.idea/' \
  "$SRC/" "$HF_DIR/"

# Commit (only if files changed), then always push — this also flushes any
# commit made on a previous run whose push failed (e.g. missing token).
git -C "$HF_DIR" add -A
if git -C "$HF_DIR" diff --cached --quiet; then
  echo "No file changes to commit."
else
  git -C "$HF_DIR" commit -m "$MSG"
fi

if [ -z "$(git -C "$HF_DIR" rev-list "origin/$BRANCH..$BRANCH" 2>/dev/null)" ]; then
  echo "Space already up to date — nothing to push."
  exit 0
fi
echo "Pushing to HF Space ..."
git -C "$HF_DIR" push origin "$BRANCH"
echo "Done. The Space will rebuild automatically; watch: $HF_URL"
