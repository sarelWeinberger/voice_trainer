#!/usr/bin/env bash
# Publish only the runtime file list; never mirror the entire working tree.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_DIR="${HF_DIR:-$SRC/../../.deploy/voice-trainer}"
HF_URL="https://huggingface.co/spaces/Voice-trainer/voice-trainer"
BRANCH=main
MSG="${1:-Update Voice Trainer runtime}"

command -v rsync >/dev/null || { echo "ERROR: rsync is required"; exit 1; }
if [ ! -d "$HF_DIR/.git" ]; then
    mkdir -p "$(dirname "$HF_DIR")"
    GIT_LFS_SKIP_SMUDGE=1 git clone "$HF_URL" "$HF_DIR"
fi
REMOTE_URL="$(git -C "$HF_DIR" remote get-url origin)"
if [ "$REMOTE_URL" != "$HF_URL" ] && [ "$REMOTE_URL" != "$HF_URL.git" ]; then
    echo "ERROR: deployment checkout does not point to the expected Space"
    exit 1
fi
if [ -n "$(git -C "$HF_DIR" status --porcelain)" ]; then
    echo "ERROR: deployment checkout has uncommitted changes: $HF_DIR"
    exit 1
fi
git -C "$HF_DIR" pull --ff-only origin "$BRANCH"

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/voice-runtime.XXXXXX")"
trap 'rm -rf -- "$STAGE"' EXIT
rsync -ar --files-from="$SRC/runtime-files.txt" "$SRC/" "$STAGE/"
if [ -f "$SRC/grbas_model.joblib" ]; then
    cp "$SRC/grbas_model.joblib" "$STAGE/"
fi

# Preserve repository metadata and any existing local state in the checkout.
rsync -a --delete \
    --exclude='.git/' --exclude='.gitattributes' --exclude='.env' --exclude='.env.*' \
    --exclude='service-account-key.json' --exclude='*.db*' \
    --exclude='venv/' --exclude='__pycache__/' --exclude='data/' \
    "$STAGE/" "$HF_DIR/"
git -C "$HF_DIR" add -A
if ! git -C "$HF_DIR" diff --cached --quiet; then
    git -C "$HF_DIR" commit -m "$MSG"
fi
git -C "$HF_DIR" push origin "$BRANCH"
echo "Published. Build status: $HF_URL"
