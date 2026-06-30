#!/usr/bin/env bash
set -euo pipefail

# --- Instructions ---
# 1. Create a DOCKER space at https://huggingface.co/new-space
# 2. Set HF_SPACE_URL below
# 3. Run from project root: bash ./publish_to_hf.sh
# --------------------

# --- Configuration ---
HF_SPACE_URL="https://huggingface.co/spaces/Sana2704/elevate"
SOURCE_DIR="ai"
# --------------------

if [ "$HF_SPACE_URL" = "YOUR_HUGGING_FACE_SPACE_URL" ]; then
  echo "ERROR: Replace YOUR_HUGGING_FACE_SPACE_URL with your real HF Space URL."
  exit 1
fi

if [ ! -d "$SOURCE_DIR" ]; then
  echo "ERROR: SOURCE_DIR '$SOURCE_DIR' does not exist."
  exit 1
fi

if [ ! -f "$SOURCE_DIR/models/llm.py" ]; then
  echo "ERROR: Missing $SOURCE_DIR/models/llm.py."
  exit 1
fi

if [ -f "$SOURCE_DIR/.dockerignore" ] && grep -E '^models/?$' "$SOURCE_DIR/.dockerignore" >/dev/null 2>&1; then
  echo "ERROR: $SOURCE_DIR/.dockerignore excludes 'models'. Remove that line."
  exit 1
fi

# Generate tar exclude patterns from .dockerignore
TAR_EXCLUDES=""
if [ -f "$SOURCE_DIR/.dockerignore" ]; then
    echo "Processing .dockerignore for tar exclusions..."
    while IFS= read -r line; do
        # Strip comments and trim whitespace
        line=$(echo "$line" | sed -e 's/#.*$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')
        if [ -n "$line" ]; then
            TAR_EXCLUDES+=" --exclude='$line'"
        fi
    done < "$SOURCE_DIR/.dockerignore"
fi

TMP_DIR=""
cleanup() {
  if [ -n "$TMP_DIR" ] && [ -d "$TMP_DIR" ]; then
    rm -rf "$TMP_DIR"
  fi
}
trap cleanup EXIT

TMP_DIR=$(mktemp -d 2>/dev/null || mktemp -d -t hfspace)
echo "--- Preparing temporary workspace: $TMP_DIR ---"

# Copy source contents (including dotfiles) into temp workspace.
(cd "$SOURCE_DIR" && tar -cf - . $TAR_EXCLUDES) | (cd "$TMP_DIR" && tar -xf -)

# Ensure no nested git metadata is carried over.
if [ -d "$TMP_DIR/.git" ]; then
  rm -rf "$TMP_DIR/.git"
fi

echo "Initializing temporary git repository..."
git -C "$TMP_DIR" init

echo "Setting temporary git author for this deployment..."
git -C "$TMP_DIR" config user.name "Sana-ai-coder"
git -C "$TMP_DIR" config user.email "sanagirish0@gmail.com"

git -C "$TMP_DIR" remote add origin "$HF_SPACE_URL"
git -C "$TMP_DIR" add -A

if ! git -C "$TMP_DIR" diff --cached --quiet; then
  git -C "$TMP_DIR" commit -m "Deploy MCQ service"
else
  echo "No file changes detected in temporary workspace; creating an empty deploy commit."
  git -C "$TMP_DIR" commit --allow-empty -m "Deploy MCQ service"
fi

echo "--- Pushing to Hugging Face Space ---"
git -C "$TMP_DIR" push --force -u origin main

echo ""
echo "--- Done ---"
echo "HF Space push complete. Check build logs on Hugging Face."
echo "Your local '$SOURCE_DIR' directory was not modified with nested .git metadata."
