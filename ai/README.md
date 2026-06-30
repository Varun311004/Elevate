---
title: Elevate Topic MCQ Service
sdk: docker
app_port: 7860
---

# Elevate Topic MCQ Service

This is a FastAPI-based service for generating Multiple Choice Questions from topics.

## Runtime Notes

- The service preloads the model during startup and only reports ready after model load completes.
- Set `MODELS_CACHE_DIR=/data/elevate_models_cache` on Hugging Face Spaces to reuse cache across restarts.

## Optional Bucket Restore

You can restore cache from a Hugging Face bucket at container start by setting:

- `HF_BUCKET_URI=hf://buckets/<username>/<bucket-name>`
- `HF_BUCKET_CACHE_PREFIX=elevate_models_cache`

When configured, `start.sh` runs `hf sync` from that bucket path into the configured model cache directory.

## Deploy Checklist (Render + HF Space)

1. Configure Hugging Face Space runtime env vars.
   - Required: `MODELS_CACHE_DIR=/data/elevate_models_cache`
   - Optional (recommended for faster cold starts):
     - `HF_BUCKET_URI=hf://buckets/<username>/<bucket-name>`
     - `HF_BUCKET_CACHE_PREFIX=elevate_models_cache`
   - Optional for private models: `HF_TOKEN=<token>`

2. Confirm readiness behavior.
   - The API becomes available only after model preload completes.
   - `GET /health` returns `503` with `status=starting` until `model_ready=true`.

3. Configure Render backend env vars.
   - `AI_TOPIC_SERVICE_URL=https://<owner>-<space>.hf.space`
   - `AI_TOPIC_SERVICE_TIMEOUT_SECONDS=45` (increase if your Space is slow to respond)
   - If AI auth is enabled on the Space:
     - `AI_TOPIC_SERVICE_TOKEN=<shared-token>`
     - `AI_TOPIC_SERVICE_AUTH_SCHEME=Bearer`

4. Run a smoke test after deploy.
   - Verify Space: `GET /health` reports `model_ready=true`.
   - Verify Space generation: `POST /mcq/generate` returns valid MCQs.
   - Verify backend integration: create a teacher test/question-bank and confirm service-generated questions.

## Strict ML Training API

This Space also exposes a strict training relay API used by Render backend predeploy.

Endpoints:

- `POST /training/strict/start`
- `GET /training/strict/status/{job_id}`

Behavior:

- Training is strict and fail-fast: if any stage fails, job status becomes `failed` with stderr/stdout tails.
- Render waits by polling status until terminal state (`succeeded` or `failed`) with no internal max-wait cutoff in backend relay.

Required environment variables for strict training job execution:

- `HF_ML_TRAINING_GITHUB_REPO_URL` (for example: `https://github.com/<owner>/<repo>.git`)
- Optional: `HF_ML_TRAINING_GITHUB_REF` (default: `main`)
- Optional: `HF_ML_TRAINING_WORKSPACE` (default: `/data/elevate_training_workspace`)
- Optional: `HF_ML_TRAINING_OUTPUT_ROOT` (default: `/data/elevate_models_v3/strict_training`)

The training runner clones the repo, installs requirements, runs `scripts/train_strict_pipeline.py`, and syncs resulting artifacts to persistent HF storage.
