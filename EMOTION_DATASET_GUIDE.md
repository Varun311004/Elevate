# Emotion Dataset Guide

This project uses a fixed 7-class emotion taxonomy:

- happy
- angry
- neutral
- confused
- bored
- focused
- surprised

## Required Folder Layout

Store images in one folder per class under `dataset/`:

- `dataset/happy/`
- `dataset/angry/`
- `dataset/neutral/`
- `dataset/confused/`
- `dataset/bored/`
- `dataset/focused/`
- `dataset/surprised/` or `dataset/surprise/` (both are supported)

Supported formats: `.jpg`, `.jpeg`, `.png`.

## Recommended Distribution

To reduce class dominance (for example angry over-prediction):

- Keep at least 2,000 images per class.
- Prefer 3,000-6,000 images per class.
- Keep largest class no more than about 2.5x the smallest class.

Quick class count check (PowerShell):

```powershell
Get-ChildItem -Path dataset -Directory | ForEach-Object {
  $count = (Get-ChildItem -Path $_.FullName -File | Measure-Object).Count
  "{0}: {1}" -f $_.Name, $count
}
```

## Training Commands

Fast local training (recommended for iteration):

```bash
python scripts/train_emotion_fast.py
```

MobileNetV2 server model training:

```bash
python train_emotion_model.py
```

## Deployment Workflow (Render + HF)

Training is not automatic when backend starts. Your start command (`python -m backend.app`) only runs inference.

Use the deploy preparation script to get clear logs in build pipelines:

```bash
python scripts/prepare_emotion_deploy.py
```

Strict full training (fail-fast) command:

```bash
python scripts/train_strict_pipeline.py --min-emotion-accuracy 0.90 --processes 4
```

Strict verification-only command (fails if artifacts or accuracy gate are not met):

```bash
python scripts/prepare_emotion_deploy.py
```

Recommended split for your setup:

- Train on HF or local machine (faster and better control for quality).
- Commit/publish model artifacts.
- Render build runs verification only.

Suggested Render backend build command:

```bash
pip install --upgrade pip && pip install -r requirements.txt && python scripts/prepare_emotion_deploy.py
```

Suggested frontend build command (always strict for TFJS artifact presence):

```bash
python scripts/render_frontend_build.py
```

### Strict Env Variable Map (Render vs HF)

Set these on **Render backend** service:

- `HF_ML_TRAINING_SERVICE_URL=https://<owner>-<space>.hf.space`
- `HF_ML_TRAINING_SERVICE_TOKEN=<same-secret-used-by-HF-service-auth>`
- `HF_ML_TRAINING_GITHUB_REPO_URL=https://github.com/<owner>/<repo>.git`
- `HF_ML_TRAINING_GITHUB_REF=main`
- `HF_ML_MIN_EMOTION_ACCURACY=0.90`
- `HF_ML_TRAINING_PROCESSES=4`
- `HF_ML_TRAINING_POLL_SECONDS=20`

Set these on **HF Space** (training service):

- `HF_ML_TRAINING_GITHUB_REPO_URL=https://github.com/<owner>/<repo>.git`
- `HF_ML_TRAINING_GITHUB_REF=main`
- `HF_ML_TRAINING_WORKSPACE=/data/elevate_training_workspace`
- `HF_ML_TRAINING_OUTPUT_ROOT=/data/elevate_models_v3/strict_training`
- `AI_TOPIC_SERVICE_TOKEN=<same-secret-used-on-render>`

Must be the same on both sides:

- The service auth secret token (`HF_ML_TRAINING_SERVICE_TOKEN` on Render == `AI_TOPIC_SERVICE_TOKEN` on HF)
- Repo URL/ref values should point to the same branch you want to train.

Can be different on each side:

- Poll interval / request timeout behavior on Render.
- Workspace/storage paths on HF (`/data/...` paths are HF-only).

## Distribution Guidance For Future Users

For production and repository hygiene:

- Do not commit raw image datasets to the repository.
- Share dataset acquisition instructions and source links separately.
- Keep model artifacts and metrics in repository when needed.
- Include provenance and license metadata for each data source.

Suggested manifest fields:

- class_name
- source_name
- source_url
- license
- image_count
- preprocessing_notes
- collection_date

## Troubleshooting Dominant Emotions

If one class dominates predictions:

- Re-check class counts and improve minority coverage.
- Retrain so new calibration maps are exported in TFJS metadata.
- Inspect per-class recall in `backend/ai_models/emotion_model_info.json`.
- Make sure the browser is loading the latest TFJS model files.
