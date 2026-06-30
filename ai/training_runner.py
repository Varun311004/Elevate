"""Run strict Elevate ML training from Hugging Face Space runtime.

This runner clones/updates the GitHub repository, executes the strict training
pipeline, and copies resulting model artifacts into persistent HF storage.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download


DEFAULT_WORKSPACE = Path("/data/elevate_training_workspace")
DEFAULT_OUTPUT_ROOT = Path("/data/elevate_models_v3/strict_training")


def _run(command: list[str], cwd: Path) -> None:
    print(f"[hf-train-runner] {' '.join(command)}")
    result = subprocess.run(command, cwd=str(cwd), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")


def _ensure_repo(repo_url: str, repo_ref: str, repo_dir: Path) -> None:
    repo_dir.parent.mkdir(parents=True, exist_ok=True)

    if (repo_dir / ".git").exists():
        _run(["git", "fetch", "--all", "--tags", "--prune"], cwd=repo_dir)
        _run(["git", "checkout", repo_ref], cwd=repo_dir)
        origin_ref = f"origin/{repo_ref}"
        _run(["git", "reset", "--hard", origin_ref], cwd=repo_dir)
    else:
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        _run(["git", "clone", repo_url, str(repo_dir)], cwd=repo_dir.parent)
        _run(["git", "checkout", repo_ref], cwd=repo_dir)


def _copy_path(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Missing expected training artifact path: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


def _is_range_not_satisfiable_error(exc: Exception) -> bool:
    message = str(exc or "").lower()
    return "416" in message and "range" in message and "satisfiable" in message


def _download_snapshot_with_retry(*, repo_id: str, repo_type: str, local_dir: Path, token: str | None) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(local_dir),
            token=token,
        )
        return
    except Exception as exc:
        if not _is_range_not_satisfiable_error(exc):
            raise
        print("[hf-train-runner] WARN: got HTTP 416 during dataset download. Retrying with clean local dir + force download.")

    # Retry path for stale partial downloads / invalid byte-range resume state.
    if local_dir.exists():
        shutil.rmtree(local_dir, ignore_errors=True)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
        token=token,
        force_download=True,
    )


def _sync_artifacts(repo_dir: Path, output_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = output_root / stamp
    latest = output_root / "latest"

    targets = [
        Path("backend/models/bkt"),
        Path("backend/models/dkt"),
        Path("backend/models/at_risk_predictor"),
        Path("backend/models/training"),
        Path("backend/ai_models"),
        Path("frontend/js/emotion_tfjs"),
    ]

    for rel in targets:
        _copy_path(repo_dir / rel, target / rel)

    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(target, latest)
    return target


def _prepare_emotion_dataset(repo_dir: Path) -> None:
    """Ensure repo/dataset exists before strict training starts."""
    target_dir = repo_dir / "dataset"
    target_dir.mkdir(parents=True, exist_ok=True)

    repo_id = str(os.environ.get("HF_EMOTION_DATASET_REPO") or "Sana2704/elevate-emotion-dataset").strip()
    if not repo_id:
        print("[hf-train-runner] No dataset repo configured; skipping emotion dataset download")
        return

    token = os.environ.get("AI_TOPIC_SERVICE_TOKEN") or os.environ.get("HF_TOKEN")
    print(f"[hf-train-runner] Downloading emotion dataset repo={repo_id} -> {target_dir}")
    _download_snapshot_with_retry(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=target_dir,
        token=token,
    )
    print("[hf-train-runner] Emotion dataset ready")


def _load_summary(repo_dir: Path) -> dict[str, Any]:
    summary_path = repo_dir / "backend" / "models" / "training" / "strict_training_summary_latest.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HF strict training runner")
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--repo-ref", default="main")
    parser.add_argument("--min-emotion-accuracy", type=float, default=0.95)
    parser.add_argument("--processes", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    repo_url = str(args.repo_url).strip()
    repo_ref = str(args.repo_ref).strip() or "main"
    if not repo_url:
        print("[hf-train-runner] ERROR: repo url is required")
        return 2

    workspace = Path(os.environ.get("HF_ML_TRAINING_WORKSPACE") or DEFAULT_WORKSPACE)
    output_root = Path(os.environ.get("HF_ML_TRAINING_OUTPUT_ROOT") or DEFAULT_OUTPUT_ROOT)
    python_bin = sys.executable
    repo_dir = workspace / "repo"

    try:
        print(f"[hf-train-runner] workspace={workspace}")
        print(f"[hf-train-runner] output_root={output_root}")
        print(f"[hf-train-runner] repo_url={repo_url}")
        print(f"[hf-train-runner] repo_ref={repo_ref}")

        _ensure_repo(repo_url, repo_ref, repo_dir)

        _run([python_bin, "-m", "pip", "install", "--upgrade", "pip"], cwd=repo_dir)
        _run([python_bin, "-m", "pip", "install", "-r", "requirements.txt"], cwd=repo_dir)
        _run([python_bin, "-m", "pip", "install", "-r", "ai/requirements.txt"], cwd=repo_dir)
        _prepare_emotion_dataset(repo_dir)

        _run(
            [
                python_bin,
                "scripts/train_strict_pipeline.py",
                "--min-emotion-accuracy",
                str(args.min_emotion_accuracy),
                "--processes",
                str(max(1, int(args.processes))),
            ],
            cwd=repo_dir,
        )

        artifact_path = _sync_artifacts(repo_dir, output_root)
        summary = _load_summary(repo_dir)

        print(f"[hf-train-runner] artifacts_synced={artifact_path}")
        if summary:
            print(f"[hf-train-runner] summary={json.dumps(summary)}")
        return 0
    except Exception as exc:
        print(f"[hf-train-runner] ERROR: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())