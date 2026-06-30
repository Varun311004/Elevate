"""Generate frontend runtime config for Render static-site builds.

Environment variables consumed:
- FRONTEND_API_BASE_URL: full backend API base URL, e.g. https://my-backend.onrender.com/api
"""

from __future__ import annotations

import os
from pathlib import Path


def normalize_api_base(value: str) -> str:
    return (value or "").strip().rstrip("/")



def report_emotion_tfjs_assets(repo_root: Path) -> int:
    tfjs_dir = repo_root / "frontend" / "js" / "emotion_tfjs"
    model_json = tfjs_dir / "model.json"
    weights_bin = tfjs_dir / "group1-shard1of1.bin"

    has_model = model_json.exists()
    has_weights = weights_bin.exists()

    print(f"[render-frontend-build] emotion tfjs model.json={'present' if has_model else 'missing'} ({model_json})")
    print(f"[render-frontend-build] emotion tfjs weights={'present' if has_weights else 'missing'} ({weights_bin})")

    if not (has_model and has_weights):
        print("[render-frontend-build] ERROR: required TFJS artifacts are missing")
        return 1

    return 0


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / "frontend" / "js" / "runtime-config.js"

    api_base = normalize_api_base(os.environ.get("FRONTEND_API_BASE_URL", ""))
    content = (
        "// Auto-generated during Render static-site build.\n"
        "window.ELEVATE_RUNTIME_CONFIG = window.ELEVATE_RUNTIME_CONFIG || {};\n"
        f"window.ELEVATE_RUNTIME_CONFIG.API_BASE_URL = '{api_base}';\n"
        f"window.__ELEVATE_API_BASE_URL__ = '{api_base}';\n"
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"[render-frontend-build] Wrote {target}")
    print(f"[render-frontend-build] FRONTEND_API_BASE_URL={'(empty)' if not api_base else api_base}")

    return report_emotion_tfjs_assets(repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
