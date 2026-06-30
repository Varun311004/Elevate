#!/usr/bin/env bash
set -euo pipefail

MODEL_CACHE_DIR="${MODELS_CACHE_DIR:-/data/elevate_models_cache}"
BUCKET_URI="${HF_BUCKET_URI:-}"
BUCKET_CACHE_PREFIX="${HF_BUCKET_CACHE_PREFIX:-elevate_models_cache}"

if [ -n "$BUCKET_URI" ]; then
	restore_from_bucket() {
		echo "[topic-mcq] bucket restore enabled"
		echo "[topic-mcq] source=${BUCKET_URI%/}/${BUCKET_CACHE_PREFIX} target=${MODEL_CACHE_DIR}"
		mkdir -p "$MODEL_CACHE_DIR"

		if command -v hf >/dev/null 2>&1; then
			if hf sync "${BUCKET_URI%/}/${BUCKET_CACHE_PREFIX}" "$MODEL_CACHE_DIR"; then
				echo "[topic-mcq] bucket restore completed"
			else
				echo "[topic-mcq] bucket restore failed; continuing with local/runtime cache"
			fi
		else
			echo "[topic-mcq] hf CLI not found; skipping bucket restore"
		fi
	}
	restore_from_bucket
fi

uvicorn app:app --host 0.0.0.0 --port "${PORT:-7860}"
