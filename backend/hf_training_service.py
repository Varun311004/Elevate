"""
backend/hf_training_service.py
===============================
Thin HTTP client that talks to the Hugging Face Spaces training service.
All config comes from environment variables — zero hardcoding.
Supabase-safe: no DB calls here.
"""

from __future__ import annotations

import os
import time
import requests
from flask import current_app


def _start_request_timeout_sec() -> float:
    raw = str(
        os.environ.get("HF_TRAINING_START_TIMEOUT_SEC")
        or os.environ.get("AI_TOPIC_SERVICE_START_TIMEOUT_SEC")
        or "300"
    ).strip()
    try:
        return max(30.0, min(float(raw), 900.0))
    except ValueError:
        return 300.0


def _status_request_timeout_sec() -> float:
    raw = str(
        os.environ.get("HF_TRAINING_STATUS_TIMEOUT_SEC")
        or os.environ.get("AI_TOPIC_SERVICE_STATUS_TIMEOUT_SEC")
        or "60"
    ).strip()
    try:
        return max(10.0, min(float(raw), 300.0))
    except ValueError:
        return 60.0


def get_hf_training_service_url() -> str:
    return os.environ.get(
        "AI_TOPIC_SERVICE_URL",
        current_app.config.get("AI_TOPIC_SERVICE_URL", "")
    ).rstrip("/")


def _hf_headers() -> dict:
    # FIX 1: Look for AI_TOPIC_SERVICE_TOKEN to match your Render environment variables!
    token = os.environ.get(
        "AI_TOPIC_SERVICE_TOKEN",
        current_app.config.get("AI_TOPIC_SERVICE_TOKEN", "")
    )
    h = {"Content-Type": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def start_hf_strict_training(payload: dict | None = None) -> dict:
    """
    POST to /training/strict/start on the HF training service.
    Returns: { ok, payload, latency_ms, error, status_code }

    Uses a long default timeout: HF Spaces may cold-start and only accept HTTP after
    the app finishes bootstrapping (can be minutes on free tier).
    """
    base_url = get_hf_training_service_url()
    if not base_url:
        return {
            "ok": False,
            "error": "AI_TOPIC_SERVICE_URL not configured",
            "status_code": 503,
        }

    url = f"{base_url}/training/strict/start"
    timeout = _start_request_timeout_sec()
    t0 = time.time()
    try:
        resp = requests.post(
            url,
            json=payload or {},
            headers=_hf_headers(),
            timeout=timeout,
        )
        latency = int((time.time() - t0) * 1000)

        # Already running: Space returns 409 with existing job_id — treat as success for admin UI.
        if resp.status_code == 409:
            try:
                data = resp.json()
            except Exception:
                data = {}
            if isinstance(data, dict) and data.get("job_id"):
                return {
                    "ok": True,
                    "payload": data,
                    "latency_ms": latency,
                    "status_code": resp.status_code,
                }

        if not resp.ok:
            return {
                "ok": False,
                "error": resp.text[:512],
                "status_code": resp.status_code,
                "latency_ms": latency,
            }
        try:
            body = resp.json()
        except Exception:
            body = {}
        return {
            "ok": True,
            "payload": body if isinstance(body, dict) else {},
            "latency_ms": latency,
            "status_code": resp.status_code,
        }
    except requests.exceptions.Timeout:
        return {
            "ok": False,
            "error": (
                f"Request timed out after {int(timeout)}s waiting for the training service "
                "(Space cold start or overload). Retry, or raise HF_TRAINING_START_TIMEOUT_SEC."
            ),
            "status_code": 504,
        }
    except requests.exceptions.ConnectionError as exc:
        return {"ok": False, "error": f"Connection error: {exc}", "status_code": 503}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "status_code": 500}


def get_hf_strict_training_status(job_id: str) -> dict:
    """
    GET /training/strict/status/<job_id> on the HF training service.
    Returns: { ok, payload, latency_ms, error, status_code }
    """
    base_url = get_hf_training_service_url()
    if not base_url:
        return {
            "ok": False,
            "error": "AI_TOPIC_SERVICE_URL not configured",
            "status_code": 503,
        }

    # FIX 3: Match the app.py exact path for status checking!
    url = f"{base_url}/training/strict/status/{job_id}"
    timeout = _status_request_timeout_sec()
    t0 = time.time()
    try:
        resp = requests.get(url, headers=_hf_headers(), timeout=timeout)
        latency = int((time.time() - t0) * 1000)
        if not resp.ok:
            return {
                "ok": False,
                "error": resp.text[:512],
                "status_code": resp.status_code,
                "latency_ms": latency,
            }
        return {
            "ok": True,
            "payload": resp.json(),
            "latency_ms": latency,
            "status_code": resp.status_code,
        }
    except requests.exceptions.Timeout:
        return {"ok": False, "error": "Request timed out", "status_code": 504}
    except requests.exceptions.ConnectionError as exc:
        return {"ok": False, "error": f"Connection error: {exc}", "status_code": 503}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "status_code": 500}


def trigger_and_wait_hf_strict_training(
    payload: dict | None = None,
    *,
    poll_interval_sec: float = 12.0,
    max_wait_sec: float | None = None,
) -> dict:
    """
    Start strict training on the HF service and poll until the job reaches a terminal state.
    Used by bootstrap scripts when ELEVATE_STRICT_REMOTE_HF=1.
    """
    start = start_hf_strict_training(payload)
    if not start.get("ok"):
        return {**start, "endpoint": "start"}

    body = start.get("payload") if isinstance(start.get("payload"), dict) else {}
    job_id = str(body.get("job_id") or "").strip()
    if not job_id:
        return {
            "ok": False,
            "error": "HF start response missing job_id",
            "status_code": 500,
            "endpoint": "start",
        }

    raw_max = str(os.environ.get("HF_TRAINING_WAIT_MAX_SEC") or "").strip()
    try:
        limit = float(raw_max) if raw_max else (max_wait_sec if max_wait_sec is not None else 14400.0)
    except ValueError:
        limit = 14400.0
    limit = max(60.0, min(limit, 86400.0))

    t0 = time.time()
    last_payload: dict = dict(body)

    while (time.time() - t0) < limit:
        status = get_hf_strict_training_status(job_id)
        if not status.get("ok"):
            return {**status, "endpoint": "status", "job_id": job_id}

        last_payload = status.get("payload") if isinstance(status.get("payload"), dict) else {}
        state = str(last_payload.get("status") or "").strip().lower()

        if state in {"succeeded", "completed", "failed"}:
            metrics = last_payload.get("metrics")
            summary = metrics if isinstance(metrics, dict) else last_payload.get("summary")
            return {
                "ok": state in {"succeeded", "completed"},
                "payload": {**last_payload, "summary": summary},
                "status_code": 200,
                "job_id": job_id,
            }

        time.sleep(max(3.0, float(poll_interval_sec)))

    return {
        "ok": False,
        "error": f"Timed out waiting for job {job_id} after {int(limit)}s",
        "status_code": 504,
        "job_id": job_id,
        "payload": last_payload,
        "endpoint": "wait",
    }