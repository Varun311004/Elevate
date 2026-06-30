"""Public web context helpers for topic-driven MCQ generation."""

from __future__ import annotations

import json
from functools import lru_cache
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from config import REQUEST_TIMEOUT_SECONDS, WEB_CONTEXT_MAX_CHARS


def _fetch_json(url: str) -> dict:
    req = urlrequest.Request(
        url,
        headers={
            "User-Agent": "Elevate-AI-MCQ/1.0",
            "Accept": "application/json",
        },
        method="GET",
    )
    with urlrequest.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        raw = response.read().decode("utf-8", errors="ignore")
    return json.loads(raw)


def _duckduckgo_summary(topic: str) -> str:
    query = urlparse.quote_plus(topic)
    url = (
        "https://api.duckduckgo.com/"
        f"?q={query}&format=json&no_html=1&skip_disambig=1&no_redirect=1"
    )
    try:
        data = _fetch_json(url)
    except (urlerror.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return ""

    abstract = str(data.get("AbstractText") or "").strip()
    heading = str(data.get("Heading") or "").strip()
    related = data.get("RelatedTopics") or []

    pieces = []
    if heading and abstract:
        pieces.append(f"{heading}: {abstract}")
    elif abstract:
        pieces.append(abstract)

    for item in related[:4]:
        if isinstance(item, dict):
            text = str(item.get("Text") or "").strip()
            if text:
                pieces.append(text)

    return "\n".join(pieces).strip()


def _wikipedia_summary(topic: str) -> str:
    safe_topic = urlparse.quote(topic.replace(" ", "_"))
    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{safe_topic}"
    try:
        data = _fetch_json(summary_url)
    except (urlerror.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return ""

    extract = str(data.get("extract") or "").strip()
    title = str(data.get("title") or "").strip()
    if title and extract:
        return f"{title}: {extract}"
    return extract


@lru_cache(maxsize=128)
def _build_topic_web_context_cached(normalized_topic: str) -> str:
    safe_topic = str(normalized_topic or "").strip()
    if not safe_topic:
        return "Topic: general science and mathematics"

    snippets = []
    wiki = _wikipedia_summary(safe_topic)
    if wiki:
        snippets.append(wiki)

    ddg = _duckduckgo_summary(safe_topic)
    if ddg:
        snippets.append(ddg)

    if not snippets:
        return f"Topic: {safe_topic}\nNo reliable web context was retrieved."

    combined = "\n\n".join(snippets).strip()
    if len(combined) > WEB_CONTEXT_MAX_CHARS:
        combined = combined[:WEB_CONTEXT_MAX_CHARS].rsplit(" ", 1)[0].strip()

    return f"Topic: {safe_topic}\n\n{combined}"


def build_topic_web_context(topic: str) -> str:
    normalized_topic = " ".join(str(topic or "").strip().lower().split())
    return _build_topic_web_context_cached(normalized_topic)
