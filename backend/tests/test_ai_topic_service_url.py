import pytest

from backend.ai_topic_service import get_topic_ai_service_url


def test_hf_spaces_page_url_is_normalized(monkeypatch):
    monkeypatch.setenv("AI_TOPIC_SERVICE_URL", "https://huggingface.co/spaces/Sana2704/elevate")
    monkeypatch.delenv("TOPIC_AI_SERVICE_URL", raising=False)

    assert get_topic_ai_service_url() == "https://sana2704-elevate.hf.space"


def test_hf_spaces_compact_owner_space_format(monkeypatch):
    monkeypatch.setenv("AI_TOPIC_SERVICE_URL", "Sana2704/elevate")
    monkeypatch.delenv("TOPIC_AI_SERVICE_URL", raising=False)

    assert get_topic_ai_service_url() == "https://sana2704-elevate.hf.space"


def test_hf_space_host_is_kept_as_origin(monkeypatch):
    monkeypatch.setenv("AI_TOPIC_SERVICE_URL", "https://sana2704-elevate.hf.space/some/path")
    monkeypatch.delenv("TOPIC_AI_SERVICE_URL", raising=False)

    assert get_topic_ai_service_url() == "https://sana2704-elevate.hf.space"
