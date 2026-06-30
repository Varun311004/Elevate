import pytest

from backend.config import normalize_database_url


def test_normalize_postgres_scheme_and_ssl():
    url = normalize_database_url("postgres://user:pass@db.example.com:5432/mydb")
    assert url.startswith("postgresql://")
    assert "sslmode=require" in url


def test_normalize_supabase_project_url_with_password_helper(monkeypatch):
    monkeypatch.setenv("SUPABASE_DB_PASSWORD", "abc123")
    normalized = normalize_database_url("https://cdwdludsyivlamcnizhk.supabase.co")
    assert normalized.startswith("postgresql://postgres:")
    assert "@db.cdwdludsyivlamcnizhk.supabase.co:5432/postgres" in normalized
    assert "sslmode=require" in normalized


def test_invalid_http_database_url_raises(monkeypatch):
    monkeypatch.delenv("SUPABASE_DB_PASSWORD", raising=False)
    monkeypatch.delenv("SUPABASE_DIRECT_CONNECTION_STRING", raising=False)
    with pytest.raises(ValueError):
        normalize_database_url("https://example.com")
