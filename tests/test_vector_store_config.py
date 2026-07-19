"""Tests for local vs Qdrant Cloud connection selection."""
import importlib
from unittest.mock import MagicMock

import qdrant_client


def _reload_store(monkeypatch, url, api_key):
    """Reload vector_store with given settings and capture client kwargs."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "qdrant_url", url)
    monkeypatch.setattr(settings, "qdrant_api_key", api_key)
    captured = {}

    def fake_client(*args, **kwargs):
        captured.update(kwargs)
        return MagicMock()

    monkeypatch.setattr(qdrant_client, "QdrantClient", fake_client)
    from app.services import vector_store

    importlib.reload(vector_store)
    return captured


def test_uses_cloud_url_when_set(monkeypatch):
    captured = _reload_store(
        monkeypatch, "https://demo.cloud.qdrant.io:6333", "secret-key"
    )
    assert captured.get("url") == "https://demo.cloud.qdrant.io:6333"
    assert captured.get("api_key") == "secret-key"


def test_falls_back_to_host_port_when_no_url(monkeypatch):
    captured = _reload_store(monkeypatch, "", "")
    assert "url" not in captured
    assert captured.get("host")  # host/port mode
