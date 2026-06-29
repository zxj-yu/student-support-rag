"""Shared test setup: stub heavy optional dependencies before any app import.

In CI these packages are installed via requirements.txt, so the real services
work. These stubs simply let the unit tests run fast and in isolation, without
needing Qdrant, sentence-transformers, anthropic, or an API key.
"""
import sys
import types
from unittest.mock import MagicMock


def _stub(name, **attrs):
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    for key, val in attrs.items():
        setattr(mod, key, val)
    sys.modules[name] = mod


# sentence_transformers.SentenceTransformer
_stub("sentence_transformers", SentenceTransformer=MagicMock())

# qdrant_client and its models submodule
_stub("qdrant_client", QdrantClient=MagicMock())
_models = types.ModuleType("qdrant_client.models")
_models.Distance = MagicMock()
_models.PointStruct = MagicMock()
_models.VectorParams = MagicMock()
sys.modules.setdefault("qdrant_client.models", _models)

# anthropic
_stub("anthropic", Anthropic=MagicMock())