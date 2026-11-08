from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run
from test_core import CharacterTokenizer


def encoder(tmp_path, monkeypatch, token_count=3):
    monkeypatch.setattr(run, "RESULTS", tmp_path)
    enc = run.Encoder.__new__(run.Encoder)
    enc.identity = {"model_digest": "fixture", "protocol_sha256": "fixture"}
    enc.root = tmp_path
    enc.protocol = {"model": "synthetic-test-only", "options": {"num_ctx": 4096}}
    enc.monitor = SimpleNamespace(check=lambda: None)
    enc.tokenizer = CharacterTokenizer()
    enc.native_token_count = lambda text: 3
    seen = []

    def fake_api(route, payload):
        seen.append(payload)
        assert route == "/api/embed"
        assert payload["truncate"] is False and payload["dimensions"] == 4096
        assert payload["input"] == "abc"
        return {"embeddings": [np.ones(4096).tolist()], "prompt_eval_count": token_count,
                "total_duration": 1000000000, "load_duration": 0}

    monkeypatch.setattr(run, "api", fake_api)
    return enc, seen


def test_resume_does_not_request_embedding_again_and_rejects_corruption(tmp_path, monkeypatch):
    enc, seen = encoder(tmp_path, monkeypatch)
    v, first = enc.encode("abc", "document")
    v2, second = enc.encode("abc", "document")
    assert len(seen) == 1
    assert np.array_equal(v, v2) and second["cache_hit"]
    assert first["prompt_eval_count"] == 3
    path = next(tmp_path.glob("*.npy"))
    path.write_bytes(b"corrupt checkpoint")
    with pytest.raises(ValueError, match="Corrupt cached"):
        enc.encode("abc", "document")


def test_runtime_token_loss_is_rejected_before_checkpoint(tmp_path, monkeypatch):
    enc, seen = encoder(tmp_path, monkeypatch, token_count=2)
    with pytest.raises(ValueError, match="token mismatch"):
        enc.encode("abc", "document")
    assert len(seen) == 1 and not list(tmp_path.glob("*.npy"))


def test_configuration_changes_invalidate_cache(tmp_path, monkeypatch):
    enc, seen = encoder(tmp_path, monkeypatch)
    enc.encode("abc", "document")
    enc.identity = {**enc.identity, "model_digest": "changed-model-fixture"}
    enc.encode("abc", "document")
    assert len(seen) == 2 and len(list(tmp_path.glob("*.npy"))) == 2


def test_native_special_token_count_can_differ_without_truncation(tmp_path, monkeypatch):
    enc, seen = encoder(tmp_path, monkeypatch, token_count=2)
    enc.native_token_count = lambda text: 2
    vector, receipt = enc.encode("abc", "document")
    assert receipt["original_tokens"] == 3
    assert receipt["prompt_eval_count"] == receipt["native_tokenizer_verified_count"] == 2
    assert vector.shape == (4096,) and len(seen) == 1
