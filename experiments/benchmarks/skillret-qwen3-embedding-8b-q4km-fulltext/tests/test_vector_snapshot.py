from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from core import digest, file_hash
from vector_snapshot import load_archive, write_archive


def test_lossless_archive_is_deterministic_and_preserves_float_bits(tmp_path):
    vector = np.zeros((1, 4096), dtype=np.float32)
    vector[0, :3] = [.6, .8, -0.0]
    arrays = {key: vector.copy() for key in ("document_vectors", "official_query_vectors", "panel_query_vectors")}
    arrays["skill_ids"] = np.asarray(["public-id-中文"])
    paths = [tmp_path / "first.npz", tmp_path / "second.npz"]
    for path in paths:
        write_archive(path, arrays)
    assert file_hash(paths[0]) == file_hash(paths[1])
    manifest = {"sha256": file_hash(paths[0]), "arrays": {
        key: {"shape": list(value.shape), "dtype": str(value.dtype),
              "sha256_contiguous_bytes": digest(value.tobytes())} for key, value in arrays.items()}}
    loaded = load_archive(paths[0], manifest)
    for key, value in arrays.items():
        assert loaded[key].dtype == value.dtype
        assert loaded[key].tobytes() == value.tobytes()
    assert np.signbit(loaded["document_vectors"][0, 2])
    paths[0].write_bytes(paths[0].read_bytes() + b"corruption")
    with pytest.raises(AssertionError, match="corrupt vector archive"):
        load_archive(paths[0], manifest)


def test_archive_refuses_pickle_object_payloads(tmp_path):
    with pytest.raises(AssertionError):
        write_archive(tmp_path / "unsafe.npz", {"object": np.array([{"value": 1}], dtype=object)})
