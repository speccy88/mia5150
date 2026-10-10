"""Small offline regression tests for baseline integrity and resource guards.

No model, CUDA context, external command, network call, or credential access.
All embedding/telemetry operations are mocked or use tiny local numpy arrays.
"""

import contextlib
import hashlib
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

import resource_guard as guard
import run_skillret_baseline as baseline


def unit(index=0, dtype="float32"):
    vector = np.zeros(1024, dtype=dtype)
    vector[index] = 1
    return vector


def journal_row(index, vector):
    return {"index": index, "seconds": 0.1, "utc": 1.0,
            "sha256": hashlib.sha256(vector.tobytes()).hexdigest()}


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cache = Path(self.directory.name)
        self.texts = ["name | description | full skill one", "name | description | full skill two"]

    def write_cache(self, dtype="float32"):
        values = np.lib.format.open_memmap(self.cache / "documents.npy", mode="w+", dtype=dtype, shape=(2, 1024))
        values[0] = unit(dtype=dtype)
        values.flush()
        del values
        self.write_journal([journal_row(0, unit(dtype=dtype))])

    def write_journal(self, rows):
        (self.cache / "documents-journal.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def encode(self, **kwargs):
        return baseline.cached_encode(None, None, self.texts, self.cache, "documents", 32768, **kwargs)

    def test_resume_reuses_verified_rows_and_encodes_only_missing(self):
        self.write_cache()
        with mock.patch.object(baseline, "encode_one", return_value=(unit(1), 0.2)) as encoder:
            vectors, done = self.encode()
            self.assertEqual(encoder.call_count, 1)
            self.assertEqual(encoder.call_args.args[2], self.texts[1])
            np.testing.assert_array_equal(vectors[0], unit())
            np.testing.assert_array_equal(vectors[1], unit(1))
            self.assertEqual(set(done), {0, 1})
            for index, row in done.items():
                self.assertEqual(hashlib.sha256(vectors[index].tobytes()).hexdigest(), row["sha256"])
            del vectors

    def assert_refuses_corruption(self):
        with mock.patch.object(baseline, "encode_one", return_value=(unit(1), 0.2)):
            with self.assertRaises((ValueError, RuntimeError, AssertionError, OSError)):
                self.encode()

    def test_surviving_journal_with_missing_array_is_rejected(self):
        self.write_journal([journal_row(0, unit())])
        self.assert_refuses_corruption()

    def test_corrupted_unit_vector_with_stale_digest_is_rejected(self):
        self.write_cache()
        values = np.lib.format.open_memmap(self.cache / "documents.npy", mode="r+")
        values[0] = unit(5)  # Still finite and unit norm, but wrong cached content.
        values.flush()
        del values
        self.assert_refuses_corruption()

    def test_wrong_dtype_is_rejected_even_if_digest_matches(self):
        self.write_cache(dtype="float64")
        self.assert_refuses_corruption()

    def test_out_of_range_journal_index_is_rejected(self):
        self.write_cache()
        self.write_journal([journal_row(0, unit()), journal_row(7, unit())])
        self.assert_refuses_corruption()

    def test_nonunit_cached_vector_is_rejected_even_if_digest_matches(self):
        self.write_cache()
        values = np.lib.format.open_memmap(self.cache / "documents.npy", mode="r+")
        values[0] *= 3
        self.write_journal([journal_row(0, values[0])])
        values.flush()
        del values
        self.assert_refuses_corruption()

    def test_nonfinite_cached_vector_is_rejected_even_if_digest_matches(self):
        self.write_cache()
        values = np.lib.format.open_memmap(self.cache / "documents.npy", mode="r+")
        values[0, 0] = np.nan
        self.write_journal([journal_row(0, values[0])])
        values.flush()
        del values
        self.assert_refuses_corruption()


class EmbeddingContractTests(unittest.TestCase):
    def test_protocol_records_exact_fulltext_and_query_contract(self):
        manifest = {"files": {"config.json": {"sha256": "fixture"}}}
        protocol = baseline.protocol(manifest, 32768)
        self.assertEqual(protocol["query_prefix"],
                         "Instruct: Given a skill search query, retrieve relevant skills that match the query\nQuery: ")
        self.assertIn("skill_md", protocol["document_format"])
        self.assertEqual(protocol["pooling"], "lasttoken")
        self.assertTrue(protocol["normalize_embeddings"])
        self.assertEqual(protocol["normalization_dtype"], "float32")
        self.assertEqual(protocol["max_seq_length"], 32768)
        self.assertEqual(protocol["batch_size"], 1)
        self.assertEqual(protocol["top_k"], 50)
        self.assertNotEqual(protocol, baseline.protocol(manifest, 8192))

    def test_protocol_binds_cache_to_encoder_source_and_runtime(self):
        manifest = {"files": {}}
        with mock.patch.object(baseline.importlib.metadata, "version", return_value="1.0"):
            first = baseline.protocol(manifest, 32768)
        with mock.patch.object(baseline.importlib.metadata, "version", return_value="2.0"):
            upgraded = baseline.protocol(manifest, 32768)
        self.assertEqual(set(first["runtime"]), {"torch", "transformers", "numpy", "tokenizers", "accelerate", "safetensors"})
        self.assertNotEqual(first, upgraded)
        self.assertEqual(len(first["encoder_source_sha256"]), 64)
        with mock.patch("inspect.getsource", return_value="deliberately changed encoder source"):
            changed = baseline.protocol(manifest, 32768)
        self.assertNotEqual(first["encoder_source_sha256"], changed["encoder_source_sha256"])

    def test_last_token_converts_to_float32_before_normalization(self):
        observed = {}

        class Tensor:
            def __init__(self, array):
                self.array = np.asarray(array)
            def __getitem__(self, index):
                return Tensor(self.array[index])
            def float(self):
                return Tensor(self.array.astype(np.float32))
            def cpu(self):
                return self
            def numpy(self):
                return self.array

        class Inputs(dict):
            def to(self, device):
                observed["device"] = device
                return self

        hidden = np.zeros((1, 3, 1024), dtype=np.float16)
        hidden[0, 0, 0] = 9
        hidden[0, -1, 1:3] = [3, 4]
        def tokenize(text, **kwargs):
            observed["tokenizer"] = kwargs
            return Inputs(input_ids="mock input IDs")
        def forward(**kwargs):
            observed["forward"] = kwargs
            return types.SimpleNamespace(last_hidden_state=Tensor(hidden))
        def normalize(tensor, p, dim):
            observed["normalization_dtype"] = tensor.array.dtype
            self.assertEqual((p, dim), (2, 1))
            return Tensor(tensor.array / np.linalg.norm(tensor.array, axis=1, keepdims=True))
        fake_torch = types.SimpleNamespace(inference_mode=contextlib.nullcontext,
            nn=types.SimpleNamespace(functional=types.SimpleNamespace(normalize=normalize)))
        with mock.patch.dict("sys.modules", {"torch": fake_torch}):
            vector, seconds = baseline.encode_one(forward, tokenize, "fixture", 32768)
        self.assertEqual(observed["normalization_dtype"], np.dtype("float32"))
        self.assertEqual(observed["device"], "cuda")  # Request captured, no device was accessed.
        self.assertFalse(observed["forward"]["use_cache"])
        self.assertEqual(observed["tokenizer"]["max_length"], 32768)
        self.assertTrue(observed["tokenizer"]["truncation"])
        self.assertEqual(vector.dtype, np.dtype("float32"))
        self.assertEqual(vector[0], 0)
        np.testing.assert_allclose(vector[1:3], [0.6, 0.8], atol=1e-7)
        self.assertTrue(math.isfinite(seconds) and seconds >= 0)

    @contextlib.contextmanager
    def mocked_model_loader(self, *, effective_theta=None, raw_rope=None):
        """Simulate Transformers 4's wrong default; never initialize torch/CUDA."""
        model = mock.Mock()
        model.eval.return_value = model
        cfg = types.SimpleNamespace(rope_theta=10000, rope_scaling={"legacy": "default"})
        auto_config = mock.Mock()
        auto_config.from_pretrained.return_value = cfg
        auto_model = mock.Mock()

        def from_pretrained(path, **kwargs):
            configured = kwargs["config"]
            theta = configured.rope_theta if effective_theta is None else effective_theta
            frequencies = np.reciprocal(np.power(np.float32(theta), np.arange(0, 128, 2, dtype=np.float32) / 128))
            model.rotary_emb = types.SimpleNamespace(inv_freq=types.SimpleNamespace(float=lambda: frequencies))
            return model

        auto_model.from_pretrained.side_effect = from_pretrained
        cuda_backend = types.SimpleNamespace(enable_flash_sdp=mock.Mock(), enable_math_sdp=mock.Mock(),
            enable_mem_efficient_sdp=mock.Mock(), matmul=types.SimpleNamespace(allow_tf32=True))

        def arange(start, stop, step, *, dtype, device):
            self.assertEqual(device, "cuda")  # Only inspect the requested device.
            return np.arange(start, stop, step, dtype=dtype)

        fake_torch = types.SimpleNamespace(float16="float16", float32=np.float32,
            arange=arange, allclose=np.allclose, set_num_threads=mock.Mock(),
            set_num_interop_threads=mock.Mock(), cuda=types.SimpleNamespace(set_per_process_memory_fraction=mock.Mock()),
            backends=types.SimpleNamespace(cuda=cuda_backend))
        sdpa = types.SimpleNamespace(use_gqa_in_sdpa=lambda attention_mask, key: True)
        modules = {"torch": fake_torch, "transformers": types.SimpleNamespace(AutoModel=auto_model, AutoConfig=auto_config),
                   "transformers.integrations": types.SimpleNamespace(sdpa_attention=sdpa)}
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict("sys.modules", modules):
            path = Path(directory)
            rope = {"rope_theta": 1000000, "rope_type": "default"} if raw_rope is None else raw_rope
            (path / "config.json").write_text(json.dumps({"rope_parameters": rope}), encoding="utf-8")
            yield {"path": path, "model": model, "cfg": cfg, "auto_config": auto_config,
                   "auto_model": auto_model, "torch": fake_torch, "cuda_backend": cuda_backend, "sdpa": sdpa}

    def test_model_load_is_offline_and_memory_efficient_only(self):
        with self.mocked_model_loader() as fixture:
            loaded, seconds = baseline.load_model(fixture["path"])
        self.assertIs(loaded, fixture["model"])
        options = fixture["auto_model"].from_pretrained.call_args.kwargs
        self.assertTrue(options["local_files_only"])
        self.assertFalse(options["trust_remote_code"])
        self.assertEqual(options["torch_dtype"], "float16")
        self.assertEqual(options["attn_implementation"], "sdpa")
        self.assertEqual(options["device_map"], "cuda")
        fixture["torch"].cuda.set_per_process_memory_fraction.assert_called_once_with(0.80)
        cuda_backend = fixture["cuda_backend"]
        cuda_backend.enable_math_sdp.assert_called_once_with(False)
        cuda_backend.enable_flash_sdp.assert_called_once_with(False)
        cuda_backend.enable_mem_efficient_sdp.assert_called_once_with(True)
        self.assertFalse(fixture["sdpa"].use_gqa_in_sdpa(None, None))
        self.assertTrue(math.isfinite(seconds) and seconds >= 0)

    def test_nested_model_rope_theta_overrides_transformers4_default(self):
        with self.mocked_model_loader() as fixture:
            self.assertEqual(fixture["cfg"].rope_theta, 10000)
            baseline.load_model(fixture["path"])
        config_options = fixture["auto_config"].from_pretrained.call_args.kwargs
        self.assertTrue(config_options["local_files_only"])
        self.assertFalse(config_options["trust_remote_code"])
        effective_cfg = fixture["auto_model"].from_pretrained.call_args.kwargs["config"]
        self.assertIs(effective_cfg, fixture["cfg"])
        self.assertEqual(effective_cfg.rope_theta, 1000000)
        self.assertIsNone(effective_cfg.rope_scaling)

    def test_actual_rotary_frequencies_with_wrong_theta_are_rejected(self):
        # Merely correcting cfg is insufficient: inspect the loaded buffer too.
        with self.mocked_model_loader(effective_theta=10000) as fixture:
            with self.assertRaisesRegex(AssertionError, "rotary frequencies"):
                baseline.load_model(fixture["path"])

    def test_unexpected_nested_rope_contract_is_rejected(self):
        for rope in ({"rope_theta": 10000, "rope_type": "default"},
                     {"rope_theta": 1000000, "rope_type": "dynamic"}):
            with self.subTest(rope=rope), self.mocked_model_loader(raw_rope=rope) as fixture:
                with self.assertRaises(AssertionError):
                    baseline.load_model(fixture["path"])
                fixture["auto_model"].from_pretrained.assert_not_called()


class ResourceGuardTests(unittest.TestCase):
    def sample(self, *, age=1, host_available=2 * 1024**3, guest_available=2 * 1024**3,
               cpu=20.0, gpu_mib=7168, temperature=80):
        host = {"utc": 100.0 - age, "available_bytes": host_available, "cpu_pct": cpu}
        meminfo = f"MemAvailable: {guest_available // 1024} kB\nSwapTotal: 2097152 kB\nSwapFree: 2097152 kB\n"
        vmstat = "pswpin 0\npswpout 0\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "host.jsonl"
            path.write_text(json.dumps(host) + "\n", encoding="utf-8")
            with mock.patch.object(guard.time, "time", return_value=100.0), \
                 mock.patch.object(guard.pathlib.Path, "read_text", side_effect=[meminfo, vmstat]), \
                 mock.patch.object(guard.subprocess, "check_output", return_value=f"{gpu_mib}, {temperature}, 0"):
                return guard.sample(path)

    def test_limits_are_inclusive(self):
        self.assertEqual(self.sample()["gpu_used_mib"], 7168)

    def test_stale_or_future_host_telemetry_is_rejected(self):
        for age in (8, 9, -1):
            with self.subTest(age=age), self.assertRaises(RuntimeError):
                self.sample(age=age)

    def test_memory_and_gpu_limit_breaches_stop(self):
        for kwargs in ({"host_available": 2 * 1024**3 - 1},
                       {"guest_available": 2 * 1024**3 - 1024},
                       {"gpu_mib": 7169}, {"temperature": 81}):
            with self.subTest(kwargs=kwargs), self.assertRaises(RuntimeError):
                self.sample(**kwargs)

    def test_nonfinite_host_cpu_is_rejected(self):
        for cpu in (float("nan"), float("inf"), -1.0, 101.0, True):
            with self.subTest(cpu=cpu), self.assertRaises(RuntimeError):
                self.sample(cpu=cpu)

    def test_active_swap_or_sustained_cpu_terminates_worker(self):
        for trigger in ("swap", "cpu"):
            with self.subTest(trigger=trigger), tempfile.TemporaryDirectory() as directory:
                destination = Path(directory) / "attempt"
                before = {"host": {"cpu_pct": 20}, "swap": {"pswpin": 0, "pswpout": 0}}
                next_row = {"host": {"cpu_pct": 81 if trigger == "cpu" else 20},
                            "swap": {"pswpin": 1 if trigger == "swap" else 0, "pswpout": 0}}
                child = mock.Mock(pid=4242)
                child.poll.return_value = None
                samples = [before] + [next_row] * 6
                with mock.patch.object(guard.sys, "argv", ["guard", "--host", "fixture", "--run-dir", str(destination), "--", "fixture-worker"]), \
                     mock.patch.object(guard, "sample", side_effect=samples), \
                     mock.patch.object(guard.subprocess, "Popen", return_value=child) as spawn, \
                     mock.patch.object(guard.time, "sleep"), \
                     mock.patch.object(guard.os, "killpg") as kill:
                    code = guard.main()
                self.assertEqual(code, 1)
                spawn.assert_called_once()
                kill.assert_called_once_with(4242, guard.signal.SIGTERM)
                child.wait.assert_called_once()
                error = json.loads((destination / "guard-stop.json").read_text())["error"]
                self.assertIn("swap" if trigger == "swap" else "CPU", error)


if __name__ == "__main__":
    unittest.main()
