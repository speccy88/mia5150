"""Strict batch-one official BGE-M3 forward calls, resumable unit-vector cache."""
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import threading
import time

import numpy as np
import psutil
from common import STUDY, MODEL_REV, MODEL_SHA, atomic_json, digest, file_hash, unit


def utc():
    return datetime.now(timezone.utc).isoformat()


def public(value):
    if isinstance(value, dict): return {k: public(v) for k, v in value.items()}
    if isinstance(value, list): return [public(v) for v in value]
    if isinstance(value, str): return re.sub(r'/(?:home|Users)/[^/\s]+', '<user-home>', value)
    return value


class Monitor:
    def __init__(self, phase):
        self.phase = phase
        self.stop = threading.Event()
        self.problem = None
        self.baseline = psutil.swap_memory()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def sample(self):
        memory, swap = psutil.virtual_memory(), psutil.swap_memory()
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu,temperature.gpu,power.draw', '--format=csv,noheader,nounits'], text=True)
        gpu = [float(x.strip()) for x in raw.strip().split(',')]
        if memory.available < 1024 * 1024**2: self.problem = 'Available host RAM below 1 GiB'
        if swap.sout - self.baseline.sout > 64 * 1024**2 or swap.sin - self.baseline.sin > 64 * 1024**2: self.problem = 'Host swap traffic exceeded 64 MiB'
        if gpu[0] > 7168 or gpu[2] > 80: self.problem = 'GPU reserve/temperature guard'
        return {'utc': utc(), 'phase': self.phase, 'gpu_memory_mib': gpu[0], 'gpu_utilization_percent': gpu[1],
                'gpu_temperature_c': gpu[2], 'gpu_power_w': gpu[3], 'system_used_bytes': memory.used,
                'system_available_bytes': memory.available, 'swap_used_bytes': swap.used,
                'swap_in_bytes': swap.sin, 'swap_out_bytes': swap.sout}

    def loop(self):
        with (STUDY / 'results/telemetry.jsonl').open('a') as f:
            while not self.stop.is_set():
                try:
                    f.write(json.dumps(self.sample()) + '\n'); f.flush()
                except Exception as e:
                    self.problem = 'Resource monitoring unavailable: ' + type(e).__name__
                self.stop.wait(1)

    def check(self):
        if self.problem: raise RuntimeError(self.problem)

    def close(self):
        self.stop.set(); self.thread.join()


class Encoder:
    def __init__(self, cache, monitor, context=8192, frozen=True):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        import torch
        from FlagEmbedding import BGEM3FlagModel
        from tokenizers import Tokenizer
        from transformers import AutoModel
        from unittest.mock import patch
        self.torch, self.monitor, self.context = torch, monitor, context
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.use_deterministic_algorithms(True)
        assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 5)
        self.model_dir = cache / 'model' / MODEL_REV
        acquisition = json.loads((cache / 'model-acquisition.json').read_text())
        assert acquisition['revision'] == MODEL_REV and acquisition['model'] == 'BAAI/bge-m3'
        assert next(x['sha256'] for x in acquisition['files'] if x['file'] == 'pytorch_model.bin') == MODEL_SHA
        for item in acquisition['files']:
            assert file_hash(self.model_dir / item['file']) == item['sha256']
        started = time.perf_counter()
        original_loader = AutoModel.from_pretrained
        def load_backbone(*args, **kwargs):
            kwargs.update(torch_dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation='sdpa', device_map={'': 'cuda:0'})
            return original_loader(*args, **kwargs)
        # FlagEmbedding 1.3.5 does not forward backbone loader kwargs. Keep its
        # official model and CLS/norm code, while avoiding an FP32 model copy.
        with patch.object(AutoModel, 'from_pretrained', side_effect=load_backbone):
            self.flag = BGEM3FlagModel(str(self.model_dir), use_fp16=True, normalize_embeddings=True,
                devices='cuda:0', batch_size=1, pooling_method='cls', query_instruction_for_retrieval=None,
                return_dense=True, return_sparse=False, return_colbert_vecs=False)
        self.flag.model.half().to('cuda:0').eval()
        torch.cuda.synchronize()
        self.load_s = time.perf_counter() - started
        self.tokenizer = self.flag.tokenizer
        self.audit_tokenizer = Tokenizer.from_file(str(self.model_dir / 'tokenizer.json'))
        self.audit_tokenizer.no_truncation(); self.audit_tokenizer.no_padding()
        base = self.flag.model.model
        assert base.config._attn_implementation == 'sdpa', 'Require memory-efficient SDPA, no eager fallback'
        assert self.flag.pooling_method == 'cls'
        assert all(p.device.type == 'cuda' and p.dtype == torch.float16 for p in self.flag.model.parameters())
        self.identity = {'model': 'BAAI/bge-m3', 'revision': MODEL_REV, 'context': context,
            'checkpoint_sha256': next(x['sha256'] for x in acquisition['files'] if x['file'] == 'pytorch_model.bin'),
            'tokenizer_sha256': file_hash(self.model_dir / 'tokenizer.json'), 'pooling': 'official CLS',
            'precision': 'FP16; official normalization then float32 L2', 'dimensions': 1024,
            'attention': 'Transformers SDPA; PyTorch efficient attention only', 'batch_size': 1,
            'backbone_load_dtype': 'torch.float16', 'low_cpu_mem_usage': True,
            'backbone_device_map': {'': 'cuda:0'},
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'flagembedding': importlib.metadata.version('FlagEmbedding'),
            'transformers': importlib.metadata.version('transformers'),
            'tokenizers': importlib.metadata.version('tokenizers'), 'numpy': np.__version__,
            'deterministic_algorithms': True, 'tf32': False, 'cpu_threads': 2,
            'protocol_sha256': file_hash(STUDY / 'protocol.json') if frozen else None}
        self.root = cache / 'vectors' / digest(self.identity)
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_json(self.root / 'identity.json', self.identity)
        if frozen:
            atomic_json(STUDY / 'results/model.json', {**self.identity, 'cache_identity': self.identity, 'parameter_count': sum(p.numel() for p in base.parameters()),
                'all_parameters_cuda_fp16': True, 'attention_class': type(base.encoder.layer[0].attention.self).__name__,
                'model_loading_s': self.load_s, 'os_file_cache_evicted': False})

    def encode(self, text, kind='document', use_cache=True, force=False):
        cache_hash = digest({'text_sha256': digest(text), 'identity': self.identity, 'kind': kind})
        path = self.root / (cache_hash + '.npy'); meta = self.root / (cache_hash + '.json')
        if use_cache and not force and path.exists() and meta.exists():
            started = time.perf_counter(); row = json.loads(meta.read_text())
            assert row['text_sha256'] == digest(text) and row['vector_sha256'] == file_hash(path)
            vector = np.load(path, allow_pickle=False)
            assert vector.dtype == np.float32 and vector.shape == (1024,) and np.isfinite(vector).all()
            assert abs(float(np.linalg.norm(vector)) - 1) < 2e-5
            return vector, {**row, 'cache_hit': True, 'cache_read_s': time.perf_counter() - started}
        self.monitor.check(); torch = self.torch
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        inputs = self.tokenizer(text, add_special_tokens=True, truncation=False, padding=False, return_tensors='pt')
        ids = inputs['input_ids'][0].tolist()
        if len(ids) > self.context: raise ValueError(f'Overlength input: {len(ids)} > {self.context}; no truncation allowed')
        with torch.inference_mode():
            output = self.flag.model(inputs.to('cuda:0'), return_dense=True, return_sparse=False, return_colbert_vecs=False)['dense_vecs']
        assert output.shape == (1, 1024)
        vector = unit(output[0].float().cpu().numpy(), dimensions=1024)
        torch.cuda.synchronize(); elapsed = time.perf_counter() - started
        expected = self.audit_tokenizer.encode(text, add_special_tokens=False).ids
        assert ids[0] == self.tokenizer.cls_token_id and ids[-1] == self.tokenizer.eos_token_id
        assert ids[1:-1] == expected, 'Independent full-input tokenizer parity failed'
        self.monitor.check()
        row = {'utc': utc(), 'phase': self.monitor.phase, 'kind': kind, 'cache_sha256': cache_hash,
            'text_sha256': digest(text), 'input_ids_sha256': digest(ids), 'content_tokens': len(expected),
            'input_tokens': len(ids), 'encoding_s': elapsed, 'cache_hit': False,
            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'truncation': False}
        if use_cache:
            if force and path.exists():
                assert np.array_equal(vector, np.load(path, allow_pickle=False)), 'Fresh query differs from verified cached vector'
            with path.with_suffix('.tmp').open('wb') as f:
                np.save(f, vector, allow_pickle=False); f.flush(); os.fsync(f.fileno())
            path.with_suffix('.tmp').replace(path)
            row['vector_sha256'] = file_hash(path)
            row['recomputed_for_uncached_latency'] = force
            atomic_json(meta, row)
        with (STUDY / 'results/embedding_calls.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        return vector, row
