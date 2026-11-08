"""Read-only reuse of the pinned Qwen corpus/ranking and independent metric code."""
from pathlib import Path
import importlib.util
import sys

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[2]
QWEN = STUDY.parent / 'skillret-qwen3-embedding-8b-q4km-fulltext'
QWEN_REV = 'e42e68cad43b7bff285777ac4e20c5722963b9f4'
MODEL_REV = '5617a9f61b028005a4858fdac845db406aefb181'
MODEL_SHA = 'b5e0ce3470abf5ef3831aa1bd5553b486803e83251590ab7ff35a117cf6aad38'
CONFIGS = ('A_metadata', 'B_matched', 'C_native8k')

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

qcore = module('qwen_shared_core', QWEN / 'src/core.py')
sys.modules.setdefault('core', qcore)
qvalidate = module('qwen_independent_metrics', QWEN / 'src/validate.py')
digest, file_hash, atomic_json, jsonl = qcore.digest, qcore.file_hash, qcore.atomic_json, qcore.jsonl
full_text, chunks, unit = qcore.full_text, qcore.chunks, qcore.unit
SIZES, rank_skills, evaluate = qcore.SIZES, qcore.rank_skills, qcore.evaluate
bootstrap, latency_summary = qcore.bootstrap, qcore.latency_summary
independent, compare_evaluation = qvalidate.independent, qvalidate.compare_evaluation


def panel():
    import json
    return json.loads((REPO / 'experiments/benchmarks/skill-discovery-oauth-v1-20261103/data/test_queries.json').read_text())
