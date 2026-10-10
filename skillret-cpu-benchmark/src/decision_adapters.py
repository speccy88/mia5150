"""CPU binary-relevance adapters, restored from the authored benchmark module.

These rerank a fixed external shortlist; they are not dense retrievers. Pass
complete local checkpoints. This module performs no downloads, installation,
training, quantization, or implicit model substitution. The caller owns serial
model loading and timing. Sources and immutable revisions are in
docs/model_sources.json.

score_pairs([(query, candidate_header), ...], batch_size=8) returns P(relevant)
in input order. last_audit records exact input lengths and truncation. Explicit
truncate_query=True permits only query right-prefix truncation; candidate text
and every label are preserved. Prefix fitting is deterministic and safe, not a
claim to find the longest possible prefix under nonmonotonic tokenization.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

QUESTION = "Is the candidate skill relevant to fulfilling the user request?"
RELEVANT = "A relevant skill that helps fulfill the user request"
IRRELEVANT = "An irrelevant skill that does not help fulfill the user request"
CRITERIA = {"A": RELEVANT, "B": IRRELEVANT}


def pair_state(query: str, candidate: str) -> str:
    if not isinstance(query, str) or not isinstance(candidate, str):
        raise TypeError("Each pair must contain query and candidate text strings")
    return f"User request:\n{query}\n\nCandidate skill:\n{candidate}"


def softmax(values: Sequence[float], temperature: float = 1.0) -> list[float]:
    if len(values) == 0 or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Softmax needs finite logits and a positive temperature")
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Nonfinite model logits")
    maximum = max(values)
    exponentials = [math.exp((float(value) - maximum) / temperature) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def _local(path: str | Path) -> Path:
    root = Path(path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"A complete local checkpoint is required: {root}")
    return root


def _chunks(items, size):
    if size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), size):
        yield items[start:start + size]


class _QueryFitMixin:
    """Exact per-pair native-token fitting, with no candidate truncation."""

    def prepare_pairs(self, pairs):
        fitted, audits = [], []
        for query, candidate in pairs:
            pair_state(query, candidate)
            original = self._measure_pair(query, candidate)
            kept = query
            if original > self.max_length:
                if not self.truncate_query:
                    raise ValueError(
                        f"Lossless input overflow: {original} > {self.max_length}; "
                        "set truncate_query=True for audited query-only truncation")
                if self._measure_pair("", candidate) > self.max_length:
                    raise ValueError(
                        "Complete candidate and label prompt exceed token budget even "
                        "with an empty query; candidate truncation is forbidden")
                low, high = 0, len(query)
                while low < high:
                    midpoint = (low + high + 1) // 2
                    if self._measure_pair(query[:midpoint], candidate) <= self.max_length:
                        low = midpoint
                    else:
                        high = midpoint - 1
                kept = query[:low]
            final = self._measure_pair(kept, candidate) if kept != query else original
            if final > self.max_length:
                raise ValueError("Query fitting did not satisfy the exact token budget")
            fitted.append((kept, candidate))
            audits.append({
                "tokens_original": original,
                "tokens": final,
                "query_chars_original": len(query),
                "query_chars_used": len(kept),
                "query_truncated": kept != query,
                "truncated": kept != query,
                "candidate_preserved": True,
                "truncation_policy": "query_only_right",
            })
        return fitted, audits


class VerdictONNXAdapter(_QueryFitMixin):
    """Publisher GLiClass serialization and per-cardinality calibration.

    Keep the abstention slot in the denominator. Discard unused exported slots
    before softmax. There is no fallback to a different checkpoint or backend.
    """
    model_id = "heman10x/rlcd-modernbert-151m"

    def __init__(self, model_dir, *, threads=4, max_length=512, strict=True,
                 truncate_query=False):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        root = _local(model_dir)
        self.max_length, self.strict = int(max_length), bool(strict)
        self.truncate_query = bool(truncate_query)
        self.tokenizer = AutoTokenizer.from_pretrained(
            root, local_files_only=True, trust_remote_code=False)
        if self.tokenizer.convert_tokens_to_ids("<<LABEL>>") != 50368:
            raise ValueError("Verdict tokenizer lacks its expected LABEL marker")
        if self.tokenizer.convert_tokens_to_ids("<<SEP>>") != 50369:
            raise ValueError("Verdict tokenizer lacks its expected SEP marker")
        calibration = json.loads((root / "calibrator.json").read_text())
        self.temperature = float(calibration.get("per_k", {}).get(
            "3", calibration["temperature"]))
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(threads)
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(root / "model.onnx"), options, providers=["CPUExecutionProvider"])
        self.last_audit = []

    @staticmethod
    def prompt(query, candidate):
        labels = ["It is " + RELEVANT.lower(), "It is " + IRRELEVANT.lower(),
                  "insufficient evidence"]
        prefix = "".join("<<LABEL>>" + label for label in labels)
        return (prefix + "<<SEP>>Question: " + QUESTION + "\n\nContext:\n"
                + pair_state(query, candidate))

    def _measure_pair(self, query, candidate):
        return len(self.tokenizer(self.prompt(query, candidate),
                                  truncation=False)["input_ids"])

    def score_pairs(self, pairs, *, batch_size=8):
        self.last_audit, result = [], []
        for chunk in _chunks(list(pairs), batch_size):
            chunk, fit_audits = self.prepare_pairs(chunk)
            prompts = [self.prompt(*pair) for pair in chunk]
            raw = self.tokenizer(prompts, truncation=False, padding=False)["input_ids"]
            lengths = [len(ids) for ids in raw]
            if any(length > self.max_length for length in lengths):
                raise ValueError("Verdict fitted input unexpectedly exceeds context budget")
            inputs = self.tokenizer(prompts, padding=True, truncation=False,
                                    return_tensors="np")
            if any(int((row == 50368).sum()) != 3 for row in inputs["input_ids"]):
                raise ValueError("Verdict input has an unexpected option-marker count")
            feed = {name: inputs[name] for name in ("input_ids", "attention_mask")}
            logits = self.session.run(None, feed)[0]
            if len(logits) != len(chunk) or logits.shape[-1] < 3:
                raise ValueError(f"Unexpected Verdict logits shape: {logits.shape}")
            for row, fit in zip(logits, fit_audits):
                probabilities = softmax(row[:3].tolist(), self.temperature)
                result.append(probabilities[0])
                self.last_audit.append({
                    "p_abstain": probabilities[2], "temperature": self.temperature,
                    "score": "P(relevant), including abstention denominator", **fit})
        return result


class LayaAdapter(_QueryFitMixin):
    """Direct multilingual model, with the native public batched choice API.

    Native probabilities are rounded to four decimals. Use stable ranking ties.
    The two-option choice avoids the documented multilingual noul issue.
    """
    model_id = "convaiinnovations/laya-multilingual"

    def __init__(self, model_dir, *, threads=4, max_length=1024, head_length=256,
                 strict=True, truncate_query=False):
        import torch
        import laya
        torch.set_num_threads(int(threads))
        self.agent = laya.load(str(_local(model_dir)), device="cpu", compile=False, fast=False)
        self.max_length, self.head_length = int(max_length), int(head_length)
        self.strict, self.truncate_query = bool(strict), bool(truncate_query)
        self.questions = {"relevance": {
            "type": "choice", "instructions": QUESTION, "criteria": dict(CRITERIA)}}
        self.last_audit = []

    def _measure_pair(self, query, candidate):
        from laya.common import build_sequence
        internal = {"t": "choice", "ins": QUESTION, "crit": dict(CRITERIA)}
        ids, markers, options, stats = build_sequence(
            self.agent.tok, pair_state(query, candidate), internal, 1_000_000,
            self.head_length, return_stats=True, return_truncation_stats=True)
        if len(markers) != 2 or options.get("tokens_per_option") is not None:
            raise ValueError("Laya labels do not fit losslessly in the head budget")
        return len(ids)

    def score_pairs(self, pairs, *, batch_size=8):
        from laya.common import build_sequence
        self.last_audit, result = [], []
        internal = {"t": "choice", "ins": QUESTION, "crit": dict(CRITERIA)}
        for chunk in _chunks(list(pairs), batch_size):
            chunk, fit_audits = self.prepare_pairs(chunk)
            states = [pair_state(*pair) for pair in chunk]
            audits = []
            for state in states:
                ids, markers, options, stats = build_sequence(
                    self.agent.tok, state, internal, self.max_length, self.head_length,
                    return_stats=True, return_truncation_stats=True)
                if len(markers) != 2 or options.get("tokens_per_option") is not None:
                    raise ValueError("Laya failed to preserve both binary options")
                if stats["truncated"]:
                    raise ValueError("Laya fitted input unexpectedly truncates state")
                audits.append({"tokens": len(ids), **stats,
                               "option_stats": options, "probability_decimals": 4})
            predictions = self.agent.predict_batch(
                states, self.questions, batch_size=batch_size,
                max_len=self.max_length, head_max_len=self.head_length)
            if len(predictions) != len(chunk):
                raise ValueError("Laya output count does not match input")
            for prediction, audit, fit in zip(predictions, audits, fit_audits):
                if prediction.get("usage", {}).get("truncated"):
                    raise ValueError("Laya reports unexpected truncation after preflight")
                probability = float(prediction["answers"]["relevance"]["probabilities"]["A"])
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("Invalid Laya relevance probability")
                result.append(probability)
                self.last_audit.append({**audit, **fit})
        return result


class JuliaAdapter(_QueryFitMixin):
    """Official resident Torch CPU runtime, with encoding caches disabled."""
    model_id = "SupersonicLabs/Julia-1"

    def __init__(self, model_dir, *, threads=4, max_length=1024, head_length=256,
                 strict=True, batch_size=8, truncate_query=False):
        import torch
        from julia import load_model
        if not strict:
            raise ValueError("JuliaAdapter requires strict=True for audited lossless inputs")
        torch.set_num_threads(int(threads))
        self.max_length, self.head_length = int(max_length), int(head_length)
        self.truncate_query = bool(truncate_query)
        self.engine = load_model(
            str(_local(model_dir)), device="cpu", backend="torch",
            max_length=self.max_length, head_length=self.head_length,
            strict_encoding=True, batch_size=int(batch_size), compile_model=False,
            encoding_cache=0, token_cache=0)
        self.last_audit = []

    def _measure_pair(self, query, candidate):
        from julia.data import sequence
        row = {"state": pair_state(query, candidate), "question": QUESTION,
               "type": "choice", "options": [RELEVANT, IRRELEVANT]}
        encoded = sequence(self.engine.tokenizer, row, 1_000_000,
                           self.head_length, strict=True)
        return len(encoded["ids"])

    def score_pairs(self, pairs, *, batch_size=8):
        self.last_audit, result = [], []
        for chunk in _chunks(list(pairs), batch_size):
            chunk, fit_audits = self.prepare_pairs(chunk)
            rows = [{"state": pair_state(*pair), "question": QUESTION,
                     "type": "choice", "options": [RELEVANT, IRRELEVANT]}
                    for pair in chunk]
            audits = self.engine.encoding_info(rows)
            logits = self.engine.logits(rows)
            if len(logits) != len(rows):
                raise ValueError("Julia output count does not match input")
            for row, audit, fit in zip(logits, audits, fit_audits):
                if len(row) != 2:
                    raise ValueError("Julia failed to preserve both binary options")
                result.append(softmax(row)[0])
                self.last_audit.append({**audit, **fit})
        return result


class GLiNERAdapter(_QueryFitMixin):
    """Public batched classification, compatible with installed gliner2==2.0.0.

    max_len in GLiNER is a WORD cap. Keep it None, and use the public strict
    collator to count the exact encoded sequence. Both fitting and preflight
    use the same path; no unavailable transform_record method is called.
    """
    model_id = "fastino/gliner2.5-small-v1"

    def __init__(self, model_dir, *, threads=4, max_length=1024, strict=True,
                 truncate_query=False):
        import torch
        from gliner2.classification import Classifier, ClassificationSchema
        if not strict:
            raise ValueError("GLiNERAdapter requires strict=True for audited lossless inputs")
        torch.set_num_threads(int(threads))
        self.classifier = Classifier.from_pretrained(
            str(_local(model_dir)), device="cpu", map_location="cpu", quantize=False,
            compile=False, local_files_only=True, token=False).to("cpu").eval()
        self.schema = ClassificationSchema().single(
            "relevance", dict(CRITERIA), instruction=QUESTION, activation="softmax")
        self.compiled = self.classifier.compile_schema(self.schema)
        self.max_length = int(max_length)
        self.truncate_query = bool(truncate_query)
        self.last_audit = []

    def _state_token_count(self, state):
        batch = self.classifier.model.processor.collate_fn_inference(
            [(state, self.compiled.build())], max_len=None, error_policy="raise")
        if int(batch.attention_mask.shape[0]) != 1:
            raise ValueError("GLiNER strict preflight returned an unexpected batch size")
        return int(batch.attention_mask[0].sum())

    def _measure_pair(self, query, candidate):
        return self._state_token_count(pair_state(query, candidate))

    def score_pairs(self, pairs, *, batch_size=8):
        from gliner2.classification import ClassificationConfig
        self.last_audit, result = [], []
        for chunk in _chunks(list(pairs), batch_size):
            chunk, fit_audits = self.prepare_pairs(chunk)
            states = [pair_state(*pair) for pair in chunk]
            audits = []
            for state in states:
                length = self._state_token_count(state)
                if length > self.max_length:
                    raise ValueError("GLiNER fitted input unexpectedly exceeds token budget")
                audits.append({"tokens": length, "truncated": False, "word_cap": None,
                               "preflight_error_policy": "raise"})
            scores = self.classifier.batch_score(
                states, self.schema,
                config=ClassificationConfig(batch_size=batch_size, max_len=None))
            if len(scores) != len(chunk):
                raise ValueError("GLiNER output count does not match input")
            for score, audit, fit in zip(scores, audits, fit_audits):
                if set(score.tasks["relevance"]) != {"A", "B"}:
                    raise ValueError("GLiNER failed to preserve both binary options")
                probability = float(score.probability("relevance", "A"))
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("Invalid GLiNER relevance probability")
                result.append(probability)
                self.last_audit.append({**audit, **fit})
        return result


def load_adapter(name: str, model_dir, **kwargs):
    adapters = {
        "verdict": VerdictONNXAdapter, "verdict-151m-onnx": VerdictONNXAdapter,
        "laya": LayaAdapter, "laya-multilingual": LayaAdapter,
        "julia": JuliaAdapter, "julia-1": JuliaAdapter,
        "gliner": GLiNERAdapter, "gliner2.5-small": GLiNERAdapter,
    }
    if name not in adapters:
        raise ValueError(f"Unknown decision model {name!r}; choose from {sorted(adapters)}")
    return adapters[name](model_dir, **kwargs)
