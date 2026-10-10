"""Offline contract tests with synthetic fixtures only; zero hosted/API activity."""
import argparse
import ast
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import run_jev_rerank as jev


class PreparationContract(unittest.TestCase):
    def setUp(self):
        self.skills = {str(i): {"id": str(i), "name": "Skill " + str(i),
                               "description": "Description " + str(i),
                               "skill_md": "# Capability\nEntire unshortened skill body."} for i in range(50)}
        self.pool = [{"skill_id": str(i), "dense_rank": i + 1, "dense_score": 1 - i / 100} for i in range(20)]

    def build(self, **changes):
        args = {"query_id": "q1", "query": "Original task\nKeep this verbatim.", "pool": self.pool,
                "skills": self.skills, "model": "jev-latest", "context_limit": 64000,
                "reserved_output_tokens": 2048, "overhead_tokens": 2048}
        args.update(changes)
        return jev.build_batches(**args)

    def response(self, batch, **changes):
        result = {"model": "jev-1.13.0", "usage": {"input_tokens": 100, "output_tokens": 12},
                  "answers": {qid: {"type": "score", "score": 3.1, "confidence": 0.8}
                              for qid in batch["question_ids"]}}
        result.update(changes)
        return result

    def test_exact_state_full_text_fixed_rubric_and_no_labels(self):
        self.skills["0"]["gold_label"] = "this must not enter prompt"
        batches, audit = self.build()
        self.assertEqual(len(batches), 1)
        packet = batches[0]["request"]
        self.assertEqual(packet["state"], {"task": "Original task\nKeep this verbatim."})
        self.assertEqual(packet["questions"]["s0"]["criteria"], jev.RUBRIC)
        self.assertEqual(packet["questions"]["s0"]["instructions"]["question"],
                         jev.QUESTION + "Skill 0 | Description 0 | # Capability\nEntire unshortened skill body.")
        self.assertNotIn("gold_label", json.dumps(packet))
        self.assertTrue(all(not row["truncated"] and row["original_characters"] == row["prepared_characters"] for row in audit))

    def test_deterministic_batching_keeps_complete_frozen_pool(self):
        first, _ = self.build(context_limit=6000)
        second, _ = self.build(context_limit=6000)
        self.assertGreater(len(first), 1)
        self.assertEqual(first, second)
        self.assertEqual([sid for batch in first for sid in batch["candidate_ids"]], [str(i) for i in range(20)])
        self.assertTrue(all(b["estimated_input_token_upper_bound"] + 2048 <= 6000 for b in first))

    def test_single_candidate_overflow_rejects_whole_query(self):
        self.skills["19"]["skill_md"] = "x" * 64000
        with self.assertRaisesRegex(jev.ContractError, "no_truncation"):
            self.build()

    def test_metadata_only_evidence_is_rejected(self):
        self.skills["0"].pop("skill_md")
        with self.assertRaises(jev.ContractError):
            self.build()

    def test_selection_is_hash_order_and_smoke_subset(self):
        ids = ["q" + str(i) for i in range(250)]
        expected = sorted(ids, key=lambda s: (hashlib.sha256(s.encode()).hexdigest(), s))[:200]
        self.assertEqual(jev.select_ids(ids), expected)
        self.assertEqual(jev.select_ids(list(reversed(ids)), 10), expected[:10])

    def test_version_usage_answer_and_range_validation(self):
        batch = self.build()[0][0]
        response = self.response(batch)
        scores, usage = jev.parse_response(response, batch, "jev-1.13.0", 64000)
        self.assertEqual(scores[0]["relevance_score"], 4.1)
        self.assertEqual(usage["input_tokens"], 100)
        invalid = []
        r = copy.deepcopy(response); r["model"] = "jev-latest"; invalid.append(r)
        r = copy.deepcopy(response); r["model"] = "jev-other"; invalid.append(r)
        r = copy.deepcopy(response); r["usage"]["input_tokens"] = True; invalid.append(r)
        r = copy.deepcopy(response); r["answers"].pop("s0"); invalid.append(r)
        r = copy.deepcopy(response); r["answers"]["s20"] = r["answers"]["s0"]; invalid.append(r)
        r = copy.deepcopy(response); r["answers"]["s0"]["score"] = 4.01; invalid.append(r)
        r = copy.deepcopy(response); r["answers"]["s0"]["confidence"] = float("nan"); invalid.append(r)
        for response in invalid:
            with self.assertRaises(jev.ContractError):
                jev.parse_response(response, batch, "jev-1.13.0", 64000)

    def test_ranking_confidence_then_dense_and_rrf60_identity(self):
        scored = [{"skill_id": str(i), "native_score": 2, "relevance_score": 3, "confidence": None} for i in range(20)]
        scored[19]["confidence"] = 0.9
        jr, rr, details = jev.rank_and_fuse(self.pool, scored)
        self.assertEqual(jr[:3], ["19", "0", "1"])
        self.assertEqual(set(jr), {str(i) for i in range(20)})
        self.assertEqual(set(rr), set(jr))
        self.assertAlmostEqual(details[0]["rrf60_score"], 1 / 80 + 1 / 61)
        self.assertEqual(rr[0], "0")
        with self.assertRaises(jev.ContractError):
            jev.rank_and_fuse(self.pool, scored[:-1])

    def test_immutable_artifacts_and_strict_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "artifact.json"
            jev.write_new(target, {"a": 1})
            jev.write_new(target, {"a": 1})
            with self.assertRaises(jev.ContractError):
                jev.write_new(target, {"a": 2})
        with self.assertRaises(jev.ContractError):
            jev.loads('{"a":1,"a":2}')
        with self.assertRaises(jev.ContractError):
            jev.loads('{"a":NaN}')

    def test_execution_block_and_absence_of_credential_network_imports(self):
        with contextlib.redirect_stderr(io.StringIO()) as output:
            self.assertEqual(jev.main(["execute"]), 2)
        self.assertIn("No credential was read", output.getvalue())
        tree = ast.parse(Path(jev.__file__).read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(x.name.split(".")[0] for x in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module.split(".")[0])
        self.assertFalse(imports & {"os", "urllib", "requests", "httpx", "socket", "keyring", "subprocess"})

    def fixture_plan(self, folder):
        qpath, spath, dpath = [folder / name for name in ("queries.jsonl", "skills.jsonl", "dense.jsonl")]
        qpath.write_text(json.dumps({"id": "q1", "query": "Original request"}) + "\n", encoding="utf-8")
        spath.write_text("\n".join(json.dumps(s) for s in self.skills.values()) + "\n", encoding="utf-8")
        dpath.write_text(json.dumps({"query_id": "q1", "candidates": [{"skill_id": str(i), "score": 1 - i / 100} for i in range(50)]}) + "\n", encoding="utf-8")
        plan_dir = folder / "plan"
        args = argparse.Namespace(queries=qpath, skills=spath, dense=dpath, output_dir=plan_dir,
                                  scope="pilot", model="jev-1.13.0", model_version=None,
                                  context_limit=6000, reserved_output_tokens=2048, overhead_tokens=2048)
        result = jev.prepare(args)
        self.assertEqual(result["network_calls"], 0)
        self.assertFalse((plan_dir / "jev_rerank.json").exists())
        return plan_dir

    def write_cache(self, cache_dir, batch, **response_changes):
        raw = json.dumps(self.response(batch, **response_changes))
        jev.write_new(cache_dir / (batch["request_sha256"] + ".json"), {
            "schema_version": jev.SCHEMA_VERSION, "request_sha256": batch["request_sha256"],
            "status_code": 200, "raw_response": raw,
            "response_sha256": hashlib.sha256(raw.encode()).hexdigest(), "elapsed_seconds": 0.2})

    def test_ingestion_never_emits_partial_query_ranking_and_can_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            plan_dir = self.fixture_plan(folder)
            manifest = jev.read_json(plan_dir / "preparation_manifest.json")
            batches = manifest["rows"][0]["batches"]
            self.assertGreater(len(batches), 1)
            cache_dir = folder / "cache"
            self.write_cache(cache_dir, batches[0])
            args = argparse.Namespace(plan=plan_dir / "preparation_manifest.json", responses_dir=cache_dir,
                                      output_dir=folder / "partial")
            result = jev.ingest(args)
            self.assertEqual(result["complete_queries"], 0)
            self.assertEqual(jev.read_json(args.output_dir / "jev_rerank.json")["rankings"], {"q1": []})
            for batch in batches[1:]:
                self.write_cache(cache_dir, batch)
            args.output_dir = folder / "complete"
            result = jev.ingest(args)
            self.assertEqual(result["complete_queries"], 1)
            ranking = jev.read_json(args.output_dir / "jev_rerank.json")["rankings"]["q1"]
            self.assertEqual(ranking, [str(i) for i in range(20)])
            usage = jev.read_json(args.output_dir / "jev_usage.json")
            self.assertEqual(usage["known_input_tokens_validated_responses"], 100 * len(batches))
            self.assertEqual(usage["network_calls_this_process"], 0)

    def test_cache_binding_rejects_mismatched_response_digest(self):
        batch = self.build()[0][0]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.write_cache(folder, batch)
            path = folder / (batch["request_sha256"] + ".json")
            data = jev.read_json(path)
            data["raw_response"] += " "
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(jev.ContractError):
                jev.parse_cache(path, batch, "jev-1.13.0", 64000)


if __name__ == "__main__":
    unittest.main()
