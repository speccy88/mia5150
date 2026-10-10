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
                  "answers": {qid: {"type": "score", "score": 3.1, "confidence": 11 / 12,
                                    "legend": {str(i): text for i, text in enumerate(jev.RUBRIC)},
                                    "probabilities": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.9, "4": 0.1}}
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

    def test_opt_in_truncates_only_oversized_body_and_keeps_maximum_prefix(self):
        self.skills["0"]["skill_md"] = "start " + "x" * 33000
        original = copy.deepcopy(self.skills)
        batches, audit = self.build(oversize_policy="truncate_skill_md_right")
        self.assertEqual(self.skills, original)
        self.assertEqual([sid for batch in batches for sid in batch["candidate_ids"]], [str(i) for i in range(20)])
        questions = {qid: q for b in batches for qid, q in b["request"]["questions"].items()}
        for i in range(1, 20):
            self.assertEqual(questions["s" + str(i)]["instructions"]["question"], jev.QUESTION + jev.document(original[str(i)]))
            self.assertFalse(audit[i]["truncated"])
        row = audit[0]
        kept = row["prepared_skill_md_characters"]
        expected_body = original["0"]["skill_md"][:kept]
        expected_doc = "Skill 0 | Description 0 | " + expected_body
        self.assertEqual(questions["s0"]["instructions"]["question"], jev.QUESTION + expected_doc)
        self.assertEqual(questions["s0"]["criteria"], jev.RUBRIC)
        self.assertTrue(all(b["request"]["state"] == {"task": "Original task\nKeep this verbatim."} for b in batches))
        self.assertTrue(row["truncated"])
        self.assertEqual(row["truncation_policy_version"], jev.TRUNCATION_POLICY_VERSION)
        self.assertEqual(row["original_limits_exceeded"], ["state_plus_question"])
        self.assertLessEqual(row["prepared_estimated_pair_input_tokens"] + 2048, 32000)
        self.assertEqual(row["prepared_estimated_pair_input_tokens"] + 2048, 32000)
        self.assertGreater(row["original_estimated_pair_input_tokens"] + 2048, 32000)
        self.assertEqual(row["document_sha256"], hashlib.sha256(expected_doc.encode()).hexdigest())
        self.assertEqual(row["original_document_sha256"], hashlib.sha256(jev.document(original["0"]).encode()).hexdigest())
        self.assertEqual(row["prepared_skill_md_sha256"], hashlib.sha256(expected_body.encode()).hexdigest())
        self.assertAlmostEqual(row["retained_skill_md_character_fraction"], kept / len(original["0"]["skill_md"]))
        self.assertEqual(row["removed_skill_md_characters"], len(original["0"]["skill_md"]) - kept)
        next_doc = "Skill 0 | Description 0 | " + original["0"]["skill_md"][:kept + 1]
        request = jev.make_request("Original task\nKeep this verbatim.", [{**self.pool[0], "document": next_doc}], "jev-latest")
        self.assertGreater(jev.estimate_request_context(request)["estimated_longest_pair_token_upper_bound"] + 2048, 32000)

    def test_unicode_and_json_escaping_are_counted_without_splitting_codepoints(self):
        self.skills["0"]["skill_md"] = ('雪🙂"\\\n' * 12000)
        batches, audit = self.build(oversize_policy="truncate_skill_md_right")
        row = audit[0]
        prefix = self.skills["0"]["skill_md"][:row["prepared_skill_md_characters"]]
        self.assertEqual(row["prepared_skill_md_utf8_bytes"], len(prefix.encode("utf-8")))
        self.assertEqual(row["removed_skill_md_utf8_bytes"], len(self.skills["0"]["skill_md"].encode()) - len(prefix.encode()))
        self.assertGreater(row["prepared_utf8_bytes"], row["prepared_characters"])
        self.assertLessEqual(row["prepared_estimated_pair_input_tokens"] + 2048, 32000)
        next_prefix = self.skills["0"]["skill_md"][:row["prepared_skill_md_characters"] + 1]
        candidate = {**self.pool[0], "document": "Skill 0 | Description 0 | " + next_prefix}
        request = jev.make_request("Original task\nKeep this verbatim.", [candidate], "jev-latest")
        self.assertGreater(jev.estimate_request_context(request)["estimated_input_token_upper_bound"] + 2048, 32000)
        self.assertEqual((batches, audit), self.build(oversize_policy="truncate_skill_md_right"))

    def test_opt_in_does_not_truncate_full_candidates_merely_to_reduce_batches(self):
        for skill in self.skills.values():
            skill["skill_md"] = "x" * 10000
        default_batches, default_audit = self.build()
        opt_batches, opt_audit = self.build(oversize_policy="truncate_skill_md_right")
        self.assertEqual(default_batches, opt_batches)
        self.assertGreater(len(opt_batches), 1)
        self.assertFalse(any(row["truncated"] for row in opt_audit))
        self.assertEqual([r["document_sha256"] for r in default_audit], [r["document_sha256"] for r in opt_audit])

    def test_opt_in_rejects_immutable_overflow_without_query_or_metadata_changes(self):
        for changes in ({"query": "q" * 32000}, {}):
            original = copy.deepcopy(self.skills)
            if not changes:
                self.skills["0"]["description"] = "d" * 33000
                original = copy.deepcopy(self.skills)
            with self.assertRaisesRegex(jev.ContextAdmissionError, "immutable_content") as caught:
                self.build(oversize_policy="truncate_skill_md_right", **changes)
            self.assertEqual(self.skills, original)
            failures = caught.exception.context_failures
            self.assertEqual(len(failures), 20 if changes else 1)
            self.assertEqual(len(caught.exception.candidate_text_audit), 20)
            self.assertTrue(all(f["reason"] == "immutable_content_plus_nonempty_skill_md_cannot_fit" for f in failures))
            self.assertTrue(all(not f["truncated"] for f in failures))

    def test_empty_or_whitespace_only_prefix_is_never_metadata_only_evidence(self):
        self.skills["0"]["skill_md"] = " " * 33000 + "meaningful body"
        with self.assertRaisesRegex(jev.ContextAdmissionError, "immutable_content"):
            self.build(oversize_policy="truncate_skill_md_right")

    def test_policy_is_explicit_and_default_preservation_is_identical(self):
        self.assertEqual(self.build(), self.build(oversize_policy="preserve_and_block"))
        with self.assertRaisesRegex(jev.ContractError, "Unknown oversize policy"):
            self.build(oversize_policy="silently_summarize")
        self.assertEqual(jev.oversize_policy_contract("truncate_skill_md_right")["version"], jev.TRUNCATION_POLICY_VERSION)

    def test_pair_limit_defaults_and_invalid_overrides(self):
        self.assertEqual(jev.resolve_pair_context_limit(64000), 32000)
        self.assertEqual(jev.resolve_pair_context_limit(6000), 6000)
        self.assertEqual(jev.resolve_pair_context_limit(64000, 12000), 12000)
        for total, pair in ((64001, None), (True, None), (64000, 32768), (6000, 7000), (64000, True)):
            with self.assertRaises(jev.ContractError):
                jev.resolve_pair_context_limit(total, pair)

    def test_pair_overflow_blocks_candidate_that_fits_total(self):
        self.skills["0"]["skill_md"] = "x" * 33000
        with self.assertRaises(jev.ContextAdmissionError) as caught:
            self.build()
        error = caught.exception
        self.assertEqual(len(error.candidate_text_audit), 20)
        self.assertEqual(len(error.context_failures), 1)
        failure = error.context_failures[0]
        self.assertEqual(failure["skill_id"], "0")
        self.assertEqual(failure["limits_exceeded"], ["state_plus_question"])
        self.assertGreater(failure["estimated_pair_input_tokens"] + failure["reserved_output_tokens"], 32000)
        self.assertLess(failure["estimated_single_request_input_tokens"] + failure["reserved_output_tokens"], 64000)
        self.assertEqual(failure["original_characters"], failure["prepared_characters"])
        self.assertFalse(failure["truncated"])

    def test_large_state_is_counted_for_every_pair(self):
        with self.assertRaises(jev.ContextAdmissionError) as caught:
            self.build(query="q" * 32000)
        self.assertEqual(len(caught.exception.context_failures), 20)
        self.assertTrue(all(row["limits_exceeded"] == ["state_plus_question"] for row in caught.exception.context_failures))

    def test_context_estimator_includes_full_question_rubric_state_and_actual_id(self):
        candidate = {**self.pool[0], "document": jev.document(self.skills["0"])}
        request = jev.make_request("é" * 100, [candidate], "jev-test-version")
        request["questions"]["long_actual_question_identifier"] = request["questions"].pop("s0")
        admission = jev.estimate_request_context(request, 30)
        expected = len(jev.canonical(request)) + 30
        self.assertEqual(admission["estimated_input_token_upper_bound"], expected)
        self.assertEqual(admission["estimated_longest_pair_token_upper_bound"], expected)
        self.assertEqual(admission["per_question_estimated_pair_tokens"], {"long_actual_question_identifier": expected})
        self.assertGreater(admission["serialized_utf8_bytes"], len(jev.canonical(request).decode("utf-8")))

    def test_long_total_is_split_while_each_full_pair_fits(self):
        for skill in self.skills.values():
            skill["skill_md"] = "x" * 3000
        batches, audit = self.build()
        self.assertGreater(len(batches), 1)
        self.assertEqual([sid for batch in batches for sid in batch["candidate_ids"]], [str(i) for i in range(20)])
        self.assertTrue(all(batch["estimated_input_token_upper_bound"] + 2048 <= 64000 for batch in batches))
        self.assertTrue(all(batch["estimated_longest_pair_token_upper_bound"] + 2048 <= 32000 for batch in batches))
        self.assertTrue(all(row["original_characters"] == row["prepared_characters"] for row in audit))

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

    def test_score_answer_requires_legend_probabilities_and_confidence(self):
        batch = self.build()[0][0]
        for field in ("legend", "probabilities", "confidence"):
            for missing in (True, False):
                with self.subTest(field=field, missing=missing):
                    response = self.response(batch)
                    if missing:
                        response["answers"]["s0"].pop(field)
                    else:
                        response["answers"]["s0"][field] = None
                    with self.assertRaises(jev.ContractError):
                        jev.parse_response(response, batch, "jev-1.13.0", 64000)

    def test_score_legend_and_probability_keys_match_exact_five_level_rubric(self):
        batch = self.build()[0][0]
        valid = self.response(batch)["answers"]["s0"]
        for field in ("legend", "probabilities"):
            for change in ("missing", "extra", "numeric_keys", "list"):
                with self.subTest(field=field, change=change):
                    answer = copy.deepcopy(valid)
                    if change == "missing":
                        answer[field].pop("4")
                    elif change == "extra":
                        answer[field]["5"] = "extra" if field == "legend" else 0
                    elif change == "numeric_keys":
                        answer[field] = {int(k): v for k, v in answer[field].items()}
                    else:
                        answer[field] = list(answer[field].values())
                    with self.assertRaises(jev.ContractError):
                        jev.validate_score_answer(answer)
        answer = copy.deepcopy(valid)
        answer["legend"]["0"], answer["legend"]["4"] = answer["legend"]["4"], answer["legend"]["0"]
        with self.assertRaisesRegex(jev.ContractError, "frozen ordered criteria"):
            jev.validate_score_answer(answer)

    def test_probability_values_are_finite_numeric_and_normalized(self):
        valid = self.response(self.build()[0][0])["answers"]["s0"]
        for value in (True, "0.1", -0.01, 1.01, float("nan"), float("inf")):
            with self.subTest(value_type=type(value).__name__):
                answer = copy.deepcopy(valid)
                answer["probabilities"]["0"] = value
                with self.assertRaisesRegex(jev.ContractError, "probability"):
                    jev.validate_score_answer(answer)
        answer = copy.deepcopy(valid)
        answer["probabilities"]["0"] = 0.001
        with self.assertRaisesRegex(jev.ContractError, "sum to one"):
            jev.validate_score_answer(answer)

    def test_score_and_confidence_must_match_distribution(self):
        valid = self.response(self.build()[0][0])["answers"]["s0"]
        for field, value, message in (("score", 3.2, "probability-weighted"),
                                      ("confidence", 0.9, "MAD formula"),
                                      ("confidence", True, "finite numeric"),
                                      ("confidence", -0.1, "outside permitted range"),
                                      ("confidence", 1.1, "outside permitted range")):
            with self.subTest(field=field, value=value):
                answer = copy.deepcopy(valid)
                answer[field] = value
                with self.assertRaisesRegex(jev.ContractError, message):
                    jev.validate_score_answer(answer)

    def test_mad_confidence_uses_first_maximum_and_uniform_midpoint_denominator(self):
        answer = self.response(self.build()[0][0])["answers"]["s0"]
        # First tied mode 0 gives MAD=.8, D=1.2 and confidence=1/3.
        # Choosing tied mode 1 would give .5; max probability would give .4.
        answer.update(score=0.8, confidence=1 / 3,
                      probabilities={"0": 0.4, "1": 0.4, "2": 0.2, "3": 0.0, "4": 0.0})
        self.assertEqual(jev.validate_score_answer(answer)["confidence"], 1 / 3)
        for wrong_confidence in (0.4, 0.5):
            with self.assertRaisesRegex(jev.ContractError, "MAD formula"):
                jev.validate_score_answer({**answer, "confidence": wrong_confidence})
        answer.update(score=2.0, confidence=0.0, probabilities={str(i): 0.2 for i in range(5)})
        self.assertEqual(jev.validate_score_answer(answer)["confidence"], 0.0)
        for mode in range(5):
            answer.update(score=mode, confidence=1.0,
                          probabilities={str(i): float(i == mode) for i in range(5)})
            self.assertEqual(jev.validate_score_answer(answer)["confidence"], 1.0)

    def test_numeric_tolerance_preserves_provider_values_without_replacement(self):
        answer = self.response(self.build()[0][0])["answers"]["s0"]
        answer["score"] += 4e-7
        answer["confidence"] -= 3e-7
        parsed = jev.validate_score_answer(answer)
        self.assertEqual(parsed["native_score"], answer["score"])
        self.assertEqual(parsed["confidence"], answer["confidence"])
        self.assertEqual(parsed["probabilities"], answer["probabilities"])
        self.assertEqual(parsed["legend"], answer["legend"])
        self.assertIsNot(parsed["probabilities"], answer["probabilities"])

    def test_raw_cache_with_valid_hash_still_requires_complete_score_schema(self):
        batch = self.build()[0][0]
        response = self.response(batch)
        response["answers"]["s0"].pop("probabilities")
        raw = jev.canonical(response).decode("utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.json"
            jev.write_new(cache, {"schema_version": jev.SCHEMA_VERSION,
                                  "request_sha256": batch["request_sha256"], "status_code": 200,
                                  "raw_response": raw, "response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                                  "elapsed_seconds": 0.2})
            with self.assertRaisesRegex(jev.ContractError, "probabilities"):
                jev.parse_cache(cache, batch, "jev-1.13.0", 64000)

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

    def test_prepare_default_dual_limits_and_blocked_count_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.fixture_plan(folder)
            output_dir = folder / "default-limits"
            command = ["prepare", "--queries", str(folder / "queries.jsonl"),
                       "--skills", str(folder / "skills.jsonl"), "--dense", str(folder / "dense.jsonl"),
                       "--output-dir", str(output_dir)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(jev.main(command), 0)
            manifest = jev.read_json(output_dir / "preparation_manifest.json")
            self.assertEqual(manifest["context"]["limit_supplied"], 64000)
            self.assertEqual(manifest["context"]["pair_limit_supplied"], 32000)
            self.assertFalse(manifest["context"]["native_tokenizer_verified"])
            self.skills["0"]["skill_md"] = "x" * 33000
            (folder / "skills.jsonl").write_text("\n".join(json.dumps(s) for s in self.skills.values()) + "\n", encoding="utf-8")
            command[-1] = str(folder / "blocked-limits")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(jev.main(command), 0)
            manifest = jev.read_json(folder / "blocked-limits" / "preparation_manifest.json")
            self.assertEqual(manifest["counts"]["blocked"], 1)
            self.assertEqual(manifest["counts"]["context_failure_candidate_count"], 1)
            row = manifest["rows"][0]
            self.assertEqual(row["batches"], [])
            self.assertEqual(len(row["candidate_text_audit"]), 20)
            self.assertEqual(row["context_failures"][0]["limits_exceeded"], ["state_plus_question"])
            self.assertFalse((folder / "blocked-limits" / "requests").exists())

    def test_opt_in_prepares_entire_synthetic_200_and_10_scopes_and_freezes_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            self.skills["0"]["skill_md"] = "x" * 33000
            qpath, spath, dpath = [folder / name for name in ("queries.jsonl", "skills.jsonl", "dense.jsonl")]
            query_ids = ["synthetic-q" + str(i) for i in range(205)]
            qpath.write_text("\n".join(json.dumps({"id": qid, "query": "Original task " + qid}) for qid in query_ids) + "\n", encoding="utf-8")
            spath.write_text("\n".join(json.dumps(s) for s in self.skills.values()) + "\n", encoding="utf-8")
            dpath.write_text("\n".join(json.dumps({"query_id": qid, "candidates": [{"skill_id": str(i), "score": 1 - i / 100} for i in range(50)]}) for qid in query_ids) + "\n", encoding="utf-8")
            plans = {}
            for scope, expected in (("pilot", 200), ("smoke", 10)):
                command = ["prepare", "--queries", str(qpath), "--skills", str(spath), "--dense", str(dpath),
                           "--output-dir", str(folder / scope), "--scope", scope,
                           "--oversize-policy", "truncate_skill_md_right"]
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(jev.main(command), 0)
                plan = jev.read_json(folder / scope / "preparation_manifest.json")
                plans[scope] = plan
                self.assertEqual(plan["counts"]["selected"], expected)
                self.assertEqual(plan["counts"]["prepared"], expected)
                self.assertEqual(plan["counts"]["blocked"], 0)
                self.assertEqual(plan["counts"]["prepared_candidate_pairs"], expected * 20)
                self.assertEqual(plan["counts"]["queries_with_truncation"], expected)
                self.assertEqual(plan["counts"]["truncated_candidate_pairs"], expected)
                self.assertEqual(plan["context"]["truncation_policy"], jev.oversize_policy_contract("truncate_skill_md_right"))
                self.assertEqual(plan["context"]["truncation_policy_sha256"], jev.digest(plan["context"]["truncation_policy"]))
                self.assertEqual(plan["network_calls"], 0)
                self.assertFalse(plan["credentials_accessed"])
                self.assertTrue(all(len(row["candidate_text_audit"]) == 20 for row in plan["rows"]))
            self.assertEqual(plans["smoke"]["rows"], plans["pilot"]["rows"][:10])


if __name__ == "__main__":
    unittest.main()
