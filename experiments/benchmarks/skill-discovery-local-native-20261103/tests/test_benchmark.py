import contextlib
import csv
import io
import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench
import oauth_chatgpt as oauth
import summarize

TEST_TEMP = Path(__file__).resolve().parents[1]/'.test-tmp'
TEST_TEMP.mkdir(exist_ok=True)


def fixture():
    skills = [{"id": str(i), "name": f"skill-{i}", "description": "Description " + "é"*350} for i in range(6006)]
    queries = [{"query": f"task {i}", "gold_skill": f"skill-{i}", "gold_id": str(i)} for i in range(20)]
    return skills, queries


CFG = {"backend": "llama", "model": "synthetic", "base_url": "http://127.0.0.1:18080", "context_window": 32768}


def fake_result(config, trial):
    return {"text": "1. " + trial["query"]["gold_skill"],
            "usage": {"input_tokens": 101, "output_tokens": 7}, "llm_s": .125,
            "e2e_s": .25, "preflight_tokens": 101}


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.skills, cls.queries = fixture()
        cls.plan = bench.make_plan(cls.skills, cls.queries)

    def test_exact_prompt_and_140_trials(self):
        self.assertEqual(len(self.plan), 140)
        t = self.plan[0]
        self.assertTrue(t["messages"][0]["content"].startswith(bench.SYSTEM))
        self.assertIn("1. " + t["candidates"][0]["name"] + ": Description " + "é"*288, t["messages"][0]["content"])
        self.assertEqual(t["messages"][1]["content"], self.queries[0]["query"])
        self.assertNotIn("gold_id", t["messages"][0]["content"])
        self.assertEqual([c["id"] for c in self.plan[0]["candidates"][:-1]],
                         [c["id"] for c in self.plan[1]["candidates"][:-1]])
        for t in self.plan:
            self.assertIn(t["query"]["gold_id"], [c["id"] for c in t["candidates"]])
            if t["gold_replaced_last"]:
                self.assertEqual(t["candidates"][-1]["id"], t["query"]["gold_id"])
        self.assertEqual(self.skills[-1]["id"], "6005")

    def test_gold_already_present_is_not_moved(self):
        import random
        gold = random.Random(42).sample(self.skills, 50)[3]
        q = {"query": "keep", "gold_skill": gold["name"], "gold_id": gold["id"]}
        t = bench.make_plan(self.skills, [q], [50])[0]
        self.assertFalse(t["gold_replaced_last"])
        self.assertEqual(t["candidates"][3]["id"], gold["id"])

    def test_real_zero_and_estimated_block_are_distinct(self):
        blocked = self.plan[-1]
        with patch.object(bench, "infer") as infer:
            row, _ = bench.evaluate(CFG, blocked)
        infer.assert_not_called()
        self.assertTrue(row["context_limit"])
        self.assertEqual(row["error"], "CONTEXT_LIMIT_ESTIMATED")
        self.assertEqual(row["input_tokens"], "")
        self.assertEqual(row["latency_s"], "0.000000")
        t = self.plan[0]
        wrong = next(c["name"] for c in t["candidates"] if c["name"] != t["query"]["gold_skill"])
        with patch.object(bench, "infer", return_value={**fake_result(CFG, t), "text": "1. " + wrong}):
            row, _ = bench.evaluate(CFG, t)
        self.assertFalse(row["context_limit"])
        self.assertEqual(row["error"], "")
        self.assertFalse(row["exact_match"])

    def test_invalid_first_does_not_promote_gold(self):
        t = self.plan[0]
        row = bench.empty_row(CFG, t)
        bench.score(row, "1. hallucinated-name\n2. " + row["gold_skill"], t["candidates"])
        self.assertEqual(row["selected_skill"], "hallucinated-name")
        self.assertFalse(row["exact_match"])
        self.assertEqual(row["rank_of_gold"], 2)
        self.assertEqual(row["recall@1"], 0)
        self.assertEqual(row["precision@3"], 1/3)
        self.assertEqual(row["map@3"], .5)
        self.assertEqual(row["ndcg@3"], 1/math.log2(3))

    def test_markdown_suffix_case_and_duplicate_slots(self):
        candidates = [{"name": "My-Skill"}, {"name": "other"}]
        entries = bench.parse_ranked('```text\n**1. `My-Skill`**: Description\n2) "other"\n3. My-Skill\n```', candidates)
        self.assertEqual(entries[0]["name"], "My-Skill")
        self.assertTrue(entries[0]["exact"])
        self.assertIsNone(entries[2]["name"])
        entries = bench.parse_ranked('1. my-skill', candidates)
        self.assertEqual(entries[0]["name"], "My-Skill")
        self.assertFalse(entries[0]["exact"])

    def test_duplicate_names_are_retained(self):
        skills = [{"id": "a", "name": "same", "description": "a"},
                  {"id": "b", "name": "same", "description": "b"}]
        t = bench.make_plan(skills, [{"query": "x", "gold_id": "b", "gold_skill": "same"}], [2])[0]
        self.assertEqual(len(t["candidates"]), 2)
        with patch.object(bench, "infer", return_value={**fake_result(CFG, t), "text": "1. same"}):
            _, audit = bench.evaluate(CFG, t)
        self.assertEqual(audit["ambiguous_returned_names"], ["same"])

    def test_usage_and_error_classification(self):
        t = self.plan[0]
        with patch.object(bench, "infer", return_value=fake_result(CFG, t)):
            row, _ = bench.evaluate(CFG, t)
        self.assertEqual(row["total_tokens"], 108)
        with patch.object(bench, "infer", side_effect=bench.BenchError("HARDWARE_OOM")):
            row, _ = bench.evaluate(CFG, t)
        self.assertFalse(row["context_limit"])
        self.assertEqual(row["error"], "HARDWARE_OOM")
        self.assertEqual(bench.classify_error("CUDA out of memory", 500), "HARDWARE_OOM")
        self.assertEqual(bench.classify_error("context_length_exceeded", 400), "CONTEXT_LIMIT_ACTUAL")
        self.assertEqual(bench.classify_error("limit", 429), "RATE_LIMITED")

    def test_sse_completion_required(self):
        payload = (b'data: {"type":"response.output_text.delta","delta":"1. abc"}\n\n'
                   b'data: {"type":"response.completed","response":{"usage":{"input_tokens":12,"output_tokens":3}}}\n\n')
        text, usage = bench.read_sse(io.BytesIO(payload))
        self.assertEqual(text, "1. abc")
        self.assertEqual(usage["input_tokens"], 12)
        with self.assertRaises(bench.BenchError):
            bench.read_sse(io.BytesIO(b'data: {"type":"response.output_text.delta","delta":"x"}\n\n'))

    def test_oauth_payload_omits_unsupported_fields(self):
        t = self.plan[0]
        cfg = {**CFG, "backend": "chatgpt-oauth", "base_url": "https://api.openai.com/v1"}
        stream = io.BytesIO(b'data: {"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":0}}}\n\n')
        with patch.object(oauth, "access_token", return_value="SYNTHETIC"), patch.object(bench, "request", return_value=stream) as req:
            bench.infer(cfg, t)
        body = req.call_args.args[1]
        self.assertEqual(set(body), {"model", "instructions", "input", "store", "stream"})
        self.assertEqual(body["instructions"], t["messages"][0]["content"])
        self.assertTrue(body["stream"])
        self.assertFalse(body["store"])

    def test_real_tokenized_overflow_does_not_call_completion(self):
        t = self.plan[0]
        with patch.object(bench, "check_gpu_proof"), patch.object(bench, "request", side_effect=[{"prompt": "x"}, {"tokens": [0]*33000}]) as req:
            row, _ = bench.evaluate(CFG, t)
        self.assertEqual(req.call_count, 2)
        self.assertEqual(row["error"], "CONTEXT_LIMIT_TOKENIZED")
        self.assertTrue(row["context_limit"])

    def test_csv_shape_resume_no_replay_and_config_binding(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder, contextlib.redirect_stdout(io.StringIO()):
            with patch.object(bench, "infer", side_effect=fake_result) as infer:
                bench.run(CFG, self.skills, self.queries, folder)
                self.assertEqual(infer.call_count, 80)  # 60 estimated context blocks.
                bench.run(CFG, self.skills, self.queries, folder, resume=True)
                self.assertEqual(infer.call_count, 80)
            with (Path(folder)/"details.csv").open(newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                self.assertEqual(reader.fieldnames, bench.COLUMNS)
                self.assertEqual(len(reader.fieldnames), 45)
                self.assertEqual(len(list(reader)), 140)
            with self.assertRaises(ValueError):
                bench.run({**CFG, "context_window": 8192}, self.skills, self.queries, folder, resume=True)

    def test_interrupted_request_is_never_automatically_replayed(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder, contextlib.redirect_stdout(io.StringIO()):
            with patch.object(bench, "evaluate", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    bench.run(CFG, self.skills, self.queries, folder)
            with patch.object(bench, "infer") as infer:
                with self.assertRaisesRegex(ValueError, 'Uncertain previous request'):
                    bench.run(CFG, self.skills, self.queries, folder, resume=True)
                infer.assert_not_called()
            with patch.object(bench, "infer", side_effect=fake_result) as infer:
                bench.run(CFG, self.skills, self.queries, folder, resume=True, resolve=True)
                self.assertEqual(infer.call_count, 79)
            with (Path(folder)/"details.csv").open(newline='', encoding='utf-8') as f:
                first = next(csv.DictReader(f))
            self.assertEqual(first["error"], "INTERRUPTED_REQUEST_OUTCOME_UNKNOWN")

    def test_summary_distinguishes_real_zero_and_block(self):
        real = bench.empty_row(CFG, self.plan[0])
        blocked = bench.empty_row(CFG, self.plan[-1])
        blocked.update(error="CONTEXT_LIMIT_ESTIMATED", context_limit=True)
        groups = summarize.summarize([real, blocked])
        self.assertEqual(groups[0]["noncontext_accuracy"], 0)
        self.assertTrue(groups[0]["real_zero_noncontext"])
        self.assertIsNone(groups[-1]["noncontext_accuracy"])
        hardware = bench.empty_row(CFG, self.plan[0])
        hardware['error'] = 'HARDWARE_OOM'
        g = summarize.summarize([hardware])[0]
        self.assertIsNone(g['selection_precision_valid'])
        self.assertFalse(g['real_zero_noncontext'])
        self.assertEqual(g['noncontext_error_rate'], 1)

    def test_config_rejects_oauth_token_exfiltration_target(self):
        with self.assertRaises(ValueError):
            bench.validate_config({**CFG, "backend": "chatgpt-oauth", "base_url": "https://example.invalid/v1"})


class OAuthTests(unittest.TestCase):
    def test_signed_synthetic_id_token_validated(self):
        import time
        import jwt
        from cryptography.hazmat.primitives.asymmetric import rsa
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
        jwk['kid'] = 'synthetic-key'
        claims = {'iss': oauth.ISSUER, 'aud': 'synthetic-client', 'sub': 'synthetic-user',
                  'exp': int(time.time()) + 300, 'nonce': 'correct-nonce'}
        signed = jwt.encode(claims, key, algorithm='RS256', headers={'kid': 'synthetic-key'})
        def fetch(_url):
            return {'keys': [jwk]} if _url.endswith('/jwks') else {'jwks_uri': oauth.ISSUER+'/jwks'}
        with patch.object(oauth, 'http_json', side_effect=fetch):
            result = oauth.validate_id_token(signed, 'synthetic-client', 'correct-nonce')
            self.assertEqual(result['sub'], 'synthetic-user')
            with self.assertRaises(oauth.OAuthError):
                oauth.validate_id_token(signed, 'synthetic-client', 'wrong-nonce')
            with self.assertRaises(oauth.OAuthError):
                oauth.validate_id_token(signed, 'wrong-client', 'correct-nonce')
            with self.assertRaises(oauth.OAuthError):
                oauth.validate_id_token(signed, 'synthetic-client', 'correct-nonce', 'different-user')

    def test_pkce_own_registration_and_scope(self):
        p = oauth.auth_params("urn:uuid:test", "http://127.0.0.1:1455/auth/callback", "verifier", "state", "nonce")
        self.assertEqual(p["client_id"], "dynamic_agent_client")
        self.assertEqual(p["agent_name_hint"], oauth.APP_NAME)
        self.assertEqual(p["code_challenge_method"], "S256")
        self.assertNotIn("verifier", p.values())
        self.assertIn("chatgpt.tokens.use.direct", p["scope"])
        self.assertIn("offline_access", p["scope"])
        returning = oauth.auth_params("h", "r", "v", "s", "n", "own-issued-client")
        self.assertNotIn("agent_name_hint", returning)
        self.assertEqual(returning["client_id"], "own-issued-client")

    def test_missing_plan_consent_is_rejected(self):
        with self.assertRaises(oauth.OAuthError):
            oauth.token_profile({"scope": "openid email"}, "own-client", "host", {"sub": "synthetic"})

    @unittest.skipUnless(sys.platform == 'win32', 'Windows DPAPI')
    def test_dpapi_roundtrip_synthetic_only(self):
        data = b'synthetic-test-value-not-a-real-credential'
        encrypted = oauth.dpapi(data, True)
        self.assertNotIn(data, encrypted)
        self.assertEqual(oauth.dpapi(encrypted, False), data)


if __name__ == '__main__':
    unittest.main()
