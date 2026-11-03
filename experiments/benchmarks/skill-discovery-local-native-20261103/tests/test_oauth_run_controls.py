import contextlib
import csv
import io
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bench
from test_benchmark import fixture, CFG, TEST_TEMP


class OAuthRunControlTests(unittest.TestCase):
    def test_quota_failure_stops_after_one_call_and_preserves_pending_rows(self):
        skills, queries=fixture()
        cfg={**CFG,'backend':'chatgpt-oauth','stop_on_any_error':True}
        first=bench.make_plan(skills,queries)[0]
        row=bench.empty_row(cfg,first)
        row.update(error='RATE_LIMITED',rate_limited=True)
        audit={'provider_code':'subscription_sharing_usage_limit_exceeded','http_status':429,'retry_after_s':60}
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as folder, contextlib.redirect_stdout(io.StringIO()), \
             patch.object(bench,'evaluate',return_value=(row,audit)) as evaluate:
            status=bench.run(cfg,skills,queries,folder)
            self.assertEqual(evaluate.call_count,1)
            self.assertEqual(status['not_run_trials'],139)
            self.assertFalse(status['complete'])
            with (Path(folder)/'details.csv').open(newline='',encoding='utf-8') as f:
                reader=csv.DictReader(f)
                self.assertEqual(reader.fieldnames,bench.COLUMNS)
                rows=list(reader)
            self.assertEqual(len(rows),140)
            self.assertEqual(rows[0]['error'],'RATE_LIMITED')
            self.assertTrue(all(r['error']=='NOT_RUN_AFTER_MODEL_STOP' for r in rows[1:]))
            db=sqlite3.connect(Path(folder)/'state.sqlite3')
            self.assertEqual(db.execute('SELECT COUNT(*) FROM trials').fetchone()[0],1)
            db.close()

    def test_stream_quota_failure_keeps_machine_readable_reason(self):
        raw=b'data: {"type":"response.failed","response":{"error":{"code":"subscription_sharing_usage_limit_exceeded"}}}\n\n'
        with self.assertRaises(bench.BenchError) as caught:
            bench.read_sse(io.BytesIO(raw))
        self.assertEqual(caught.exception.code,'RATE_LIMITED')
        self.assertEqual(caught.exception.provider_code,'subscription_sharing_usage_limit_exceeded')

    def test_completed_stream_audits_actual_server_model_reasoning_and_usage(self):
        raw=b'data: {"type":"response.completed","response":{"id":"synthetic-response","model":"synthetic-model","reasoning":{"effort":"medium"},"usage":{"input_tokens":143,"output_tokens":9}}}\n\n'
        audit={}
        _,usage=bench.read_sse(io.BytesIO(raw),audit=audit)
        self.assertEqual(audit['reasoning_effort'],'medium')
        self.assertEqual(audit['server_model'],'synthetic-model')
        self.assertEqual(usage['input_tokens'],143)

    def test_model_refusal_is_not_a_valid_empty_answer(self):
        raw=b'data: {"type":"response.refusal.delta","delta":"synthetic refusal"}\n\n'
        with self.assertRaises(bench.BenchError) as caught:
            bench.read_sse(io.BytesIO(raw))
        self.assertEqual(caught.exception.code,'MODEL_REFUSAL')


if __name__=='__main__':
    unittest.main()
