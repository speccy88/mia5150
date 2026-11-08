"""Offline tests; synthetic values are test-only and never published as results."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from PIL import Image

MODULE_PATH = Path(__file__).resolve().parents[1] / "visualizations/plot_audit.py"
spec = importlib.util.spec_from_file_location("audit_visualizations", MODULE_PATH)
viz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viz)


class AuditVisualizationTests(unittest.TestCase):
    def test_missing_is_not_zero(self):
        import math
        for value in [None, "", "NA", "null"]:
            self.assertTrue(math.isnan(viz.number(value)))
        self.assertEqual(viz.number("0"), 0)

    def test_absent_expected_cell_fails(self):
        with self.assertRaisesRegex(ValueError, "Missing expected audit cell"):
            viz.get({}, "model", 50)

    def test_coverage_preserves_different_missing_states(self):
        base = {"n_expected": 20, "n_completed": 20, "n_context_excluded": 0, "n_unattempted": 0}
        self.assertEqual(viz.coverage_state(base), ("complete", "20/20"))
        partial = dict(base, n_completed=13, n_unattempted=7)
        self.assertEqual(viz.coverage_state(partial), ("partial", "13/20\n7 absent"))
        context = dict(base, n_completed=0, n_context_excluded=20)
        self.assertEqual(viz.coverage_state(context), ("blocked", "0/20\ncontext"))
        absent = dict(base, n_completed=0, n_unattempted=20)
        self.assertEqual(viz.coverage_state(absent), ("unattempted", "0/20\nno result"))

    def test_rejects_zero_quality_for_empty_cohort(self):
        record = {"model": "m", "library_size": 50, "cohort": "no_reported_invalid",
                  "n_expected": 20, "n_completed": 0, "n_scored": 0, "top1_accuracy": 0}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            with path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=record)
                writer.writeheader(); writer.writerow(record)
            with self.assertRaisesRegex(ValueError, "Empty quality cohort"):
                viz.load_summary(path)

    def test_matched_retrieval_uses_only_the_same_primary_queries(self):
        scores = {"recall@5": 1, "ndcg@5": 1, "mrr@5": 1}
        prompts = [{"model": "m", "library_size": 50, "query_id": "q01", "status": "completed", "scores": {viz.PRIMARY: scores}},
                   {"model": "m", "library_size": 50, "query_id": "q02", "status": "completed", "scores": {viz.PRIMARY: None}}]
        retrieval = [{"method": method, "size": 50, "k": 5, "query_index": q,
                      "label_contract": "name_equivalent_single_target", "recall": q, "ndcg": q, "mrr": q}
                     for method in viz.RETRIEVAL_METHODS for q in [0, 1]]
        with tempfile.TemporaryDirectory() as tmp:
            p, r = Path(tmp) / "p.jsonl", Path(tmp) / "r.csv"
            p.write_text("\n".join(json.dumps(row) for row in prompts))
            with r.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=retrieval[0])
                writer.writeheader(); writer.writerows(retrieval)
            matched = viz.build_matched_rows(p, r, ["m"], [50])
            self.assertEqual(len(matched), 5)
            for row in matched:
                self.assertEqual(row["n"], 1)
                self.assertEqual(row["query_indices"], [0])
                self.assertEqual(row["recall@5"], 1 if row["method"] == "prompt" else 0)

    def test_jev_requires_zero_matched_evidence_for_gap_slide(self):
        with self.assertRaisesRegex(ValueError, "Non-empty Jev evidence"):
            viz.plot_jev_gap({"matching_completed_jev_cells": 1})

    def test_published_figures_match_manifest_bytes_and_dimensions(self):
        directory = MODULE_PATH.parents[1] / "results/figures"
        manifest_path = directory / "figure_manifest.json"
        if not manifest_path.exists():
            self.skipTest("Generated figures are not installed in this checkout")
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(manifest["base_commit"], viz.BASE_COMMIT)
        for item in manifest["figures"]:
            for extension in ["png", "svg"]:
                path = directory / item[extension]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item[extension + "_sha256"])
            with Image.open(directory / item["png"]) as image:
                self.assertEqual(image.size, (1920, 1080))
            root = ET.parse(directory / item["svg"]).getroot()
            viewbox = list(map(float, root.attrib["viewBox"].split()))
            self.assertAlmostEqual(viewbox[2] / viewbox[3], 16 / 9)

    def test_all_nine_models_in_coverage_and_exact_16_9_output(self):
        models = list(viz.MODEL_NAMES)
        sizes = [50, 100, 250, 500, 1000, 2000, 5000]
        rows = [{"model": model, "library_size": str(size), "cohort": viz.SENSITIVITY,
                 "n_expected": 20, "n_completed": 20, "n_context_excluded": 0, "n_unattempted": 0}
                for model in models for size in sizes]
        fig = viz.plot_coverage(viz.index_rows(rows), models, sizes)
        with tempfile.TemporaryDirectory() as tmp:
            output = viz.save_figure(fig, Path(tmp), "test_coverage")
            with Image.open(Path(tmp) / output["png"]) as image:
                self.assertEqual(image.size, (1920, 1080))
            root = ET.parse(Path(tmp) / output["svg"]).getroot()
            viewbox = list(map(float, root.attrib["viewBox"].split()))
            self.assertAlmostEqual(viewbox[2] / viewbox[3], 16 / 9)
            svg_text = " ".join(root.itertext())
            for name in viz.MODEL_NAMES.values():
                self.assertIn(name, svg_text)
            self.assertIn("20/20", svg_text)
            self.assertEqual(len(output["png_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
