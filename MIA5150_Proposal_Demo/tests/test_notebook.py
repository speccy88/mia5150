"""Offline checks of the notebook's small Python functions.

These tests do NOT download SkillRet, install packages, run BM25/BGE, connect to
Ollama, or claim that the LangChain/Deep Agents integration has been executed.
Functions are read from the notebook so the tests do not maintain a second copy.
"""
import ast
import json
import random
import unittest
from pathlib import Path
from types import SimpleNamespace

import nbformat
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = nbformat.read(ROOT / "proposal_workflow.ipynb", as_version=4)


def load_functions():
    names = {"recall_at_k", "selection_scores", "make_catalogue", "vector_rank",
             "decode_choice", "input_tokens", "make_metadata"}
    namespace = {"random": random, "np": np}
    for cell in NOTEBOOK.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name in names]
        if selected:
            code = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
            exec(compile(code, "<notebook helper functions>", "exec"), namespace)
    return namespace


FN = load_functions()


def message(ids, name="choose_skills", invalid=False):
    return SimpleNamespace(
        tool_calls=[{"name": name, "args": {"ids": ids}}],
        invalid_tool_calls=[{}] if invalid else [],
    )


class NotebookTests(unittest.TestCase):
    def setUp(self):
        self.skills = pd.DataFrame({"id": [f"s{i}" for i in range(40)],
                                    "position": range(40),
                                    "text": [f"description {i}" for i in range(40)]})

    def test_nbformat_valid(self):
        nbformat.validate(NOTEBOOK)

    def test_all_code_cells_compile(self):
        for i, cell in enumerate(NOTEBOOK.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"cell {i}", "exec")

    def test_no_custom_classes_in_notebook(self):
        for cell in NOTEBOOK.cells:
            if cell.cell_type == "code":
                self.assertFalse(any(isinstance(n, ast.ClassDef) for n in ast.walk(ast.parse(cell.source))))

    def test_no_fabricated_executed_outputs(self):
        for cell in NOTEBOOK.cells:
            if cell.cell_type == "code":
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])

    def test_pilot_ids_preserved(self):
        ids = json.loads((ROOT / "pilot_query_ids.json").read_text())
        source = json.loads((ROOT / "source_notes/pilot_query_ids.json").read_text())
        self.assertEqual(ids, source)
        self.assertEqual(len(ids), 200)
        self.assertEqual(len(set(ids)), 200)

    def test_recall_multiple_gold(self):
        self.assertEqual(FN["recall_at_k"](["a", "x", "b"], {"a", "b"}, 2), 0.5)

    def test_recall_cutoff(self):
        self.assertEqual(FN["recall_at_k"](["x", "a"], {"a"}, 1), 0.0)

    def test_empty_ranking(self):
        self.assertEqual(FN["recall_at_k"]([], {"a"}, 5), 0.0)

    def test_recall_rejects_empty_gold(self):
        with self.assertRaises(ValueError):
            FN["recall_at_k"](["a"], set())

    def test_recall_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            FN["recall_at_k"](["a", "a"], {"a"})

    def test_recall_rejects_nonpositive_k(self):
        with self.assertRaises(ValueError):
            FN["recall_at_k"](["a"], {"a"}, 0)

    def test_selection_f1_and_exact(self):
        self.assertEqual(FN["selection_scores"](["a"], {"a", "b"}),
                         {"f1": 2 / 3, "exact": 0.0})
        self.assertEqual(FN["selection_scores"](["b", "a"], {"a", "b"}),
                         {"f1": 1.0, "exact": 1.0})

    def test_extras_penalize_precision(self):
        self.assertAlmostEqual(FN["selection_scores"](["a", "x"], {"a"})["f1"], 2 / 3)

    def test_no_selection(self):
        self.assertEqual(FN["selection_scores"]([], {"a"}), {"f1": 0.0, "exact": 0.0})

    def test_nested_library_membership(self):
        a = FN["make_catalogue"](self.skills, {"s9", "s21"}, 10, "q-test")
        b = FN["make_catalogue"](self.skills, {"s9", "s21"}, 25, "q-test")
        self.assertTrue(set(a.id) <= set(b.id))
        self.assertTrue({"s9", "s21"} <= set(a.id))
        self.assertEqual((len(a), len(b)), (10, 25))

    def test_catalogue_reproducible(self):
        a = FN["make_catalogue"](self.skills, {"s9"}, 10, "q-test")
        b = FN["make_catalogue"](self.skills, {"s9"}, 10, "q-test")
        self.assertEqual(a.id.tolist(), b.id.tolist())

    def test_catalogue_does_not_mutate_input(self):
        old = self.skills.copy(deep=True)
        FN["make_catalogue"](self.skills, {"s9"}, 10, "q-test")
        pd.testing.assert_frame_equal(old, self.skills)

    def test_catalogue_bad_size(self):
        for size in [0, 41]:
            with self.assertRaises(ValueError):
                FN["make_catalogue"](self.skills, {"s9"}, size, "q-test")

    def test_catalogue_unknown_gold(self):
        with self.assertRaises(ValueError):
            FN["make_catalogue"](self.skills, {"absent"}, 10, "q-test")

    def test_vector_search_stays_in_library(self):
        vectors = np.zeros((40, 2))
        vectors[0] = [1, 0]
        vectors[1] = [0.8, 0.6]
        vectors[2] = [0, 1]
        # s0 is globally best but isn't in this query's candidate library.
        result = FN["vector_rank"](self.skills.iloc[[2, 1]], np.array([1, 0]), vectors, 1)
        self.assertEqual(result, ["s1"])

    def test_vector_empty_pool(self):
        self.assertEqual(FN["vector_rank"](self.skills.iloc[:0], np.array([1, 0]), np.zeros((40, 2))), [])

    def test_vector_ties_are_stable(self):
        self.assertEqual(FN["vector_rank"](self.skills.iloc[[2, 1]], np.array([1, 0]), np.zeros((40, 2)), 2),
                         ["s2", "s1"])

    def test_valid_aliases(self):
        self.assertEqual(FN["decode_choice"](message(["skill-0002"]), ["skill-0001", "skill-0002"]), ["skill-0002"])

    def test_unknown_aliases_not_repaired(self):
        with self.assertRaises(ValueError):
            FN["decode_choice"](message(["not-in-catalogue"]), ["skill-0001"])

    def test_duplicate_aliases_rejected(self):
        with self.assertRaises(ValueError):
            FN["decode_choice"](message(["skill-1", "skill-1"]), ["skill-1"])

    def test_bad_tool_name_rejected(self):
        with self.assertRaises(ValueError):
            FN["decode_choice"](message(["skill-1"], name="execute"), ["skill-1"])

    def test_bad_args_rejected(self):
        for ids in ["skill-1", [1], None]:
            with self.assertRaises(ValueError):
                FN["decode_choice"](message(ids), ["skill-1"])

    def test_no_tool_and_many_tools_rejected(self):
        for calls in [[], message([]).tool_calls * 2]:
            with self.assertRaises(ValueError):
                FN["decode_choice"](SimpleNamespace(tool_calls=calls, invalid_tool_calls=[]), ["s1"])

    def test_usage_includes_routing(self):
        self.assertEqual(FN["input_tokens"]({"usage": {"input_tokens": 20}}, {"usage": {"input_tokens": 100}}), 120)

    def test_unknown_usage_is_not_zero(self):
        self.assertIsNone(FN["input_tokens"]({"usage": {"input_tokens": 20}}, {}))
        self.assertIsNone(FN["input_tokens"]())

    def test_required_native_prompt_slots(self):
        text = "\n".join(cell.source for cell in NOTEBOOK.cells if cell.cell_type == "code")
        for slot in ["{skills_locations}", "{skills_load_warnings}", "{skills_list}"]:
            self.assertIn(slot, text)

    def test_selector_signature_has_no_gold(self):
        code = next(c.source for c in NOTEBOOK.cells if c.cell_type == "code" and "def select_skills(" in c.source)
        function = next(n for n in ast.parse(code).body if isinstance(n, ast.FunctionDef))
        self.assertEqual([a.arg for a in function.args.args], ["query", "candidates", "model", "tokenizer", "trace"])
        self.assertFalse(any(isinstance(n, ast.Name) and n.id in {"gold", "qrels", "labels"} for n in ast.walk(function)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
