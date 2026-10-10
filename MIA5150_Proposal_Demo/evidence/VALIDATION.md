# Validation performed in the authoring environment

## Executed

- Parsed and validated the `.ipynb` with `nbformat`.
- Compiled every code cell without executing its top-level statements.
- Executed 32 offline unit/structure checks against helper functions extracted from the notebook.
- Checked set-based recall/F1/exact match, nested catalogue membership and deterministic order, vector-ranking index mapping using hand-built numerical vectors, tool-result alias parsing using plain Python message fixtures, and routing-plus-selection token arithmetic.
- Verified that the 200 development IDs match the copy read from the uploaded repository.
- Verified that there are no custom class definitions or fabricated executed outputs in the notebook.
- Checked the native skills prompt includes all three slots required by the inspected upstream source.

The test log is `helper_tests.txt`. Run the same checks with:

```bash
python -m unittest discover -s tests -v
```

## Not executed

- SkillRet download or full corpus preparation.
- BM25S library indexing/ranking.
- BGE model loading or real embeddings.
- Native LangChain graph, Deep Agents middleware, or filesystem backend at runtime.
- Ollama or any other LLM inference.
- Model-quality, latency, memory, or token-consumption benchmarks.
- Installation of the pinned direct dependencies on macOS.

The environment could not resolve `huggingface.co`, `pypi.org`, or the configured package index. Consequently the missing external dependencies could not be installed. No model-generated data or benchmark numbers were fabricated to replace those tests.

## Important distinction

The plain Python fixtures in the tests exercise response parsing and numerical mechanics. They are not a simulated benchmark, not a stand-in for Jev/CLM/Qwen, and not evidence of real agent accuracy. All notebook cells remain unexecuted in the delivered artifact.

## Evidence preservation

The uploaded source ZIP and its existing notebooks were read but not edited. The new notebook writes only into its own timestamped results directories; the validation process does not run any code from the old repository.
