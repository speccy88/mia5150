# Native-adapter correction: separate version

The Qwen3.5 GGUF template prefills `<think>\n`. `/completion` returns only text
generated after the prompt, excluding this prefilled opening. The v1 parser could
not recognize every reasoning block and sometimes interpreted internal lists as
a ranking. Local v1 metrics are preserved as diagnostics, rather than final
comparative results.

Evidence: `qwen35-format-audit.json` contains the actual template suffix.
62 of Qwen3.5-4B's 80 responses reach the 500-token cap. All first entries are
reasoning/non-final text; some gold items occur later in that text. Strict v1
top-1 is zero, but v1 top-10 is not a reliable measure of the final answer.

`bench_native_v2.py` and `results/local-native-v2/` isolate the correction:

- The sampler, 20 questions, candidates, truncated descriptions, prompts,
  temperature 0, seed 42 and 500-token cap are unchanged.
- The opening already supplied by the template is retained only as parser state;
  it is not added to the prompt a second time.
- `<think>…</think>` blocks and Gemma's `<|channel>thought…<channel|>` channel are
  excluded before scoring the final answer.
- Unfinished reasoning yields `REASONING_NO_FINAL_ANSWER`, retaining actual tokens.
- Invalid or repeated final-answer ranks are neither removed nor promoted.
- Raw native responses remain in `private-responses/`, excluded from Git.
- The same 45 columns and transactional SQLite/CSV persistence are retained.

All 34 tests pass, including seven dedicated regressions. Independent comparison
of v1/v2 plans confirms the same 140 prompt hashes and candidate IDs. Previously
executed v1 models are preserved. The v1 suite stopped cleanly after Ling; the
corrected native suite uses separate aliases, proofs, configurations and results.
This correction did not alter the separate OAuth scripts or results.
