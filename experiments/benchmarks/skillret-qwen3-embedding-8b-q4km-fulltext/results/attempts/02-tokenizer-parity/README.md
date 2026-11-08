# Verified native tokenizer parity stop

The first full invocation stopped after 1,680 complete skills because the pinned
Hugging Face audit tokenizer counted 3,411 content tokens in the next skill,
while Ollama reported 3,410 evaluated tokens. The truncate:false request never
truncated source text. Native llama-server tokenization of the complete exact
input gives 3,409 content tokens when parse_special:true and 3,410 with EOS,
exactly matching inference. With parse_special:false it agrees with the
Hugging Face audit at 3,411 content tokens. A source control-token spelling
therefore explains the discrepancy. No text or vector value was shortened.

The amended validator queries the local native tokenizer only for count
mismatches and requires its complete-input count to equal prompt_eval_count.
The model, context, chunk spans, embeddings and cache identity are unchanged.
Existing vectors and completed scaling results are reused on resume.
