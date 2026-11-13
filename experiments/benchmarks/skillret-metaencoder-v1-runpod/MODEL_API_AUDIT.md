# MetaEncoder v1 source audit

Audited 2026-11-13. No model inference, dataset/vector processing, or statistical evaluation was performed outside RunPod. Only public model metadata/source was read.

## Sources and pin
- Model: https://huggingface.co/facebook/meta-encoder-v1
- Revision: `ed04ebd4defb8e837b5eaea164143177272696ac`
- Wrapper: https://huggingface.co/facebook/meta-encoder-v1/blob/ed04ebd4defb8e837b5eaea164143177272696ac/modeling_metaencoder.py
- Config: https://huggingface.co/facebook/meta-encoder-v1/blob/ed04ebd4defb8e837b5eaea164143177272696ac/config.json
- Dependencies: https://huggingface.co/facebook/meta-encoder-v1/blob/ed04ebd4defb8e837b5eaea164143177272696ac/requirements.txt

HF metadata: public, not gated, 28,431,795,200 BF16 parameters; storage approximately 57GB. No license field in model-card metadata and no LICENSE file in root listing. Do not infer permissive redistribution rights from public availability. Keep weights off GitHub and Drive.

## Architecture/dependencies
MuseGlimmerModel (`muse_glimmer`), text backbone 52 layers, hidden size6656, 32 attention heads, 2 KV heads, head_dim128. Alternates three sliding-attention layers (window2048) and one full attention layer. Vision component also present. Embedding head in separate safetensors file is applied to final normalized hidden state of closing EOT token. Model card says one80GB GPU holds BF16; peak batch/length memory still must be measured.

Dependencies: transformers>=5.15, torch>=2.6, torchvision, accelerate>=1.0, safetensors, huggingface_hub, pillow. Config records transformers5.15.1. Pin actual resolved versions in RunPod. Official default attention implementation SDPA.

## Reproducible loading
Wrapper `MetaEncoder.from_pretrained(path, device_map='auto', dtype=torch.bfloat16, attn_implementation='sdpa')` accepts no revision kwarg. Snapshot-download the pinned revision first (including tokenizer/processor/head), import modeling_metaencoder from that snapshot, then call from_pretrained(local_snapshot). Loading by unpinned HF name would fetch current model/head independently.

## Encoding
`encode(inputs, instruction=None, batch_size=8)` returns L2-normalized float32 tensor (N,6656). Inputs are string/PIL image/dict or lists. Dictionary allowed keys text,instruction,image,images,video. Per-item instructions override default. Exact instruction choice must be saved; no universal mandatory text-retrieval pair is prescribed, although class docstring uses 'Represent the document for retrieval.'. Do not describe chosen skill-specific instructions as an official task-tuned recipe.

Processor left-pads. Fixed-date chat template adds system instruction and user content. No truncation. Training maximum16384 includes template overhead, while text config positional capacity131072. `_encode_batch` only warns above16384; it does not refuse or truncate. Treat16384 as supported trained operating protocol; chunk long full-text documents deterministically, accounting for full rendered token length. Preserve chunk mappings. Any longer-than-trained direct experiment must be separate and labeled extrapolation.

`_encode_items` concatenates all output batches on the model device. For resumable execution call encode on bounded external batches and persist FP32 CPU arrays; do not accumulate entire raw text or decision runs needlessly on GPU. Sort/group by token length for batching then restore exact ID ordering.

## Retrieval
`retrieve(query, corpus, top_k=None, query_instruction=None, corpus_instruction=None, batch_size=8)` accepts raw corpus or precomputed torch tensor. It always re-encodes the query, multiplies query by corpus transpose, then `torch.argsort(descending=True)`. Tensor corpus produces Hit.item=None. Returns Hit.index, score, item. This is the same cosine mechanism as manual similarity on identical normalized vectors, not a second model improvement. Native argsort does not explicitly request stable tie handling; document tie differences separately from quality changes. For chunked max-sim document aggregation, native chunk ranking is not automatically equivalent to final skill aggregation; compare native cosine on identical skill-vector representation or explicit chunk aggregation consistently.

## Dimensions
Neither encode nor from_pretrained supports configurable output dimensions. embedding_dim returns embedding head output rows. No claim of Matryoshka training/support in inspected card/source. Prefix truncation+L2 norm can only be exploratory, retaining full6656 vectors.

## Decision modes
`decide(state, questions, images=None, video=None)` and `decide_batch(states, questions, images=None, video=None,batch_size=8)`.
Questions map name to `{type,instructions,criteria}`; no extra keys accepted.
- noul: criteria optional mapping true/false to descriptions, options No/Yes, boolean value.
- choice: mapping2–26 distinct option keys to descriptions, value chosen key.
- score: ordered list2–26 labels, value probability-weighted expected level index, NOT an integer score or generated judgment.
All return questions[name] containing value, probabilities, confidence, scores. scores are raw option cosine similarities; probabilities softmax(scores/temperature), initial temperature0.02; confidence1−entropy/log(num_options).

State string is preserved (strip), JSON serialized with ensure_ascii=False. String and JSON states use different fixed instructions. Query includes state+task+option labels, but NOT option descriptions. Descriptions appear in separately encoded option candidates. Thus choice with IDs as options must put the query in state and each full skill text in its candidate description. Avoid putting full text of20 skills in state (would inflate context). These are instruction-conditioned decision embeddings, distinct from retrieval embeddings.

For binary/score candidate relevance, place original query and candidate canonical document in state; instructions/rubric fixed. noul labels and score labels can be cached across examples. Wrapper caches up to4096 instruction,text option pairs (clears entire cache on overflow). Repeated choice candidate sets also reuse identical options/descriptions when possible. Never give qrels to state/option descriptions.

## Calibration
`calibrate(answers, labels)` takes per-question answer dictionaries and true answer keys/booleans/level indices. Fits a single global scalar temperature, mutates model.temperature, returns it. `fit_temperature` searches301 log-spaced temperatures0.001..1 minimizing summed NLL, grouping by option count. No model-weight training. Fit on deterministic disjoint query subset and recompute softmax on held-out saved raw scores for calibration metrics; run all these calculations on RunPod. Reset to0.02 between independent experiment arms or clearly retain calibrated status. Binary qrels legitimately calibrate binary relevance; do not treat qrels as ground-truth ordinal levels for score mode.

## Feasibility
Budget must be based on measured pilot; raw decision top50 over4392 queries is219600 state encodings per decision mode before any long-document chunk overhead. Pools10/20 are nested subsets of50 and need not re-infer identical pairs. Binary and score require different task-conditioned queries, so do not silently reuse embeddings across them. Full retrieval should be completed first. Native retrieve repeats query encoding; use a bounded equivalence/latency sample if budget disallows full duplication and mark scope explicitly.
