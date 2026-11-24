# Benchmark Subset

## Prompt-based
2026-11-05 to 2026-11-07 · Nine-model OpenRouter prompt benchmark tested 20 single-label questions at seven sizes from 50 to 5,000 skills and yielded 1,184 responses for the ranking audit. 

## BM25
2026-10-24 · October matched full-query BM25 measured lexical retrieval on all 4,392 queries across four catalog sizes and three seeds. 

## Hybrid
2026-10-24 · October matched full-query hybrid evaluated BM25 and BGE-small rank fusion on all 4,392 queries across four catalog sizes and three seeds. 

## Vector
2026-10-24 · October matched full-query vector retrieval evaluated BGE-small metadata embeddings on all 4,392 queries across four catalog sizes and three seeds. 
2026-11-08 · BGE-M3 dense document views compared metadata, matched Qwen chunks and native 8K chunks on eight scaling sizes and the full test set. (we found that good name+description is better than putting full skill.md) 

## Rerank and decision
2026-11-22 · OpenAI Decisions scaling benchmark compared BM25, vector and hybrid retrieval with and without OpenAI Decisions reranking on 200 queries across seven catalog sizes and three seeds. 
