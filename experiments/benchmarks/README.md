# Benchmark studies

[All experiments](../README.md) · [Catalog](../../docs/SkillRet-Benchmark-Catalog.md)

| Study | Start here |
| --- | --- |
| Skill discovery listing, OAuth variant | [skill-discovery-oauth-v1-20261103/REPORT.md](skill-discovery-oauth-v1-20261103/REPORT.md) |
| Skill discovery, local native answer parsing | [skill-discovery-local-native-20261103/README.md](skill-discovery-local-native-20261103/README.md) and [comparison with coverage](skill-discovery-local-native-20261103/COMPARISON.md) |
| BM25 retrieval and full listing | [benchmark_bm25/](benchmark_bm25/) |
| Full-catalog RunPod retrieval and CLM head adaptation | [skillret-full-catalog-runpod/README.md](skillret-full-catalog-runpod/README.md) |
| CPU-only pipelines and catalog sizes | [skillret-cpu-benchmark/README.md](skillret-cpu-benchmark/README.md) |
| BM25 → Jev / CLM reranking | [skillret-reranker-study/README.md](skillret-reranker-study/README.md) |
| Official embedding reproduction and hosted Jev fusion | [skillret-embedding-jev-fusion/README.md](skillret-embedding-jev-fusion/README.md) |
| Local metadata retrieval, reranking, and adaptation | [skillret-local-metadata-20261005/README.md](skillret-local-metadata-20261005/README.md) |

Protocols and completion limits differ. The metadata study includes interrupted
and unrun arms, and the full hosted expansion was not run. Follow the linked
READMEs and catalog rather than treating every implementation as a result.

## Offline OpenRouter audit (8 November 2026)

[Prompt-result audit](openrouter-prompt-audit-20261108/README.md) preserves the nine-model raw evidence and reports full metrics, coverage, clustered uncertainty, latency, and cost limitations. The [conditional retrieval comparison](openrouter-matched-retrieval-20261108/README.md) is a separate reconstruction rather than a verified identical-prompt rerun. No paid API calls were made.
