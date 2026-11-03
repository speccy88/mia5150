"""Build a Markdown section for the coordinated report; no sending or publishing."""
import collections
import datetime
import json
from pathlib import Path

from review_local_results import review, ROOT

def percent(value):
    return '—' if value is None else f'{value:.1%}'

def main():
    audited=review()
    jobs=json.loads((ROOT/'local-models.json').read_text())
    by_model=collections.defaultdict(list)
    for row in audited['summary']:
        by_model[row['model']].append(row)
    lines=['# Skill discovery — local Windows series', '',
        'Status at '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'.', '',
        'This file supplies a section of the combined report. Generating it does not send messages or publish anything.', '',
        '## Status of the 18 requested models', '',
        '| Requested model | Status | Recorded rows | Measured responses | Outputs without errors | Top-1 / measured responses |',
        '|---|---|---:|---:|---:|---:|']
    checks={r['model']:r for r in audited['checks']}
    for job in jobs:
        alias=job['alias']
        rows=by_model.get(alias,[])
        measured=sum(r['measured_inference_rows'] for r in rows)
        hit=sum(r['strict_top1_hits'] for r in rows)
        state='pending'
        if job.get('blocked'):
            state='gated access: user acceptance required'
        elif alias in checks:
            state='complete export' if checks[alias]['complete'] else 'in progress'
            errors=collections.Counter()
            for r in rows:
                errors.update(r['errors'])
            operational={k:v for k,v in errors.items() if k not in ['OK','INVALID_OUTPUT','EMPTY_OUTPUT','CONTEXT_LIMIT_ESTIMATED']}
            if operational:
                state+='; '+', '.join(f'{k}={v}' for k,v in operational.items())
        lines.append(f"| {job['requested']} | {state} | {sum(r['recorded_rows'] for r in rows)} | {measured} | {sum(r['valid_nonerror_rows'] for r in rows)} | {hit}/{measured} ({percent(hit/measured if measured else None)}) |")
    lines += ['', 'A 140-row export records all outcomes; it does not mean 140 calls succeeded. Hardware failures, blocked access and context exclusions are not observed zero-accuracy results.', '',
        '## Results by library size', '',
        '| Model | N | Measured responses | Top-1 exact | Top-10 exact | Top-1 precision, non-error outputs | Errors/limits |',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in audited['summary']:
        counts=', '.join(f'{k}: {v}' for k,v in r['errors'].items()) or 'pending'
        lines.append(f"| {r['model']} | {r['library_size']} | {r['measured_inference_rows']} | {percent(r['top1_rate_measured'])} | {percent(r['top10_rate_measured'])} | {percent(r['top1_precision_valid'])} | {counts} |")
    lines += ['', 'Measured rates use only rows with actual input and output token usage. Precision conditional on non-error outputs uses a different denominator. Invalid or duplicate ranks remain in place; removing them never promotes a correct answer.', '',
        '## Protocol and interpretation-changing limitations', '',
        '- Supplied inputs: 6,006 skills, 5,801 distinct names and 20 exact questions. Seven sizes: 50, 100, 250, 500, 1,000, 2,000 and 5,000. No deduplication.',
        '- Each size restarts `random.Random(42).sample(all_skills, N)`. Every question keeps that library, replacing the last item only when its gold ID is absent. Prompts use only the name and first 300 description characters, without full skill bodies.',
        '- **Verified position bias:** the 20 supplied gold IDs are already the first 20 seed-42 samples. They occupy positions 1–20 at every size; no final replacement occurs. The runner does not add a gold prefix. Examined local and Luna/Sol/Terra plans are identical. This series does not test uniformly distributed answer positions. A corrected variant needs a separate version and execution.',
        '- The originally requested and verified server profile uses context 32,768, one slot, transformer/output layers and f16 KV on CUDA, fit=off, no context shift, warmup, RAM cache or checkpoints, and batch 512 / microbatch 128. Later hardware auditing clarifies that default CPU_Mapped input/embedding weights remain; see the final hardware-scope note. Transformer-layer CPU fallback is not allowed.',
        '- Actual tokenization uses `/apply-template` then `/tokenize`, with IDs passed to `/completion`, temperature 0, seed 42 and a 500-token output cap; truncation is rejected. At this context, the estimated v1 gate `44.5*N+200 > 90% of context` excludes sizes 1,000/2,000/5,000, or 60 rows per model (1,080 planned rows across all 18 models). The estimate is not an actual token count.',
        '- Default model thinking is retained. Reasoning can consume the 500-token cap before the expected list; invalid outputs and outputs reaching the cap remain visible in the evidence.',
        '- Available artifacts use different quantizations. Results compare those specific variants, not parameter counts alone.',
        '- The hosted OAuth variant imposes neither temperature 0 nor a 500-token output limit. Identical candidates therefore do not make it strictly equivalent to local inference. The original reference runner was not supplied; exact reproduction of that execution is not claimed.',
        '- Twenty questions are a small, partly lexical sample; duplicate names prevent exact identification of some IDs in a name-only response format.',
        '- Timings describe this Windows environment with 16 GB RAM. Some downloads, installations and preparatory CPU conversions ran alongside model calls; these are not isolated speed measurements. Local service charges are USD 0, excluding electricity.', '',
        '## Artifacts and identities', '',
        '| Model | Quantization / provenance | Verified publisher source |',
        '|---|---|---|']
    for j in jobs:
        source=j['source']
        # Resolve the known official identity; installed quantizations remain labelled as such.
        if j['requested'].startswith('Qwen3.5-'):
            source='Qwen/'+j['requested']
        elif j['requested']=='Ling-3.0-tiny':
            source='inclusionAI/Ling-3.0-tiny-GGUF'
        label=j.get('quantization','not downloaded: gated access')
        if j['requested'].startswith('Qwen3.5-'):
            label+='; existing Ollama weights, not a publisher-provided Qwen GGUF'
        lines.append(f"| {j['requested']} | {label} | [{source}](https://huggingface.co/{source}) |")
    lines += ['', 'Source hashes, commits, sizes and licenses are recorded in the source inventories and weight provenance files. Installed Ling bytes exactly match the official file hash. Phi, Mistral v0.3 and Nanbeige were converted/quantized from pinned official sources. H2O uses its official stock ordinary-chat export, without requesting its typed-decision API. Multimodal models are tested as text only, without a projector.', '',
        '## Validation and incidents', '',
        'The 27 synthetic engine tests passed before inference. `review_local_results.py` recalculates metrics from ranks and checks CSV width, unique keys, per-size cardinality, token consistency, SQLite integrity, CSV/SQLite agreement and candidate/prompt plans. Its scope and timestamp are recorded in `evidence/result-review.json`.', '',
        'Qwen3-1.7B incident: 66 measured responses; CUDA OOM during prefill of request 67 (`500:6`), followed by 13 `GPU_SERVER_NOT_RUNNING` rows and 60 estimated-context exclusions. The original CSV is retained and no call was replayed. These 14 hardware/runtime errors are excluded from measured-output rates; original incident and CUDA logs remain in the private source archive.', '',
        'Historical launch logs, CUDA proofs, SQLite, CSV, manifests and hashes were retained. No OAuth token was copied into this local series. Benchmark execution itself sent no email or push. Weights, original source data, environments and downloads are excluded from the public repository; integration and report delivery are separate operations.', '']
    (ROOT/'LOCAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Wrote LOCAL_REPORT.md')

if __name__=='__main__':
    main()
