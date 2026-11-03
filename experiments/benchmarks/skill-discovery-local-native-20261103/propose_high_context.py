"""Write a proposal after all accessible models are resolved. Starts no runtime."""
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    state = json.loads((ROOT/'results/local-native-v2/suite-status.json').read_text(encoding='utf-8-sig'))
    if state['state'] != 'completed':
        raise SystemExit('Proposal is finalized only after completion of the existing suite.')
    jobs = json.loads((ROOT/'local-models.json').read_text())
    contexts = {m['model']: m for m in json.loads((ROOT/'evidence/context-window-audit.json').read_text())['models']}
    allocations = {m['alias']: m for m in json.loads((ROOT/'evidence/native-buffer-audit.json').read_text())['models']}
    statuses = {m['alias']: m for m in state['models']}
    lines = ['# Proposal for measurements at library sizes 1,000 and above', '',
        'Prepared at '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'. **Proposal only: this document initiates no calls, larger-context loads or new downloads. Execution requires a separate decision.**', '',
        'The existing 32K series remains unchanged. Its 60 exclusions per model come from the estimated gate, without tokenizing N=1,000/2,000/5,000 inputs. They establish neither the native model limit nor the maximum usable VRAM context.', '',
        '## Proposed steps', '',
        '1. In a new directory/version, prepare the same 20 prompts at each size with unchanged candidates, seed and descriptions. Use the actual template and tokenizer of the same runtime to count each input without generation. Tokenization can use an admitted 32K load: tokenizing a string does not require a KV cache of the same length. For models already rejected at 32K loading, consider vocabulary-only tokenization without executing layers; do not force an oversized GPU load.',
        '2. Choose the smallest context within the official window that covers actual inputs plus 500 output tokens. In this separate version, the approximate 44.5*N+200 filter is replaced by the exact reserve gate. A size might fit within 32K or 40,960 after counting; do not assume 64K is required.',
        '3. Load one model at a time. Preserve weight quantization, f16 KV, transformer/output layers on CUDA, default CPU_Mapped input weights, temperature 0, seed 42, a 500-token cap, one slot and fit=off. Reject declared CUDA-buffer totals above 8,192 MiB before generation and leave room for the driver and transient allocations. Also inspect observed dedicated-memory usage and WDDM messages; successful cudaMalloc does not prove residency.',
        '4. Start with N=1,000, then N=2,000 if native and hardware limits allow. N=5,000 is a separate phase after preflight. Save each response and intent transactionally; a hardware error stops the model, and an uncertain request is never replayed automatically.', '',
        'Indicative context tiers 65,536 / 131,072 / 262,144 correspond to historical estimates 44.700 / 89.200 / 222.700, not actual token counts. LFM3 is restricted here to its 128,000 text configuration; proposed tiers are adjusted to that window.', '',
        '## Model-by-model review', '',
        '| Model | Published window / text config / GGUF | Declared CUDA at 32K (MiB) | f16 projection at 64K / 128K / 256K (MiB) | Initial proposal |',
        '|---|---|---:|---|---|']
    for job in jobs:
        ctx = contexts[job['requested']]
        a = allocations.get(job['alias'], {})
        status = statuses[job['alias']]
        published = ctx.get('published_statement') or {}
        limits = ctx.get('released_config_limits', {})
        native = published.get('native') or limits.get('text_config.max_position_embeddings') or limits.get('max_position_embeddings')
        if native is None:
            native = next(iter(ctx.get('gguf_context_metadata', {}).values()), None)
        window = str(native) if native else 'not inspected: gated access'
        if job['requested'] in ('Qwen3-0.6B', 'Qwen3-1.7B'):
            window = '32,768 advertised; config/GGUF 40,960'
        if published.get('extension'):
            window += '; extension '+str(published['extension'])+' separate'
        buffers = a.get('cuda_buffer_records', [])
        declared = a.get('declared_cuda_buffers_mib')
        kv = sum(b['mib'] for b in buffers if b['kind'].upper() == 'KV')
        projection = []
        if declared is not None and kv:
            for target in (65536, 131072, 262144):
                if native and target > native:
                    projection.append('outside window')
                else:
                    estimate = declared + kv * (target / 32768 - 1)
                    projection.append(f'{estimate:.0f}'+(' >8 GB' if estimate > 8192 else ''))
        else:
            projection = ['unavailable']
        if job.get('blocked'):
            recommendation = 'Wait for official acceptance and authenticated access; no substitute.'
        elif a.get('physical_capacity_exceeded'):
            recommendation = 'The 32K f16 profile is already too large; no larger context with this profile.'
        elif status['state'] != 'completed':
            recommendation = 'Resolve the runtime/hardware block before any larger-context measurement.'
        elif job['requested'] in ('Qwen3-0.6B', 'Qwen3-1.7B'):
            recommendation = 'Tokenize N=1,000; check 32K/40,960. A larger-context extension is separate, not implicit.'
        elif job['requested'] == 'Mistral 7B Instruct':
            recommendation = 'Tokenize N=1,000 against 32K; no implicit native-window extension.'
        else:
            viable=[]
            for target, label in ((65536, '1 000'), (min(131072, native or 131072), '2 000')):
                if native and target <= native and declared is not None and kv and declared + kv*(target/32768-1) < 8192:
                    viable.append(label)
            recommendation = 'Preflight N='+', then '.join(viable)+'; confirm actual allocation before calls.' if viable else 'Tokenizer preflight at 32K; the f16 projection selects no larger tier.'
        lines.append(f"| {job['requested']} | {window} | {declared if declared is not None else '-'} | {' / '.join(projection)} | {recommendation} |")
    lines += ['',
        'Projection = observed 32K CUDA allocation + observed KV × (future context / 32,768 − 1). This is a planning approximation, not a measurement, admission guarantee or maximum VRAM estimate. Sliding-window caches, recurrent state, compute graphs and transient allocations can scale differently. No larger context was tested in this series.', '',
        '## Profiles requiring separate consideration', '',
        '- KV-cache quantization, a smaller Qwen3-8B/Mistral context, different weight quantization, disabling thinking or raising the generation cap changes the profile. Each requires a new directory and explicitly separate comparison.',
        '- YaRN/RoPE extensions, including Ling beyond 131,072, require the official recipe and separate validation. A large published window does not guarantee that it fits in 8 GB.',
        '- Keeping the sampler preserves the existing comparison but keeps all golds at positions 1–20. Testing distributed positions requires a separate approved protocol, which must not replace or be mixed into this series.', '',
        'Inspectable sources: `evidence/context-window-audit.json`, `evidence/native-buffer-audit.json`, `evidence/model-artifact-provenance.json` and pinned public-source records under `evidence/sources/`. Text limits take precedence over secondary vision/audio fields.', '']
    (ROOT/'HIGH_CONTEXT_PROPOSAL.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote proposal; no model action performed.')


if __name__ == '__main__':
    main()
