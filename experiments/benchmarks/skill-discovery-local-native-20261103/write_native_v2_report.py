"""Regenerate the local Markdown handoff from independently checked native-v2 results."""
import collections
import datetime
import json
from pathlib import Path
from review_local_results import review, ROOT

def pct(value):
    return '-' if value is None else f'{value:.2%}'

def main():
    audit=review('local-native-v2')
    jobs=json.loads((ROOT/'local-models.json').read_text())
    state=json.loads((ROOT/'results/local-native-v2/suite-status.json').read_text(encoding='utf-8-sig'))
    statuses={x['alias']:x for x in state['models']}
    grouped=collections.defaultdict(list)
    for r in audit['summary']:
        grouped[r['model']].append(r)
    lines=['# Skill discovery — local results with the corrected native adapter', '',
        'Status verified at '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'. Suite: `'+state['state']+'`.', '',
        'The comparative series is `results/local-native-v2/`. The earlier `local-v1/` series was preserved after four models as a diagnostic: its adapter could count internal reasoning as ranked answers. Its scores are not the final comparative results. See `evidence/NATIVE_PARSER_V2.md`.', '',
        '## Model coverage', '',
        '| Requested model | Status | Saved rows | Measured responses | Strict top-1 / measured responses | Top-10 / measured responses | Outputs without errors |',
        '|---|---|---:|---:|---:|---:|---:|']
    scorecard=[]
    for j in jobs:
        rows=grouped.get(j['alias'],[])
        n=sum(r['measured_inference_rows'] for r in rows)
        hits=sum(r['strict_top1_hits'] for r in rows)
        hits10=sum(r['strict_top10_hits'] for r in rows)
        s=statuses.get(j['alias'],{'state':'en attente'})
        label=s['state']+(': '+s['reason'] if s.get('reason') else '')
        score=f'{hits}/{n} ({pct(hits/n)})' if n else '-'
        score10=f'{hits10:g}/{n} ({pct(hits10/n)})' if n else '-'
        lines.append(f"| {j['requested']} | {label} | {sum(r['recorded_rows'] for r in rows)} | {n} | {score} | {score10} | {sum(r['valid_nonerror_rows'] for r in rows)} |")
        errors=collections.Counter()
        for r in rows:
            errors.update(r['errors'])
        scorecard.append({'requested':j['requested'],'alias':j['alias'],'state':label,
            'recorded_rows':sum(r['recorded_rows'] for r in rows),'measured_responses':n,
            'strict_top1_hits':hits,'strict_top10_hits':hits10,
            'top1_rate_measured':hits/n if n else None,'top10_rate_measured':hits10/n if n else None,
            'valid_nonerror_rows':sum(r['valid_nonerror_rows'] for r in rows),'issues':dict(errors)})
    lines[4:4]=[f"Of the {len(jobs)} requested models, {sum(m['measured_responses']>0 for m in scorecard)} provided measured responses. The suite contains {sum(m['recorded_rows'] for m in scorecard)} rows with 45 columns and {sum(m['measured_responses'] for m in scorecard)} responses with actual usage. Blocks and exclusions remain separate from measurements.", '']
    lines += ['', 'A measured response has actual input and output token usage. OOM, gated access, context exclusions and unattempted cases have no measured accuracy; they are not observed zero-accuracy results. A complete 140-row export does not imply 140 successful calls.', '',
        '## Details by library size', '',
        '| Model | N | Measured responses | Strict top-1 | Top-10 recall | Top-1 precision, non-error outputs | Outputs reaching the 500-token cap | Actual input tokens, min–max | Median LLM call (s) | Recorded issues |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in audit['summary']:
        issues=', '.join(f'{k}={v}' for k,v in r['errors'].items()) or 'en attente'
        token_range=f"{r['input_token_min']}-{r['input_token_max']}" if r['input_token_min'] is not None else '-'
        llm_median=f"{r['median_llm_s']:.2f}" if r['median_llm_s'] is not None else '-'
        lines.append(f"| {r['model']} | {r['library_size']} | {r['measured_inference_rows']} | {pct(r['top1_rate_measured'])} | {pct(r['top10_rate_measured'])} | {pct(r['top1_precision_valid'])} | {r['outputs_at_500']} | {token_range} | {llm_median} | {issues} |")
    lines += ['', 'Rates over measured outputs include missing final lists and formatting failures. Precision conditional on non-error outputs uses another denominator. Errors and invalid final-answer ranks are never removed to promote a correct answer.', '',
        '## Shared conditions and limitations', '',
        '- 6,006 skills, 5,801 distinct names and the 20 exact questions; sizes 50, 100, 250, 500, 1,000, 2,000 and 5,000; seed 42; top 10. No full bodies: names and descriptions capped at 300 characters.',
        '- `random.Random(42).sample(all_skills,N)` is restarted at each size. Each question gets a copy, with last-item replacement if and only if its gold ID is missing. **Verified intrinsic bias: the 20 supplied gold IDs are already the first 20 seed-42 draws; their positions remain 1–20 at every size.** The runner adds no gold prefix. This is not a test of uniformly distributed answer positions.',
        '- All 140 v1/native-v2 candidate plans and prompt hashes are identical. The corrected adapter only adds parsing state for reasoning already prefilled by the template; prompts and thinking mode are unchanged. Delimited internal blocks are excluded from ranks. An unfinished reasoning block at the cap produces `REASONING_NO_FINAL_ANSWER` while retaining actual usage.',
        '- RTX 2060 SUPER, 8,192 MiB; driver 610.88; llama.cpp b11960 CUDA 12.4. Transformer/output layers, f16 KV and observed recurrent state are on CUDA. Effective context 32,768, one slot, fit=off, no context shift, warmup, RAM cache or checkpoints; batch 512 / microbatch 128. **Exact scope: llama.cpp retains default CPU_Mapped input/embedding weights**, even with all transformer/output layers on GPU. This profile is not CPU-free computation. Transformer-layer fallback to fit VRAM is rejected. The buffer audit records these allocations. Preparatory conversion and quantization also use CPU.',
        '- `/apply-template` then `/tokenize` provide actual counts; IDs are sent to `/completion`. Temperature 0, seed 42, 500-token output cap, no truncation. The estimated v1 gate `44.5*N+200 > 90% of context` excludes 60 rows at sizes 1,000/2,000/5,000; those are not measured input-token counts.',
        '- **The 60 rows reflect the chosen n_ctx filter, not intrinsic model limits.** Qwen3.5 advertises 262,144 native tokens (an extension to 1,010,000); Ling configures 131,072 native tokens and a YaRN recipe for 262.144. This series loads only 32,768, a uniform conservative choice without per-model maximum-VRAM search. No tokenizer request or inference measures excluded sizes. Published windows, publisher configs, GGUF values and observed n_ctx are separated in the context-window audit and notes.',
        '- Reasoning models can spend all 500 tokens before the final list. Budget/format failures describe this configuration, not their best possible ability. Quantization differs and is identified per model.',
        '- Multimodal models are tested with text only. H2O uses its official ordinary-chat export without a typed-decision shim. Instruct/IT versions are used when available; Mistral is explicitly v0.3.',
        '- The 20 questions form a small, partly lexical sample; duplicate names prevent recovery of some gold IDs through name-only responses. The original reference cloud runner was not supplied.',
        '- OAuth Luna/Sol/Terra uses the same candidates, but its route guarantees neither temperature 0 nor a 500-token output cap. It remains a separate variant; strict cross-route equivalence is not claimed.',
        '- Latencies describe this Windows environment with 16 GB RAM. Downloads and CPU conversions could run concurrently, so this is not an isolated speed benchmark. Local service charges are USD 0, excluding electricity.', '',
        '## Weights and provenance', '', '| Model | Quantization | Verified source |', '|---|---|---|']
    for j in jobs:
        src=j['source']
        if j['requested'].startswith('Qwen3.5-'):
            src='Qwen/'+j['requested']
        elif j['requested']=='Ling-3.0-tiny':
            src='inclusionAI/Ling-3.0-tiny-GGUF'
        quant=j.get('quantization','not downloaded: gated access')
        if j['requested'].startswith('Qwen3.5-'):
            quant+='; installed Ollama artifact, digest confirmed against the official registry'
        lines.append(f"| {j['requested']} | {quant} | [{src}](https://huggingface.co/{src}) |")
    lines += ['', 'Commits, hashes, sizes and licenses are recorded in source inventories, verified Ollama manifests and weight provenance files. Installed Ling matches the official GGUF. Llama 3.2 3B and Gemma 3 4B require manual gated-access acceptance; no bypass or ungated substitute was used.', '',
        '## Evidence, validation and delivery', '',
        'All 34 synthetic tests pass, including seven native-parser regressions. `evidence/result-review-native-v2.json` checks the 45 columns, unique keys, cardinalities, actual usage, SQLite integrity and CSV/SQLite equality, and independently recalculates rank metrics. SQLite is the transactional authority; CSV is replaced atomically after each row.', '',
        'Each model retains configuration, manifest and source-linked launch/GPU evidence. Initial failures are preserved. Raw outputs are held in ignored private response captures, without full prompts; no OAuth credential was copied into this local series. Public exports remove process identifiers and private paths while retaining audit hashes.', '',
        'The benchmark/report generators do not send email or push code. Weights, archives, tools, environments, private full input files and raw captures are excluded. Public integration and the consolidated Markdown report are handled separately.', '']
    lines += [
        '## Physical-VRAM admission', '',
        'Under Windows WDDM, CUDA allocation can succeed despite a declared budget exceeding physical VRAM. The launcher therefore checks the sum of model, KV, recurrent-state and maximum compute CUDA buffers before inference. Excess produces `HARDWARE_VRAM_CAPACITY_EXCEEDED`, distinct from an observed CUDA OOM. It does not silently offload layers or change context.', '',
        'Qwen3-8B at 32K declares 9,097.34 MiB against 8,192 MiB physical VRAM. The 21 responses acquired before this control remain in `results/diagnostic-vram-oversubscription/`, with the next unresolved intent unreplayed. They are excluded from the comparative series; its canonical ledger records hardware outcomes without measured accuracy. Corrected guard validation rejected the same load without inference. Exact launch hashes and allocations are audited in the native-buffer evidence.', '',
        'Mistral-7B-Instruct-v0.3 is likewise rejected before generation: 8,230.52 MiB declared against 8,192 MiB physical VRAM. It is not assigned an observed zero; measured accuracy is absent. Converted originals and startup allocations are preserved.', '',
        'Starting with Ornith, historical WDDM telemetry and its summary record verified-process dedicated/shared counters. Nonzero shared usage can include host allocations or driver staging; alone it proves neither transformer offload nor paging of a specific CUDA buffer. This sampled collection cannot reconstruct residency for earlier models.', '',
        'Consolidated public provenance, including local conversions and installed Ollama artifacts, is in `evidence/model-artifact-provenance.json`. It remains available when ignored weight directories are not transferred.', '']
    (ROOT/'LOCAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    progress={'utc':audit['reviewed_utc'],'sampler_audit':'CONFORME; supplied gold/seed42 position bias documented',
              'native_v2_state':state,'native_v2_checks':audit['checks'],'native_v2_summary':audit['summary'],
              'v1_state':'preserved diagnostic after four models; native reasoning contamination demonstrated',
              'raw_response_audit':'evidence/offline-reasoning-audit.json',
              'report':'LOCAL_REPORT.md'}
    (ROOT/'evidence/progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
    (ROOT/'evidence/model-scorecard.json').write_text(json.dumps({'utc':audit['reviewed_utc'],'models':scorecard},indent=2),encoding='utf-8')
    print('Wrote native-v2 LOCAL_REPORT.md')

if __name__=='__main__':
    main()
