"""Independent checks and report for isolated, small-N context/KV profiles."""
import collections
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics

import bench_profiles_v3 as b

ROOT=Path(__file__).resolve().parent


def pct(h,n):
    return f'{h}/{n} ({h/n:.2%})' if n else '-'


def mib(value):
    return '-' if value is None else f'{value:.2f}'


def main():
    state=json.loads((ROOT/'evidence/profile-status.json').read_text(encoding='utf-8-sig'))
    telemetry=collections.defaultdict(list)
    stream_path=ROOT/'evidence/wddm-telemetry.jsonl'
    if stream_path.exists():
        for line in stream_path.read_text(encoding='utf-8').splitlines():
            sample=json.loads(line)
            telemetry[sample['model']].append(sample)
    summary=[]
    stages=[]
    raw_count=0
    for entry in state['stages']:
        name=entry['model']+'-'+entry['profile']
        record=dict(entry)
        launch=ROOT/entry['launch'] if entry.get('launch') else None
        proof_path=launch/'gpu-proof.json' if launch else None
        if proof_path and proof_path.exists():
            proof=json.loads(proof_path.read_text(encoding='utf-8-sig'))
            raw=(launch/'server.stderr.log').read_bytes()[:proof['log_bytes']]
            assert hashlib.sha256(raw).hexdigest()==proof['log_sha256']
            assert proof['all_layers_gpu'] and proof['kv_gpu']
            assert proof['kv_type']==entry['kv'] and proof['context_window']==entry['context']
            assert proof['declared_cuda_buffers_mib']<=8192 and proof['wddm_committed_mib']<=8192
            record.update(cuda_mib=proof['declared_cuda_buffers_mib'],wddm_mib=proof['wddm_committed_mib'],
                kv_mib=sum(x['mib'] for x in proof['cuda_buffer_records'] if x['kind']=='KV'),
                sealed_log_hash_valid=True,artifact_sha256=proof['model_sha256'],
                cpu_mapped_lines=[x for x in raw.decode('utf-8',errors='replace').splitlines() if 'CPU_Mapped' in x and 'buffer size' in x])
            observations=list(telemetry.get(name,[]))
            for sample_file in launch.glob('wddm-*.json'):
                observations.append(json.loads(sample_file.read_text(encoding='utf-8-sig')))
            committed=[]
            for sample in observations:
                observed_mib=sum(v['bytes']/1048576 for v in sample['samples'] if v['counter']=='total committed' and v['status']==0)
                assert 0<observed_mib<=8192, (name,'Sampled WDDM committed capacity exceeded')
                committed.append(observed_mib)
            record.update(wddm_sample_count=len(observations),sampled_wddm_peak_mib=max(committed,default=None))
        folder=ROOT/'results'/name
        if (folder/'details.csv').exists():
            with (folder/'details.csv').open(encoding='utf-8-sig',newline='') as stream:
                reader=csv.DictReader(stream); columns=reader.fieldnames; rows=list(reader)
            assert len(columns)==45 and len(set(columns))==45
            manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8-sig'))
            assert manifest['protocol']==b.LOCAL_PROTOCOL
            assert manifest['implementation_sha256']['bench_profiles_v3.py']==b.file_hash(ROOT/'bench_profiles_v3.py')
            planned=20*len(manifest['config']['library_sizes'])
            if entry['state']=='completed' or entry['state']=='blocked': assert len(rows)==planned
            db=sqlite3.connect((folder/'state.sqlite3').as_uri()+'?mode=ro',uri=True)
            assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
            saved=[(k,json.loads(r),json.loads(a)) for k,r,a in db.execute('SELECT key,row_json,audit FROM trials WHERE row_json IS NOT NULL ORDER BY ordinal')]
            # CSV is replaced after the SQLite commit; a running writer can be one row ahead.
            assert len(saved)>=len(rows)
            if entry['state'] in ('completed','blocked'): assert len(saved)==len(rows)
            preflight=json.loads((ROOT/'evidence'/(entry['model']+'-token-preflight.json')).read_text(encoding='utf-8'))
            expected_tokens={t['key']:t['input_tokens'] for t in preflight['trials']}
            for (key,stored,audit),row in zip(saved,rows):
                assert all(str(stored[k])==row[k] for k in columns)
                rank=int(row['rank_of_gold'])
                assert 0<=rank<=10
                for k in (1,3,5,10):
                    hit=0<rank<=k
                    metrics={'precision':1/k if hit else 0,'recall':float(hit),'ndcg':1/math.log2(rank+1) if hit else 0,'map':1/rank if hit else 0,'mrr':1/rank if hit else 0}
                    for metric,value in metrics.items(): assert math.isclose(float(row[f'{metric}@{k}']),value,abs_tol=1e-12)
                if row['input_tokens']:
                    assert int(row['input_tokens'])==expected_tokens[key]
                    assert int(row['total_tokens'])==int(row['input_tokens'])+int(row['output_tokens'])
                    assert int(row['input_tokens']) + 500<=entry['context'] and int(row['output_tokens'])<=500
                    path=Path(audit['raw_response_path'])
                    raw_path=ROOT.joinpath(*path.parts[path.parts.index('private-responses'):])
                    capture=json.loads(raw_path.read_text(encoding='utf-8-sig'))
                    assert b.digest(capture['text'].encode())==audit['response_sha256']
                    final,error=b.final_answer(capture['text'],capture['reasoning_prefix'])
                    assert b.digest(final.encode())==audit['final_answer_sha256'] and error==audit['reasoning_error']
                    assert all(capture['usage'][x]==stored[x] for x in ('input_tokens','output_tokens'))
                    raw_count+=1
            pending=[x[0] for x in db.execute('SELECT key FROM trials WHERE row_json IS NULL')]
            db.close()
            if entry['state']=='completed': assert not pending
            measured=[x for x in rows if x['input_tokens']]
            summary.append({'model':entry['model'],'profile':entry['profile'],'sizes':entry['sizes'],
                'context':entry['context'],'kv':entry['kv'],'rows':len(rows),'measured':len(measured),
                'top1':sum(x['exact_match']=='True' for x in measured),'top10':sum(float(x['recall@10']) for x in measured),
                'valid':sum(not x['error'] for x in measured),'issues':dict(collections.Counter(x['error'] or 'OK' for x in rows)),
                'median_llm_s':statistics.median(float(x['llm_latency_s']) for x in measured) if measured else None,
                'input_min':min((int(x['input_tokens']) for x in measured),default=None),
                'input_max':max((int(x['input_tokens']) for x in measured),default=None),
                'unresolved_intents':pending,'checks_passed':True})
        stages.append(record)
    paired=[]
    for quant in [s for s in summary if s['kv']=='q8_0' and s['measured']]:
        if quant['measured']!=20: continue
        baseline=next((x for x in summary if x['model']==quant['model'] and x['sizes']==quant['sizes'] and x['kv']=='f16' and x['measured']),None)
        if not baseline: continue
        if baseline['measured']!=20: continue
        assert baseline['context']==quant['context']
        def records(profile):
            folder=ROOT/'results'/(profile['model']+'-'+profile['profile'])
            manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8-sig'))
            db=sqlite3.connect((folder/'state.sqlite3').as_uri()+'?mode=ro',uri=True)
            rows={key:(json.loads(r),json.loads(a)) for key,r,a in db.execute('SELECT key,row_json,audit FROM trials WHERE row_json IS NOT NULL')}
            db.close()
            return manifest['plan'],rows
        fp,fr=records(baseline); qp,qr=records(quant)
        assert fp==qp
        common=sorted(fr.keys()&qr.keys())
        paired.append({'model':quant['model'],'size':quant['sizes'][0],'context':quant['context'],'paired_queries':len(common),
            'f16_top1':baseline['top1'],'q8_top1':quant['top1'],'f16_top10':baseline['top10'],'q8_top10':quant['top10'],
            'identical_raw_texts':sum(fr[k][1].get('response_sha256')==qr[k][1].get('response_sha256') for k in common),
            'f16_hit_q8_miss':sum(fr[k][0]['exact_match'] and not qr[k][0]['exact_match'] for k in common),
            'q8_hit_f16_miss':sum(qr[k][0]['exact_match'] and not fr[k][0]['exact_match'] for k in common),
            'same_candidate_prompt_plans':True})
    output={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'suite_state':state['state'],
        'summary':summary,'stages':stages,'paired_quality':paired,'raw_responses_verified':raw_count,
        'columns':45,'generation_cap':200,'measured_total':sum(x['measured'] for x in summary)}
    assert output['measured_total']<=200
    b.atomic_write(ROOT/'evidence/profile-review.json',json.dumps(output,indent=2))
    lines=['# Separate local context and KV profiles — results','',
        f"State: {state['state']}; checked {output['utc']}. {output['measured_total']} measured responses, maximum 200.",'',
        'Initial 32K/f16 loads exceeded the 8,192 MiB capacity: 9,097.34 MiB of declared CUDA buffers for Qwen3-8B and 8,230.52 for Mistral v0.3. This supplement measures smaller f16 contexts and a separate Q8_0 cache condition while keeping the same weights.','',
        'The historical 32K/f16 series and archives are preserved. These measurements use a different context protocol: actual tokenizer preflight plus 500 tokens, rounded up to a multiple of 256. The 20 questions, candidates, prompts, seed 42, temperature 0, ranking rules and 500-token cap remain identical. Results are not merged with the native-v2 series.','',
        'N is the presented library size. Validated inputs contain 6,006 skills and 5,801 distinct names; the 20 original questions are reused. Prompts contain only names and descriptions capped at 300 characters, without full bodies. Sampling and scoring remain unchanged, including duplicate names and gold positions 1–20.','',
        'Explicit deviation: the original estimated gate `44.5×N+200 > 90% of n_ctx` is not applied in v3. The control becomes `actual input tokens + 500 ≤ n_ctx`, without an additional 10% margin. Contexts can be close to this exact reserve; each input is retokenized before generation and truncation is rejected. This supplement is therefore not an unchanged execution of the initial protocol.','',
        'Hardware: RTX 2060 SUPER, 8,192 MiB. Runtime: llama.cpp b11960 CUDA 12.4, commit `8642541b255bfb5dc39cd351a178be09013a55d6`. Q4_K_M weights are unchanged from the earlier series: [official Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B-GGUF) and a pinned local conversion of [official Mistral v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3). Weight, executable and sealed log-prefix hashes are linked to each launch. Q8_0 here refers to the KV cache, not a different weight quantization.','',
        '## Measurements by profile','',
        '| Model | Profile | N | n_ctx | KV | Responses | Top-1 | Top-10 | Without errors | Median LLM (s) | Issues |',
        '|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|']
    for s in summary:
        latency=f"{s['median_llm_s']:.2f}" if s['median_llm_s'] is not None else '-'
        issues=', '.join(f'{k}={v}' for k,v in s['issues'].items())
        lines.append(f"| {s['model']} | {s['profile']} | {','.join(map(str,s['sizes']))} | {s['context']} | {s['kv']} | {s['measured']} | {pct(s['top1'],s['measured'])} | {pct(s['top10'],s['measured'])} | {s['valid']} | {latency} | {issues} |")
    lines+=['','## f16 totals across four sizes','',
        '| Model | Responses | Top-1 | Top-10 | Outputs without errors |','|---|---:|---:|---:|---:|']
    for model in sorted({s['model'] for s in summary}):
        group=[s for s in summary if s['model']==model and s['kv']=='f16']
        count=sum(s['measured'] for s in group)
        lines.append(f"| {model} | {count} | {pct(sum(s['top1'] for s in group),count)} | {pct(sum(s['top10'] for s in group),count)} | {sum(s['valid'] for s in group)} |")
    lines+=['','These totals combine the four adjusted f16 contexts; Q8_0 results remain in their separate paired comparison.','',
        '## Hardware admission','',
        '| Model | Profile | n_ctx | KV | Status | Declared CUDA MiB | Admission WDDM MiB | Maximum sampled WDDM MiB |',
        '|---|---|---:|---|---|---:|---:|---:|']
    for s in stages:
        lines.append(f"| {s['model']} | {s['profile']} | {s['context']} | {s['kv']} | {s['state']} | {mib(s.get('cuda_mib'))} | {mib(s.get('wddm_mib'))} | {mib(s.get('sampled_wddm_peak_mib'))} |")
    lines+=['','Admission-only loads and preflights perform no generation. K and V must have the requested types in the logs, Flash Attention must be enabled, and transformer/output layers and KV must remain on CUDA. Declared CUDA buffers and WDDM committed memory are checked before inference. Default CPU_Mapped input/embedding weights are explicitly retained; no transformer-layer fallback is allowed. These counters are distinct from an exhaustive residency trace.','',
        'Latencies include prefill and generation, with variable output lengths capped at 500 tokens. Weight loading and hashing are excluded from LLM time. They describe this Windows session with telemetry, without complete machine isolation.','',
        '## Paired quality comparison','']
    if paired:
        lines+=['| Model | N | Shared n_ctx | Paired questions | Top-1 f16 / Q8 | Top-10 f16 / Q8 | Identical raw outputs |','|---|---:|---:|---:|---|---|---:|']
        for p in paired: lines.append(f"| {p['model']} | {p['size']} | {p['context']} | {p['paired_queries']} | {p['f16_top1']} / {p['q8_top1']} | {p['f16_top10']:g} / {p['q8_top10']:g} | {p['identical_raw_texts']} |")
    else: lines+=['Comparaison encore en attente.']
    lines+=['','Q8_0 uses rounded quantization: 32 int8 values and an f16 scale, or 8.5 bits/value versus 16 for f16. The [initial llama.cpp experiments](https://github.com/ggml-org/llama.cpp/pull/7412) motivate this one quantized format here without guaranteeing equivalence. The same 20 questions at the same context permit a local comparison; this small set does not replace a general perplexity evaluation.','',
        'Outputs with no final answer at the 500-token cap are generation/format failures for this profile. Hardware rejections have no measured accuracy. The seed-42 gold-position bias at positions 1–20 remains. No N≥1000 context, new weights, other weight quantization or hosted inference is tested.','',
        'Evidence: `evidence/profile-review.json`, `evidence/runtime-kv-support.json`, `evidence/candidate-comparison.json`, preflights and historical runtime records; transactional SQLite and 45-column CSV files under `results/`. Raw outputs were hash-checked and remain private. The historical baseline-preservation records verify the earlier archives.','',
        'Llama 3.2 and Gemma 3 remain unexecuted at user-requested closure; see [CLOSURE.md](CLOSURE.md). This report generator performs no connection, terms acceptance, publication or email delivery.','']
    (ROOT/'PROFILE_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'state':state['state'],'measured':output['measured_total'],'verified_raw':raw_count,'profiles':len(summary),'checks_passed':True}),flush=True)


if __name__=='__main__': main()

