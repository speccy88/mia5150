"""Bounded, sequential context/KV comparison. Never replays an acquired trial."""
import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import bench_profiles_v3 as b

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent/'local-benchmark'
PYTHON = Path(sys.executable)
SERVER = BASE/'tools/llama-b11960-cuda12.4/llama-server.exe'
PORT = 28085
STATE = ROOT/'evidence/profile-status.json'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write(path, value):
    b.atomic_write(path, json.dumps(value, indent=2))


def stop(owned):
    subprocess.run(['pwsh', '-NoProfile', '-File', str(ROOT/'stop_owned.ps1'), str(owned['pid']),
        owned['executable_path'], owned['process_start']], cwd=ROOT, check=True)


def start(job, label, context, kv, sizes):
    alias = job['alias']+'-'+label
    config = {'backend': 'llama', 'model': alias, 'context_window': context, 'kv_type': kv,
        'base_url': f'http://127.0.0.1:{PORT}', 'quantization': job['quantization'],
        'gpu_proof': str(ROOT/'runtime'/alias/'gpu-proof.json'), 'library_sizes': sizes,
        'raw_response_dir': str(ROOT/'private-responses'/alias), 'timeout_s': 1200,
        'require_token_usage': True, 'source': job}
    config_path = ROOT/'configs'/(alias+'.json')
    assert not config_path.exists(), 'No silent restart of an existing profile.'
    write(config_path, config)
    foreign = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True)
    assert not any(x.strip().isdigit() for x in foreign.stdout.splitlines()), 'FOREIGN_GPU_WORKLOAD_PRESENT'
    runtime = ROOT/'runtime'/alias
    with (ROOT/'evidence'/(alias+'-launcher.log')).open('w', encoding='utf-8') as log:
        result = subprocess.run(['pwsh', '-NoProfile', '-File', str(ROOT/'Start-ProfileServer.ps1'),
            '-ServerExe', str(SERVER), '-ModelFile', job['path'], '-Config', str(config_path)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    launches = list(runtime.glob('launch-*'))
    assert len(launches) == 1
    launch = launches[0]
    owned = json.loads((launch/'owned-process.json').read_text(encoding='utf-8-sig'))
    if result.returncode:
        stop(owned)
        admission = launch/'admission-failure.json'
        if not admission.exists():
            raise RuntimeError('Inspect non-capacity startup failure; no retry: '+str(launch))
        verdict = json.loads(admission.read_text(encoding='utf-8-sig'))
        config.update(startup_error=verdict['kind'], admission_evidence=verdict)
        write(config_path, config)
        return config, launch, None
    return config, launch, owned


def sample(config, destination):
    result = subprocess.run(['pwsh', '-NoProfile', '-File', str(ROOT/'sample_wddm_memory.ps1'),
        '-ProofPath', config['gpu_proof']], cwd=ROOT, capture_output=True, text=True, check=True)
    write(destination, json.loads(result.stdout))


def main():
    assert not STATE.exists(), 'Existing profile state must be inspected; no automatic replay.'
    jobs = [j.copy() for j in json.loads((BASE/'local-models.json').read_text())
        if j['alias'] in ('qwen3-8b-q4km', 'mistral-7b-instruct-v03-q4km')]
    provenance = {x['alias']: x for x in json.loads((BASE/'evidence/model-artifact-provenance.json').read_text())['artifacts']}
    for job in jobs:
        path = Path(job['path'])
        if not path.is_absolute():
            path = BASE/path
        job['path'] = str(path)
        expected = provenance[job['alias']]['verified_manifest']['sha256']
        assert b.file_hash(path) == expected
        job['sha256'] = expected
    for name, expected in b.SOURCE_HASHES.items():
        assert b.file_hash(ROOT/'data'/name) == expected
    skills, queries = b.load_data(ROOT/'data/test.jsonl', ROOT/'data/test_queries.json')
    all_trials = b.make_plan(skills, queries)
    reference = json.loads((BASE/'results/local-native-v2/qwen3-0.6b-q8/manifest.json').read_text())
    reference_plan = {t['key']: t for t in reference['plan']}
    for t in all_trials:
        saved = reference_plan[t['key']]
        assert t['prompt_sha256'] == saved['prompt_sha256']
        assert [s['id'] for s in t['candidates']] == saved['candidate_ids']
    write(ROOT/'evidence/candidate-comparison.json', {'trials_checked':len(all_trials),
        'candidate_ids_and_prompt_hashes_identical_to_v2':True, 'new_inference_calls':0})
    package = json.loads((ROOT.parent/'deliverables/package-manifest-20261103T095448Z.json').read_text())
    preserved = [{'path':a['path'],'expected_sha256':a['sha256'],'observed_sha256':b.file_hash(Path(a['path']))} for a in package['artifacts']]
    assert all(a['expected_sha256']==a['observed_sha256'] for a in preserved)
    write(ROOT/'evidence/baseline-preservation-before.json', preserved)
    state = {'state': 'running', 'started_utc': now(), 'max_generation_calls': 200,
        'stages': [], 'original_archives_untouched': True}
    write(STATE, state)

    def stage(job, label, context, kv, sizes, generate=False):
        entry = {'model': job['alias'], 'profile': label, 'context': context, 'kv': kv,
            'sizes': sizes, 'generation_planned': generate, 'state': 'loading', 'started_utc': now()}
        state['stages'].append(entry)
        write(STATE, state)
        cfg, launch, owned = start(job, label, context, kv, sizes)
        entry['launch'] = str(launch.relative_to(ROOT))
        entry['config'] = str((ROOT/'configs'/(cfg['model']+'.json')).relative_to(ROOT))
        if not owned:
            entry.update(state='blocked', reason=cfg['startup_error'], finished_utc=now())
            if generate:
                b.run(cfg, skills, queries, ROOT/'results'/cfg['model'])
            write(STATE, state)
            return cfg, None, entry
        entry['state'] = 'benchmarking' if generate else 'admitted_no_generation'
        write(STATE, state)
        if generate:
            try:
                sample(cfg, launch/'wddm-before-generation.json')
                with (ROOT/'evidence'/(cfg['model']+'-benchmark.log')).open('w', encoding='utf-8') as log:
                    run = subprocess.run([str(PYTHON), str(ROOT/'bench_profiles_v3.py'), 'run',
                        '--config', str(ROOT/entry['config']), '--out', str(ROOT/'results'/cfg['model'])],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if run.returncode:
                    entry['state'] = 'unresolved'; write(STATE, state)
                    raise RuntimeError('Benchmark failed; inspect acquired/pending state before any action.')
                sample(cfg, launch/'wddm-after-generation.json')
                entry.update(state='completed', finished_utc=now())
            finally:
                stop(owned)
            write(STATE, state)
            print(json.dumps(entry), flush=True)
            return cfg, None, entry
        return cfg, owned, entry

    for job in jobs:
        # Real-token preflight only: no /completion endpoint called here.
        cfg, owned, entry = stage(job, 'preflight-f16-4096', 4096, 'f16', [50,100,250,500])
        assert owned
        counts = []
        contexts = {}
        try:
            b.check_gpu_proof(cfg)
            for trial in all_trials:
                templated = b.request(cfg['base_url']+'/apply-template', {'messages':trial['messages']}, timeout=30)['prompt']
                tokens = b.request(cfg['base_url']+'/tokenize', {'content':templated,'add_special':True,'parse_special':True}, timeout=30)['tokens']
                counts.append({'key':trial['key'], 'size':trial['n'], 'input_tokens':len(tokens),
                    'prompt_sha256':trial['prompt_sha256'], 'templated_prompt_sha256':b.digest(templated.encode()),
                    'token_ids_sha256':b.digest(b.canonical(tokens).encode()), 'reasoning_prefix':b.reasoning_prefix(templated)})
                write(ROOT/'evidence'/(job['alias']+'-token-preflight.json'), {'generation_calls':0,'trials':counts})
            for n in b.SIZES:
                maximum = max(x['input_tokens'] for x in counts if x['size']==n)
                contexts[n] = math.ceil((maximum+500)/256)*256
            entry.update(state='completed_no_generation', finished_utc=now(), adjusted_contexts=contexts)
            write(STATE,state)
        finally:
            stop(owned)
        # Prove a 32K quantized-cache loading option; do not repeat all sizes with it.
        cfg, owned, entry = stage(job, 'q8-32768-admission-only', 32768, 'q8_0', [50,100,250,500])
        if owned:
            sample(cfg, ROOT/entry['launch']/'wddm-admission-only.json')
            stop(owned)
            entry.update(state='completed_no_generation', finished_utc=now())
            write(STATE,state)
        successful = []
        blocked = []
        for n in b.SIZES:
            cfg, _, entry = stage(job, 'f16-adjusted-n'+str(n), contexts[n], 'f16', [n], True)
            (successful if entry['state']=='completed' else blocked).append(n)
        # Exactly one paired-quality size, plus only sizes not measurable in f16.
        quant_sizes = sorted(set(blocked + ([max(successful)] if successful else [])))
        for n in quant_sizes:
            stage(job, 'q8-adjusted-n'+str(n), contexts[n], 'q8_0', [n], True)
    state.update(state='completed', finished_utc=now())
    after = [{'path':a['path'],'expected_sha256':a['expected_sha256'],'observed_sha256':b.file_hash(Path(a['path']))} for a in preserved]
    assert all(a['expected_sha256']==a['observed_sha256'] for a in after)
    write(ROOT/'evidence/baseline-preservation-after.json', after)
    write(STATE,state)


if __name__ == '__main__':
    main()
