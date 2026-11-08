"""Run dense scaling A/B/C first, then official A/B/C, with verified caches."""
import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import psutil
from common import (STUDY, QWEN, REPO, CONFIGS, SIZES, atomic_json, digest, file_hash,
                    full_text, jsonl, panel, evaluate, rank_skills)
from engine import Encoder, Monitor, public, utc

R = STUDY / 'results'


def csv_write(path, rows):
    if rows:
        with path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(rows)


def main(cache):
    started = time.monotonic(); monitor = Monitor('loading')
    try:
        protocol = json.loads((STUDY / 'protocol.json').read_text())
        for path,expected in protocol['shared_helper_sha256'].items():
            assert file_hash(REPO/path)==expected
        for name, expected in protocol['data_sha256'].items():
            assert file_hash(cache / 'data' / (name + '.jsonl')) == expected
        assert file_hash(R / 'candidate_pools.json') == protocol['candidate_pools_sha256']
        for config in CONFIGS:
            assert file_hash(R / ('documents-' + config + '.jsonl')) == protocol['document_manifest_sha256'][config]
        sources = {str(p.relative_to(STUDY)): file_hash(p) for p in sorted((STUDY / 'src').glob('*.py'))}
        atomic_json(R / ('execution-source-' + digest(sources)[:16] + '.json'), sources)
        atomic_json(R / 'hardware.json', {'os': platform.freedesktop_os_release()['PRETTY_NAME'],
            'cpu': 'Intel Core i3-3225, 2 cores / 4 threads', 'python': platform.python_version(),
            'gpu': subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total,compute_cap','--format=csv,noheader'], text=True).strip(),
            'system_ram_bytes': psutil.virtual_memory().total, 'swap_bytes': psutil.swap_memory().total,
            'execution': 'Directly on the existing Ubuntu GPU server; no cloud or Windows GPU stack',
            'gpu_preflight': 'No compute process; 9 MiB display allocation', 'torch_cpu_threads': 2,
            'openblas_threads': 1, 'telemetry_interval_s': 1})
        encoder = Encoder(cache, monitor, context=protocol['context_including_special_tokens'])
        skills = jsonl(cache / 'data/skills.jsonl'); questions = panel()
        queries = jsonl(cache / 'data/queries.jsonl'); raw_qrels = jsonl(cache / 'data/qrels.jsonl')
        gold_full = {}
        for row in raw_qrels:
            gold_full.setdefault(row['query_id'], {})[row['skill_id']] = row['relevance']
        assert len(skills) == 6006 and len(queries) == len(gold_full) == 4392 and len(raw_qrels) == 7187
        skill_index = {s['id']: i for i, s in enumerate(skills)}
        pools = json.loads((R / 'candidate_pools.json').read_text())
        original_docs = jsonl(QWEN / 'results/documents.jsonl')
        ties = np.asarray([d['source_sha256'] for d in original_docs])
        matrices, ownership = {}, {}
        for config in CONFIGS:
            encoder.torch.cuda.empty_cache()
            folder = R / config; folder.mkdir(exist_ok=True)
            docs = jsonl(R / ('documents-' + config + '.jsonl'))
            vectors, owners, completed = [], [], set()
            for size in SIZES:
                monitor.phase = config + ':indexing'
                index_started = time.perf_counter()
                counts = {'uncached_calls': 0, 'cached_calls': 0, 'uncached_tokens': 0,
                          'encoding_s_this_stage': 0.0, 'logical_encoding_s_from_measured_receipts': 0.0}
                additional = [sid for sid in pools[str(size)]['ids'] if sid not in completed]
                for sid in additional:
                    i = skill_index[sid]; skill = skills[i]; doc = docs[i]
                    view = skill['name'] + ' | ' + skill['description'] if config == 'A_metadata' else full_text(skill)
                    assert digest(view) == doc['view_sha256']
                    for chunk in doc['chunks']:
                        part = view[chunk['start']:chunk['end']]
                        assert digest(part) == chunk['text_sha256']
                        vector, receipt = encoder.encode(part)
                        assert receipt['content_tokens'] == chunk['tokens']
                        vectors.append(vector); owners.append(i)
                        counts['logical_encoding_s_from_measured_receipts'] += receipt['encoding_s']
                        if receipt['cache_hit']:
                            counts['cached_calls'] += 1
                        else:
                            counts['uncached_calls'] += 1; counts['uncached_tokens'] += receipt['input_tokens']
                            counts['encoding_s_this_stage'] += receipt['encoding_s']
                    completed.add(sid)
                    if len(completed) % 100 == 0:
                        print(config, 'indexed', len(completed), '/6006', flush=True)
                        atomic_json(R / 'progress.json', {'stage': 'indexing', 'config': config, 'skills': len(completed), 'utc': utc()})
                stage_row = {'utc': utc(), 'config': config, 'library_size': size, 'additional_skills': len(additional),
                             'cumulative_skills': len(completed), 'index_wall_s': time.perf_counter() - index_started, **counts}
                with (R / 'indexing_history.jsonl').open('a') as f: f.write(json.dumps(stage_row) + '\n')
                matrix, owner = np.stack(vectors), np.asarray(owners, dtype=np.int64)
                saved = folder / f'scaling-n{size}.json'
                if saved.exists():
                    prior = json.loads(saved.read_text()); assert prior['identity'] == encoder.identity
                    print(config, 'resumed scaling', size, flush=True); continue
                monitor.phase = config + ':scaling'
                rankings, gold, rows = {}, {}, []
                for qi, question in enumerate(questions):
                    total_started = time.perf_counter()
                    vector, receipt = encoder.encode(question['query'], 'query', force=True)
                    search_started = time.perf_counter()
                    selected, scores = rank_skills(matrix, owner, vector, [skill_index[sid] for sid in pools[str(size)]['ids']], ties)
                    search_s = time.perf_counter() - search_started
                    total_s = time.perf_counter() - total_started
                    cached_started = time.perf_counter()
                    cached_selected, _ = rank_skills(matrix, owner, vector, [skill_index[sid] for sid in pools[str(size)]['ids']], ties)
                    cached_search_s = time.perf_counter() - cached_started
                    assert np.array_equal(cached_selected, selected)
                    qid = f'panel-{qi:02d}'
                    ranked = [skills[i]['id'] for i in selected]
                    rankings[qid] = ranked; gold[qid] = {question['gold_id']: 1}
                    rows.append({'model': 'BAAI/bge-m3', 'config': config, 'library_size': size, 'query_id': qid,
                        'gold_id': question['gold_id'], 'query_sha256': digest(question['query']), 'ranking': ranked,
                        'scores': scores.tolist(), 'query_embedding_uncached_s': receipt['encoding_s'],
                        'similarity_search_s': search_s, 'total_retrieval_uncached_s': total_s,
                        'cached_query_search_s': cached_search_s, 'query_cache_persistence_included': True})
                result = evaluate(gold, rankings)
                for row in rows: row.update(result['per_query'][row['query_id']])
                atomic_json(saved, {'status': 'completed', 'identity': encoder.identity, 'config': config,
                                   'library_size': size, 'evaluation': result, 'rows': rows})
                print('SCALING', config, size, result['aggregate']['Top1_ID_accuracy'], result['aggregate']['NDCG@10'], flush=True)
            assert len(completed) == 6006
            matrices[config], ownership[config] = matrix, owner
            atomic_json(folder / 'content_coverage.json', {'skills': 6006, 'chunks': len(vectors), 'coverage_percent': 100,
                'view': docs[0]['view'], 'missing_or_failed_documents': 0, 'original_tokens': sum(d['original_tokens'] for d in docs),
                'source_characters': sum(d['source_characters'] for d in docs), 'protocol_sha256': encoder.identity['protocol_sha256']})
        atomic_json(R / 'progress.json', {'stage': 'scaling_completed', 'cases': 480, 'utc': utc()})
        for config in CONFIGS:
            encoder.torch.cuda.empty_cache()
            monitor.phase = config + ':official_queries'; folder = R / config
            output = folder / 'full_rankings.jsonl'
            saved = {r['query_id']: r for r in jsonl(output)} if output.exists() else {}
            assert set(saved) <= {q['id'] for q in queries}
            rankings, rows = {}, []
            with output.open('a') as f:
                for question in queries:
                    qid = question['id']
                    if qid in saved:
                        row = saved[qid]; assert row['query_sha256'] == digest(question['query'])
                    else:
                        total_started = time.perf_counter()
                        vector, receipt = encoder.encode(question['query'], 'query', force=True)
                        search_started = time.perf_counter()
                        selected, scores = rank_skills(matrices[config], ownership[config], vector, range(6006), ties)
                        search_s = time.perf_counter() - search_started
                        total_s = time.perf_counter() - total_started
                        cached_started = time.perf_counter()
                        again, _ = rank_skills(matrices[config], ownership[config], vector, range(6006), ties)
                        cached_s = time.perf_counter() - cached_started
                        assert np.array_equal(selected, again)
                        row = {'model': 'BAAI/bge-m3', 'config': config, 'library_size': 6006, 'query_id': qid,
                            'query_sha256': digest(question['query']), 'ranking': [skills[i]['id'] for i in selected],
                            'scores': scores.tolist(), 'query_embedding_uncached_s': receipt['encoding_s'],
                            'similarity_search_s': search_s, 'total_retrieval_uncached_s': total_s,
                            'cached_query_search_s': cached_s, 'query_cache_persistence_included': True}
                        f.write(json.dumps(row) + '\n'); f.flush()
                        import os
                        os.fsync(f.fileno())
                    rankings[qid] = row['ranking']; rows.append(row)
                    if len(rows) % 100 == 0:
                        print(config, 'official queries', len(rows), '/4392', flush=True)
                        atomic_json(R / 'progress.json', {'stage': 'official_queries', 'config': config, 'queries': len(rows), 'utc': utc()})
            assert len(rankings) == 4392
            result = evaluate(gold_full, rankings)
            atomic_json(folder / 'full_evaluation.json', result)
            csv_write(folder / 'full_per_query.csv', [{**{k:v for k,v in row.items() if k not in ('ranking','scores')},
                        **result['per_query'][row['query_id']]} for row in rows])
            print('OFFICIAL_COMPLETE', config, result['aggregate']['NDCG@10'], result['aggregate']['Recall@10'], result['aggregate']['MRR@10'], flush=True)
        atomic_json(R / 'progress.json', {'stage': 'completed', 'configs': list(CONFIGS), 'skills_per_config': 6006,
            'scaling_cases': 480, 'official_cases': 13176, 'wall_seconds_this_invocation': time.monotonic()-started, 'utc': utc()})
    except Exception as error:
        atomic_json(R / 'failure.json', {'exception': type(error).__name__, 'message': public(str(error)),
            'phase': monitor.phase, 'utc': utc(), 'wall_s_this_invocation': time.monotonic()-started})
        raise
    finally:
        monitor.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--cache', type=Path, default=Path.home()/'.cache/mia-bgem3')
    main(parser.parse_args().cache)
