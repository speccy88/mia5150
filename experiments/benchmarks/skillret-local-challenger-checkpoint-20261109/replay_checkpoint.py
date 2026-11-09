"""Replay the completed checkpoint from published vectors and candidate scores."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from common import STUDY, evaluate, iter_rows, mix_query_vectors, np, read, retrieve, sha
from replay_development_reranker import replay as replay_reranker_panels
from score_reuse import verify_confirmation_records
from snapshot import replay_development


def replay_official(archive, lock):
    documents = archive['test_documents']
    document_ids = archive['test_documents_ids'].tolist()
    base = archive['test_base_queries']
    query_ids = archive['test_base_queries_ids'].tolist()
    assert documents.shape == (6006, 1024) and base.shape == (4392, 1024)
    gold = {qid: set() for qid in query_ids}
    for qid, sid in zip(archive['test_qrel_query_ids'], archive['test_qrel_skill_ids'], strict=True):
        gold[qid].add(sid)
    assert len(gold) == 4392 and all(gold.values()) and sum(map(len, gold.values())) == 7187

    cases = [('reconstructed-base', base)]
    for suffix, array_name, config in (
        ('', 'test_adapted_queries', lock['query']),
        ('-seed43', 'test_adapted_queries_seed43', lock['secondary_query']),
    ):
        adapted = archive[array_name]
        assert adapted.shape == base.shape and archive[array_name + '_ids'].tolist() == query_ids
        cases.append((f'adapted-query{suffix}', adapted))
        cases.append((f'mixture-query{suffix}', mix_query_vectors(base, adapted, config['mixture_weight'])))

    results = {}
    for name, vectors in cases:
        path = STUDY / 'results' / 'final' / f'{name}.jsonl'
        rankings = {}
        max_score_error = 0.0
        for qid, vector, row in zip(query_ids, vectors, iter_rows(path), strict=True):
            rank, scores = retrieve(vector, documents, document_ids)
            assert row['query_id'] == qid and row['status'] == 'ok' and row['ranking'] == rank
            error = max(abs(a - b) for a, b in zip(scores, row['scores'], strict=True))
            max_score_error = max(max_score_error, error)
            rankings[qid] = rank
        assert max_score_error < 1e-6
        metrics = evaluate(gold, rankings)
        saved = read(path.with_suffix('.metrics.json'))
        assert metrics == saved, name
        results[name] = {'rankings_identical': len(rankings), 'metrics_identical': True, 'max_score_error': max_score_error}
    return results


def main():
    archive_path = STUDY / 'vectors' / 'dense-vectors-v1.npz'
    lock_path = STUDY / 'FINAL_EVAL_LOCK.json'
    manifest = read(archive_path.with_name('manifest.json'))
    assert sha(archive_path) == manifest['archive_sha256'] and sha(lock_path) == manifest['lock_sha256']
    lock = read(lock_path)
    with np.load(archive_path, allow_pickle=False) as archive:
        official = replay_official(archive, lock)
        development = replay_development(archive, manifest['development_query_panels'])
        rerankers = replay_reranker_panels(archive, manifest, lock)
    reuse = verify_confirmation_records(STUDY)
    assert reuse == read(STUDY / 'results' / 'CONFIRMATION.json')['offline_score_reuse_verified']
    print(json.dumps({
        'status': 'passed',
        'official_query_conditions': official,
        'development_query_panels': development,
        'development_reranker_panels': rerankers,
        'confirmation_score_reuse': reuse,
    }, indent=2))


if __name__ == '__main__':
    main()
