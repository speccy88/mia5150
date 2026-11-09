"""Audit the complete training miner output without loading a model."""
from common import CACHE, STUDY, data, digest, np, read, record_path, rows, save, sha
import unicodedata


def main():
    membership = read(STUDY / 'manifests/split_membership.json')
    source = record_path(STUDY / 'manifests/reranker-negatives.jsonl')
    raw = rows(source)
    records = {r['query_id']: r for r in raw}
    order = membership['train_query_ids']
    assert len(raw) == len(records) == len(order) == 46431 and list(records) == order
    train_ids = {sid for sid, part in membership['skill_partition'].items() if part == 'train'}
    assert len(train_ids) == 8687
    skills, queries, gold = data('train')
    fingerprints = {
        x['id']: digest(' '.join(unicodedata.normalize('NFKC', x['skill_md']).casefold().split()))
        for x in skills
    }
    del skills, queries
    hard, moderate, inserted = [], [], 0
    for qid, r in records.items():
        positive = gold[qid]
        high, mid, rnd = r['hard_ids'], r['moderate_ids'], r['random_id']
        assert positive == set(r['positive_ids']) and positive <= train_ids
        negative = set(high + mid + [rnd])
        assert len(set(high + mid)) == len(high) + len(mid)
        assert negative <= train_ids and not negative & positive
        positive_bodies = {fingerprints[sid] for sid in positive}
        assert positive_bodies.isdisjoint(fingerprints[sid] for sid in negative)
        injected = set(r['training_gold_inserted_if_missing'])
        assert injected <= positive
        inserted += len(injected)
        hard.append(len(high))
        moderate.append(len(mid))
    assert set(hard) == {20} and 10 <= min(moderate) <= max(moderate) <= 30

    def vectors(name):
        folder = CACHE / 'vectors' / name
        path = folder / 'vectors.npy'
        assert read(folder / 'complete.json')['array_sha256'] == sha(path)
        return np.load(path, mmap_mode='r', allow_pickle=False), read(folder / 'contract.json')['ids']

    documents, document_ids = vectors('train-documents')
    queries, query_ids = vectors('train-queries')
    query_row = {q: i for i, q in enumerate(query_ids)}
    eligible = [i for i, sid in enumerate(document_ids) if sid in train_ids]
    exact_corpus = np.array(documents[eligible])
    eligible_ids = [document_ids[i] for i in eligible]
    position = np.arange(len(eligible))
    sample = sorted(order, key=lambda q: digest('independent-negative-audit:' + q))[:128]
    for qid in sample:
        scores = exact_corpus @ queries[query_row[qid]]
        top = [eligible_ids[i] for i in np.lexsort((position, -scores))[:50]]
        positive = gold[qid]
        positive_bodies = {fingerprints[sid] for sid in positive}
        filtered = [sid for sid in top if sid not in positive and fingerprints[sid] not in positive_bodies]
        assert records[qid]['hard_ids'] == filtered[:20] and records[qid]['moderate_ids'] == filtered[20:50]
        assert set(records[qid]['training_gold_inserted_if_missing']) == positive - set(top)
    save(STUDY / 'results/reranker/mining_verification.json', {
        'status': 'passed', 'source_sha256': sha(source), 'all_training_queries_checked': len(records),
        'train_only_corpus_size': len(train_ids), 'independent_top50_sample_size': len(sample),
        'independent_top50_sample_query_ids': sample, 'heldout_or_positive_equivalent_negatives': 0,
        'hard_count_per_query': 20, 'moderate_count_min': min(moderate), 'moderate_count_max': max(moderate),
        'training_positive_insertions': inserted, 'model_loaded': False, 'official_test_accessed': False,
    })
    print('Verified all mining memberships and 128 independent exact top-50 cases', flush=True)


if __name__ == '__main__':
    main()
