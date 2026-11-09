"""Replay every complete development reranking panel with vectors and saved scores."""
from common import STUDY, evaluate, math, mix_query_vectors, np, read, records_exist, retrieve, rows


def fusion(dense, reranker, weight):
    n = len(dense)
    order_dense = sorted(range(n), key=lambda i: -dense[i])
    order_reranker = sorted(range(n), key=lambda i: -reranker[i])
    dr = {idx: rank for rank, idx in enumerate(order_dense)}
    rr = {idx: rank for rank, idx in enumerate(order_reranker)}
    return sorted(range(n), key=lambda i: -((1-weight)*(1-dr[i]/max(1,n-1)) + weight*(1-rr[i]/max(1,n-1))))


def replay(archive, manifest, lock):
    documents = archive['train_documents']
    document_ids = archive['train_documents_ids'].tolist()
    anchors = archive['train_queries']
    anchor_row = {qid: i for i, qid in enumerate(archive['train_queries_ids'].tolist())}
    gold = {}
    for qid, sid in zip(archive['train_qrel_query_ids'].tolist(), archive['train_qrel_skill_ids'].tolist(), strict=True):
        gold.setdefault(qid, set()).add(sid)
    cache, query_ids, ordered_ids = {}, {}, {}
    for part in ('selection','confirmation'):
        meta = manifest['development_query_panels'][part]
        key = meta['array_key']
        ids = archive[key+'_ids'].tolist()
        query_ids[part] = set(ids)
        ordered_ids[part] = ids
        base = np.array([anchors[anchor_row[qid]] for qid in ids])
        combined = mix_query_vectors(base, archive[key], lock['query']['mixture_weight'])
        for name, vectors in [('frozen', base), ('combined', combined)]:
            cache[part, name] = {qid: retrieve(v, documents, document_ids) for qid, v in zip(ids, vectors, strict=True)}

    directory = STUDY/'results/reranker'
    output = {}
    for summary in sorted(directory.rglob('*.summary.json')):
        relative = summary.relative_to(directory)
        source = summary.with_name(summary.name.replace('.summary.json', '.jsonl'))
        if 'interrupted' in relative.parts or not records_exist(source):
            continue
        records = rows(source)
        ids = {r['query_id'] for r in records}
        reload_fixture = relative.parts[0] == 'reload-selection'
        if reload_fixture:
            part = 'selection'
            assert len(records) == len(ids) == 64 and [r['query_id'] for r in records] == ordered_ids[part][:64]
        else:
            assert len(records) == len(ids) == 512
            parts = [part for part, members in query_ids.items() if ids == members]
            assert len(parts) == 1, relative
            part = parts[0]
        query_mode = 'combined' if any('combined' in component for component in relative.parts) else 'frozen'
        for r in records:
            candidates, scores = cache[part, query_mode][r['query_id']]
            assert r['status'] == 'ok' and r['candidates'] == candidates
            assert len(r['dense_scores']) == len(scores) == 50
            assert max(abs(x-y) for x,y in zip(scores,r['dense_scores'],strict=True)) < 1e-6
        expected = read(summary)
        relevant = {r['query_id']: gold[r['query_id']] for r in records}
        for config, saved in expected.items():
            pool, weight = config.split('-')
            k, w = int(pool[1:]), float(weight[1:])
            rankings = {}
            hits = []
            for r in records:
                qid = r['query_id'];candidates = r['candidates']
                assert len(r['reranker_scores']) >= k
                rankings[qid] = [candidates[i] for i in fusion(r['dense_scores'][:k],r['reranker_scores'][:k],w)]+candidates[k:]
                hits.append((len(relevant[qid] & set(candidates[:k])),len(relevant[qid])))
            actual = evaluate(relevant, rankings)
            reference = saved['evaluation']
            for qid in relevant:
                assert all(abs(v-reference['per_query'][qid][m])<1e-12 for m,v in actual['per_query'][qid].items())
            assert all(abs(v-reference['aggregate'][m])<1e-12 for m,v in actual['aggregate'].items())
            ceiling = math.fsum(sum(1/math.log2(i+2) for i in range(min(10,found)))/sum(1/math.log2(i+2) for i in range(min(10,total))) for found,total in hits)/len(hits)
            assert abs(saved['candidate_recall']-math.fsum(found/total for found,total in hits)/len(hits))<1e-12
            assert abs(saved['ideal_candidate_ndcg10']-ceiling)<1e-12
        output[relative.as_posix()] = {'part': part, 'queries': len(ids), 'reload_fixture': reload_fixture, 'dense_query_vectors': query_mode, 'configurations': len(expected), 'all_candidates_and_metrics_match': True}

    mandatory = {'released1024/selection.summary.json','released-fullcoverage/selection.summary.json','released-confirmation/panel.summary.json','trained-confirmation/panel.summary.json','reload-selection/panel.summary.json','combined-query/selection.summary.json'}
    if any(r['tokens']==2048 for r in read(STUDY/'results/reranker_head_parity.json')['budget_admission']):
        mandatory.add('released2048/selection.summary.json')
    if lock['primary_condition']=='trained-combined':
        mandatory.update({'released-combined-confirmation/panel.summary.json','trained-combined-confirmation/panel.summary.json'})
    mandatory.add(f'{lock["reranker"]["trial"]}/selection-{lock["reranker"]["best_step"]:05d}.summary.json')
    assert mandatory <= output.keys(), mandatory-output.keys()
    return output
