"""Freeze a family split, cache dense BGE-M3 vectors, and convert natural top-15 lists."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from skillret_decider.core import (read_jsonl, write_jsonl, write_json, index_unique, load_qrels,
    skill_text, split_queries, select_ids, make_example, upstream_row, digest, sha256_file)

def dense_rank(queries: np.ndarray, skills: np.ndarray, ids: list[str], top_k: int = 100) -> list[list[str]]:
    """Stable tie-break follows sorted skill IDs; no gold labels enter retrieval."""
    if queries.ndim != 2 or skills.ndim != 2 or queries.shape[1] != skills.shape[1]:
        raise ValueError('Bad embedding shapes')
    if not np.isfinite(queries).all() or not np.isfinite(skills).all(): raise ValueError('Nonfinite vectors')
    if ids != sorted(set(ids)) or len(ids) != skills.shape[0]: raise ValueError('Unstable/duplicate skill index')
    result = []
    for start in range(0, len(queries), 64):
        scores = queries[start:start+64] @ skills.T
        order = np.argsort(-scores, axis=1, kind='stable')[:, :min(top_k, len(ids))]
        result += [[ids[i] for i in row] for row in order]
    return result

def cached_encode(model, texts, ids, kind, cache: Path, provenance: dict, batch_size):
    fingerprint = digest({'kind':kind, 'ids':ids, 'texts':texts, **provenance})
    folder = cache / fingerprint
    if folder.exists():
        m = json.loads((folder/'manifest.json').read_text())
        if m['vectors_sha256'] != sha256_file(folder/'vectors.npy') or m['fingerprint'] != fingerprint:
            raise ValueError('Embedding-cache hash mismatch')
        return np.load(folder/'vectors.npy', allow_pickle=False)
    t = time.monotonic()
    token_lengths = [len(model.tokenizer.encode(text, add_special_tokens=True, truncation=False)) for text in texts]
    vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    if len(vectors) != len(ids) or not np.isfinite(vectors).all(): raise ValueError('Invalid embedding result')
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1, atol=1e-3): raise ValueError('Expected unit-normalized embeddings')
    folder.mkdir(parents=True)
    np.save(folder/'vectors.npy', vectors, allow_pickle=False)
    write_json(folder/'ids.json', ids)
    write_json(folder/'manifest.json', {'fingerprint':fingerprint, **provenance, 'kind':kind,
        'count':len(ids), 'shape':list(vectors.shape), 'seconds':time.monotonic()-t,
        'vectors_sha256':sha256_file(folder/'vectors.npy'),
        'inputs_over_token_budget':sum(n>provenance['max_tokens'] for n in token_lengths),
        'longest_untruncated_input_tokens':max(token_lengths)})
    return vectors

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--cache', default='workspace/embedding-cache', type=Path)
    p.add_argument('--prior-split', type=Path)
    p.add_argument('--train-raw', type=Path, help='Required for test: verify catalog disjointness and dataset revision')
    p.add_argument('--device', default='cuda', choices=['cuda','cpu'])
    p.add_argument('--allow-final-test', action='store_true')
    args = p.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    config = json.loads(args.config.read_text()); raw = json.loads((args.raw/'manifest.json').read_text())
    for part, item in raw['files'].items():
        if sha256_file(args.raw/f'{part}.jsonl') != item['sha256']: raise ValueError('Raw file hash mismatch')
    skills = index_unique(read_jsonl(args.raw/'skills.jsonl'))
    queries = index_unique(read_jsonl(args.raw/'queries.jsonl'))
    gold = load_qrels(read_jsonl(args.raw/'qrels.jsonl'),set(queries),set(skills))
    is_test = raw['split'] == 'test'
    split = None
    if is_test:
        if not args.allow_final_test or args.train_raw is None:
            p.error('Test conversion requires --allow-final-test and --train-raw')
        train_manifest = json.loads((args.train_raw/'manifest.json').read_text())
        if train_manifest['revision'] != raw['revision']: raise ValueError('Train/test revision mismatch')
        train_skills = index_unique(read_jsonl(args.train_raw/'skills.jsonl'))
        if set(train_skills) & set(skills): raise ValueError('Official train/test skills overlap')
        groups = {'test': sorted(queries)}
    else:
        if args.train_raw is not None: p.error('--train-raw applies only to test')
        prior = json.loads(args.prior_split.read_text()) if args.prior_split else None
        split = split_queries(list(queries.values()),config['seed'],config['validation_fraction'],prior)
        groups = {'train':select_ids(split['train_query_ids'],config['train_limit'],config['seed']),
                  'validation':select_ids(split['validation_query_ids'],config['validation_limit'],config['seed'])}
    qids = sorted(q for ids in groups.values() for q in ids)
    if len(qids) != len(set(qids)): raise ValueError('Overlapping query groups')
    from sentence_transformers import SentenceTransformer
    retriever = config['retriever']
    model = SentenceTransformer(retriever['id'], revision=retriever['revision'], device=args.device)
    model.float()
    model.max_seq_length = retriever['max_tokens']
    provenance = {**retriever, 'representation':'name: description', 'normalize_embeddings':True,
                  'implementation':'SentenceTransformer.encode dense only; no query instruction',
                  'precision':'FP32', 'device':args.device}
    sids = sorted(skills)
    sv = cached_encode(model, [skill_text(skills[s]) for s in sids],sids,'skills',args.cache,provenance,retriever['batch_size'])
    qv = cached_encode(model,[queries[q]['query'] for q in qids],qids,'queries',args.cache,provenance,retriever['batch_size'])
    ranking = dict(zip(qids,dense_rank(qv,sv,sids,config['mining_top_k'])))
    # Do not keep the retriever resident during LoRA training (separate process is even safer).
    del model, sv, qv
    args.out.mkdir(parents=True)
    if split: write_json(args.out/'split.json',split)
    write_jsonl(args.out/'retrieved_top100.jsonl', ({'query_id':q,'skill_ids':ranking[q]} for q in qids))
    for group, ids in groups.items():
        labelled = [make_example(queries[q],ranking[q],skills,gold[q],config['seed'],config['candidates'],split=group) for q in ids]
        # Inference files omit all qrel-derived target fields. Test labels go only to the scorer.
        inference = [{k:v for k,v in e.items() if k not in ('gold_probs','expected')} for e in labelled]
        write_jsonl(args.out/f'{group}.jsonl', labelled if group=='train' else inference)
        if group=='train':
            write_jsonl(args.out/'train.upstream.jsonl',(upstream_row(e) for e in labelled))
        else:
            write_json(args.out/f'{group}_labels.json',{q:gold[q] for q in ids})
        if group=='train' and config.get('augmented_training_view',False):
            augmented = [make_example(queries[q],ranking[q],skills,gold[q],config['seed'],config['candidates'],True,'train') for q in ids]
            write_jsonl(args.out/'train_augmented_only_not_primary.jsonl',augmented)
    files = {p.name:sha256_file(p) for p in args.out.iterdir() if p.is_file()}
    write_json(args.out/'manifest.json', {'dataset':raw,'config':config, 'config_sha256':sha256_file(args.config),
        'retriever':provenance,'groups':{k:len(v) for k,v in groups.items()},'is_test':is_test,
        'primary_candidates':'natural top-15; no gold injection','files':files})
    print(json.dumps({'groups':{k:len(v) for k,v in groups.items()}, 'out':str(args.out)},indent=2))

if __name__ == '__main__': main()
