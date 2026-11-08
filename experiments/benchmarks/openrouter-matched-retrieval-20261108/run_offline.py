#!/usr/bin/env python3
"""Offline-only reconstruction and retrieval. No API clients or network execution.

The source corpus and encoder are supplied locally; hashes are mandatory. Saved
results are a NEW experiment, not evidence of the original hosted prompts.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, io, json, math, platform, random, re, statistics, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SNAPSHOT = '3fb7e66a1a219bc43d7c2057dca3d1c70a3c6c45'
CORPUS_SHA = '1960453e422e4ba01977197580934ca3c02b2ac302dfa9c0d074707d961b7c81'
MODEL_SHA = '828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35'
TOKENIZER_SHA = 'd241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66'
BUILDER_SHA = '6743c3e933d730bbe88073b102de47725114502795d5d66cbe2d46c72472d3c9'
QUERY_SHA = '7ef69814dd8f93f4a8dadc290567f158e78986898aa76a33bc95cf883716c28b'
REV = '6583d7d2ed07644d0fb8938ed8178f3a7dc42a12'
MODEL_REV = '5c38ec7c405ec4b44b94cc5a9bb96e735b38267a'
SIZES = [50, 100, 250, 500, 1000, 2000, 5000]
PREFIX = 'Represent this sentence for searching relevant passages: '


def canonical(x):
    return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(x):
    return hashlib.sha256(canonical(x).encode()).hexdigest()


def filehash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def require_hash(path, expected):
    actual = filehash(path)
    if actual != expected:
        raise ValueError(f'Input hash mismatch for {Path(path).name}: {actual}')


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def write_csv(path, rows):
    if not rows:
        raise ValueError('Refuse empty CSV')
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def literal_builder(skills, queries, size, seed=42):
    """Independent transcription to cross-check the source AST."""
    rng = random.Random(seed)
    names = {q['gold_skill'] for q in queries}
    positive = [s for s in skills if s['name'] in names]
    negative = [s for s in skills if s['name'] not in names]
    if size < len(positive):
        raise ValueError('Too many preserved gold names')
    library = positive + rng.sample(negative, min(size-len(positive), len(negative)))
    rng.shuffle(library)
    return library


def source_builder(source, skills, queries, size, seed=42):
    """Compile ONLY the inspected deterministic library block, never API code.

    Fixed SHA of the complete source is recorded in results. The AST node bounds
    are checked and compared with the independent literal builder for every size.
    No downloaded data is evaluated as code.
    """
    require_hash(source, BUILDER_SHA)
    module = ast.parse(Path(source).read_text())
    func = next(n for n in module.body if isinstance(n, ast.FunctionDef)
                and n.name == 'evaluate_model_openrouter')
    body = []
    started = False
    for node in func.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'rng':
            started = True
        if started:
            body.append(node)
        if (started and isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'shuffle'):
            break
    if len(body) != 9 or ast.unparse(body[-1]) != 'rng.shuffle(library)':
        raise ValueError('Original builder AST changed; re-audit before proceeding')
    # Includes an unreachable early return for impossible sizes: wrap in function.
    args = ast.arguments(posonlyargs=[], args=[ast.arg(arg=n) for n in
        ['all_skills', 'queries', 'library_size', 'seed', 'gold_names_override', 'model_info']],
        kwonlyargs=[], kw_defaults=[], defaults=[])
    fn = ast.FunctionDef(name='rebuild', args=args,
                         body=body + [ast.Return(value=ast.Name(id='library', ctx=ast.Load()))],
                         decorator_list=[])
    namespace = {'random': random}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                 '<verified-library-builder>', 'exec'), namespace)
    return namespace['rebuild'](skills, queries, size, seed, None, None)


def text_view(skill):
    # Preserve original metadata up to source OpenRouter's 1024-character limit.
    # Whitespace-normalize only; do not infer unavailable _clean_text semantics.
    return skill['name'].strip() + ' | ' + re.sub(r'\s+', ' ', skill['description'][:1024]).strip()


class BM25:
    """Frozen gap-study formula: Unicode word tokens, k1=1.5, b=.75."""
    def __init__(self, texts):
        self.tokens = [Counter(re.findall(r'\w+', x.casefold())) for x in texts]
        self.lengths = np.array([sum(x.values()) for x in self.tokens], dtype=np.float64)
        self.average = self.lengths.mean() or 1
        self.df = Counter(t for row in self.tokens for t in row)

    def scores(self, query):
        out = np.zeros(len(self.tokens))
        denominator = 1.5 * (.25 + .75 * self.lengths/self.average)
        for term, qtf in Counter(re.findall(r'\w+', query.casefold())).items():
            tf = np.array([x[term] for x in self.tokens], dtype=np.float64)
            df = self.df[term]
            out += qtf * math.log1p((len(out)-df+.5)/(df+.5)) * tf * 2.5/(tf+denominator)
        return out


def rank(scores, ids):
    return [ids[i] for i in sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))]


def rrf(a, b, depth=100):
    """Fixed equal-weight RRF60, input top100 per retriever, no query tuning."""
    scores = defaultdict(float)
    for ranking in [a[:depth], b[:depth]]:
        for i, sid in enumerate(ranking):
            scores[sid] += 1/(60+i+1)
    return sorted(scores, key=lambda sid: (-scores[sid], sid))[:100]


def single_target_metrics(ranking, acceptable_ids, k):
    """Name-equivalence is ONE target; duplicate-name IDs are not extra qrels."""
    acceptable_ids = set(acceptable_ids)
    first = next((i+1 for i,sid in enumerate(ranking[:k]) if sid in acceptable_ids), None)
    hit = float(first is not None)
    return {'recall':hit, 'hit':hit, 'precision':hit/k, 'f1':2*hit/(k+1),
            'map':1/first if first else 0., 'mrr':1/first if first else 0.,
            'ndcg':1/math.log2(first+1) if first else 0., 'completeness':hit}


class Encoder:
    def __init__(self, assets):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        require_hash(assets/'model.onnx', MODEL_SHA)
        require_hash(assets/'tokenizer.json', TOKENIZER_SHA)
        self.raw = Tokenizer.from_file(str(assets/'tokenizer.json'))
        self.tok = Tokenizer.from_file(str(assets/'tokenizer.json'))
        self.tok.enable_truncation(max_length=512)
        self.tok.enable_padding(pad_id=0, pad_token='[PAD]')
        opt = ort.SessionOptions()
        opt.intra_op_num_threads = 1
        opt.inter_op_num_threads = 1
        opt.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opt.enable_cpu_mem_arena = False
        opt.enable_mem_pattern = False
        self.session = ort.InferenceSession(str(assets/'model.onnx'), sess_options=opt,
                                            providers=['CPUExecutionProvider'])
        self.inputs = {x.name for x in self.session.get_inputs()}
        self.version = ort.__version__

    def encode(self, texts, query=False, batch=8):
        values, timing = [], []
        for i in range(0, len(texts), batch):
            start = time.perf_counter()
            chunk = [(PREFIX if query else '') + x for x in texts[i:i+batch]]
            raw = self.raw.encode_batch(chunk)
            enc = self.tok.encode_batch(chunk)
            data = {'input_ids': np.array([x.ids for x in enc], dtype=np.int64),
                    'attention_mask': np.array([x.attention_mask for x in enc], dtype=np.int64),
                    'token_type_ids': np.array([x.type_ids for x in enc], dtype=np.int64)}
            value = self.session.run(['last_hidden_state'], {k:v for k,v in data.items() if k in self.inputs})[0][:, 0]
            value = value / np.linalg.norm(value, axis=1, keepdims=True)
            if value.shape != (len(chunk), 384) or not np.isfinite(value).all():
                raise ValueError('Invalid model output')
            values.append(value)
            elapsed = time.perf_counter()-start
            for row in raw:
                timing.append({'raw_tokens':len(row.ids), 'used_tokens':min(512,len(row.ids)),
                               'truncated':len(row.ids)>512, 'batch_seconds':elapsed,
                               'batch_size':len(chunk)})
            if len(texts)>100 and i%512 == 0:
                print(f'Encoded {i}/{len(texts)}', flush=True)
        return np.concatenate(values), timing


def prepare_vectors(assets, skills, queries, cache, out):
    cache.mkdir(parents=True, exist_ok=True)
    contract = {'source_sha':CORPUS_SHA, 'document_ids':[s['id'] for s in skills],
                'texts_sha':digest([text_view(s) for s in skills]),
                'queries_sha':digest([q['query'] for q in queries]), 'model_sha':MODEL_SHA,
                'tokenizer_sha':TOKENIZER_SHA, 'prefix':PREFIX, 'max_length':512,
                'pooling':'normalized CLS FP32', 'batch':8, 'protocol_version':1}
    manifest_path = cache/'embedding-manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest['contract'] != contract:
            raise ValueError('Embedding cache input contract mismatch')
        for name, checksum in manifest['vector_files'].items():
            require_hash(cache/name, checksum)
        docs = np.load(cache/'documents.npy', allow_pickle=False)
        qs = np.load(cache/'queries.npy', allow_pickle=False)
        write_json(out/'embedding-manifest.json', {**manifest,'cache_reused_this_execution':True})
        return docs, qs
    start = time.perf_counter()
    encoder = Encoder(assets)
    startup = time.perf_counter()-start
    docs, dtime = encoder.encode([text_view(s) for s in skills])
    # Fresh per-query batch1 timings separately recorded; not amortized batch8.
    qs, qtime = encoder.encode([q['query'] for q in queries], query=True, batch=1)
    np.save(cache/'documents.npy', docs)
    np.save(cache/'queries.npy', qs)
    manifest = {'contract':contract, 'onnxruntime':encoder.version, 'numpy':np.__version__,
        'startup_seconds':startup, 'document_batches_seconds':sum(x['batch_seconds']/x['batch_size'] for x in dtime),
        'document_truncated_count':sum(x['truncated'] for x in dtime),
        'query_timings':qtime, 'vector_files':{n:filehash(cache/n) for n in ['documents.npy','queries.npy']},
        'source':'New CPU encoding; not reused historical vector cache',
        'timing_note':'Startup and document indexing separate; query timing includes tokenization and encoder, batch1, no warmup removal'}
    write_json(manifest_path, manifest)
    write_json(out/'embedding-manifest.json', {**manifest, 'cache_reused_this_execution':False})
    return docs, qs


def audit_jev(queries, root, out):
    texts = {q['query'] for q in queries}
    sources = []
    # Compare canonical request text, rather than matching gold names or query IDs.
    candidate_file = root/'experiments/benchmarks/skillret-reranker-study/reference_run/jev/candidates.json'
    historical = json.loads(candidate_file.read_text())
    matching = [x for x in historical if x['query'] in texts]
    sources.append({'path':str(candidate_file.relative_to(root)), 'sha256':filehash(candidate_file),
                    'queries':len(historical), 'exact_query_text_matches':len(matching)})
    official = root/'experiments/benchmarks/skillret-local-metadata-20261005/data/queries/test.jsonl'
    records = [json.loads(x) for x in official.open() if x.strip()]
    values = {str(x.get('text', x.get('query', ''))) for x in records}
    sources.append({'path':str(official.relative_to(root)), 'sha256':filehash(official),
                    'queries':len(records), 'exact_query_text_matches':len(texts & values)})
    all_matches = []
    exact_leaf_matches = []
    def exact_leaves(value, location='$'):
        found=[]
        if isinstance(value,str) and value in texts:
            found.append(location)
        elif isinstance(value,dict):
            for key,item in value.items():found.extend(exact_leaves(item,location+'.'+str(key)))
        elif isinstance(value,list):
            for i,item in enumerate(value):found.extend(exact_leaves(item,location+'['+str(i)+']'))
        return found
    # Inspect JSON/JSONL/CSV artifacts in documented Jev studies for exact strings.
    # No external calls, model loading, or deserialization of pickle files.
    prefixes = ['experiments/benchmarks/skillret-reranker-study',
                'experiments/benchmarks/skillret-embedding-jev-fusion',
                'experiments/pilots/skillret-local-rubrics-20261006']
    for prefix in prefixes:
        for path in sorted((root/prefix).rglob('*')):
            if not path.is_file() or path.suffix not in {'.json','.jsonl','.csv'}:
                continue
            data = path.read_text(errors='replace')
            found = [i for i,q in enumerate(queries) if q['query'] in data or
                     json.dumps(q['query'],ensure_ascii=True)[1:-1] in data]
            if found:
                leaves=[]
                if path.suffix=='.json':
                    leaves=exact_leaves(json.loads(data))
                elif path.suffix=='.jsonl':
                    for line in path.open():
                        if line.strip():leaves.extend(exact_leaves(json.loads(line)))
                elif path.suffix=='.csv':
                    for record in csv.DictReader(io.StringIO(data)):leaves.extend(exact_leaves(record))
                item={'path':str(path.relative_to(root)), 'query_indices':found,'exact_string_leaf_paths':leaves,
                      'interpretation':'candidate-content substring only; no exact query value' if not leaves else 'inspect exact leaf paths'}
                all_matches.append(item)
                if leaves:exact_leaf_matches.append(item)
    report = {'method':'Exact canonical query text required before candidate/order/schema matching',
              'sources':sources, 'broader_string_matches':all_matches,
              'exact_leaf_matches':exact_leaf_matches,
              'matching_completed_jev_cells':0 if not exact_leaf_matches and not matching else None,
              'status':'unavailable_no_exact_matched_query_evidence' if not exact_leaf_matches and not matching else 'requires_manual_candidate_schema_audit',
              'caveat':'Text scan is a bounded evidence search, not proof that unarchived responses never existed. No mismatched historical score is reused.',
              'fresh_api_calls':0, 'new_api_cost_usd':0,
              'paid_reranking':'Not attempted; requires explicit budget and verified current pricing.'}
    write_json(out/'jev-audit.json', report)


def run(args):
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    require_hash(args.corpus, CORPUS_SHA)
    source = ROOT/'benchmark_openrouter/benchmark_models_testplan_openrouter.py'
    query_path = ROOT/'benchmark_openrouter/tests_openrouter/test_queries.json'
    all_skills = [json.loads(x) for x in args.corpus.open() if x.strip()]
    skills = [{k:s[k] for k in ['id','name','description','major']} for s in all_skills]
    require_hash(query_path, QUERY_SHA)
    queries = json.loads(query_path.read_text())
    if len(skills)!=6006 or len(queries)!=20 or len({s['id'] for s in skills})!=6006:
        raise ValueError('Expected source dimensions and unique document IDs')
    by_id = {s['id']:s for s in skills}
    for q in queries:
        if by_id[q['gold_id']]['name'] != q['gold_skill']:
            raise ValueError('Gold identity mismatch')
    equivalent = {i:[s['id'] for s in skills if s['name']==q['gold_skill']] for i,q in enumerate(queries)}
    catalogs = []
    for size in SIZES:
        library = source_builder(source, skills, queries, size)
        if library != literal_builder(skills,queries,size):
            raise ValueError('Source AST versus independent builder mismatch')
        ids = [s['id'] for s in library]
        if len(ids)!=size or len(set(ids))!=size:
            raise ValueError('Catalog cardinality failure')
        catalogs.append({'size':size,'seed':42,'ordered_ids':ids,'ordered_sha256':digest(ids),
                         'gold_positions':{str(i):ids.index(q['gold_id'])+1 for i,q in enumerate(queries)}})
    write_json(out/'catalogs.json',catalogs)
    metadata = {'snapshot':SNAPSHOT,'source_dataset':{'repository':'ThakiCloud/SKILLRET','revision':REV,
        'path':'data/skills/test.jsonl','sha256':CORPUS_SHA,'bytes':args.corpus.stat().st_size},
        'query_file':str(query_path.relative_to(ROOT)),'query_sha256':filehash(query_path),
        'builder_source':str(source.relative_to(ROOT)),'builder_sha256':filehash(source),
        'python':platform.python_version(),'platform':platform.platform(),
        'script_sha256':filehash(Path(__file__)),'seed':42,'sizes':SIZES,'top_k':[5,10],
        'data_exposure':'20 synthetic single-target tasks; not official held-out query panel',
        'comparison_status':'new internally matched reconstructed-input panel; historical prompt equivalence conditional',
        'historical_identity_gaps':['benchmark_models.py imported loader/cleaner missing',
            'Original hosted run has no dataset hash or ordered catalog/prompt hash',
            'Pinned official corpus matches other preserved source receipts; does not prove historical loading semantics'],
        'document_view':'name | whitespace-normalized description[:1024]; same field/cap but unavailable historical cleaner not claimed identical',
        'tie_break':'descending score then UUID; original candidate order retained in manifests, not used as relevance tie-break',
        'domain_supplement':'nearest normalized mean document-embedding category centroid; constructed within each catalog; no gold labels or query outcomes used; not original LLM-routed domain_vector',
        'original_domain_vector_status':'not_run: exact original query router outputs and local router runtime absent',
        'hybrid':'equal RRF60 on BM25 top100 and vector top100; top100 result retained; no tuning',
        'gold_name_records_preserved':sum(s['name'] in {q['gold_skill'] for q in queries} for s in skills),
        'nonunique_gold_names':{q['gold_skill']:len(equivalent[i]) for i,q in enumerate(queries) if len(equivalent[i])>1},
        'label_contracts':['uuid_single_target','name_equivalent_single_target'],
        'name_scoring':'First occurrence of any ID with gold_skill name is one target; duplicates never add recall credit',
        'new_api_calls':0,'new_api_cost_usd':0}
    write_json(out/'provenance.json',metadata)
    docs, qvectors = prepare_vectors(args.assets,skills,queries,args.cache,out)
    positions = {s['id']:i for i,s in enumerate(skills)}
    all_rows, raw_rows, plans, prep_rows = [],[],[],[]
    for catalog in catalogs:
        ids = catalog['ordered_ids']; ix = [positions[sid] for sid in ids]
        values = docs[ix]; texts = [text_view(by_id[sid]) for sid in ids]
        start = time.perf_counter(); bm = BM25(texts); bm_prep = time.perf_counter()-start
        start = time.perf_counter()
        categories = sorted({by_id[sid]['major'] for sid in ids})
        local = {c:np.array([i for i,sid in enumerate(ids) if by_id[sid]['major']==c]) for c in categories}
        centroids = np.stack([values[local[c]].mean(axis=0) for c in categories])
        centroids = centroids/np.linalg.norm(centroids,axis=1,keepdims=True)
        prep_rows.append({'size':catalog['size'],'bm25_index_seconds':bm_prep,
                          'centroid_preparation_seconds':time.perf_counter()-start})
        for qi,q in enumerate(queries):
            t=time.perf_counter(); br=rank(bm.scores(q['query']),ids); bt=time.perf_counter()-t
            t=time.perf_counter(); vr=rank(values@qvectors[qi],ids); vt=time.perf_counter()-t
            t=time.perf_counter(); hr=rrf(br,vr); ht=time.perf_counter()-t
            t=time.perf_counter(); route=categories[int(np.argmax(centroids@qvectors[qi]))]; ri=local[route]
            dr=rank(values[ri]@qvectors[qi],[ids[i] for i in ri]); dt=time.perf_counter()-t
            for method,ranking,elapsed in [('bm25',br,bt),('vector',vr,vt),('hybrid_rrf60',hr,bt+vt+ht),('domain_vector_centroid',dr,dt)]:
                row={'method':method,'size':catalog['size'],'query_index':qi,'query_sha256':digest(q['query']),
                     'catalog_sha256':catalog['ordered_sha256'],'gold_id':q['gold_id'],'gold_name':q['gold_skill'],'name_equivalent_ids':equivalent[qi],
                     'ranking20':ranking[:20],'search_seconds':elapsed,
                     'stage':'ranked_retrieval','final_selection':None,
                     'scope':'supplement_new_router' if method=='domain_vector_centroid' else 'primary_new_retrieval',
                     'routed_category':route if method=='domain_vector_centroid' else None,
                     'candidate_recall20':float(q['gold_id'] in ranking[:20]),
                     'name_candidate_recall20':float(bool(set(equivalent[qi])&set(ranking[:20]))),
                     'oracle_recall5_from20':float(q['gold_id'] in ranking[:20]),
                     'oracle_mrr5_from20':float(q['gold_id'] in ranking[:20])}
                raw_rows.append(row)
                for contract,targets in [('uuid_single_target',[q['gold_id']]),('name_equivalent_single_target',equivalent[qi])]:
                    for k in [5,10]:
                        all_rows.append({'method':method,'size':catalog['size'],'k':k,'query_index':qi,
                                         'label_contract':contract,
                                         **single_target_metrics(ranking,targets,k),'search_ms':elapsed*1000})
                plans.append({'query_index':qi,'query':q['query'],'source_method':method,'size':catalog['size'],
                    'catalog_sha256':catalog['ordered_sha256'],'candidate_ids':ranking[:20],
                    'candidate_text_sha256':digest([text_view(by_id[sid]) for sid in ranking[:20]]),
                    'candidate_order_sha256':digest(ranking[:20]),'return_k':[5,10],
                    'status':'prepared_not_sent','api_latency_seconds':None,'observed_cost_usd':None})
    with (out/'per_case.jsonl').open('w') as f:
        for r in raw_rows:f.write(json.dumps(r)+'\n')
    write_csv(out/'per_query_metrics.csv',all_rows)
    write_csv(out/'index_preparation.csv',prep_rows)
    grouped=defaultdict(list)
    for r in all_rows:grouped[(r['method'],r['size'],r['k'],r['label_contract'])].append(r)
    summary=[]
    for (method,size,k,contract),rows in grouped.items():
        summary.append({'method':method,'size':size,'k':k,'label_contract':contract,'n':len(rows),
            **{key:statistics.mean(r[key] for r in rows) for key in ['recall','hit','mrr','map','ndcg','precision','f1','completeness']},
            'search_median_ms':statistics.median(r['search_ms'] for r in rows),
            'search_p95_ms':float(np.percentile([r['search_ms'] for r in rows],95))})
    write_csv(out/'summary.csv',summary)
    write_json(out/'jev-rerank-plan.json',{'requests':plans,'request_count':len(plans),
        'network_calls':0,'fresh_paid_reranker_status':'not_run_budget_not_approved',
        'charge_estimate_usd':None,'reason_estimate_missing':'Current provider model, tokenizer and pricing not verified',
        'latency_boundary':'retrieval search measured; query encoding separate; actual rerank/end-to-end latency not measured',
        'reuse_gate':'Require exact query, candidate IDs and order, candidate text digest, model/schema/version; failure and cost records retained',
        'labels':'Gold labels absent from requests. Oracle metrics are evaluator-side ceilings, not observed reranker quality.'})
    audit_jev(queries,ROOT,out)
    print(json.dumps({'cases':len(raw_rows),'metric_rows':len(all_rows),'cells':len(summary),'output':str(out)}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',required=True,type=Path)
    p.add_argument('--assets',required=True,type=Path)
    p.add_argument('--cache',required=True,type=Path)
    p.add_argument('--output',type=Path,default=HERE/'results')
    args=p.parse_args()
    run(args)

if __name__=='__main__':main()
