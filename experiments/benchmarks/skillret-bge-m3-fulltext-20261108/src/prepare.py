"""Audit exact pinned sources and all three views before freezing inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import statistics

from tokenizers import Tokenizer
from common import (STUDY, REPO, QWEN, QWEN_REV, MODEL_REV, MODEL_SHA, CONFIGS, SIZES,
                    atomic_json, chunks, digest, file_hash, full_text, jsonl, panel)


def prepare(cache, context, freeze):
    results = STUDY / 'results'
    results.mkdir(exist_ok=True, parents=True)
    original = json.loads((QWEN / 'protocol.json').read_text())
    # Every published Qwen evidence byte remains immutable, including its vectors.
    original_files = json.loads((QWEN / 'FILE_MANIFEST.json').read_text())['files']
    for row in original_files:
        path = QWEN / row['path']
        with path.open('rb') as f:
            prefix = f.read(200)
        if prefix.startswith(b'version https://git-lfs.github.com/spec/v1\n'):
            pointer = path.read_text()
            assert 'oid sha256:' + row['sha256'] in pointer and 'size ' + str(row['bytes']) in pointer
        else:
            assert file_hash(path) == row['sha256'], row['path']
    (cache / 'data').mkdir(exist_ok=True, parents=True)
    for name, expected in original['data_sha256'].items():
        target = cache / 'data' / (name + '.jsonl')
        if not target.exists():
            old = Path.home() / '.cache/mia-qwen/data' / (name + '.jsonl')
            if old.exists():
                shutil.copyfile(old, target)
            else:
                import requests
                url = f"https://huggingface.co/datasets/ThakiCloud/SKILLRET/resolve/{original['dataset_revision']}/data/{name}/test.jsonl"
                response = requests.get(url, timeout=180)
                response.raise_for_status()
                target.write_bytes(response.content)
        assert file_hash(target) == expected, name
    model_dir = cache / 'model' / MODEL_REV
    tokenizer = Tokenizer.from_file(str(model_dir / 'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    budget = context - 2
    skills = jsonl(cache / 'data/skills.jsonl')
    original_docs = jsonl(QWEN / 'results/documents.jsonl')
    assert len(skills) == len(original_docs) == 6006
    output = {config: [] for config in CONFIGS}
    matched_lengths, subdivisions = [], []
    for i, (skill, old) in enumerate(zip(skills, original_docs)):
        text = full_text(skill)
        assert old['skill_id'] == skill['id'] and old['source_sha256'] == digest(text)
        header = skill['name'] + ' | ' + skill['description']
        for config in CONFIGS:
            view = header if config == 'A_metadata' else text
            if config == 'B_matched':
                spans = []
                for ci, c in enumerate(old['chunks']):
                    part = text[c['start']:c['end']]
                    n = len(tokenizer.encode(part, add_special_tokens=False).ids)
                    matched_lengths.append(n)
                    if n <= budget:
                        spans.append({**c, 'qwen_tokens': c['tokens'], 'tokens': n, 'qwen_chunk_index': ci})
                    else:
                        divisions = chunks(part, tokenizer, budget=budget, overlap=128)
                        for sub in divisions:
                            spans.append({**sub, 'start': sub['start'] + c['start'], 'end': sub['end'] + c['start'], 'qwen_chunk_index': ci})
                        subdivisions.append({'skill_id': skill['id'], 'qwen_chunk_index': ci, 'bge_tokens': n, 'subdivisions': len(divisions)})
            else:
                spans = chunks(view, tokenizer, budget=budget, overlap=128)
            end = 0
            for span in spans:
                assert 0 <= span['start'] <= end < span['end'] <= len(view)
                assert digest(view[span['start']:span['end']]) == span['text_sha256']
                assert span['tokens'] <= budget
                end = span['end']
            assert end == len(view)
            output[config].append({'skill_id': skill['id'], 'source_index': i,
                'source_full_sha256': digest(text), 'view_sha256': digest(view),
                'source_characters': len(view), 'original_tokens': len(tokenizer.encode(view, add_special_tokens=False).ids),
                'chunks': spans, 'view': 'metadata' if config == 'A_metadata' else 'complete full text'})
        if (i + 1) % 500 == 0:
            print('audited skills', i + 1, flush=True)
    audit = {'skills': 6006, 'official_queries': 4392, 'qrels': 7187, 'context_including_special_tokens': context,
             'content_token_budget': budget, 'tokenizer_sha256': file_hash(model_dir / 'tokenizer.json'),
             'qwen_files_verified': len(original_files), 'matched_qwen_chunks': len(matched_lengths),
             'matched_chunk_bge_tokens_max': max(matched_lengths), 'matched_chunk_subdivisions': subdivisions,
             'data_sha256': original['data_sha256'], 'views': {}, 'provisional': not freeze}
    for config, docs in output.items():
        target = results / ('documents-' + config + '.jsonl')
        with target.open('w') as f:
            for doc in docs:
                f.write(json.dumps(doc, sort_keys=True) + '\n')
        lengths = [d['original_tokens'] for d in docs]
        audit['views'][config] = {'documents': len(docs), 'chunks': sum(len(d['chunks']) for d in docs),
            'chunked_documents': sum(len(d['chunks']) > 1 for d in docs), 'original_tokens': sum(lengths),
            'median_original_tokens': statistics.median(lengths), 'max_original_tokens': max(lengths),
            'chunk_tokens_with_overlap': sum(c['tokens'] for d in docs for c in d['chunks']),
            'source_coverage_percent': 100, 'manifest_sha256': file_hash(target)}
    queries = jsonl(cache / 'data/queries.jsonl')
    qrels = jsonl(cache / 'data/qrels.jsonl')
    assert len(queries) == len({q['id'] for q in queries}) == 4392 and len(qrels) == 7187
    query_rows = [{'query_id': q['id'], 'query_sha256': digest(q['query']),
                   'tokens_including_special': len(tokenizer.encode(q['query'], add_special_tokens=True).ids)} for q in queries]
    assert max(q['tokens_including_special'] for q in query_rows) <= context
    for q in panel():
        assert len(tokenizer.encode(q['query'], add_special_tokens=True).ids) <= context
    atomic_json(results / 'query_audit.json', query_rows)
    audit['maximum_official_query_tokens'] = max(q['tokens_including_special'] for q in query_rows)
    shutil.copyfile(QWEN / 'results/candidate_pools.json', results / 'candidate_pools.json')
    atomic_json(results / 'input_audit.json', audit)
    protocol = {'model': 'BAAI/bge-m3', 'model_revision': MODEL_REV, 'precision': 'FP16 CUDA inference; float32 normalized stored vectors',
        'model_checkpoint_sha256': MODEL_SHA,
        'dimensions': 1024, 'pooling': 'Official FlagEmbedding CLS', 'flagembedding_version': '1.3.5',
        'backbone_loading': {'torch_dtype': 'float16', 'low_cpu_mem_usage': True, 'attn_implementation': 'sdpa', 'device_map': {'': 'cuda:0'}},
        'query_prefix': '', 'batch_size': 1, 'context_including_special_tokens': context,
        'content_token_budget': budget, 'overlap_tokens': 128, 'markdown_lookback_tokens': 256,
        'configs': {'A_metadata': 'name + | + description, original whitespace',
                    'B_matched': 'exact Qwen full-text source-character spans, with explicitly recorded overlength subdivisions only if required',
                    'C_native8k': 'complete full text; native BGE tokenizer chunks at tested context'},
        'aggregation': 'max float32 chunk cosine per skill, deduplicated top50', 'ties': original['ties'],
        'dataset_revision': original['dataset_revision'], 'data_sha256': original['data_sha256'],
        'candidate_pools_sha256': file_hash(results / 'candidate_pools.json'),
        'panel_sha256': file_hash(REPO / 'experiments/benchmarks/skill-discovery-oauth-v1-20261103/data/test_queries.json'),
        'qwen_reference_revision': QWEN_REV, 'qwen_protocol_sha256': file_hash(QWEN / 'protocol.json'),
        'source_repository_base_revision': QWEN_REV,
        'shared_helper_sha256': {str(p.relative_to(REPO)): file_hash(p) for p in (QWEN/'src/core.py', QWEN/'src/validate.py', QWEN.parent/'skillret-embedding-jev-fusion/evaluate.py')},
        'document_manifest_sha256': {c: audit['views'][c]['manifest_sha256'] for c in CONFIGS},
        'tokenizer_sha256': audit['tokenizer_sha256'], 'sizes': list(SIZES),
        'bootstrap': {'unit': 'question cluster paired across encoders and repeated sizes', 'seed': 42, 'resamples': 10000},
        'primary_contrast': {'treatment': 'B_matched', 'reference': 'archived Qwen3', 'metric': 'NDCG@10'},
        'secondary_contrasts': ['A-Qwen', 'C-Qwen', 'B-A', 'C-B; exploratory intervals without multiplicity correction'],
        'optional_sparse_hybrid_colbert': 'unattempted; dense A/B/C required first'}
    if freeze:
        feasibility = json.loads((results / 'feasibility.json').read_text())
        assert feasibility['selected_context'] == context and feasibility['status'] == 'passed'
        target = STUDY / 'protocol.json'
        if target.exists():
            assert json.loads(target.read_text()) == protocol, 'Frozen scientific protocol changed'
        else:
            atomic_json(target, protocol)
    print(json.dumps(audit, indent=2), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, default=Path.home() / '.cache/mia-bgem3')
    parser.add_argument('--context', type=int, default=8192)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    prepare(args.cache, args.context, args.freeze)
