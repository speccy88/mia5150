"""Audit split overlap and freeze train-only examples and validation membership."""
from collections import Counter,defaultdict
import json
import math
import re
import shutil
import unicodedata
import numpy as np
from common import *

def normalize(text):return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())
def words(text):return set(re.findall(r'\w+',normalize(text)))

def near_rows(left,right,threshold=.95):
    """Exhaustive sparse TF-IDF cosine audit, a lexical proxy for duplication."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    if not left or not right:return []
    texts=[normalize(x['query']) for x in left+right]
    vectorizer=TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,max_features=100000,dtype=np.float32)
    matrix=vectorizer.fit_transform(texts);other=matrix[len(left):].T.tocsr();found=[]
    for start in range(0,len(left),128):
        matches=(matrix[start:min(start+128,len(left))]@other).tocoo()
        for i,j,value in zip(matches.row,matches.col,matches.data):
            if value>=threshold:
                found.append({'left_query_id':left[start+int(i)]['id'],'right_query_id':right[int(j)]['id'],
                              'cosine':float(value),'exact_normalized':texts[start+int(i)]==texts[len(left)+int(j)]})
    return found

def split_queries(queries,gold,train_ids,val_ids,excluded):
    train=[];valid=[];reasons=Counter()
    for q in queries:
        positives=set(gold[q['id']])
        if q['id'] in excluded:reasons['test_query_overlap']+=1
        elif positives<=train_ids:train.append(q)
        elif positives<=val_ids:valid.append(q)
        else:reasons['cross_group_or_quarantined_positive']+=1
    return train,valid,dict(reasons)

def prepare():
    from tokenizers import Tokenizer
    audit=verify_baseline();atomic_json(STUDY/'results/baseline_reuse.json',audit)
    skills=jsonl(CACHE/'data/train-skills.jsonl');queries=jsonl(CACHE/'data/train-queries.jsonl')
    pairs=jsonl(CACHE/'data/train-qrels.jsonl')
    assert (len(skills),len(queries),len(pairs))==(10123,63259,127190)
    assert len({s['id'] for s in skills})==10123 and len({q['id'] for q in queries})==63259
    gold=defaultdict(dict)
    for p in pairs:
        assert p['relevance']>0
        gold[p['query_id']][p['skill_id']]=p['relevance']
    assert set(gold)=={q['id'] for q in queries}
    assert sum(map(len,gold.values()))==127190
    test_skills=jsonl(CACHE/'data/skills.jsonl');test_queries=jsonl(CACHE/'data/queries.jsonl')
    def identities(s):return [('id',s['id']),('source',s['source_url']),('body',digest(normalize(s['skill_md'])))]
    test_identity={item for s in test_skills for item in identities(s)}
    overlap=Counter();quarantine=set();parents={s['id']:s['id'] for s in skills};seen={}
    def root(sid):
        while parents[sid]!=sid:parents[sid]=parents[parents[sid]];sid=parents[sid]
        return sid
    for s in skills:
        for kind,key in identities(s):
            if (kind,key) in test_identity:overlap[kind]+=1;quarantine.add(s['id'])
            if (kind,key) in seen:parents[root(s['id'])]=root(seen[kind,key])
            else:seen[kind,key]=s['id']
    groups=defaultdict(list)
    for s in skills:groups[root(s['id'])].append(s['id'])
    # Whole duplicate/identity groups stay together. Test overlap is quarantined.
    val_ids=set();train_ids=set()
    for members in groups.values():
        if quarantine.intersection(members):quarantine.update(members);continue
        destination=val_ids if int(digest({'seed':42,'group_ids':sorted(members)})[:8],16)%10==0 else train_ids
        destination.update(members)
    print('Auditing training/test query overlap',flush=True)
    matches=near_rows(queries,test_queries)
    excluded={x['left_query_id'] for x in matches}
    train,valid,reasons=split_queries(queries,gold,train_ids,val_ids,excluded)
    print('Auditing training/validation query overlap',len(train),len(valid),flush=True)
    val_near=near_rows(train,valid);val_excluded={x['left_query_id'] for x in val_near}
    train=[q for q in train if q['id'] not in val_excluded]
    valid=sorted(valid,key=lambda q:digest({'validation_seed':42,'query':q['id']}))
    validation=valid[:128];assert len(validation)==128 and len(train)>3200
    tokenizer=Tokenizer.from_file(str(CACHE/'model'/MODEL_REV/'tokenizer.json'));tokenizer.no_truncation();tokenizer.no_padding()
    (CACHE/'prepared').mkdir(exist_ok=True)
    docs={};df=Counter()
    for s in skills:
        if s['id'] not in train_ids|val_ids:continue
        text=full_text(s);spans=chunks(text,tokenizer,budget=382,overlap=64)
        docs[s['id']]={'id':s['id'],'source_sha256':digest(text),'chunks':spans,
                       'representative_chunk':0,'tokens_original':len(tokenizer.encode(text,add_special_tokens=False).ids)}
        df.update(words(text))
    idf={word:math.log(1+len(docs)/(1+count)) for word,count in df.items()}
    skillmap={s['id']:s for s in skills};examples=[];selection=[]
    for split,items in [('train',train),('validation',validation)]:
        for q in items:
            qspans=chunks(q['query'],tokenizer,budget=254,overlap=0)
            query_span=qspans[0];qw=words(q['query'][query_span['start']:query_span['end']]);positive=[]
            for sid in sorted(gold[q['id']]):
                text=full_text(skillmap[sid]);spans=docs[sid]['chunks']
                scores=[sum(idf.get(word,0) for word in words(text[c['start']:c['end']])&qw) for c in spans]
                best=max(range(len(spans)),key=lambda i:(scores[i],-i))
                positive.append({'skill_id':sid,'chunk_index':best,'source_sha256':docs[sid]['source_sha256'],
                                 'passage_sha256':spans[best]['text_sha256']})
            row={'query_id':q['id'],'query_sha256':digest(q['query']),'query_span':query_span,
                 'query_original_tokens':sum(c['tokens'] for c in qspans),'positives':positive,'all_positive_ids':sorted(gold[q['id']]),'split':split}
            examples.append(row)
    atomic_json(CACHE/'prepared/documents.json',docs)
    with (CACHE/'prepared/examples.jsonl').open('w') as f:
        for row in examples:f.write(json.dumps(row)+'\n')
    membership={'training_query_ids':[q['id'] for q in train],'validation_query_ids':[q['id'] for q in validation],
                'heldout_unused_query_ids':[q['id'] for q in valid[128:]],'training_skill_ids':sorted(train_ids),
                'validation_skill_ids':sorted(val_ids),'quarantined_skill_ids':sorted(quarantine)}
    atomic_json(STUDY/'manifests/split_membership.json',membership)
    with (STUDY/'manifests/positive_passage_selection.jsonl').open('w') as f:
        for row in examples:f.write(json.dumps(row)+'\n')
    atomic_json(STUDY/'manifests/training_document_chunks.json',docs)
    atomic_json(STUDY/'results/overlap_audit.json',{'raw_counts':{'train_skills':len(skills),'train_queries':len(queries),'train_qrels':len(pairs)},
         'skill_identity_overlap_with_test':dict(overlap),'quarantined_training_skills':len(quarantine),
         'train_duplicate_identity_groups':sum(len(x)>1 for x in groups.values()),
         'training_test_query_matches':matches,'training_validation_query_matches':val_near,
         'near_duplicate_method':'Exhaustive sparse TF-IDF unigram/bigram cosine >=0.95, 100000 feature cap; lexical proxy, not a semantic-duplicate guarantee',
         'test_qrels_read':False,'test_used_for_selection_or_mining':False,
         'test_text_use':'Overlap audit and fixed quarantine only; no test relevance or retrieval scores inspected',
         'eligible_training_queries':len(train),'validation_queries_used':128,'heldout_unused_queries':len(valid)-128,
         'training_skills':len(train_ids),'validation_skills':len(val_ids),'excluded_queries':reasons,
         'additional_train_queries_near_validation_removed':len(val_excluded),
         'split_method':'Seed42 hash of entire ID/source URL/normalized-body duplicate component; 10% heldout components; mixed-positive queries excluded',
         'positive_selection':'Maximum train-corpus IDF lexical overlap over explicit source chunks, ties first chunk. Skill-level labels weakly supervise only the selected passage.',
         'training_view':'382-content-token Markdown chunks, 64 overlap; queries explicitly select first <=254-token source span; complete test inputs remain unchanged',
         'positive_manifest_sha256':file_hash(STUDY/'manifests/positive_passage_selection.jsonl')})
    print('Prepared',len(train),'training queries and',len(validation),'validation queries',flush=True)

if __name__=='__main__':prepare()
