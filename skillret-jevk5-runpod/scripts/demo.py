"""CPU-only synthetic schema/conversion demo. NOT a SkillRet or JevK5 model result."""
from __future__ import annotations
import argparse
from pathlib import Path
from skillret_decider.core import *

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);args=p.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    skills=[{'id':'s-review','name':'staged-review','description':'Review staged Git changes without committing them.'},
            {'id':'s-security','name':'security-review','description':'Find security issues in a code diff.'},
            {'id':'s-commit','name':'git-commit','description':'Create a Git commit from staged changes.'},
            {'id':'s-status','name':'git-status','description':'Show modified and staged files.'},
            {'id':'s-pump','name':'pump-report','description':'Analyze pump measurements and produce a report.'}]
    queries=[{'id':'q-multi','query':'Review staged changes and check them for security problems.',
              'original_id':'f1','skill_ids':['s-review','s-security'],'skill_names':['do not leak'],'k':2},
             {'id':'q-miss','query':'Analyze the pump measurements.', 'original_id':'f2','skill_ids':['s-pump']}]
    qrels=[{'query_id':'q-multi','skill_id':s,'relevance':1} for s in ('s-review','s-security')]
    qrels += [{'query_id':'q-miss','skill_id':'s-pump','relevance':1}]
    ix=index_unique(skills);gold=load_qrels(qrels,set(index_unique(queries)),set(ix))
    retrieved={'q-multi':['s-commit','s-review','s-security','s-status','s-pump'],
               'q-miss':['s-review','s-security','s-commit','s-status','s-pump']}
    examples=[make_example(q,retrieved[q['id']],ix,gold[q['id']],k=3) for q in queries]
    args.out.mkdir(parents=True)
    write_jsonl(args.out/'skills.synthetic.jsonl',skills);write_jsonl(args.out/'queries.synthetic.jsonl',queries)
    write_jsonl(args.out/'qrels.synthetic.jsonl',qrels);write_jsonl(args.out/'converted.synthetic.jsonl',examples)
    write_jsonl(args.out/'upstream.synthetic.jsonl',[upstream_row(e) for e in examples])
    write_json(args.out/'README.json',{'status':'SYNTHETIC_SCHEMA_EXAMPLE_ONLY','real_models_executed':False,
        'expected':'q-multi has two targets with probability .5; q-miss has target NONE=1; no IDs or answer fields are inserted in state.'})
    print(args.out)

if __name__=='__main__': main()
