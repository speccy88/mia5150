"""Score a frozen candidate list, without reading relevance labels. Never execute skill code."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
from skillret_decider.core import read_jsonl,write_jsonl,write_json,sha256_file,digest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True,type=Path); p.add_argument('--data',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path); p.add_argument('--which',choices=['zero','selected'],required=True)
    p.add_argument('--split',choices=['validation','test'],required=True)
    p.add_argument('--approve-final-test',action='store_true')
    args=p.parse_args()
    if args.out.exists(): raise FileExistsError(args.out)
    cfg=json.loads((args.run/'config.json').read_text()); selection=json.loads((args.run/'selection.json').read_text())
    data=json.loads((args.data/'manifest.json').read_text())
    if digest(cfg)!=selection['config_sha256']: raise ValueError('Run config changed after model selection')
    if data['dataset']['revision'] != selection['dataset_revision']: raise ValueError('Dataset revision mismatch')
    for k in ('id','revision','max_tokens'):
        if data['retriever'][k]!=selection['retriever'][k]: raise ValueError('Retriever changed after selection')
    if args.split=='test':
        if not args.approve_final_test or not data['is_test']: p.error('Final test requires explicit approval and official test data')
        if cfg['train_limit'] is not None: p.error('Pilot is not a final test run. Complete/freeze the intended training protocol first.')
    elif data['is_test']: p.error('Cannot label test data as validation')
    source=args.data/f'{args.split}.jsonl'
    if sha256_file(source)!=data['files'][source.name]: raise ValueError('Candidate-file hash mismatch')
    examples=read_jsonl(source)
    if any('gold_probs' in e or 'expected' in e or e['gold_injected'] for e in examples):
        raise ValueError('Inference data must not contain gold targets or injected candidates')
    from skillret_decider.model import load_backbone,LetterReader,encode_example,predict,torch
    model,tok=load_backbone(cfg)
    adapter=selection['selected']['adapter'] if args.which=='selected' else None
    if adapter:
        for name,checksum in selection['adapter_files'].items():
            if sha256_file(args.run/adapter/name)!=checksum: raise ValueError('Selected adapter checksum mismatch')
        from peft import PeftModel
        model=PeftModel.from_pretrained(model,args.run/adapter,is_trainable=False)
    reader=LetterReader(model,tok).eval()
    encoded=[encode_example(tok,e,cfg) for e in examples]
    args.out.mkdir(parents=True)
    write_jsonl(args.out/'tokenization_audit.jsonl',(r['audit'] for r in encoded))
    # One unscored warm-up; per-batch times exclude tokenization, loading and networking.
    predict(reader,examples[:1],encoded[:1],1)
    t=time.monotonic()
    predictions=predict(reader,examples,encoded,cfg['training']['eval_batch_size'])
    write_jsonl(args.out/'predictions.jsonl',predictions)
    write_json(args.out/'manifest.json',{'which':args.which,'split':args.split,
        'selected_is_unadapted':adapter is None, 'model':cfg['model'], 'selection_sha256':sha256_file(args.run/'selection.json'),
        'candidate_file_sha256':sha256_file(source),'count':len(predictions),'seconds':time.monotonic()-t,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'labels_read_by_inference':False})
    if args.split=='validation' and args.which=='selected':
        expected=read_jsonl(args.run/'selected_validation_predictions.jsonl')
        a={x['query_id']:x for x in expected}; b={x['query_id']:x for x in predictions}
        if set(a)!=set(b): raise ValueError('Reload verification query mismatch')
        maximum=max(abs(a[q]['option_logits'][k]-b[q]['option_logits'][k]) for q in a for k in a[q]['option_logits'])
        agreement=sum(a[q]['ranked_skill_ids']==b[q]['ranked_skill_ids'] for q in a)/len(a)
        write_json(args.out/'reload_check.json',{'max_abs_logit_difference':maximum,'full_ranking_agreement':agreement,
            'passed_tolerance':maximum<=.05 and agreement>=.99,
            'note':'Any failure needs inspection before the official test; near-ties can move in BF16.'})
        if maximum>.05 or agreement<.99: raise RuntimeError('Adapter reload parity failed: inspect before test')
    print(json.dumps({'out':str(args.out),'count':len(predictions)},indent=2))

if __name__ == '__main__': main()
