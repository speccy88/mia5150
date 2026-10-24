"""Create auditable proposal/gap crosswalks; never invent experiment completion."""
import csv,json,pathlib,hashlib
from core import SIZES,METHODS,filehash
HERE=pathlib.Path(__file__).resolve().parent

def main():
    fields=['proposal_model','method','size','status','new_query_count','old_evidence','new_evidence','limitation']
    with (HERE/'proposal-80-cells.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for model in ['gpt-4o-mini','gpt-4.1-mini','gpt-4.1-nano','DeepSeek V4']:
            for method in METHODS:
                for n in SIZES:
                    status='not_run; no_execution_evidence_found' if model in ['gpt-4.1-mini','gpt-4.1-nano'] else 'not_run_new_hosted; related_historical_partial_not_exact_cell'
                    old='No execution found at audited base.' if model in ['gpt-4.1-mini','gpt-4.1-nano'] else ('Historical generated/name-based listing or BM25 exists at different panels/sizes; no exact matched completion.' if model=='gpt-4o-mini' and method in ['listing','bm25'] else 'Related revised-model native pilot or retrieval studies only; no exact matched completion.')
                    w.writerow(dict(zip(fields,[model,method,n,status,0,old,'protocol.json; results/','Local Qwen is a labeled substitute. Historical generated/name-based or revised-model panels do not complete this exact cell. DeepSeek level requires exact available route before admission.'])))
    scope=[
        ['methods','five methods','four methods','New local study retains all five; no retroactive proposal rewrite.'],
        ['sizes','10/100/1000/6006','50/250/1000','New study uses original grid; existing CPU/scaling coverage reused as historical support.'],
        ['model','four named hosted models','DeepSeek V4.1 Flash primary','Two local Qwen sensitivity tiers; all original hosted cells pending new authorization/access.'],
        ['panel','4392 local; about2000 hosted','20 intended new IDs;200 consumed','Primary200 exposed queries atseed42 with98/78/24 strata; separate24 balanced seed/model sensitivity. Full4392 retrieval and12 generated probes remain separate. Old4192 reservation obsolete.'],
        ['description','1024 character description limit','1024 combined chars and128 BGE tokens','New name plus description[:1024], BGE512tokens with truncation measured; new representation not pooled with old scores.'],
        ['integration','discovery','native middleware and bounded read','Direct isolated local JSON selection, no tools. Existing native pilot remains separate; no native execution claim.'],
        ['scenarios','single/multi-domain,overlap,cascade','no dependency chains','Gold-count/major proxies and12 silver compositional probes; real overlap, ordering, human labels and untouched external panel pending.'],
        ['scope refinement','explicitly permitted','bounded pilot','Local completion is a reduced study, not completion of80 hosted cells or original full-selection scale.']]
    with (HERE/'scope-drift.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['factor','supplied_proposal','revised_proposal','new_scope']);w.writerows(scope)
    # Retain the baseline audit; a separate closure field must point to new evidence.
    matrix=json.loads((HERE/'gap_matrix.json').read_text())
    # Existing evidence-linked closure is a publication artifact; do not replace
    # measured statuses with the baseline placeholder on a reconciliation rerun.
    if not (HERE/'gap-closure-matrix.json').exists():
        (HERE/'gap-closure-matrix.json').write_text(json.dumps({'baseline_audit':matrix,'new_study':'protocol.json','status':'pending evidence-linked closure'},indent=2))
    print('80 original cells; scope drift preserved; independent audit retained')

if __name__=='__main__':main()
