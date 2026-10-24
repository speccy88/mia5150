"""Reconcile final publication against validated completed retrieval evidence."""
import csv,json,pathlib
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[2]
def main():
    validation=json.loads((HERE/'report/full-retrieval-validation.json').read_text());assert validation['status']=='passed' and validation['query_seed_size_rows']==52704
    matrix=json.loads((HERE/'gap-closure-matrix.json').read_text())
    for r in matrix['rows']:
        if r['id']=='A3.1':
            r.update(status='measured_full_query_retrieval',closure_and_limits='New pinned metadata representation completes all4392 exposed queries, four original sizes and three catalog seeds for BM25/vector/hybrid. Prior full-query studies remain separate.',new_evidence=['report/full-retrieval.csv','report/full-retrieval-validation.json','results/retrieval.jsonl.gz','AUDIT.md'])
        if r['id']=='A3.3':r['new_evidence']+=['report/seed-query-bootstrap.csv']
        if r['id']=='A5.2':r['new_evidence']+=['../../../paper/presentation/research_draft.tex']
        if r['id']=='N8':r['new_evidence']+=['report/engineering-validation.json','audit/INDEPENDENT_SENSITIVITY_README.md']
    (HERE/'gap-closure-matrix.json').write_text(json.dumps(matrix,indent=2))
    lines=['# Gap closure matrix','','Submitted and revised proposal scopes remain versioned. Local evidence is independently audited within the stated limits; human, hosted and execution evidence remains pending.','','|ID|Topic|Status|Evidence and limits|','|---|---|---|---|']
    for r in matrix['rows']:
        links=', '.join('['+p+']('+p+')' for p in r['new_evidence']);lines.append('|'+r['id']+'|'+r['topic']+'|'+r['status']+'|'+r['closure_and_limits']+' '+links+'|')
    (HERE/'GAP_CLOSURE.md').write_text('\n'.join(lines)+'\n')
    with (HERE/'report/full-retrieval.csv').open() as f:full=[r for r in csv.DictReader(f) if r['seed']=='42' and r['size']=='6006']
    summary='; '.join(r['method']+' Recall@5 '+f"{float(r['macro_recall5']):.4f}"+', MRR@5 '+f"{float(r['mrr5']):.4f}" for r in full)
    report=HERE/'report/RESEARCH_DRAFT.md';text=report.read_text();text+='\n## Completed full-query retrieval\n\nAll 4,392 exposed official queries × three catalog seeds × four original sizes are complete (52,704 rows, three rankers per row). Seed42 at 6,006 skills: '+summary+'. These are full-query retrieval metrics, not final selection.\n\nIndependent positive-qrel arithmetic passes for 158,112 ranker rows. All '+str(sum(validation['primary_overlap_checked_by_method'].values()))+' available BM25/vector/hybrid primary seed42 top-five rankings and retrieval metrics match exactly. See [full-query CSV](full-retrieval.csv), [validation](full-retrieval-validation.json) and [compressed exact raw evidence](../results/retrieval.jsonl.gz). Search timing uses sparse BM25 and offline batched query embeddings; do not pool it with primary fresh discovery latency.\n';report.write_text(text)
    primary=json.loads((HERE/'embedding-manifest.json').read_text());fullmanifest=json.loads((HERE/'embedding-full-manifest.json').read_text());assert primary['arrays']['docs.npy']==fullmanifest['arrays']['docs.npy']
    (HERE/'results/embedding-preparation-summary.json').write_text(json.dumps({'documents':6006,'document_vectors_reused_and_hash_verified':True,'document_cumulative_successful_checkpoint_seconds':fullmanifest['document_stats']['seconds'],'query_vectors_including12_generated':4404,'query_cumulative_successful_checkpoint_seconds':fullmanifest['query_stats']['seconds'],'query_truncated_count_scope_all4404':fullmanifest['query_stats']['truncated'],'scope':'Successful CPU checkpoint encoding time only; not last-invocation wall. Document timing is reused from the completed primary index. Pauses, loading, warmup and failed setup remain separately recorded.'},indent=2))
    readme=ROOT/'README.md';text=readme.read_text();text=text.replace('The separate local seed/model sensitivity is complete.','The separate local seed/model sensitivity and full 4,392-query retrieval pass are complete.');readme.write_text(text)
    index=ROOT/'EXPERIMENTS.md';text=index.read_text();text=text.replace('4B sensitivity complete.','4B sensitivity complete; full 4,392-query retrieval completes 52,704 query/seed/size rows with three rankers each.');index.write_text(text)
    tasks=HERE/'TASKS.md';text=tasks.read_text();text=text.replace('- [ ] Publish research draft','- [x] Publish research draft').replace('- [ ] Return compact evidence ZIP and raw results to Library; verify main/wiki remote SHA and readback.','- [x] Primary and sensitivity evidence ZIPs/raw saved to Library; main/wiki commits verified. Final extended full-retrieval delivery is recorded in the final response.');tasks.write_text(text)
    from polish_publication import main as polish
    polish();print('Final closure and full-query narrative reconciled against passed validation')
if __name__=='__main__':main()
