"""Nine presentation figures and a measured, source-backed research report."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import STUDY,QWEN,CONFIGS,SIZES,atomic_json,file_hash,bootstrap

R=STUDY/'results';F=STUDY/'figures'
COLORS={'Qwen':'#777777','A_metadata':'#D08A00','B_matched':'#176FB0','C_native8k':'#00836B'}
LABELS={'Qwen':'Qwen3 8B Q4 full text','A_metadata':'BGE-M3 A: metadata','B_matched':'BGE-M3 B: matched chunks','C_native8k':'BGE-M3 C: native context'}


def read(p):return json.loads(p.read_text())


def build():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'})
    F.mkdir(exist_ok=True);manifest=[]
    performance=read(R/'performance.json');comparison=read(R/'full_comparison.json')
    paired=read(R/'paired_official.json');scaling_paired=read(R/'paired_scaling.json')
    audit=read(R/'input_audit.json');protocol=read(STUDY/'protocol.json');feasi=read(R/'feasibility.json')
    summaries={'Qwen':{n:read(QWEN/f'results/scaling-n{n}.json') for n in SIZES}}
    for c in CONFIGS:summaries[c]={n:read(R/c/f'scaling-n{n}.json') for n in SIZES}
    def save(fig,name,sources):
        for suffix in ('png','svg'):
            path=F/(name+'.'+suffix);fig.savefig(path,dpi=300,bbox_inches='tight',metadata={'Creator':'Research Team'})
            if suffix=='svg':path.write_text('\n'.join(s.rstrip() for s in path.read_text().splitlines())+'\n')
            manifest.append({'file':path.name,'sha256':file_hash(path),
                'sources':[{'path':str(p.relative_to(STUDY.parents[2])),'sha256':file_hash(p)} for p in sources]})
        plt.close(fig)
    def axis(ax,title,ylabel,percent=False):
        ax.set_title(title,loc='left',weight='bold');ax.set_xscale('log');ax.set_xlim(40,7600)
        ax.set_xticks(SIZES,[f'{n:,}' for n in SIZES],fontsize=9,rotation=40,ha='right');ax.minorticks_off();ax.grid(axis='y',alpha=.18)
        ax.set_xlabel('Eligible library size (skills; logarithmic scale)');ax.set_ylabel(ylabel)
        if percent:
            from matplotlib.ticker import PercentFormatter
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(-.025,1.05)
    def curves(ax,metric):
        for model,sizes in summaries.items():
            values=[sizes[n]['evaluation']['aggregate'][metric] for n in SIZES]
            ax.plot(SIZES,values,marker={'Qwen':'D','A_metadata':'s','B_matched':'o','C_native8k':'^'}[model],
                linestyle='--' if model=='Qwen' else '-',color=COLORS[model],label=LABELS[model])
    footer='20 repeated exposed questions per size · exact skill IDs · historical nonnested, gold-conditioned pools'
    source_scaling=[R/'scaling_summary.csv',QWEN/'results/scaling_summary.csv']
    for metric,name,title in [('Top1_ID_accuracy','01_top1_accuracy','Exact-ID Top-1 accuracy'),
                              ('NDCG@10','03_ndcg10','NDCG@10'),('MRR@10','04_mrr10','MRR@10')]:
        fig,ax=plt.subplots(figsize=(14,5),layout='constrained');axis(ax,title,'Accuracy' if metric=='Top1_ID_accuracy' else 'Query mean (0–1)',metric=='Top1_ID_accuracy')
        curves(ax,metric);ax.legend(frameon=False,ncol=2,fontsize=10);fig.supxlabel(footer,fontsize=10)
        save(fig,name,source_scaling)
    fig,axes=plt.subplots(1,2,figsize=(15,5),layout='constrained')
    for ax,metric in zip(axes,('Recall@5','Recall@10')):
        axis(ax,metric,'Macro recall',True);curves(ax,metric)
    axes[0].legend(frameon=False,fontsize=9);fig.supxlabel(footer,fontsize=10)
    save(fig,'02_recall5_recall10',source_scaling)
    # Prespecified matched-chunk contrast: paired questions and repeated sizes.
    key='B_matched-minus-Qwen';data=paired[key];metrics=data['metrics'];means=np.asarray(data['paired_question_bootstrap']['mean']);ci=np.asarray(data['paired_question_bootstrap']['per_column_95ci'])
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained');y=np.arange(len(metrics))
    axes[0].errorbar(means,y,xerr=np.stack([means-ci[:,0],ci[:,1]-means]),fmt='o',capsize=4,color=COLORS['B_matched'])
    axes[0].set_yticks(y,metrics);axes[0].axvline(0,color='#777777',ls='--');axes[0].set_xlabel('BGE matched − Qwen · metric units')
    axes[0].set_title('Official test: 4,392 paired questions',loc='left',weight='bold')
    data=scaling_paired[key]['Top1_ID_accuracy'];means=np.asarray(data['mean']);ci=np.asarray(data['per_column_95ci'])
    axes[1].errorbar(SIZES,means,yerr=np.stack([means-ci[:,0],ci[:,1]-means]),fmt='o-',capsize=4,color=COLORS['B_matched'])
    axes[1].set_xscale('log');axes[1].set_xticks(SIZES,[f'{n:,}' for n in SIZES],fontsize=8,rotation=40,ha='right');axes[1].minorticks_off();axes[1].axhline(0,color='#777777',ls='--')
    axes[1].set_xlabel('Eligible skills');axes[1].set_ylabel('Top-1 difference');axes[1].set_title('Scaling: 20 paired question clusters',loc='left',weight='bold')
    fig.supxlabel('95% paired question-bootstrap intervals · 10,000 resamples · seed 42 · encoder, precision and query conventions differ',fontsize=9)
    save(fig,'05_paired_differences',[R/'paired_official.json',R/'paired_scaling.json'])
    # Encoding throughput and actual cache-aware wall time use separate boundaries.
    qperf=read(QWEN/'results/performance.json');labels=['Qwen3 8B Q4','BGE A','BGE B','BGE C']
    throughput=[qperf['chunk_embeddings_per_second_active']];wall=[qperf['index_wall_seconds_completed_stages']/60]
    for c in CONFIGS:
        p=performance['configurations'][c];throughput.append(p['uncached_calls']/p['encoding_s_this_stage'] if p['encoding_s_this_stage'] else np.nan);wall.append(p['index_wall_s']/60)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');colors=[COLORS[c] for c in ('Qwen',*CONFIGS)]
    axes[0].bar(labels,throughput,color=colors);axes[0].set_ylabel('Uncached chunk encodings / second');axes[0].set_title('Measured active encoding throughput',loc='left',weight='bold')
    axes[1].bar(labels,wall,color=colors);axes[1].set_ylabel('Observed indexing wall time (minutes)');axes[1].set_title('Execution includes verified cache reuse',loc='left',weight='bold')
    cached=performance['configurations']['C_native8k']['cached_calls'];fig.supxlabel(f'Document views/lengths differ · BGE C reused {cached:,} cached chunks · C wall time is incremental, not a fresh standalone run',fontsize=9)
    save(fig,'06_indexing_throughput_time',[R/'performance.json',QWEN/'results/performance.json'])
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    total=[qperf['full_uncached_retrieval_including_cache_persistence']['median_s']];p95=[qperf['full_uncached_retrieval_including_cache_persistence']['p95_s']]
    search=[qperf['full_similarity_search']['median_s']]
    for c in CONFIGS:
        p=performance['configurations'][c];total.append(p['uncached_query_total']['median_s']);p95.append(p['uncached_query_total']['p95_s']);search.append(p['query_search']['median_s'])
    axes[0].bar(labels,np.asarray(total)*1000,color=colors);axes[0].scatter(labels,np.asarray(p95)*1000,marker='_',s=250,color='black',label='p95 observation spread');axes[0].legend(frameon=False)
    axes[0].set_ylabel('Milliseconds per query');axes[0].set_title('Fresh embedding + persistence + search',loc='left',weight='bold')
    axes[1].bar(labels,np.asarray(search)*1000,color=colors);axes[1].set_ylabel('Milliseconds per query');axes[1].set_title('Similarity search component only',loc='left',weight='bold')
    fig.supxlabel('4,392 queries per condition · batch one · warm model · Qwen local HTTP versus BGE in-process PyTorch; runtime remains a confound',fontsize=9)
    save(fig,'07_query_latency',[R/'performance.json',QWEN/'results/performance.json'])
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained');memory=[qperf['primary_indexing_query_gpu_peak_sampled_mib']]+[performance['configurations'][c]['gpu_peak_sampled_mib'] for c in CONFIGS]
    ax.bar(labels,memory,color=colors);ax.axhline(8192,color='#666666',ls='--',label='Physical GPU: 8,192 MiB');ax.legend(frameon=False)
    ax.set_ylabel('Whole-device sampled peak VRAM (MiB)');ax.set_ylim(0,8800);ax.set_title('RTX 2060 SUPER: selected dense configurations',loc='left',weight='bold')
    fig.supxlabel('One-second samples may miss short peaks · allocator peaks are recorded separately · Qwen feasibility peak 5,614 MiB is excluded here',fontsize=9)
    save(fig,'08_peak_gpu_memory',[R/'performance.json',QWEN/'results/performance.json'])
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained');metrics=('NDCG@10','Recall@10','MRR@10');x=np.arange(3)
    for i,c in enumerate(CONFIGS):
        full=read(R/c/'full_evaluation.json');axes[0].bar(x+(i-1)*.24,[full['aggregate'][m] for m in metrics],width=.24,label=LABELS[c],color=COLORS[c])
    axes[0].set_xticks(x,metrics);axes[0].set_ylim(0,1);axes[0].set_ylabel('Official query mean (0–1)');axes[0].legend(frameon=False,fontsize=9);axes[0].set_title('Metadata and complete-source representations',loc='left',weight='bold')
    for i,(contrast,label,color) in enumerate([('B_matched-minus-A_metadata','B − A',COLORS['B_matched']),('C_native8k-minus-B_matched','C − B',COLORS['C_native8k'])]):
        p=paired[contrast];indices=[p['metrics'].index(m) for m in metrics];d=p['paired_question_bootstrap'];means=np.asarray(d['mean'])[indices];ci=np.asarray(d['per_column_95ci'])[indices]
        axes[1].errorbar(means,x+(i-.5)*.13,xerr=np.stack([means-ci[:,0],ci[:,1]-means]),fmt='o',capsize=4,label=label,color=color)
    axes[1].set_yticks(x,metrics);axes[1].axvline(0,color='#777777',ls='--');axes[1].set_xlabel('Paired difference and 95% interval');axes[1].legend(frameon=False);axes[1].set_title('4,392 paired questions',loc='left',weight='bold')
    fig.supxlabel('Same BGE model, precision and raw query convention · B − A changes full text plus chunk aggregation; C − B changes context and boundaries',fontsize=9)
    save(fig,'09_metadata_fulltext_ablation',[R/'full_comparison.json',R/'paired_official.json'])
    atomic_json(F/'manifest.json',manifest)
    header=['# BGE-M3 dense SkillRet benchmark against Qwen3 8B','','Execution: **completed** — three configurations, 480 scaling cases and 13,176 official query/configuration cases.','',
        'All runs use the pinned 6,006 skills, 4,392 official questions and 7,187 positive judgments from the immutable Qwen study. No model weights or raw restricted text are published. Sparse, hybrid and ColBERT experiments remain unattempted.','',
        '## Official full-catalog results','','| Model / configuration | Hit@1 | NDCG@10 | Recall@10 | MRR@10 | Median query latency |','|---|---:|---:|---:|---:|---:|']
    for row in comparison:
        latency=row['median_query_total_s'];latency=f'{latency*1000:.2f} ms' if latency is not None else 'Not recorded at matched boundary'
        hit=row.get('Top1_ID_accuracy');hit=f'{hit:.4f}' if hit is not None else 'Not recorded'
        header.append(f"| {row['model_configuration']} | {hit} | {row['NDCG@10']:.6f} | {row['Recall@10']:.6f} | {row['MRR@10']:.6f} | {latency} |")
    header+=['','Qwen values are unchanged archived measurements. SkillRet 0.6B is contextual: it strips fields, uses a 32K cap and records two truncated documents. No missing historical latency is invented.','',
        '## Scaling accuracy','','Twenty exposed single-positive questions repeat at each size. All conditions retain the exact published pool IDs/order. The seven smaller libraries are not fully nested and place golds at positions 1–20; the 6,006-skill condition preserves Qwen’s seeded full-catalog order. Rankings use source text and cosine scores, with full-source SHA-256 only for exact ties.','',
        '| Skills | Qwen Top-1 | BGE A | BGE B | BGE C | A − Qwen | B − Qwen | C − Qwen |','|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n in SIZES:
        vals={m:summaries[m][n]['evaluation']['aggregate']['Top1_ID_accuracy'] for m in summaries}
        header.append(f"| {n:,} | {vals['Qwen']:.0%} | {vals['A_metadata']:.0%} | {vals['B_matched']:.0%} | {vals['C_native8k']:.0%} | {(vals['A_metadata']-vals['Qwen'])*100:+.1f} pp | {(vals['B_matched']-vals['Qwen'])*100:+.1f} pp | {(vals['C_native8k']-vals['Qwen'])*100:+.1f} pp |")
    for name,title in [('01_top1_accuracy','Top-1'),('02_recall5_recall10','Recall at five and ten'),('03_ndcg10','NDCG@10'),('04_mrr10','MRR@10')]:header+=['',f'![{title}](figures/{name}.png)']
    header+=['','## Model, exact inputs and complete coverage','',
        f"The official BAAI/bge-m3 revision `{protocol['model_revision']}` runs in FP16 on the local Ubuntu RTX 2060 SUPER 8 GB. The loader uses reduced CPU memory, CUDA-only parameters, official FlagEmbedding CLS pooling/normalization, and a final float32 L2 normalization. Dense vectors have 1,024 dimensions. Batch size is one; no automatic CPU fallback or batch retries are permitted.",'',
        f"The tested input capacity is **{protocol['context_including_special_tokens']:,} tokens**, including two CLS/SEP slots. Explicit full-input tokenization disables truncation and checks every model input ID against the pinned audit tokenizer. Context feasibility was tested conservatively at 512/1,024/2,048/4,096/6,144/8,192, stopping on the first OOM. Model loading took {feasi['model_loading_s']:.3f} seconds; OS file caches were not evicted.",'',
        'FlagEmbedding 1.3.5’s convenience encode method enables truncation and has automatic batch retries. This study instantiates BGEM3FlagModel but calls its official dense forward directly on verified complete inputs. The unchanged backbone/CLS/normalization run with Transformers SDPA and PyTorch memory-efficient attention; math/flash/cuDNN attention fallback is disabled. CPU threads are two, OpenBLAS one, and Torch deterministic mode is enabled.','',
        'A embeds `name + " | " + description` only. B embeds the exact Qwen source-character slices of `name + " | " + description + " | " + skill_md`. C retains the same full original text with deterministic BGE-token-aware Markdown chunks and 128-token overlap. Source whitespace is never stripped.','',
        f"All {audit['matched_qwen_chunks']:,} Qwen chunks fit unchanged; the largest has {audit['matched_chunk_bge_tokens_max']:,} BGE content tokens. Required subdivisions: {len(audit['matched_chunk_subdivisions'])}. The longest official query has {audit['maximum_official_query_tokens']} tokens including specials. BGE receives the original query without Qwen’s instruction prefix, following its documented convention.",'',
        '| Configuration | Skills | Logical chunks | Chunked skills | Audited content tokens with overlap | Source coverage |','|---|---:|---:|---:|---:|---:|']
    for c in CONFIGS:
        a=audit['views'][c];header.append(f"| {c} | 6,006 | {a['chunks']:,} | {a['chunked_documents']:,} | {a['chunk_tokens_with_overlap']:,} | 100% of its defined view |")
    header+=['','Metadata deliberately excludes skill bodies; B and C cover every original full-text character. Max chunk cosine produces one score per skill, and top-50 rankings deduplicate by exact ID. Duplicate names remain distinct IDs. Candidate position, skill IDs and relevance labels never enter embeddings or similarity scores.','',
        '## Paired uncertainty and interpretation','',
        'The prespecified primary contrast is B matched chunks minus archived Qwen, using official NDCG@10. Every full-test difference pairs the same query ID; 10,000 question bootstrap resamples use seed 42. Scaling resamples 20 question clusters and retains their eight repeated sizes. It never treats 160 outcomes as independent questions. Secondary metrics and A/C/ablation contrasts are exploratory without multiplicity correction.','',
        '| Official B − Qwen metric | Mean difference | 95% paired interval |','|---|---:|---|']
    p=paired['B_matched-minus-Qwen'];d=p['paired_question_bootstrap']
    for i,m in enumerate(p['metrics']):lo,hi=d['per_column_95ci'][i];header.append(f"| {m} | {d['mean'][i]:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    header+=['','![Paired differences](figures/05_paired_differences.png)','','![Metadata/full-text ablation](figures/09_metadata_fulltext_ablation.png)','',
        'Exact B source spans, queries, candidates, qrels and metric formulas match Qwen. Remaining differences include encoder training/architecture, tokenizer, FP16 versus Q4_K_M, CLS versus the Qwen embedding pathway, 1,024 versus 4,096 dimensions, query instruction and PyTorch in-process versus Ollama HTTP runtime. Paired score differences describe these measured protocols; they do not isolate model architecture or demonstrate blinded generalization.','',
        'Within BGE, B − A measures the combined complete-source representation and max-chunk retrieval policy. C − B additionally changes context and chunk boundaries. Longer skills have more max-score opportunities. Exposed synthetic questions, incomplete judgments and gold-conditioned scaling pools limit generalization; bootstrap intervals do not remove those biases.','',
        '## Computational cost and latency','',
        '| Configuration | Actual index wall | New encoded chunks | Reused cached chunks | Peak sampled VRAM | Median / p95 full query |','|---|---:|---:|---:|---:|---:|']
    for c in CONFIGS:
        p=performance['configurations'][c];lat=p['uncached_query_total'];header.append(f"| {c} | {p['index_wall_s']/60:.2f} min | {p['uncached_calls']:,} | {p['cached_calls']:,} | {p['gpu_peak_sampled_mib']:.0f} MiB | {lat['median_s']*1000:.2f} / {lat['p95_s']*1000:.2f} ms |")
    telemetry=performance['all_monitored_phases_including_loading_and_feasibility']
    header+=['',f"Across the successful monitored phases, whole-host used RAM peaked at {telemetry['system_peak_used_gib']:.3f} GiB, minimum available RAM was {telemetry['system_min_available_gib']:.3f} GiB, and GPU temperature peaked at {telemetry['gpu_peak_temperature_c']:.0f} °C. Whole-host swap-in/out deltas were {telemetry['host_swap_in_delta_bytes']/2**20:.3f}/{telemetry['host_swap_out_delta_bytes']/2**20:.3f} MiB. These host-wide counters are not process-private allocations. The failed initial loader is excluded and reported separately below.",'',
        '| Configuration | New document encoding | Chunks / active second | Input tokens / active second | Mean sampled GPU use |','|---|---:|---:|---:|---:|']
    for c in CONFIGS:
        p=performance['configurations'][c]
        header.append(f"| {c} | {p['encoding_s_this_stage']:.2f} s | {p['uncached_chunks_per_active_second']:.2f} | {p['uncached_input_tokens_per_active_second']:.0f} | {p['gpu_utilization_mean_percent']:.1f}% |")
    header+=['','Fresh query totals include encoding, full-input validation, vector-cache persistence and CPU similarity search, ending before ranking-file persistence, as in the Qwen receipt boundary. Every configuration makes fresh query measurements; cached search is separately timed and is never substituted for end-to-end query latency. Model startup and corpus indexing remain separate.','',
        'Identical text vectors are shared between B and C. C’s observed index wall therefore measures incremental work with verified cache reuse, not a fresh standalone index. The performance JSON separately records new encoding time, measured tokens/second and the logical sum of original measured per-vector receipts; the latter is a derived sum, not an observed standalone wall run. Short metadata and long full-text chunk throughputs are different workloads. Token throughput is tokenizer-specific, so BGE and Qwen tokens/second are not model-neutral units.','',
        'One-second telemetry records whole-host RAM/swap and GPU memory/utilization/temperature. CUDA allocator peaks capture shorter model allocations separately. Guards retain at least 1 GiB available RAM and 1 GiB device headroom, stop above 80 °C or 64 MiB additional host swap traffic, and never stop another workload. Electricity spend is unmeasured; no paid inference service is used.','',
        '![Indexing](figures/06_indexing_throughput_time.png)','','![Query latency](figures/07_query_latency.png)','','![VRAM](figures/08_peak_gpu_memory.png)','',
        '## Contextual BGE references','',
        'The repository’s preserved official SkillRet README reports BGE-small NDCG@10 0.5451 / Recall@10 0.6001 and BGE-large 0.5900 / 0.6437 on the stated 4,392-query, 6,006-skill test. These are published contextual references, not new local measurements or matched complete-source controls. Source paths and checksums are in contextual_bge_references.json.','',
        '## Validation, vectors and reproducibility','',
        'Independent arithmetic re-scores all saved rankings, including multi-positive AP/NDCG/completeness and macro/micro candidate coverage. The official denominator remains 4,392 questions and 7,187 positive pairs; scaling keeps one positive ID per question. Official Top-1 means the first ID is any positive; completeness means every positive is recovered within K. NDCG uses linear positive gains, AP divides by all positives, precision divides by K and reciprocal rank uses the first positive. Missing questions receive zero and remain in the denominator. Full metric tables, per-query outcomes and performance components are retained as JSON and CSV.','',
        'Lossless normalized float32 vectors for all A/B/C documents and shared official/scaling queries are published through Git LFS with IDs, chunk ownership and public qrels. The snapshot replay checks every ranking and metric without a GPU, model, network call or original vector cache. See vectors/README.md and results/published_vector_replay.json. Model checkpoints, credentials, personal paths and raw source text stay outside Git.','']
    if (R/'published_vector_replay.json').exists():
        replay=read(R/'published_vector_replay.json');vm=read(STUDY/'vectors/manifest.json');header += [f"The verified archive is {vm['bytes']:,} bytes. Replay completed all 480 scaling and 13,176 official query/configuration outcomes in {replay['total_wall_s']:.3f} seconds with zero model/API/network requests.",'']
    primary=paired['B_matched-minus-Qwen']['paired_question_bootstrap'];i=paired['B_matched-minus-Qwen']['metrics'].index('NDCG@10');lo,hi=primary['per_column_95ci'][i]
    wording='above zero' if lo>0 else ('below zero' if hi<0 else 'including zero')
    header+=['## Findings','',f"The matched-chunk B − Qwen NDCG@10 difference is {primary['mean'][i]:+.6f}, with a 95% paired interval [{lo:+.6f}, {hi:+.6f}] {wording}. This is evidence about the tested exposed protocols, with the encoder and runtime differences stated above.",'']
    for contrast in ('B_matched-minus-A_metadata','C_native8k-minus-B_matched'):
        p=paired[contrast];i=p['metrics'].index('NDCG@10');d=p['paired_question_bootstrap'];lo,hi=d['per_column_95ci'][i]
        header += [f"{contrast}: NDCG@10 difference {d['mean'][i]:+.6f}, exploratory 95% paired interval [{lo:+.6f}, {hi:+.6f}].",'']
    header+=['The official test favors archived Qwen for Top-1, relevant-skill coverage and ranking quality. The 20-question panel reverses that ordering at the larger catalog sizes: at 6,006 skills, A/B/C Top-1 is 65%/60%/55%, versus Qwen 40%. These are different exposed question panels; the small-panel result does not overturn the 4,392-question result. No monotonic library-size improvement is established.','',
        'Full text increases Recall@10 over metadata, but B-minus-A NDCG@10 has no clear gain and Hit@1 falls. Native 8K adds a small exploratory NDCG@10 gain over matched chunks while changing boundaries as well as context. All BGE views have lower observed query latency and sampled VRAM than archived Qwen; B indexing takes 16.75 minutes versus Qwen 163.31 minutes. This is a measured runtime/workload comparison, not an isolated architecture effect or an electricity-cost estimate.','']
    header+=['## Failed setup attempts and unattempted work','',
        'The first PyTorch dependency download exhausted the small temporary-filesystem quota before model inference. A task-specific disk-backed TMPDIR completed installation. The first model-loading feasibility invocation reached CUDA but the host swap guard stopped it before any encoding; the retained telemetry shows 894.4 MiB extra swap-out. Cgroup limits were unrestricted. Loading the backbone directly to CUDA through the official device_map argument then passed all context tests with zero additional swap-out. These setup/loading failures produced no retrieval scores and remain separate from the completed benchmark.', '',
        'Sparse lexical, BM25/dense RRF, dense/sparse fusion and ColBERT reranking were not attempted. They are not reported as zero scores. Any additional interrupted execution receipts are retained separately and do not replace completed measurements.','',
        'Primary references: [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3), [official FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding), [pinned Qwen study](../skillret-qwen3-embedding-8b-q4km-fulltext/REPORT.md).','']
    (STUDY/'REPORT.md').write_text('\n'.join(header));print('Nine PNG/SVG figures and completed report generated',flush=True)

if __name__=='__main__':build()
