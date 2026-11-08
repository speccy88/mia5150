"""Twelve measured PNG/SVG figures and a reproducible paired-training report."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from common import *

R=STUDY/'results';F=STUDY/'figures'
COLORS={'A_metadata':'#D08A00','B_matched':'#176FB0','C_native8k':'#00836B'}
SHORT={'A_metadata':'A: metadata','B_matched':'B: matched','C_native8k':'C: native 8K'}

def build():
    assert read(R/'validation.json')['completed_requested_scope']
    train=read(R/'training_summary.json');architecture=read(R/'architecture.json');smoke=read(R/'smoke.json')
    curve=jsonl(R/'training_curve.jsonl');validation=jsonl(R/'validation_curve.jsonl')
    comparison=read(R/'full_comparison.json');paired=read(R/'paired_base_vs_lora.json')
    performance=read(R/'performance.json');baseperf=read(BASE/'results/performance.json');resources=read(R/'resources.json')
    overlap=read(R/'overlap_audit.json');exact=read(R/'exact_overlap_verification.json')
    within=read(R/'paired_within_model_views.json');manifest=[]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'})
    F.mkdir(exist_ok=True)
    def save(fig,name,sources,caption):
        fig.supxlabel(caption,fontsize=9)
        for extension in ('png','svg'):
            path=F/(name+'.'+extension);fig.savefig(path,dpi=300,bbox_inches='tight',metadata={'Creator':'Research Team'})
            if extension=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
            manifest.append({'file':path.name,'sha256':file_hash(path),
                'sources':[{'path':str(p.relative_to(REPO)),'sha256':file_hash(p)} for p in sources]})
        plt.close(fig)
    aggregates={label:{c:read(folder/'results'/c/'full_evaluation.json')['aggregate'] for c in CONFIGS} for label,folder in [('Base',BASE),('LoRA',STUDY)]}
    fig,ax=plt.subplots(figsize=(11,4.2),layout='constrained');steps=[r['optimizer_step'] for r in curve];losses=np.asarray([r['loss'] for r in curve])
    ax.plot(steps,losses,color=COLORS['B_matched'],alpha=.3,label='Per optimizer step')
    if len(losses)>=20:ax.plot(steps[19:],np.convolve(losses,np.ones(20)/20,mode='valid'),color=COLORS['B_matched'],label='20-step mean')
    ax.axvline(200,color='#777777',ls='--',label='Pilot boundary');ax.set(xlabel='Completed optimizer step',ylabel='InfoNCE loss',title='Main LoRA training loss');ax.legend(frameon=False)
    save(fig,'01_training_loss',[R/'training_curve.jsonl'],'One query group × 16 accumulation · one negative per group · temperature 0.05 · smoke updates excluded')
    fig,ax=plt.subplots(figsize=(11,4.2),layout='constrained');ax.plot([r['optimizer_step'] for r in validation],[r['metrics']['NDCG@10'] for r in validation],marker='o',color=COLORS['B_matched'])
    ax.axhline(train['frozen_validation_ndcg10'],color='#777777',ls='--',label='Frozen reference');ax.scatter([train['best_step']],[train['best_validation_ndcg10']],marker='*',s=180,color=COLORS['A_metadata'],label='Selected adapter',zorder=5)
    ax.set(xlabel='Optimizer step',ylabel='Validation NDCG@10 (0–1)',ylim=(0,1),title='Training-derived validation selects the adapter');ax.legend(frameon=False)
    save(fig,'02_validation_ndcg',[R/'validation_curve.jsonl',R/'training_summary.json'],'128 heldout training queries · 1,014 heldout skills · fixed short-passage proxy · official test never selects checkpoints')
    def paired_bars(metric,title,name):
        fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained');x=np.arange(3)
        ax.bar(x-.18,[aggregates['Base'][c][metric] for c in CONFIGS],.36,color='#999999',label='Frozen base')
        ax.bar(x+.18,[aggregates['LoRA'][c][metric] for c in CONFIGS],.36,color=[COLORS[c] for c in CONFIGS],label='Validation-selected LoRA')
        ax.set_xticks(x,[SHORT[c] for c in CONFIGS]);ax.set(ylabel=metric+' (0–1)',ylim=(0,1),title=title);ax.legend(frameon=False)
        save(fig,name,[R/'full_comparison.json'],'4,392 official queries × 6,006 skills per view · compare base and LoRA within the same view')
    paired_bars('NDCG@10','Official skill ordering: identical-view before/after','03_base_lora_ndcg')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained');x=np.arange(3)
    for ax,metric in zip(axes,('Recall@5','Recall@10')):
        ax.bar(x-.18,[aggregates['Base'][c][metric] for c in CONFIGS],.36,color='#999999',label='Base')
        ax.bar(x+.18,[aggregates['LoRA'][c][metric] for c in CONFIGS],.36,color=[COLORS[c] for c in CONFIGS],label='LoRA')
        ax.set_xticks(x,[SHORT[c] for c in CONFIGS]);ax.set(title=metric,ylabel='Macro relevant-skill recall',ylim=(0,1));ax.legend(frameon=False)
    save(fig,'04_base_lora_recall',[R/'full_comparison.json'],'4,392 official multi-positive queries · 7,187 positive judgments · exact skill-ID relevance')
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained');scale={}
    for ax,c in zip(axes,CONFIGS):
        scale[c]={}
        for label,folder in [('Base',BASE),('LoRA',STUDY)]:
            scale[c][label]={n:read(folder/'results'/c/f'scaling-n{n}.json')['evaluation']['aggregate'] for n in SIZES}
            ax.plot(SIZES,[scale[c][label][n]['Top1_ID_accuracy']*100 for n in SIZES],marker='o',color='#888888' if label=='Base' else COLORS[c],label=label)
        ax.set_xscale('log');ax.set_xticks(SIZES,[f'{n:,}' for n in SIZES],rotation=40,ha='right',fontsize=8)
        ax.minorticks_off();ax.set(title=SHORT[c],xlabel='Skills (log scale)',ylabel='Top-1 accuracy (%)',ylim=(-2,102));ax.legend(frameon=False)
    save(fig,'05_scaling_accuracy',[R/'scaling_summary.csv'],'20 repeated exposed questions · exact original nonnested pools · smaller pools place golds at positions 1–20')
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for ax,m in zip(axes,('Top1_ID_accuracy','Recall@10','NDCG@10')):
        for label,color in [('Base','#888888'),('LoRA',COLORS['B_matched'])]:
            ax.plot(range(3),[aggregates[label][c][m] for c in CONFIGS],marker='o',label=label,color=color)
        ax.set_xticks(range(3),['Metadata','Matched full text','Native full text'],rotation=15,ha='right');ax.set(title=m,ylabel='Official mean (0–1)',ylim=(0,1));ax.legend(frameon=False)
    save(fig,'06_metadata_fulltext',[R/'full_comparison.json'],'4,392 questions · representation and max-chunk retrieval change across views · training gains require matching views')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for i,label in enumerate(('Base','LoRA')):
        rows=within[label+' C_native8k minus B_matched']['metrics'];row=next(x for x in rows if x['metric']=='NDCG@10');effect=row['absolute_difference']
        axes[0].errorbar(effect,i,xerr=[[effect-row['ci95_low']],[row['ci95_high']-effect]],fmt='o',color='#888888' if label=='Base' else COLORS['C_native8k'],capsize=4)
    axes[0].set_yticks([0,1],['Base','LoRA']);axes[0].axvline(0,color='#777777',ls='--');axes[0].set(title='Native C minus matched B',xlabel='NDCG@10 difference and 95% paired interval')
    labels=[];times=[]
    for label,p in [('Base',baseperf),('LoRA',performance)]:
        for c in ('B_matched','C_native8k'):labels.append(label+' '+c[0]);times.append(p['configurations'][c]['index_wall_s']/60)
    axes[1].bar(labels,times,color=['#888888','#BBBBBB',COLORS['B_matched'],COLORS['C_native8k']]);axes[1].set(title='Observed index wall',ylabel='Minutes (cache-aware)')
    save(fig,'07_matched_native_context',[R/'paired_within_model_views.json',R/'performance.json',BASE/'results/performance.json'],'C changes context and boundaries · exploratory intervals · C reuses shared B vectors; not fresh standalone indexing')
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    deltas=__import__('csv').DictReader((R/'per_query_lora_differences.csv').open());deltas=list(deltas)
    for ax,c in zip(axes,CONFIGS):
        values=[float(row['NDCG@10']) for row in deltas if row['configuration']==c]
        ax.hist(values,bins=np.linspace(-1,1,42),color=COLORS[c]);ax.axvline(0,color='#555555',ls='--')
        row=next(r for r in paired[c]['metrics'] if r['metric']=='NDCG@10')
        ax.set(title=SHORT[c],xlabel='LoRA − base NDCG@10',ylabel='Questions')
        ax.text(.03,.96,f"Improved {row['improved_queries']:,}\nDegraded {row['degraded_queries']:,}\nUnchanged {row['unchanged_queries']:,}",transform=ax.transAxes,va='top')
    save(fig,'08_per_query_improvement',[R/'per_query_lora_differences.csv',R/'paired_base_vs_lora.json'],'4,392 exact paired query IDs per view · zero differences remain visible; no failed queries are discarded')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    for ax,key,title in zip(axes,('before','after'),('Before smoke','After 64 smoke updates')):
        for field,color in [('positive',COLORS['B_matched']),('negative','#BC5662')]:
            ax.hist([r[field] for r in smoke[key]],bins=np.linspace(-.2,1,25),alpha=.6,label=field,color=color)
        ax.set(title=title,xlabel='Cosine similarity',ylabel='Training query groups');ax.legend(frameon=False)
    save(fig,'09_positive_negative_similarity',[R/'smoke.json'],'32 real training queries · functional smoke only · these updates were discarded; this is not a generalization result')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    mining=read(R/'mining.json');axes[0].bar(['Smoke','Train-only mining','Main training'],[smoke['wall_s']/60,mining['wall_s']/60,train['main_wall_s']/60],color=['#999999',COLORS['A_metadata'],COLORS['B_matched']]);axes[0].set(ylabel='Measured wall minutes',title='Training pipeline components')
    names=['Main training','Final inference'];axes[1].bar(names,[resources['main_training']['gpu_peak_sampled_mib'],resources['final_inference']['gpu_peak_sampled_mib']],color=[COLORS['B_matched'],COLORS['C_native8k']]);axes[1].axhline(8192,color='#777777',ls='--',label='Device capacity');axes[1].set(ylabel='Whole-device sampled MiB',title='GPU memory on RTX 2060 SUPER',ylim=(0,8500));axes[1].legend(frameon=False)
    save(fig,'10_training_time_memory',[R/'training_summary.json',R/'smoke.json',R/'mining.json',R/'resources.json'],'Training excludes one-time corpus indexing · one-second whole-device samples; CUDA allocator peaks recorded separately')
    fig,ax=plt.subplots(figsize=(11,4.5),layout='constrained');x=np.arange(3)
    for shift,label,p,color in [(-.18,'Base',baseperf,'#999999'),(.18,'LoRA',performance,COLORS['B_matched'])]:
        med=[p['configurations'][c]['uncached_query_total']['median_s']*1000 for c in CONFIGS]
        tail=[p['configurations'][c]['uncached_query_total']['p95_s']*1000 for c in CONFIGS]
        ax.bar(x+shift,med,.36,label=label+' median',color=color);ax.scatter(x+shift,tail,marker='_',s=150,color='black',label='p95 observation spread' if label=='Base' else None)
    ax.set_xticks(x,[SHORT[c] for c in CONFIGS]);ax.set(ylabel='Milliseconds per fresh query',title='Fresh embedding + validation + persistence + CPU search');ax.legend(frameon=False)
    save(fig,'11_query_latency',[R/'performance.json',BASE/'results/performance.json'],'4,392 queries per view · batch one, warm model · ends before ranking-file persistence; cached search is separate')
    fig,axes=plt.subplots(1,3,figsize=(15,6),layout='constrained');labels=[r['model_configuration'].replace('BGE-M3 ','') for r in comparison]
    for ax,m in zip(axes,('NDCG@10','Recall@10','MRR@10')):
        ax.barh(range(len(comparison)),[r[m] for r in comparison],color=[COLORS.get(next((c for c in CONFIGS if c in r['model_configuration']),''),'#777777') if 'LoRA' in r['model_configuration'] else '#AAAAAA' for r in comparison])
        ax.set_yticks(range(len(comparison)),labels,fontsize=8);ax.invert_yaxis();ax.set(title=m,xlabel='Official mean (0–1)',xlim=(0,1))
    save(fig,'12_historical_comparison',[R/'full_comparison.json'],'All historical values preserved · Qwen quantization/runtime/query convention differ; SkillRet 0.6B uses stripped fields, 32K cap and two truncated docs')
    atomic_json(F/'manifest.json',manifest)

    header=['# BGE-M3 LoRA: exact paired SkillRet evaluation','',
        '**Completed:** real CUDA LoRA training and all three paired A/B/C evaluations, 480 scaling cases and 13,176 official query/configuration cases. The completed baseline is preserved and reused; no official base inference was rerun.','',
        '## Training and checkpoint selection','',
        f"Execution used the local Ubuntu RTX 2060 SUPER 8 GB server (Intel i3-3225, approximately 7.2 GiB usable host RAM). Rank-8 LoRA with alpha 16 and dropout 0.05 targets all 48 encoder attention query/value projections. Exactly {architecture['trainable_parameters']:,} adapter parameters train; all {architecture['frozen_parameters']:,} base parameters remain frozen. The base uses FP16; PEFT adapters and AdamW moments remain FP32 for stability. CLS embeddings are normalized for the cosine InfoNCE objective, temperature 0.05.",'',
        'Physical groups contain one query, one positive passage and one negative. Sixteen-group accumulation adds no extra in-batch negatives. Query/passage caps are 256/384 tokens, gradient checkpointing is nonreentrant, AdamW starts from the prespecified 1e-4 peak learning rate with 20-step warmup and linear decay, weight decay 0.01 and clipping 1.0. Encoder dropout remains enabled during training; all dropout is disabled for validation/inference.','',
        f"The 64-update / 32-query smoke test passed and its updates were discarded. Main training completed **{train['main_optimizer_steps']:,} optimizer steps** and **{train['main_query_groups']:,} query groups** in **{train['main_wall_s']/60:.2f} minutes**, or {train['effective_query_passes']:.4f} eligible-query passes. Stop reason: `{train['stop_reason']}`. The 200-step pilot's continuation gate was {'passed' if train['full_training_gate_passed'] else 'not passed'}. The maximum was two query passes; this is not a claim that every positive pair was sampled twice.",'',
        f"The selected trained adapter is optimizer step **{train['best_step']}**, validation NDCG@10 **{train['best_validation_ndcg10']:.6f}**, against frozen **{train['frozen_validation_ndcg10']:.6f}**. Selection uses 128 fixed training-derived heldout queries and all 1,014 heldout skills, each represented by its first bounded passage. This short-passage proxy differs from complete test B/C. Ties select the earliest trained checkpoint. Official test queries/qrels/scores never train, mine negatives or select checkpoints.",'',
        '![Loss](figures/01_training_loss.png)','','![Validation](figures/02_validation_ndcg.png)','',
        '## Data and overlap','',
        f"The pinned raw training split contains 10,123 skills, 63,259 queries and 127,190 qrels. Hash-grouping entire skill identities and excluding cross-group multi-positive queries leaves **{overlap['eligible_training_queries']:,} training queries / {overlap['training_skills']:,} skills**. Validation holds out 1,014 skills; 128 queries are used and 2,185 remain heldout/unused. There are no shared skill IDs, source URLs, normalized bodies or namespace identities with test. Exact normalized query overlap with test and across train/validation is zero; the fixed TF-IDF lexical-cosine audit at 0.95 also finds none. This proxy does not guarantee absence of semantic near-duplicates.",'',
        f"The raw training split contains {exact['duplicate_normalized_query_texts_within_raw_train']} repeated normalized query texts, all with identical positive sets. Eligible training retains {exact['eligible_training_unique_normalized_queries']:,} unique normalized texts; these repeated examples are retained and explicitly counted. Every training query's positive IDs independently match its qrels. The selected validation questions and all official test question texts are unique after normalization.",'',
        'Positives come only from actual training qrels. IDF-weighted lexical overlap selects one meaningful source passage for each positive skill; a skill-level label does not establish that every chunk is positive. IDF uses the unlabeled original training-split corpus, including heldout skill documents; supervised updates and mining candidates use only the training partition. Hard negatives come from frozen BGE-M3 over training skills; deterministic random negatives alternate. Every known positive is filtered. All example IDs, negative IDs, source hashes and passage spans are published without raw training text.','',
        '## Official full-catalog results','',
        '| Model / view | Hit@1 | NDCG@10 | Recall@10 | MRR@10 | Median fresh query |','|---|---:|---:|---:|---:|---:|']
    for row in comparison:
        hit='Not recorded' if row['Top1_ID_accuracy'] is None else f"{row['Top1_ID_accuracy']:.6f}"
        latency='Not recorded at matched boundary' if row['median_query_total_s'] is None else f"{row['median_query_total_s']*1000:.2f} ms"
        header.append(f"| {row['model_configuration']} | {hit} | {row['NDCG@10']:.6f} | {row['Recall@10']:.6f} | {row['MRR@10']:.6f} | {latency} |")
    header+=['','All 4,392 official questions, 6,006 skills and 7,187 binary relevance pairs remain in every view. No missing query is dropped. Exact skill IDs distinguish duplicate names. Recall@20 equals macro candidate Coverage@20 in the saved ranking convention; macro/micro candidate coverage is also independently checked. Full Recall@1/5/10/20, NDCG@5/10, MRR@5/10, MAP@5/10, Precision@5/10, completeness and candidate @20/50 results are in JSON/CSV.','',
        '![NDCG](figures/03_base_lora_ndcg.png)','','![Recall](figures/04_base_lora_recall.png)','',
        '## Paired training effect and uncertainty','',
        'The primary contrast is **LoRA B matched minus Base B matched official NDCG@10**. Identical question IDs, source character spans, candidates, complete input tokens, precision of frozen weights, CLS pooling, float32 storage, maximum-chunk aggregation and source-hash tie-breaking isolate the adapter change. Zero-adapter PEFT parity passed against previously saved baseline vectors at absolute tolerance 3e-5 and cosine at least 0.999998. All frozen tensors hash identically before and after training.','',
        'Intervals use 10,000 paired question bootstrap resamples with seed 42. Scaling retains the 20 repeated questions as linked clusters across eight sizes. All other view/metric contrasts are exploratory without multiplicity adjustment. Relative changes use the matching base mean. A positive interval supports an increase under this measured exposed protocol, not architecture-level causation or blinded deployment performance.','',
        '| View | NDCG@10 difference (pp) | 95% paired interval (pp) | Improved / degraded / unchanged queries |','|---|---:|---|---|']
    for c in CONFIGS:
        p=next(r for r in paired[c]['metrics'] if r['metric']=='NDCG@10')
        header.append(f"| {SHORT[c]} | {p['difference_percentage_points']:+.3f} | [{p['ci95_low_percentage_points']:+.3f}, {p['ci95_high_percentage_points']:+.3f}] | {p['improved_queries']:,} / {p['degraded_queries']:,} / {p['unchanged_queries']:,} |")
    header+=['','Absolute/relative changes, paired intervals and improved/degraded/unchanged counts for every required metric are in `paired_base_vs_lora.csv`; full per-query deltas are preserved.','',
        '![Per-query change](figures/08_per_query_improvement.png)','',
        '## Scaling and representation','',
        '| Skills | A base → LoRA | A change | B base → LoRA | B change | C base → LoRA | C change |','|---:|---:|---:|---:|---:|---:|---:|']
    for n in SIZES:
        cells=[]
        for c in CONFIGS:
            b=scale[c]['Base'][n]['Top1_ID_accuracy'];t=scale[c]['LoRA'][n]['Top1_ID_accuracy'];cells.extend([f'{b:.0%} → {t:.0%}',f'{(t-b)*100:+.0f} pp'])
        header.append('| '+f'{n:,} | '+' | '.join(cells)+' |')
    header+=['','These are twenty repeated exposed single-positive questions. Pools exactly preserve baseline ID/order, nonnested membership and the smaller-pool gold-position bias. The 6,006 condition keeps the original seeded full order. They do not replace the larger multi-positive official evaluation.','',
        'A is exactly name + separator + description; B and C retain every full-source character. B uses all 6,913 original Qwen slices; C uses the baseline’s 6,258 native 8K slices, including 128-token overlap. No adapter inference silently truncates, re-chunks or reuses frozen vectors.','',
        '![Scaling](figures/05_scaling_accuracy.png)','','![Representations](figures/06_metadata_fulltext.png)','','![Context](figures/07_matched_native_context.png)','',
        '## Measured cost and hardware','',
        f"Main training throughput is {train['optimizer_steps_per_second']:.4f} optimizer steps/s and {train['query_groups_per_second']:.3f} groups/s. Active optimizer-group throughput is {train['input_tokens_per_active_group_second']:.0f} input tokens/s, excluding validation/mining; total main-wall throughput is {train['input_tokens_per_main_wall_second']:.0f} tokens/s. The selected adapter is {train['adapter_bytes']:,} bytes. Base backbone loading took {architecture['model_loading_s']:.3f} seconds after input hash checks; OS file cache was not evicted.",'']
    resource=resources['main_training']
    header += [f"During main training, sampled whole-device VRAM peaked at {resource['gpu_peak_sampled_mib']:.0f} MiB, CUDA allocated/reserved peaks were {train['cuda_peak_allocated_bytes']/2**20:.1f}/{train['cuda_peak_reserved_bytes']/2**20:.1f} MiB, whole-host used RAM peaked at {resource['host_peak_used_gib']:.3f} GiB and minimum available RAM was {resource['host_min_available_gib']:.3f} GiB. GPU temperature peaked at {resource['gpu_peak_temperature_c']:.0f} °C, mean sampled GPU use was {resource['gpu_utilization_mean_percent']:.1f}%, and host swap-in/out deltas were {resource['host_swap_in_delta_bytes']/2**20:.3f}/{resource['host_swap_out_delta_bytes']/2**20:.3f} MiB. One-second samples can miss brief peaks; host-wide counters are not process-private measurements.",'',
        '| View | Base index wall | LoRA index wall | LoRA new / cached chunks | LoRA median / p95 fresh query |','|---|---:|---:|---:|---:|']
    for c in CONFIGS:
        b=baseperf['configurations'][c];p=performance['configurations'][c];lat=p['uncached_query_total']
        header.append(f"| {SHORT[c]} | {b['index_wall_s']/60:.2f} min | {p['index_wall_s']/60:.2f} min | {p['uncached_calls']:,} / {p['cached_calls']:,} | {lat['median_s']*1000:.2f} / {lat['p95_s']*1000:.2f} ms |")
    header+=['','C index wall is incremental shared-cache work in each model, not a fresh standalone index. Every adapter document/query vector is recomputed for the selected adapter. Fresh query totals include embedding, complete-input validation, atomic vector-cache persistence and CPU search, ending before ranking persistence, as in the base study. Cached-vector search is reported separately in `query_component_comparison.csv`. Electricity is unmeasured and no paid/cloud inference was used.','',
        '![Similarity](figures/09_positive_negative_similarity.png)','','![Training resources](figures/10_training_time_memory.png)','','![Latency](figures/11_query_latency.png)','',
        '## Historical context and limitations','',
        'Archived Qwen3 8B Q4_K_M remains NDCG@10 0.642091 / Recall@10 0.701389 / MRR@10 0.732295; SkillRet 0.6B remains 0.810990 / 0.876442 / 0.856349. Qwen uses 4,096 dimensions, Q4 quantization, a 4K context and a query instruction through Ollama HTTP. BGE uses 1,024 dimensions, an FP16 base and raw queries through PyTorch. SkillRet 0.6B has specialized training, stripped source fields, a 32K cap and two truncated documents. These historical comparisons do not isolate architecture or parameter count. Their missing matched latency remains missing.','',
        'The paired base/LoRA comparison is the primary evidence. Public synthetic questions, incomplete qrels, possible unknown semantic overlap, limited training-derived validation, short training passages versus full 8K inference and adaptive checkpoint selection constrain generalization. Known-positive filtering cannot remove unjudged positives. Repeated training texts receive repeated weight. The small scaling panel has known gold-conditioned membership and position bias. No downloaded skill code was executed.','',
        '![Historical comparison](figures/12_historical_comparison.png)','',
        '## Integrity, failures and reuse','',
        f"All input hashes, exact candidate/source manifests, normalized 1,024-dimensional vectors and independent saved-ranking metrics passed. Main training recorded {train['gradient_overflow_attempts']} gradient-overflow attempts. The initial CPU overlap-audit invocation was interrupted to fix the final block boundary; its corrected rerun completed. Protocol metadata inherited from the baseline was corrected before the first main optimizer step to label the requested base-versus-LoRA contrast; training hyperparameters and evaluation inputs were unchanged. Original and corrected protocol hashes are retained.",'',
        'Best/final adapters and optimizer/scheduler/scaler/random-generator recovery states were saved outside Git. Portable best/final adapter weights are published separately from the base model; no base checkpoint, raw source corpus, credential or personal path belongs in the repository. The lossless vector archive contains all adapted A/B/C documents and shared official/scaling queries. Its replay verifies every ranking and metric without a model, GPU, network request or original cache. Sparse/hybrid/ColBERT extensions were unattempted. All baseline evidence remains unchanged.','']
    if (R/'published_vector_replay.json').exists():
        vm=read(STUDY/'vectors/manifest.json');replay=read(R/'published_vector_replay.json')
        header+=[f"The adapted archive is {vm['bytes']:,} bytes, SHA-256 `{vm['sha256']}`. Model-free replay completed all 480 scaling and 13,176 official query/configuration outcomes in {replay['total_wall_s']:.3f} seconds. See [vectors/README.md](vectors/README.md).",'']
    header+=['References: [BGE-M3](https://huggingface.co/BAAI/bge-m3), [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding), [PEFT dtype guidance](https://huggingface.co/docs/peft/developer_guides/troubleshooting), [immutable baseline](../skillret-bge-m3-fulltext-20261108/REPORT.md).','']
    (STUDY/'REPORT.md').write_text('\n'.join(header));print('Twelve PNG/SVG figures and paired LoRA report generated',flush=True)

if __name__=='__main__':build()
