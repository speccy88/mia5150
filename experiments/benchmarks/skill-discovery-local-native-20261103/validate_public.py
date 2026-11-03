"""Offline CSV/SQLite/prompt validation and descriptive cross-route comparison."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics

import bench_native_v2 as bench

ROOT=Path(__file__).resolve().parent
OAUTH=ROOT.parent/'skill-discovery-oauth-v1-20261103'


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def truth(v):
    return str(v).lower()=='true'


def read_csv(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f)
        assert reader.fieldnames==bench.COLUMNS
        return list(reader)


def main():
    skills,queries=bench.load_data(ROOT/'data/test.jsonl',ROOT/'data/test_queries.json')
    assert len(skills)==6006 and len(queries)==20
    plan=bench.make_plan(skills,queries)
    expected={t['key']:t for t in plan}
    checks=[]
    groups=[]
    coverage=[]
    canonical_totals=collections.Counter()
    for suite in ['local-native-v2','local-v1','diagnostic-vram-oversubscription']:
        for folder in sorted((ROOT/'results'/suite).iterdir()):
            if not folder.is_dir() or not (folder/'details.csv').exists():
                continue
            rows=read_csv(folder/'details.csv')
            assert len({(r['library_size'],r['query']) for r in rows})==len(rows)
            db=sqlite3.connect((folder/'state.sqlite3').as_uri()+'?mode=ro&immutable=1',uri=True)
            assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
            stored={key:json.loads(v) for key,v in db.execute('SELECT key,row_json FROM trials WHERE row_json IS NOT NULL')}
            manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
            assert len(manifest['plan'])==140
            for p in manifest['plan']:
                t=expected[p['key']]
                assert p['candidate_ids']==[s['id'] for s in t['candidates']]
                assert p['prompt_sha256']==t['prompt_sha256']
            for row in rows:
                matching=[k for k,t in expected.items() if t['n']==int(row['library_size']) and t['query']['query']==row['query']]
                assert len(matching)==1
                key=matching[0]
                assert all(str(stored[key][c])==row[c] for c in bench.COLUMNS),(folder.name,key,'persistence')
                rank=int(row['rank_of_gold'])
                assert 0<=rank<=10 and truth(row['exact_match'])==(rank==1)
                for k in [1,3,5,10]:
                    hit=0<rank<=k
                    values={'recall':float(hit),'precision':1/k if hit else 0,
                            'ndcg':1/math.log2(rank+1) if hit else 0,'mrr':1/rank if hit else 0,'map':1/rank if hit else 0}
                    for metric,value in values.items():
                        assert math.isclose(float(row[f'{metric}@{k}']),value,abs_tol=1e-12)
                for c in ['latency_s','llm_latency_s','retrieval_latency_s','e2e_latency_s']:
                    assert math.isfinite(float(row[c])) and float(row[c])>=0
                if row['input_tokens']:
                    assert int(row['total_tokens'])==int(row['input_tokens'])+int(row['output_tokens'])
                    assert int(row['output_tokens'])<=500
                    assert int(row['input_tokens'])+500<=manifest['config']['context_window']
                if truth(row['context_limit']):
                    assert row['error'].startswith('CONTEXT_LIMIT')
                assert row['gold_skill']==expected[key]['query']['gold_skill']
            intents=db.execute('SELECT count(*),sum(row_json IS NULL) FROM trials').fetchone()
            db.close()
            checks.append({'suite':suite,'model':folder.name,'rows':len(rows),'schema_columns':45,
                           'csv_matches_exported_sqlite':True,'metrics_recomputed':True,
                           'all_140_candidate_and_prompt_hashes_match':True,'trial_intents':intents[0],
                           'unresolved_intents':intents[1] or 0})
            if suite!='local-native-v2':
                continue
            assert len(rows)==140
            canonical_totals.update(r['error'] or 'SUCCESS' for r in rows)
            measured=[r for r in rows if r['input_tokens'] and r['output_tokens']]
            groups.append({'route':'local','model':folder.name,'rows':rows,'measured':measured})
    for folder in sorted(OAUTH.joinpath('results').iterdir()):
        if folder.is_dir() and (folder/'details.csv').exists():
            rows=read_csv(folder/'details.csv')
            groups.append({'route':'OAuth','model':folder.name,'rows':rows,'measured':[r for r in rows if r['input_tokens'] and r['output_tokens']]})
    common_sizes=[50,100,250,500]
    summary=[]
    import numpy as np
    indices=np.random.default_rng(20261103).integers(0,20,size=(10000,20))
    for group in groups:
        measured=group['measured']
        common=[r for r in measured if int(r['library_size']) in common_sizes]
        ci=None
        if common:
            assert len(common)==80
            lookup={(int(r['library_size']),r['query']):r for r in common}
            matrix=np.array([[float(truth(lookup[(n,q['query'])]['exact_match'])) for n in common_sizes] for q in queries])
            ci=[float(x) for x in np.quantile(matrix[indices].mean(axis=(1,2)),[.025,.975])]
        summary.append({'route':group['route'],'model':group['model'],'measured_common_sizes':len(common),
            'correct_top1':sum(truth(r['exact_match']) for r in common),
            'top1_measured':sum(truth(r['exact_match']) for r in common)/len(common) if common else None,
            'query_cluster_percentile95':ci,'unique_question_clusters':20 if common else 0,
            'valid_nonerror_common':sum(not r['error'] for r in common),
            'median_llm_s_common':statistics.median(float(r['llm_latency_s']) for r in common) if common else None,
            'input_tokens_common':sum(int(r['input_tokens']) for r in common) if common else None,
            'output_tokens_common':sum(int(r['output_tokens']) for r in common) if common else None})
        for n in bench.SIZES:
            rows=[r for r in group['rows'] if int(r['library_size'])==n]
            actual=[r for r in measured if int(r['library_size'])==n]
            errors=collections.Counter(r['error'] or 'SUCCESS' for r in rows)
            coverage.append({'route':group['route'],'model':group['model'],'N':n,'planned':20,
                'measured_with_usage':len(actual),'valid_nonerror':sum(not r['error'] for r in actual),
                'top1_measured':sum(truth(r['exact_match']) for r in actual)/len(actual) if actual else None,
                'errors':dict(errors),'median_llm_s':statistics.median(float(r['llm_latency_s']) for r in actual) if actual else None})
    assert len([g for g in groups if g['route']=='local' and g['measured']])==14
    assert len([g for g in groups if g['route']=='local' and not g['measured']])==4
    assert sum(len(g['measured']) for g in groups if g['route']=='local')==1120
    assert canonical_totals=={'SUCCESS':542,'REASONING_NO_FINAL_ANSWER':532,'INVALID_OUTPUT':42,
        'MALFORMED_REASONING_BOUNDARY':4,'ACCESS_GATED_ACCEPTANCE_REQUIRED':160,
        'HARDWARE_VRAM_CAPACITY_EXCEEDED':160,'CONTEXT_LIMIT_ESTIMATED':1080}
    result={'offline_only':True,'new_inference_calls':0,'checks':checks,'canonical_error_counts':dict(canonical_totals),
            'common_sizes':common_sizes,'common_summary':summary,'coverage':coverage,
            'uncertainty':'10000 paired question-cluster resamples, seed 20261103; descriptive, conditional on biased fixed lists; no iid n=80 claim.',
            'full_raw_response_normalization_replayed':False}
    save(ROOT/'validation/public-validation.json',result)
    with (ROOT/'validation/common-panel-summary.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]))
        writer.writeheader();writer.writerows(summary)
    plot(coverage)
    report(summary)
    print(json.dumps({'public_sqlite_checked':len(checks),'local_models_measured':14,'local_models_blocked':4,
                      'local_measured_rows':1120,'local_csv_rows':2520,'oauth_measured_rows':360,
                      'all_140_logical_prompts_match':True,'new_inference_calls':0},indent=2))


def plot(coverage):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    profile_file=ROOT/'profiles-v3/validation/public-validation.json'
    coverage=list(coverage)
    if profile_file.exists():
        profiles=json.loads(profile_file.read_text(encoding='utf-8'))['summary']
        for model in sorted({s['model'] for s in profiles}):
            for kv in ['f16','q8_0']:
                series=[s for s in profiles if s['model']==model and s['kv']==kv]
                for n in bench.SIZES:
                    s=next((x for x in series if x['N']==n),None)
                    coverage.append({'route':'v3 '+kv,'model':model,'N':n,'planned':20 if s else 0,
                        'measured_with_usage':20 if s else 0,'top1_measured':s['top1']/20 if s else None,'errors':{}})
    route_order={'OAuth':0,'local':1,'v3 f16':2,'v3 q8_0':3}
    order=sorted({(r['route'],r['model']) for r in coverage},key=lambda x:(route_order[x[0]],x[1]))
    grid=np.full((len(order),len(bench.SIZES)),np.nan)
    fig,(left,right)=plt.subplots(1,2,figsize=(16,13.0),gridspec_kw={'width_ratios':[1.15,1]},layout='constrained')
    cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#edf0f2')
    for i,(route,model) in enumerate(order):
        for j,n in enumerate(bench.SIZES):
            r=next(x for x in coverage if x['route']==route and x['model']==model and x['N']==n)
            if r['top1_measured'] is not None:
                grid[i,j]=r['top1_measured']
    image=left.imshow(np.ma.masked_invalid(grid),vmin=0,vmax=1,cmap=cmap,aspect='auto')
    right.imshow(np.zeros_like(grid),vmin=0,vmax=1,cmap=plt.get_cmap('Greys'),aspect='auto',alpha=.08)
    for i,(route,model) in enumerate(order):
        for j,n in enumerate(bench.SIZES):
            r=next(x for x in coverage if x['route']==route and x['model']==model and x['N']==n)
            val=r['top1_measured']
            left.text(j,i,f'{val:.0%}' if val is not None else '—',ha='center',va='center',fontsize=9,
                      color='white' if val is not None and val<.55 else '#172331')
            errors=r['errors']
            if r['measured_with_usage']:
                text=f"{r['measured_with_usage']}/20"
            elif r['planned']==0:
                text='NP'
            elif 'CONTEXT_LIMIT_ESTIMATED' in errors:
                text='0/20 C'
            elif 'NOT_RUN_AFTER_MODEL_STOP' in errors:
                text='0/20 H'
            else:
                text='0/20 B'
            right.text(j,i,text,ha='center',va='center',fontsize=8.5)
    labels=[({'OAuth':'OAuth','local':'Local 32K','v3 f16':'V3 f16 ajusté','v3 q8_0':'V3 KV Q8'}[route]+' · ')+model for route,model in order]
    left.set_yticks(range(len(order)),labels=labels,fontsize=9)
    right.set_yticks(range(len(order)),labels=['']*len(order))
    for ax in (left,right):
        ax.set_xticks(range(len(bench.SIZES)),labels=[str(n) for n in bench.SIZES])
        ax.set_xlabel('Nombre de skills (IDs)')
        ax.set_xticks(np.arange(-.5,len(bench.SIZES),1),minor=True)
        ax.set_yticks(np.arange(-.5,len(order),1),minor=True)
        ax.grid(which='minor',color='white',linewidth=1)
        ax.tick_params(which='minor',bottom=False,left=False)
        ax.axhline(2.5,color='#405061',linewidth=2)
        if profile_file.exists():
            ax.axhline(20.5,color='#405061',linewidth=2)
    left.set_title('Top-1 exact / réponses réellement mesurées',fontsize=12)
    right.set_title('Couverture : réponses mesurées / 20 prévues',fontsize=12)
    bar=fig.colorbar(image,ax=left,location='bottom',shrink=.75,pad=.045)
    bar.set_ticks([0,.25,.5,.75,1],labels=['0 %','25 %','50 %','75 %','100 %'])
    fig.suptitle('Skill discovery, clôture · séries OAuth, local 32K et v3 contexte/KV distinctes\n20 mêmes questions répétées ; golds toujours aux positions 1–20',fontsize=14)
    fig.supxlabel('C : exclusion estimée · B : accès gated ou VRAM · H : 1 rejet HTTP + 19 non exécutées · NP : non prévu dans cette série\nLe zéro de Nanbeige est mesuré. Les cases grises sans score ne sont pas des échecs mesurés. Les contrôles de contexte/sortie diffèrent.',fontsize=10)
    (ROOT/'figures').mkdir(exist_ok=True)
    fig.savefig(ROOT/'figures/coverage-comparison.png',dpi=165)
    plt.close(fig)


def report(summary):
    lines=["# Comparaison descriptive : bibliothèque identique, conditions d'inférence distinctes",'',
        "Les nouvelles copies publiques préservent les réponses mesurées. Cette comparaison ne valide pas une équivalence API/OAuth/local : les locaux sont quantifiés, limités à 32K et 500 tokens de sortie, avec un adaptateur de réponse finale natif ; OAuth omet température/plafond de sortie et rapporte un raisonnement medium. Aucun nouveau modèle n'a été appelé.",'',
        '![Qualité mesurée et couverture](figures/coverage-comparison.png)','',
        'Le graphique utilise les mêmes 20 questions à chaque taille. Chaque zéro coloré résulte de réponses effectivement mesurées ; les cases sans score signalent une exclusion, un blocage ou une taille non exécutée. Les erreurs de format et raisonnements sans réponse finale restent dans le dénominateur des sorties locales mesurées.','',
        '## Panneau commun N = 50, 100, 250, 500','',
        "Les scores ci-dessous utilisent 80 mesures par modèle exécuté, issues de 20 questions répétées sur quatre tailles. Les intervalles rééchantillonnent 20 blocs question avec leurs quatre tailles, 10 000 fois, graine 20261103. Ils sont descriptifs et conditionnels aux listes biaisées, sans interprétation comme 80 essais indépendants. Aucun intervalle n'est fourni aux modèles bloqués.",'',
        '| Voie | Modèle | Mesures | Top-1 exact | Bootstrap par question, 95 % | Sorties sans erreur | Médiane LLM (s) |',
        '| --- | --- | ---: | ---: | --- | ---: | ---: |']
    for s in summary:
        n=s['measured_common_sizes'];ci=s['query_cluster_percentile95']
        rate=f"{s['correct_top1']}/{n} ({s['top1_measured']:.1%})" if n else 'Non mesuré'
        interval=f'{ci[0]:.1%} à {ci[1]:.1%}' if ci else '—'
        latency=f"{s['median_llm_s_common']:.3f}" if n else '—'
        lines.append(f"| {s['route']} | {s['model']} | {n} | {rate} | {interval} | {s['valid_nonerror_common']} | {latency} |")
    lines += ['', '## Limites et preuves','',
        '- **Biais de position :** tous les golds sont aux positions 1–20, avec la même graine que la sélection des questions. Les différences ne généralisent pas à des positions aléatoires ou centrales.',
        '- **Couverture locale :** 14 modèles exécutés, 1 120 réponses avec usage réel ; quatre modèles sans inférence. Les 1 080 lignes de contexte estimé ne mesurent pas la fenêtre native maximale de chaque modèle.',
        '- **Couverture OAuth :** 360 réponses mesurées à N ≤ 2000 ; N=5000 compte un rejet de longueur de chaîne HTTP et 19 non-exécutions par modèle, pas une limite de contexte en tokens établie.',
        '- **Nanbeige :** ses 80 sorties mesurées restent sans réponse finale dans le budget 500 ; son zéro ne signifie pas une incapacité sémantique générale.',
        '- **Matériel local :** transformer, sortie et KV sur CUDA, mais poids d’entrée/embeddings CPU_Mapped et préparation CPU conservés par llama.cpp b11960. Ce profil ne prouve pas une absence totale de calcul CPU. Voir [la portée matérielle](evidence/NATIVE_GPU_SCOPE.md).',
        '- **Latence :** les temps locaux et HTTP OAuth incluent des travaux et longueurs de sortie différents. Aucun ratio de vitesse matérielle n’est revendiqué.',
        '- **Définition native-v2 :** correction du parsing du raisonnement ; le protocole scientifique biaisé v1 reste inchangé. La v2 scientifique proposée dans le rapport OAuth n’a pas été exécutée.',
        '- **Comparabilité externe :** la concordance avec le runner cloud de référence reste non vérifiée ; les noms dupliqués et indices lexicaux des questions demeurent.',
        '', '[Rapport local](LOCAL_REPORT.md) · [Rapport OAuth](../skill-discovery-oauth-v1-20261103/REPORT.md) · [Validation hors ligne](validation/public-validation.json) · [Résumé du panneau commun](validation/common-panel-summary.csv)', '',
        'Les CSV sont inchangés ; les bases distribuées sont des exports d’audit non reprisables. Les sorties HTTP privées complètes ne sont pas redistribuées, donc les vérifications publiques recalculent rangs/métriques/persistance sans rejouer chaque normalisation depuis le texte brut. Voir les transformations et empreintes pour les adaptations de confidentialité.']
    if (ROOT/'PROFILE_COMPLEMENT.md').exists():
        lines += ['',(ROOT/'PROFILE_COMPLEMENT.md').read_text(encoding='utf-8')]
    (ROOT/'COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
