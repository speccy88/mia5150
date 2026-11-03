"""Regenerate the local Markdown handoff from independently checked native-v2 results."""
import collections
import datetime
import json
from pathlib import Path
from review_local_results import review, ROOT

def pct(value):
    return '-' if value is None else f'{value:.2%}'

def main():
    audit=review('local-native-v2')
    jobs=json.loads((ROOT/'local-models.json').read_text())
    state=json.loads((ROOT/'results/local-native-v2/suite-status.json').read_text(encoding='utf-8-sig'))
    statuses={x['alias']:x for x in state['models']}
    grouped=collections.defaultdict(list)
    for r in audit['summary']:
        grouped[r['model']].append(r)
    lines=['# Skill Discovery — résultats locaux, adaptateur natif corrigé', '',
        'État vérifié au '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'. Lot : `'+state['state']+'`.', '',
        'Le lot comparatif courant est `results/local-native-v2/`. Le lot `local-v1/` est conservé après quatre modèles comme diagnostic : son adaptateur pouvait compter du raisonnement interne dans les rangs. Les scores v1 ne sont pas les résultats comparatifs finaux. Voir `evidence/NATIVE_PARSER_V2.md`.', '',
        '## Couverture des modèles', '',
        '| Modèle demandé | État | Lignes sauvegardées | Réponses mesurées | Top-1 strict / réponses mesurées | Top-10 / réponses mesurées | Sorties sans erreur |',
        '|---|---|---:|---:|---:|---:|---:|']
    scorecard=[]
    for j in jobs:
        rows=grouped.get(j['alias'],[])
        n=sum(r['measured_inference_rows'] for r in rows)
        hits=sum(r['strict_top1_hits'] for r in rows)
        hits10=sum(r['strict_top10_hits'] for r in rows)
        s=statuses.get(j['alias'],{'state':'en attente'})
        label=s['state']+(': '+s['reason'] if s.get('reason') else '')
        score=f'{hits}/{n} ({pct(hits/n)})' if n else '-'
        score10=f'{hits10:g}/{n} ({pct(hits10/n)})' if n else '-'
        lines.append(f"| {j['requested']} | {label} | {sum(r['recorded_rows'] for r in rows)} | {n} | {score} | {score10} | {sum(r['valid_nonerror_rows'] for r in rows)} |")
        errors=collections.Counter()
        for r in rows:
            errors.update(r['errors'])
        scorecard.append({'requested':j['requested'],'alias':j['alias'],'state':label,
            'recorded_rows':sum(r['recorded_rows'] for r in rows),'measured_responses':n,
            'strict_top1_hits':hits,'strict_top10_hits':hits10,
            'top1_rate_measured':hits/n if n else None,'top10_rate_measured':hits10/n if n else None,
            'valid_nonerror_rows':sum(r['valid_nonerror_rows'] for r in rows),'issues':dict(errors)})
    lines[4:4]=[f"Sur les {len(jobs)} modèles demandés, {sum(m['measured_responses']>0 for m in scorecard)} ont fourni des réponses mesurées. Le lot contient {sum(m['recorded_rows'] for m in scorecard)} lignes de 45 colonnes et {sum(m['measured_responses'] for m in scorecard)} réponses avec usage réel. Les blocages et exclusions restent séparés des mesures.", '']
    lines += ['', "Une réponse mesurée possède des tokens réels d'entrée et de sortie. Les OOM, les accès gated, les limites de contexte et les essais non exécutés restent sans taux mesuré; ils ne sont pas interprétés comme des zéros de précision. Un export complet de 140 lignes ne signifie pas 140 appels réussis.", '',
        '## Détail par taille', '',
        '| Modèle | N | Réponses mesurées | Top-1 strict | Rappel Top-10 | Précision Top-1, sorties sans erreur | Sorties au plafond 500 | Tokens entrée réels, min-max | Médiane appel LLM (s) | Issues enregistrées |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in audit['summary']:
        issues=', '.join(f'{k}={v}' for k,v in r['errors'].items()) or 'en attente'
        token_range=f"{r['input_token_min']}-{r['input_token_max']}" if r['input_token_min'] is not None else '-'
        llm_median=f"{r['median_llm_s']:.2f}" if r['median_llm_s'] is not None else '-'
        lines.append(f"| {r['model']} | {r['library_size']} | {r['measured_inference_rows']} | {pct(r['top1_rate_measured'])} | {pct(r['top10_rate_measured'])} | {pct(r['top1_precision_valid'])} | {r['outputs_at_500']} | {token_range} | {llm_median} | {issues} |")
    lines += ['', 'Les taux sur réponses mesurées incluent les réponses sans liste finale ou de mauvais format. La précision sur sorties sans erreur utilise un autre dénominateur. Les erreurs et rangs invalides de la réponse finale ne sont jamais retirés pour promouvoir une bonne réponse.', '',
        '## Conditions communes et limites', '',
        '- 6 006 skills, 5 801 noms distincts, 20 questions exactes; tailles 50, 100, 250, 500, 1 000, 2 000 et 5 000; seed 42; top 10. Aucun corps complet : noms et descriptions limitées à 300 caractères.',
        '- `random.Random(42).sample(all_skills,N)` est réinitialisé pour chaque taille. Copie par question; remplacement du dernier élément si et seulement si le gold ID manque. **Biais intrinsèque vérifié : les 20 gold IDs fournis sont déjà les 20 premiers tirages du seed 42; leurs positions restent 1–20 à toutes les tailles.** Aucun préfixe gold n’est ajouté par le runner. Ce test ne mesure pas des positions uniformément réparties.',
        '- Les 140 plans de candidats et hashes de prompts v1/native-v2 sont identiques. L’adaptateur corrigé ajoute uniquement un état de parsing pour le raisonnement déjà prérempli par le template; il ne change pas le prompt ni le mode thinking. Les blocs internes délimités sont exclus des rangs. Une pensée inachevée au plafond donne `REASONING_NO_FINAL_ANSWER`, avec les tokens réels conservés.',
        '- RTX 2060 Super 8 192 MiB; pilote 610.88; llama.cpp b11960 CUDA 12.4. Les couches transformer et de sortie, le KV f16 et les états récurrents observés sont sur CUDA; contexte effectif 32 768, un slot, `fit=off`, sans context shift, warmup, cache RAM ou checkpoints; batch 512 / microbatch 128. **Portée précise : llama.cpp garde l’entrée/les embeddings dans un buffer CPU_Mapped par défaut**, même avec toutes les couches transformer/sortie sur GPU. Ce profil n’est donc pas un calcul sans participation CPU. Aucun repli de couches transformer pour tenir en VRAM n’est accepté. Les journaux détaillent ces buffers dans `evidence/native-buffer-audit.json`. La conversion/quantification préparatoire emploie aussi le CPU.',
        '- `/apply-template` puis `/tokenize` fournissent le comptage réel; les IDs sont transmis à `/completion`. Température 0, seed 42, sortie plafonnée à 500 tokens. Aucune troncature. La barrière v1 estimée `44.5*N+200 > 90% du contexte` bloque 60 lignes aux tailles 1 000/2 000/5 000 à ce contexte; ce n’est pas un nombre de tokens mesuré.',
        '- **Ces 60 lignes sont un filtre du n_ctx choisi, pas la limite intrinsèque des modèles.** Qwen3.5 publie 262 144 tokens natifs (extension annoncée à 1 010 000); Ling configure 131 072 natifs avec une recette YaRN à 262 144. Le lot charge seulement 32 768, choix uniforme et conservateur sans recherche du maximum VRAM par modèle. Aucun tokenizer/appel ne mesure les tailles écartées. Fenêtres publiées, config éditeur, GGUF et n_ctx observé sont séparés dans `evidence/context-window-audit.json` et `evidence/CONTEXT_WINDOWS.md`.',
        '- Les modèles reasoning peuvent consommer les 500 tokens avant leur liste finale. Ces erreurs de budget/format décrivent cette configuration, pas leur meilleure capacité possible. Les quantifications sont différentes et indiquées par modèle.',
        '- Le multimodal est testé en texte seul. H2O utilise son export officiel stock de chat ordinaire, sans shim de décisions typées. Les versions Instruct/IT sont utilisées quand disponibles; Mistral est explicitement v0.3.',
        '- Les 20 questions forment un petit échantillon en partie lexical; les noms dupliqués empêchent de récupérer certains gold IDs avec une réponse par noms. Le runner original de Vincent n’a pas été fourni à ce lot.',
        '- Le lot OAuth Luna/Sol/Terra utilise les mêmes candidats, mais sa route ne garantit ni température 0 ni sortie plafonnée à 500; il doit rester une variante distincte. Aucune équivalence stricte entre routes n’est revendiquée.',
        '- Les latences sont des mesures dans cet environnement Windows 16 Go de RAM. Téléchargements/conversions CPU ont pu tourner simultanément : ce n’est pas un benchmark de vitesse isolé. Coût de service local 0 USD, hors électricité.', '',
        '## Poids et provenance', '', '| Modèle | Quantification | Source vérifiée |', '|---|---|---|']
    for j in jobs:
        src=j['source']
        if j['requested'].startswith('Qwen3.5-'):
            src='Qwen/'+j['requested']
        elif j['requested']=='Ling-3.0-tiny':
            src='inclusionAI/Ling-3.0-tiny-GGUF'
        quant=j.get('quantization','non téléchargé : accès gated')
        if j['requested'].startswith('Qwen3.5-'):
            quant+='; fichier Ollama installé, digest confirmé par le registre officiel'
        lines.append(f"| {j['requested']} | {quant} | [{src}](https://huggingface.co/{src}) |")
    lines += ['', 'Commits, SHA-256, tailles et licences : `evidence/source-inventory.json`, manifests Ollama vérifiés et fichiers `.source.json` des poids. Ling installé est identique au GGUF officiel. Llama 3.2 3B et Gemma 3 4B demandent une acceptation d’accès manuelle; aucun contournement ni substitut non gated n’a été utilisé.', '',
        '## Preuves, validation et livraison', '',
        '34 tests synthétiques passent, dont sept régressions de parsing natif. `evidence/result-review-native-v2.json` vérifie les 45 colonnes, les clés uniques, les cardinalités, les vrais tokens, l’intégrité SQLite, l’égalité CSV/SQLite et recalcule les métriques depuis les rangs. Les sources transactionnelles sont les SQLite; les CSV sont remplacés atomiquement après chaque ligne.', '',
        'Chaque modèle conserve configuration, manifest, journaux de lancement et preuve GPU liée au PID/heure/poids/exécutable. Les erreurs initiales sont préservées. Les sorties brutes se trouvent dans `private-responses/`, exclu de Git; aucun prompt complet n’y est sauvegardé. Aucun identifiant/token OAuth n’a été copié dans ce lot.', '',
        'Aucun email ni push n’est exécuté depuis ce lot. Les poids, archives, outils, environnements, données sources privées et sorties brutes sont exclus par `.gitignore`. Le parent coordonne l’intégration GitHub et le rapport Markdown commun destiné à Vincent.', '']
    lines += [
        '## Admission par la VRAM physique', '',
        'Sous Windows WDDM, une allocation CUDA peut réussir malgré un budget supérieur à la VRAM physique. Le lanceur contrôle désormais la somme des buffers CUDA modèle + KV + états récurrents + maximum compute avant toute inférence. Un dépassement produit `HARDWARE_VRAM_CAPACITY_EXCEEDED`, distinct d’un OOM CUDA réellement observé; aucun repli de couches ou changement de contexte n’est tenté.', '',
        'Qwen3-8B au profil 32K déclare 9 097,34 MiB pour 8 192 MiB physiques. Ses 21 réponses acquises avant ce contrôle sont préservées dans `results/diagnostic-vram-oversubscription/`, avec l’intention suivante non rejouée. Elles sont exclues du lot comparatif; son registre canonique contient des issues matérielles sans taux mesuré. La validation corrigée du garde-fou a refusé le même chargement sans inférence. Les empreintes et allocations des lancements exacts sont auditées dans `evidence/native-buffer-audit.json`.', '',
        'Mistral-7B-Instruct-v0.3 est également refusé avant génération : 8 230,52 MiB déclarés pour 8 192 MiB physiques. Aucun score nul ne lui est attribué : le taux mesuré reste absent. Les originaux convertis et les allocations de démarrage sont conservés.', '',
        'À partir d’Ornith, `evidence/wddm-memory-telemetry.jsonl` et son résumé enregistrent les compteurs dédiés/partagés du processus vérifié. Un usage partagé non nul peut comprendre des allocations hôte ou du staging pilote; il ne prouve à lui seul ni offload transformer ni paging d’un buffer CUDA précis. Cette collecte ponctuelle ne reconstitue pas la résidence des modèles précédents.', '',
        'La provenance publique regroupée, y compris les conversions locales et les poids Ollama installés, est livrée dans `evidence/model-artifact-provenance.json`; elle reste disponible lorsque les répertoires de poids exclus de Git ne sont pas transférés.', '']
    (ROOT/'LOCAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    progress={'utc':audit['reviewed_utc'],'sampler_audit':'CONFORME; supplied gold/seed42 position bias documented',
              'native_v2_state':state,'native_v2_checks':audit['checks'],'native_v2_summary':audit['summary'],
              'v1_state':'preserved diagnostic after four models; native reasoning contamination demonstrated',
              'raw_response_audit':'evidence/offline-reasoning-audit.json',
              'report':'LOCAL_REPORT.md'}
    (ROOT/'evidence/progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
    (ROOT/'evidence/model-scorecard.json').write_text(json.dumps({'utc':audit['reviewed_utc'],'models':scorecard},indent=2),encoding='utf-8')
    print('Wrote native-v2 LOCAL_REPORT.md')

if __name__=='__main__':
    main()
