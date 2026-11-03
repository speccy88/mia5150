"""Build a Markdown section for the coordinated report; no sending or publishing."""
import collections
import datetime
import json
from pathlib import Path

from review_local_results import review, ROOT

def percent(value):
    return '—' if value is None else f'{value:.1%}'

def main():
    audited=review()
    jobs=json.loads((ROOT/'local-models.json').read_text())
    by_model=collections.defaultdict(list)
    for row in audited['summary']:
        by_model[row['model']].append(row)
    lines=['# Skill Discovery — lot local Windows', '',
        'État au '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'.', '',
        'Ce fichier est une section destinée au rapport commun. Il ne déclenche aucun envoi ni publication.', '',
        '## État des 18 modèles', '',
        '| Modèle demandé | État | Lignes enregistrées | Réponses mesurées | Sorties sans erreur | Top-1 / réponses mesurées |',
        '|---|---|---:|---:|---:|---:|']
    checks={r['model']:r for r in audited['checks']}
    for job in jobs:
        alias=job['alias']
        rows=by_model.get(alias,[])
        measured=sum(r['measured_inference_rows'] for r in rows)
        hit=sum(r['strict_top1_hits'] for r in rows)
        state='en attente'
        if job.get('blocked'):
            state='accès gated : acceptation utilisateur requise'
        elif alias in checks:
            state='export complet' if checks[alias]['complete'] else 'en cours'
            errors=collections.Counter()
            for r in rows:
                errors.update(r['errors'])
            operational={k:v for k,v in errors.items() if k not in ['OK','INVALID_OUTPUT','EMPTY_OUTPUT','CONTEXT_LIMIT_ESTIMATED']}
            if operational:
                state+='; '+', '.join(f'{k}={v}' for k,v in operational.items())
        lines.append(f"| {job['requested']} | {state} | {sum(r['recorded_rows'] for r in rows)} | {measured} | {sum(r['valid_nonerror_rows'] for r in rows)} | {hit}/{measured} ({percent(hit/measured if measured else None)}) |")
    lines += ['', 'Un export de 140 lignes signifie que les issues ont été enregistrées, pas que 140 appels ont réussi. Les erreurs matérielles, les accès bloqués et les dépassements de contexte ne constituent pas des observations de précision nulle.', '',
        '## Résultats par taille', '',
        '| Modèle | N | Réponses mesurées | Top-1 exact | Top-10 exact | Précision Top-1, sorties sans erreur | Erreurs/limites |',
        '|---|---:|---:|---:|---:|---:|---|']
    for r in audited['summary']:
        counts=', '.join(f'{k}: {v}' for k,v in r['errors'].items()) or 'en attente'
        lines.append(f"| {r['model']} | {r['library_size']} | {r['measured_inference_rows']} | {percent(r['top1_rate_measured'])} | {percent(r['top10_rate_measured'])} | {percent(r['top1_precision_valid'])} | {counts} |")
    lines += ['', 'Les taux mesurés utilisent exclusivement les lignes portant des tokens réels d’entrée et de sortie. La précision sur les sorties sans erreur a un dénominateur distinct. Les rangs invalides ou répétés restent à leur position; ils ne sont jamais retirés pour promouvoir une réponse.', '',
        '## Protocole et limites qui changent l’interprétation', '',
        '- Données fournies : 6 006 skills, 5 801 noms distincts, 20 questions exactes. Sept tailles : 50, 100, 250, 500, 1 000, 2 000, 5 000. Aucun dédoublonnage.',
        '- Chaque taille réinitialise `random.Random(42).sample(all_skills, N)`. Chaque question garde cette bibliothèque, avec remplacement du dernier élément uniquement si son gold ID est absent. Les prompts utilisent seulement le nom et la description tronquée à 300 caractères; aucun corps complet.',
        '- **Biais de position vérifié :** les 20 gold IDs fournis sont déjà les 20 premiers éléments de cet échantillonnage de seed 42. Ils occupent donc les positions 1–20 à toutes les tailles; aucun remplacement final ne s’applique sur ces données. Ce n’est pas un préfixage ajouté au runner. Les plans locaux et Luna/Sol/Terra examinés sont identiques. Ce lot ne mesure pas une recherche à positions uniformément réparties. Toute variante corrigée devra avoir une version et un run distincts.',
        '- Serveur local demandé et vérifié quand chargé : contexte réel 32 768, un slot, toutes les couches et le KV f16 sur CUDA, `fit=off`, sans context shift, sans warmup, sans cache RAM ni checkpoints. Batch 512 / microbatch 128. CPU réservé à la tokenisation/au contrôle et à la préparation des fichiers; aucune inférence CPU ni offload partiel.',
        '- Tokenisation réelle par `/apply-template` puis `/tokenize`, passage des IDs à `/completion`, température 0, seed 42, plafond 500 tokens de sortie; aucune troncature admise. La barrière v1 estimée `44.5*N+200 > 90% du contexte` bloque les tailles 1 000/2 000/5 000 à ce contexte, soit 60 lignes par modèle. Cette estimation n’est pas un comptage de tokens.',
        '- Le mode thinking par défaut du modèle est conservé. Un plafond de 500 tokens peut être consommé par le raisonnement avant la liste attendue; les sorties invalides et le nombre de réponses au plafond restent explicitement visibles dans les preuves.',
        '- Les quantifications diffèrent selon les artefacts disponibles. Les résultats comparent ces variantes précises, pas seulement des nombres de paramètres.',
        '- La variante OAuth hôte n’impose ni température 0 ni plafond de sortie 500. Même avec des candidats identiques, ses paramètres ne sont donc pas strictement équivalents au local. Le runner original de Vincent n’a pas été fourni à ce lot; aucune reproduction octet pour octet de son exécution n’est revendiquée.',
        '- Les 20 questions constituent un petit échantillon, en partie lexical; les noms dupliqués empêchent de résoudre exactement certains IDs dans le format de réponse par noms.',
        '- Les temps sont ceux des appels dans cet environnement Windows 16 Go de RAM. Des téléchargements, installations et conversions CPU préparatoires ont eu lieu en parallèle de certains modèles; ces latences ne constituent pas un essai de vitesse isolé. Coût de service local 0 USD, hors électricité.', '',
        '## Artefacts et identités', '',
        '| Modèle | Quantification / provenance | Source éditeur vérifiée |',
        '|---|---|---|']
    for j in jobs:
        source=j['source']
        # Resolve the known official identity; installed quantizations remain labelled as such.
        if j['requested'].startswith('Qwen3.5-'):
            source='Qwen/'+j['requested']
        elif j['requested']=='Ling-3.0-tiny':
            source='inclusionAI/Ling-3.0-tiny-GGUF'
        label=j.get('quantization','non téléchargé : accès gated')
        if j['requested'].startswith('Qwen3.5-'):
            label+='; poids Ollama déjà présents, pas un GGUF publié par Qwen'
        lines.append(f"| {j['requested']} | {label} | [{source}](https://huggingface.co/{source}) |")
    lines += ['', 'SHA-256, commits, tailles et licences sont dans `evidence/source-inventory.json` et les fichiers `.source.json` des poids. Ling installé correspond exactement au SHA-256 du fichier officiel. Phi, Mistral v0.3 et Nanbeige sont convertis/quantifiés depuis les sources officielles épinglées. H2O utilise son export officiel stock de chat ordinaire; le prompt ne sollicite pas son API de décisions typées. Le multimodal est testé en texte seul, sans projecteur chargé.', '',
        '## Validation et incidents', '',
        'Les 27 tests synthétiques du moteur ont passé avant les appels. `review_local_results.py` recompute les métriques depuis les rangs et vérifie largeur du CSV, clés uniques, cardinalités par taille, cohérence tokens, intégrité SQLite, égalité CSV/SQLite et plans de candidats/prompt. Sa portée et son horodatage sont enregistrés dans `evidence/result-review.json`.', '',
        'Incident Qwen3-1.7B : 66 réponses mesurées; OOM CUDA dans la prélecture de la requête 67 (`500:6`), puis 13 lignes `GPU_SERVER_NOT_RUNNING` et 60 blocs estimés de contexte. Le CSV initial est conservé, aucun appel rejoué; voir `evidence/qwen17-runtime-incident.json` et le journal CUDA. Ces 14 erreurs matérielles/runtime sont exclues des taux mesurés.', '',
        'Les journaux de lancement, preuves CUDA, SQLite, CSV, manifests et empreintes restent locaux. Aucun token OAuth n’a été copié dans ce lot. Aucun email ni push n’a été effectué depuis ce lot. Les poids, les données sources originales, les environnements et les téléchargements sont exclus du dépôt par `.gitignore`; l’intégration et l’envoi du rapport commun relèvent de l’orchestrateur.', '']
    (ROOT/'LOCAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Wrote LOCAL_REPORT.md')

if __name__=='__main__':
    main()
