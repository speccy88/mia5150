"""Write a proposal after all accessible models are resolved. Starts no runtime."""
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    state = json.loads((ROOT/'results/local-native-v2/suite-status.json').read_text(encoding='utf-8-sig'))
    if state['state'] != 'completed':
        raise SystemExit('Proposal is finalized only after completion of the existing suite.')
    jobs = json.loads((ROOT/'local-models.json').read_text())
    contexts = {m['model']: m for m in json.loads((ROOT/'evidence/context-window-audit.json').read_text())['models']}
    allocations = {m['alias']: m for m in json.loads((ROOT/'evidence/native-buffer-audit.json').read_text())['models']}
    statuses = {m['alias']: m for m in state['models']}
    lines = ['# Proposition de mesures des tailles 1 000 et plus', '',
        'Préparée le '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'. **Proposition seulement : aucun appel, chargement de contexte supérieur ou téléchargement nouveau n’est lancé par ce document. Le parent décide de sa réalisation.**', '',
        'Le lot 32K existant reste inchangé. Ses 60 exclusions par modèle viennent du filtre estimé, sans tokenisation des tailles 1 000/2 000/5 000. Elles ne démontrent ni une limite native ni le maximum de VRAM utilisable.', '',
        '## Étapes proposées', '',
        '1. Dans un nouveau dossier/version, préparer les mêmes 20 prompts pour chaque taille avec les mêmes candidats, seed et descriptions. Utiliser le vrai template et le vrai tokenizer du même runtime pour compter chaque entrée, sans génération. Cette opération peut utiliser les endpoints du chargement 32K admissible : tokeniser une chaîne ne nécessite pas un cache KV de sa longueur. Pour les modèles déjà refusés au chargement 32K, étudier seulement un outil de tokenisation vocabulaire sans exécution des couches; ne pas forcer un chargement GPU surdimensionné.',
        '2. Choisir le plus petit contexte dans la fenêtre officielle qui couvre les entrées réelles plus 500 tokens de sortie. Le filtre approximatif 44,5*N+200 est remplacé par cette barrière exacte dans cette nouvelle version. Une taille peut donc éventuellement tenir dans 32K ou 40 960 après mesure; ne pas présumer 64K nécessaire.',
        '3. Charger un modèle à la fois. Conserver quantification, cache f16, couches transformer/sortie CUDA, entrée CPU_Mapped par défaut, température 0, seed 42, plafond 500, un slot et fit=off. Refuser avant génération toute somme CUDA déclarée dépassant 8 192 MiB; garder une marge pour le pilote et les allocations transitoires. Vérifier également la consommation dédiée observée et tout message WDDM, sans considérer cudaMalloc réussi comme preuve de résidence.',
        '4. Commencer par N=1 000, puis N=2 000 si le budget natif et matériel l’autorise. N=5 000 est une phase séparée après préflight. Sauvegarder chaque réponse et intention transactionnellement; une erreur matérielle arrête le modèle, aucune reprise automatique d’une requête incertaine.', '',
        'Les paliers indicatifs 65 536 / 131 072 / 262 144 correspondent aux estimations historiques 44 700 / 89 200 / 222 700, qui ne sont pas des comptages réels. LFM3 est limité ici à sa config texte 128 000; les paliers sont ajustés à cette fenêtre.', '',
        '## Examen par modèle', '',
        '| Modèle | Fenêtre publiée/config texte/GGUF | CUDA déclaré à 32K (MiB) | Projection f16 à 64K / 128K / 256K (MiB) | Proposition initiale |',
        '|---|---|---:|---|---|']
    for job in jobs:
        ctx = contexts[job['requested']]
        a = allocations.get(job['alias'], {})
        status = statuses[job['alias']]
        published = ctx.get('published_statement') or {}
        limits = ctx.get('released_config_limits', {})
        native = published.get('native') or limits.get('text_config.max_position_embeddings') or limits.get('max_position_embeddings')
        if native is None:
            native = next(iter(ctx.get('gguf_context_metadata', {}).values()), None)
        window = str(native) if native else 'non inspectée : accès gated'
        if job['requested'] in ('Qwen3-0.6B', 'Qwen3-1.7B'):
            window = '32 768 annoncé; config/GGUF 40 960'
        if published.get('extension'):
            window += '; extension '+str(published['extension'])+' distincte'
        buffers = a.get('cuda_buffer_records', [])
        declared = a.get('declared_cuda_buffers_mib')
        kv = sum(b['mib'] for b in buffers if b['kind'].upper() == 'KV')
        projection = []
        if declared is not None and kv:
            for target in (65536, 131072, 262144):
                if native and target > native:
                    projection.append('hors fenêtre')
                else:
                    estimate = declared + kv * (target / 32768 - 1)
                    projection.append(f'{estimate:.0f}'+(' >8 Go' if estimate > 8192 else ''))
        else:
            projection = ['non disponible']
        if job.get('blocked'):
            recommendation = 'Attendre acceptation officielle et accès authentifié; aucun substitut.'
        elif a.get('physical_capacity_exceeded'):
            recommendation = 'Profil f16 32K déjà trop grand; aucun contexte supérieur avec ce profil.'
        elif status['state'] != 'completed':
            recommendation = 'Résoudre le blocage runtime/matériel avant toute mesure supérieure.'
        elif job['requested'] in ('Qwen3-0.6B', 'Qwen3-1.7B'):
            recommendation = 'Tokeniser N=1 000; vérifier 32K/40 960. Extension supérieure séparée, pas implicite.'
        elif job['requested'] == 'Mistral 7B Instruct':
            recommendation = 'Tokeniser N=1 000 face à 32K; pas de dépassement natif implicite.'
        else:
            viable=[]
            for target, label in ((65536, '1 000'), (min(131072, native or 131072), '2 000')):
                if native and target <= native and declared is not None and kv and declared + kv*(target/32768-1) < 8192:
                    viable.append(label)
            recommendation = 'Préflight N='+', puis '.join(viable)+'; confirmer allocation réelle avant appels.' if viable else 'Préflight tokenizer à 32K; aucun palier supérieur retenu par la projection f16.'
        lines.append(f"| {job['requested']} | {window} | {declared if declared is not None else '-'} | {' / '.join(projection)} | {recommendation} |")
    lines += ['',
        'Projection = allocation CUDA observée à 32K + KV observé × (contexte futur / 32 768 − 1). Il s’agit d’une approximation de planification, pas d’une mesure, d’une garantie d’admission ou d’un maximum VRAM. Les caches à fenêtre glissante, états récurrents, graphes compute et allocations transitoires peuvent évoluer autrement. Aucun contexte supérieur n’a été essayé dans ce lot.', '',
        '## Profils à soumettre séparément', '',
        '- Quantifier le cache KV, réduire le contexte pour Qwen3-8B ou Mistral, changer la quantification des poids, désactiver le thinking ou augmenter le plafond de génération modifierait le profil. Chacun exige un nouveau dossier et une comparaison explicitement séparée.',
        '- Les extensions YaRN/RoPE, dont Ling au-delà de 131 072, nécessitent la recette officielle et une validation distincte. Une grande fenêtre publiée ne garantit pas qu’elle tienne dans 8 Go.',
        '- Garder le sampler actuel préserve la comparaison, mais les 20 gold restent aux positions 1–20. Une mesure de recherche à positions réparties demanderait un second protocole approuvé; elle ne doit pas remplacer ni être mélangée à ce lot.', '',
        'Sources inspectables : `evidence/context-window-audit.json`, `evidence/native-buffer-audit.json`, `evidence/model-artifact-provenance.json` et les snapshots publics épinglés sous `evidence/sources/`. Les limites texte ont priorité sur les champs vision/audio secondaires.', '']
    (ROOT/'HIGH_CONTEXT_PROPOSAL.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Wrote proposal; no model action performed.')


if __name__ == '__main__':
    main()
