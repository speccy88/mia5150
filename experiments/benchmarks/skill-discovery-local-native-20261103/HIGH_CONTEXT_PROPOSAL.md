# Proposition de mesures des tailles 1 000 et plus

Préparée le 2026-11-03T09:52:40.072379+00:00. **Proposition seulement : aucun appel, chargement de contexte supérieur ou téléchargement nouveau n’est lancé par ce document. Le parent décide de sa réalisation.**

Le lot 32K existant reste inchangé. Ses 60 exclusions par modèle viennent du filtre estimé, sans tokenisation des tailles 1 000/2 000/5 000. Elles ne démontrent ni une limite native ni le maximum de VRAM utilisable.

## Étapes proposées

1. Dans un nouveau dossier/version, préparer les mêmes 20 prompts pour chaque taille avec les mêmes candidats, seed et descriptions. Utiliser le vrai template et le vrai tokenizer du même runtime pour compter chaque entrée, sans génération. Cette opération peut utiliser les endpoints du chargement 32K admissible : tokeniser une chaîne ne nécessite pas un cache KV de sa longueur. Pour les modèles déjà refusés au chargement 32K, étudier seulement un outil de tokenisation vocabulaire sans exécution des couches; ne pas forcer un chargement GPU surdimensionné.
2. Choisir le plus petit contexte dans la fenêtre officielle qui couvre les entrées réelles plus 500 tokens de sortie. Le filtre approximatif 44,5*N+200 est remplacé par cette barrière exacte dans cette nouvelle version. Une taille peut donc éventuellement tenir dans 32K ou 40 960 après mesure; ne pas présumer 64K nécessaire.
3. Charger un modèle à la fois. Conserver quantification, cache f16, couches transformer/sortie CUDA, entrée CPU_Mapped par défaut, température 0, seed 42, plafond 500, un slot et fit=off. Refuser avant génération toute somme CUDA déclarée dépassant 8 192 MiB; garder une marge pour le pilote et les allocations transitoires. Vérifier également la consommation dédiée observée et tout message WDDM, sans considérer cudaMalloc réussi comme preuve de résidence.
4. Commencer par N=1 000, puis N=2 000 si le budget natif et matériel l’autorise. N=5 000 est une phase séparée après préflight. Sauvegarder chaque réponse et intention transactionnellement; une erreur matérielle arrête le modèle, aucune reprise automatique d’une requête incertaine.

Les paliers indicatifs 65 536 / 131 072 / 262 144 correspondent aux estimations historiques 44 700 / 89 200 / 222 700, qui ne sont pas des comptages réels. LFM3 est limité ici à sa config texte 128 000; les paliers sont ajustés à cette fenêtre.

## Examen par modèle

| Modèle | Fenêtre publiée/config texte/GGUF | CUDA déclaré à 32K (MiB) | Projection f16 à 64K / 128K / 256K (MiB) | Proposition initiale |
|---|---|---:|---|---|
| Qwen3-0.6B | 32 768 annoncé; config/GGUF 40 960 | 4202.65 | hors fenêtre / hors fenêtre / hors fenêtre | Tokeniser N=1 000; vérifier 32K/40 960. Extension supérieure séparée, pas implicite. |
| Qwen3-1.7B | 32 768 annoncé; config/GGUF 40 960 | 5348.77 | hors fenêtre / hors fenêtre / hors fenêtre | Tokeniser N=1 000; vérifier 32K/40 960. Extension supérieure séparée, pas implicite. |
| Qwen3.5-4B | 262144; extension 1010000 distincte | 3616.2 | 4640 / 6688 / 10784 >8 Go | Préflight N=1 000, puis 2 000; confirmer allocation réelle avant appels. |
| Ling-3.0-tiny | 131072; extension 262144 distincte | 4728.97 | 4945 / 5377 / hors fenêtre | Préflight N=1 000, puis 2 000; confirmer allocation réelle avant appels. |
| Qwen3.5-9B | 262144; extension 1010000 distincte | 5920.51 | 6945 / 8993 >8 Go / 13089 >8 Go | Préflight N=1 000; confirmer allocation réelle avant appels. |
| Llama 3.2 3B | non inspectée : accès gated | - | non disponible | Attendre acceptation officielle et accès authentifié; aucun substitut. |
| Gemma 3 4B | non inspectée : accès gated | - | non disponible | Attendre acceptation officielle et accès authentifié; aucun substitut. |
| H2O-Lightning-4B | 262144 | 3673.62 | 4698 / 6746 / 10842 >8 Go | Préflight N=1 000, puis 2 000; confirmer allocation réelle avant appels. |
| Gemma 4 E2B | 131072 | 1580.72 | 1782 / 2184 / hors fenêtre | Préflight N=1 000, puis 2 000; confirmer allocation réelle avant appels. |
| Qwen3-8B | 40960 | 9097.34 | hors fenêtre / hors fenêtre / hors fenêtre | Profil f16 32K déjà trop grand; aucun contexte supérieur avec ce profil. |
| NeoHorse-1-9B | 262144 | 5920.51 | 6945 / 8993 >8 Go / 13089 >8 Go | Préflight N=1 000; confirmer allocation réelle avant appels. |
| Ornith-1.5-9B | 262144 | 5920.51 | 6945 / 8993 >8 Go / 13089 >8 Go | Préflight N=1 000; confirmer allocation réelle avant appels. |
| Nemotron 3.5 Bolt E4B | 65536 | 3604.99 | 4147 / hors fenêtre / hors fenêtre | Préflight N=1 000, puis 2 000; confirmer allocation réelle avant appels. |
| Ternary Bonsai 2 9B | 262144 | 3263.12 | 4287 / 6335 / 10431 >8 Go | Préflight N=1 000, puis 2 000; confirmer allocation réelle avant appels. |
| LFM3-Omni-2B | 128000 | 1453.86 | 1843 / hors fenêtre / hors fenêtre | Préflight N=1 000, puis 2 000; confirmer allocation réelle avant appels. |
| Phi-4 Mini 3.8B | 131072 | 6492.44 | 10588 >8 Go / 18780 >8 Go / hors fenêtre | Préflight tokenizer à 32K; aucun palier supérieur retenu par la projection f16. |
| Mistral 7B Instruct | 32768 | 8230.52 | hors fenêtre / hors fenêtre / hors fenêtre | Profil f16 32K déjà trop grand; aucun contexte supérieur avec ce profil. |
| Nanbeige 4.2 3B | 262144 | 7839.7 | 13472 >8 Go / 24736 >8 Go / 47264 >8 Go | Préflight tokenizer à 32K; aucun palier supérieur retenu par la projection f16. |

Projection = allocation CUDA observée à 32K + KV observé × (contexte futur / 32 768 − 1). Il s’agit d’une approximation de planification, pas d’une mesure, d’une garantie d’admission ou d’un maximum VRAM. Les caches à fenêtre glissante, états récurrents, graphes compute et allocations transitoires peuvent évoluer autrement. Aucun contexte supérieur n’a été essayé dans ce lot.

## Profils à soumettre séparément

- Quantifier le cache KV, réduire le contexte pour Qwen3-8B ou Mistral, changer la quantification des poids, désactiver le thinking ou augmenter le plafond de génération modifierait le profil. Chacun exige un nouveau dossier et une comparaison explicitement séparée.
- Les extensions YaRN/RoPE, dont Ling au-delà de 131 072, nécessitent la recette officielle et une validation distincte. Une grande fenêtre publiée ne garantit pas qu’elle tienne dans 8 Go.
- Garder le sampler actuel préserve la comparaison, mais les 20 gold restent aux positions 1–20. Une mesure de recherche à positions réparties demanderait un second protocole approuvé; elle ne doit pas remplacer ni être mélangée à ce lot.

Sources inspectables : `evidence/context-window-audit.json`, `evidence/native-buffer-audit.json`, `evidence/model-artifact-provenance.json` et les snapshots publics épinglés sous `evidence/sources/`. Les limites texte ont priorité sur les champs vision/audio secondaires.
