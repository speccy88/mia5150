# Lot local Skill Discovery

Ce dossier est indépendant de `task-4/benchmark`. Les scripts, fichiers résultats,
processus et identifiants OAuth de ce dernier ne sont pas modifiés. Le README
copié à la racine décrit la préparation historique; `LOCAL_REPORT.md` décrit ce lot.

## Consulter le lot en cours

```powershell
Get-Content .\evidence\live-status.json
Get-Content .\results\local-native-v2\suite-status.json
.\.venv\Scripts\python.exe .\write_native_v2_report.py
```

Le dernier appel régénère le rapport depuis les CSV/SQLite disponibles, après
vérification indépendante des rangs, des métriques et de la persistance. Un export
complet peut contenir des erreurs ou des blocages; les réponses réellement mesurées
et leurs dénominateurs sont distincts. Ne pas réexécuter le lanceur de suite dans
le même dossier : il refuse volontairement un état existant pour éviter un replay.

`evidence/AUDIT_VERDICT.md` donne le verdict de conformité de l'échantillonnage et
le biais gold aux positions 1–20. Les originaux sont conservés. Une modification du
protocole, des poids, des paramètres ou du scoring demande un nouveau run/version.

## Reproduire dans un nouveau dossier

Python 3.12 et PowerShell 7.6.5 sont utilisés (7.5+ pour `-DateKind String`). Le moteur local emploie la bibliothèque
standard. Les dépendances de `requirements-test.txt` ne servent qu'aux tests OAuth
synthétiques conservés avec le moteur; aucune connexion réelle n'est requise.

1. Fournir les deux données privées validées sous `data/`, sans les publier.
   Les SHA-256 attendus sont vérifiés par `bench.py` avant tout appel.
2. Créer un venv local, installer le runtime CUDA avec `Install-LocalRuntime.ps1`
   et les outils de conversion uniquement si nécessaires. Le runtime officiel
   llama.cpp b11960 CUDA 12.4 est épinglé par SHA-256; aucune installation globale.
3. `inspect_sources.py` lit uniquement les métadonnées publiques des éditeurs.
   `fetch_official.py REPO FICHIER` télécharge un GGUF public épinglé; `fetch_batch.py`
   contient la sélection officielle. Les transferts utilisent des plages HTTP
   vérifiées et un SHA-256 intégral avant promotion des `.part`.
4. Les modèles sans GGUF éditeur se préparent avec `prepare_conversion.py converter`
   puis `prepare_conversion.py REPO`. `convert_official_batch.py` utilise le code
   officiel épinglé, en mode hors ligne, PyTorch CPU et un quantificateur sans DLL
   CUDA. Ces opérations préparent les poids; ce ne sont pas des inférences CPU.
5. Examiner `local-models.json` et ajuster les chemins des poids déjà présents sur
   la machine. Les Qwen3.5 sont les fichiers Ollama installés; Ling correspond
   exactement au fichier officiel. Ne pas substituer un modèle homonyme.
6. Utiliser `bench_native_v2.py` et les configurations `configs/native-v2/` pour
   le parsing corrigé des réponses finales. `run_local_suite_v2.py` crée un lot
   neuf, un modèle à la fois sur `127.0.0.1:28085`; il arrête uniquement ses
   serveurs vérifiés par PID/heure/exécutable. Une charge CUDA étrangère bloque
   le lancement. `continue_local_suite_v2.py` a servi à la reprise inspectée
   après le diagnostic Qwen3-8B; il ignore les modèles complets et refuse tout
   état transactionnel incomplet susceptible de rejouer une requête.

Le lanceur exige toutes les couches et le cache KV sur CUDA, f16, contexte réel
32 768, `fit=off`, un slot, sans context shift, warmup, cache RAM ou checkpoints.
Il ne réduit pas silencieusement le contexte ou les couches si la VRAM manque.
Chaque serveur a ses journaux et une preuve liée au PID, à l'heure de démarrage,
aux empreintes poids/exécutable et au préfixe scellé du journal.

Les accès Llama 3.2 3B et Gemma 3 4B sont manuellement gated. Aucune acceptation,
connexion ou copie tierce non gated n'est tentée pour les contourner.

## Préserver et intégrer

- `results/local-native-v2/*/state.sqlite3` est transactionnel et fait autorité;
  `details.csv` est exporté atomiquement, avec exactement 45 colonnes.
- `manifest.json` conserve configuration, identités et plans de candidats/prompt;
  toute reprise doit respecter son empreinte. Une requête interrompue d'issue
  incertaine n'est jamais rejouée automatiquement.
- Les erreurs initiales restent inchangées. L'incident OOM tardif du 1.7B est
  documenté dans `evidence/qwen17-runtime-incident.json`; les lignes consécutives
  serveur arrêté ne sont pas interprétées comme des réponses de précision nulle.
- `.gitignore` exclut poids, archives, outils, environnements, conversions et
  données privées. Aucun secret OAuth n'a été copié. Les résultats contiennent
  questions/noms et parfois des fragments de sorties : examiner avant diffusion.
- Aucun email ni push n'est exécuté ici. Le parent coordonne le rapport Markdown
  commun et l'intégration au dépôt demandé.

`local-v1/` reste un diagnostic historique du parsing; ses scores ne sont pas
comparatifs. Les réponses brutes v2 sont dans `private-responses/`, exclu de Git.
`evidence/model-artifact-provenance.json` regroupe les manifestes publics sans
les poids. `audit_native_buffers.py` vérifie les préfixes scellés et les budgets.

La portée GPU correspond aux couches transformer/sortie et KV/états récurrents.
Le buffer d'entrée reste CPU_Mapped par défaut. Le lanceur refuse aussi un budget
CUDA déclaré supérieur aux 8 192 MiB physiques : WDDM peut accepter une allocation
qui dépasse cette capacité. Le diagnostic Qwen3-8B préserve 21 réponses séparées,
sans les compter comme mesures admissibles ni rejouer l'intention interrompue.

Les 60 exclusions aux tailles 1 000/2 000/5 000 viennent du filtre de la fenêtre
configurée à 32K. Elles ne démontrent pas une limite native ni le maximum VRAM.

## Finaliser et empaqueter

Après résolution des 18 modèles, `finalize_local_handoff.py` effectue uniquement
des vérifications hors ligne et génère `evidence/final-handoff.json`, le rapport
autonome et `HIGH_CONTEXT_PROPOSAL.md`. Ce dernier est une proposition : aucun
chargement ni appel à un contexte supérieur n'est exécuté.

`package_local_handoff.py` crée les archives dans `../deliverables/` et vérifie
les CRC ZIP ainsi que chaque SHA-256 de contenu. L'archive d'intégration contient
code, tests, configurations, résultats, notes et journaux; elle exclut les poids,
outils binaires, environnements, données sources privées et réponses brutes.
Les originaux SQLite/WAL sont conservés. Le diagnostic interrompu possède aussi
une copie autonome `state.portable.sqlite3`, avec les intentions non résolues.

L'archive séparée `PRIVATE-local-response-evidence-*.zip` conserve les réponses
brutes pour vérification locale. Elle reste hors intégration GitHub et livraison
ordinaire. Les rapports copiés, manifestes et `SHA256SUMS-*.txt` accompagnent les
archives. Ces commandes ne font ni push ni envoi de message.
