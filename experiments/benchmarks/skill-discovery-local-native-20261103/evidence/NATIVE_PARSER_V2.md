# Correction de l'adaptateur natif — version séparée

Le template GGUF Qwen3.5 préremplit `<think>\n`. `/completion` retourne uniquement
le texte généré après le prompt, sans cette ouverture préremplie. Le parseur v1
ne pouvait donc pas reconnaître tous les blocs de raisonnement. Il interprétait
parfois des listes internes comme un classement; les métriques v1 locales sont
conservées comme diagnostic et ne constituent pas les résultats comparatifs finaux.

Preuve : `qwen35-format-audit.json` contient le suffixe du template réel.
62 des 80 réponses du Qwen3.5-4B atteignent le plafond de 500 tokens. Toutes les
premières entrées sont du raisonnement/texte non final; certains gold apparaissent
ensuite dans ce texte. Le Top-1 strict v1 vaut zéro, mais le Top-10 v1 n'est pas une
mesure fiable de la réponse finale.

`bench_native_v2.py` et `results/local-native-v2/` séparent cette correction :

- aucun changement du sampler, des 20 questions, des candidats, des descriptions
  tronquées, des prompts, de température 0, seed 42 ou plafond 500;
- détection de l'ouverture de raisonnement déjà fournie par le template, conservée
  comme état de parsing uniquement (pas ajoutée une seconde fois au prompt);
- exclusion des blocs `<think>…</think>` et du canal Gemma
  `<|channel>thought…<channel|>` avant scoring de la réponse finale;
- raisonnement inachevé : `REASONING_NO_FINAL_ANSWER`, avec vrais tokens conservés;
- aucune suppression/promotion des rangs invalides ou répétés dans la réponse finale;
- sorties natives brutes conservées dans `private-responses/`, exclu de Git;
- mêmes 45 colonnes et persistance SQLite/CSV transactionnelle.

Les 34 tests passent, dont sept régressions dédiées. Une comparaison indépendante
des plans v1/v2 confirme les mêmes 140 hashes de prompts et mêmes IDs candidats.
Les modèles v1 déjà exécutés sont préservés. Le lot v1 s'arrête proprement après
Ling, puis le lot natif corrigé utilise des alias/proofs/configs/résultats distincts.
Les scripts et résultats OAuth de `task-4` ne sont pas modifiés.
