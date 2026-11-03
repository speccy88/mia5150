# Fenêtre native, fenêtre configurée et limite matérielle

Les 60 lignes `CONTEXT_LIMIT_ESTIMATED` par modèle proviennent de la barrière v1
contre le **n_ctx configuré à 32 768**, pas d'une mesure de la limite native.

Le seuil vaut 29 491,2 tokens (90 % de 32 768). Les estimations v1 pour N=1000,
2000 et 5000 valent 44 700, 89 200 et 222 700, donc ces trois tailles sont écartées :
3 × 20 questions = 60 lignes. Aucun tokenizer ni appel de génération ne mesure
ces tailles écartées. Les estimations ne sont pas inscrites comme tokens réels.

| Modèle | Fenêtre publiée / configuration éditeur | Fenêtre effectivement chargée dans ce lot |
|---|---:|---:|
| Qwen3-0.6B | README : 32 768; config/GGUF : 40 960 | 32 768 |
| Qwen3-1.7B | README : 32 768; config/GGUF : 40 960 | 32 768 |
| Qwen3.5-4B | 262 144 natifs; extension annoncée à 1 010 000 | 32 768 |
| Qwen3.5-9B | 262 144 natifs; extension annoncée à 1 010 000 | 32 768 |
| Ling-3.0-tiny | config native : 131 072; recette YaRN : 262 144 | 32 768 |

Le 32K est un choix uniforme et conservateur du profil préparé pour la machine
8 Go, **sans recherche du maximum matériel par modèle**. Il ne faut donc pas
présenter ce choix comme une limite VRAM démontrée, ni les 60 lignes comme une
incapacité intrinsèque de Qwen3.5 ou Ling. Une fenêtre native plus grande ne prouve
pas non plus qu'elle tiendrait en VRAM avec les poids, KV f16 et buffers de ce lot.

Les sources éditeur, commits, limites de config, métadonnées GGUF et n_ctx observé
par preuve de lancement sont séparés dans `context-window-audit.json`. Les configs
gated non récupérées et serveurs non chargés restent explicitement non observés.

Les colonnes et erreurs protocolaires du CSV restent inchangées. Le présent audit
ajoute seulement la distinction de causalité : filtre estimé du profil configuré,
dépassement tokenisé réel, erreur de runtime/VRAM et limite native publiée sont
des notions distinctes.
