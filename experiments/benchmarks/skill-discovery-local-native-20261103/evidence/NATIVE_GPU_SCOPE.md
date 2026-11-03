# Portée matérielle exacte du profil préparé

La preuve du lanceur vérifie toutes les couches transformer et de sortie sur GPU,
le KV sur CUDA et le contexte effectif. Les journaux des modèles hybrides vérifiés
placent également les buffers récurrents RS sur CUDA. Aucun auto-fit ou repli de
couches transformer n'est autorisé par le lanceur.

L'audit des buffers révèle toutefois que l'amont llama.cpp b11960 conserve sa
couche d'entrée sur CPU par défaut. Le code `src/llama-model.cpp` explique ce choix
et assigne `dev_input` au périphérique CPU, indépendamment du nombre de couches
demandées sur GPU. Les journaux montrent ainsi un buffer modèle CPU_Mapped en plus
des buffers CUDA, par exemple 545,62 MiB pour Qwen3.5-9B.

Ce profil respecte la définition opérationnelle du script préparé (transformer,
sortie, KV sur CUDA), mais ne doit pas être présenté comme une exécution sans
aucune participation CPU. Les entrées/embeddings CPU_Mapped et la préparation CPU
sont déclarés explicitement; les mesures décrivent ce profil précis.

Les buffers observés et leurs tailles sont conservés dans `native-buffer-audit.json`.
Le runtime, les paramètres et les résultats acquis n'ont pas été modifiés pendant
cet audit. Un profil forçant aussi les poids d'entrée sur CUDA serait un profil
distinct à vérifier, notamment pour la VRAM, et ne doit pas être confondu avec ce lot.
