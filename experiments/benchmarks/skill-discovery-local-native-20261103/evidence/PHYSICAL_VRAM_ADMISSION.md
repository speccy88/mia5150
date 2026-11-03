# Admission par capacité physique

Le premier chargement Qwen3-8B a annoncé 4 455,34 MiB de modèle CUDA,
4 608 MiB de KV f16 et 34 MiB de compute : 9 097,34 MiB pour une carte de
8 192 MiB. WDDM a accepté l'allocation. Les 21 réponses acquises sont conservées
comme diagnostic séparé; l'intention `100:1` interrompue reste d'issue incertaine
et n'est pas rejouée. Aucun taux mesuré de ce diagnostic n'entre dans le lot
comparatif canonique.

Le lanceur refuse maintenant un budget CUDA déclaré supérieur à la VRAM
physique avant tout appel de génération. Ce refus est nommé
`HARDWARE_VRAM_CAPACITY_EXCEEDED`; il ne signifie pas qu'un cudaMalloc a renvoyé
OOM. Un OOM effectivement observé garde son propre type d'erreur.

Une première sonde de démarrage du garde-fou avait un bug de sommation des
OrderedDictionary PowerShell, produisant zéro malgré trois enregistrements
corrects. Elle n'a effectué aucune inférence. Sa preuve est conservée sous
`gpu-proof-diagnostic-invalid-sum.json` et explicitement invalidée dans
`invalidated-guard-probe-proof.json`. La sommation corrigée par index direct,
avec refus de tout total nul/non vérifié, a rejeté le même modèle à
9 097,34 MiB. Les journaux et le JSON du refus sont conservés.

Les modèles suivants passent ce contrôle avant génération. Les modèles déjà
terminés sont audités hors ligne à partir du préfixe de démarrage scellé; leurs
budgets déclarés sont inférieurs à la capacité physique. Les allocations restent
distinctes de la résidence réelle. Les compteurs WDDM dédié/partagé sont collectés
à partir d'Ornith, avec vérification du PID, de l'heure et de l'exécutable.
Un usage partagé non nul ne localise pas à lui seul un buffer CUDA paginé.

Les couches transformer/sortie et KV/états récurrents observés sont CUDA.
Le buffer d'entrée/embeddings CPU_Mapped est le comportement par défaut du
runtime, documenté séparément; le profil n'est pas qualifié de GPU-only absolu.
Le contexte 32K et les autres paramètres de ce lot ne sont pas modifiés.
