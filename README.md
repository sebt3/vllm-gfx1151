# vllm-gfx1151

Image d'inférence vLLM OpenAI-compatible pour AMD Strix Halo (**gfx1151 /
RDNA 3.5**), poussée sur Docker Hub + ghcr via GitHub Actions
(`sebt3/vllm-gfx1151`).

**Réécrit le 2026-09-02** : ce repo ne compile plus rien, et ne dépend plus
d'un build from-source maison. C'est un pur assemblage de wheels publiées :

- **Wheels ROCm (vllm/torch/triton/torchvision/torchaudio/AITER/flash-attn/amdsmi)**
  : viennent de l'index officiel vLLM [`wheels.vllm.ai/rocm/`](https://wheels.vllm.ai/rocm/)
  — buildées ensemble par la CI vLLM, mutuellement ABI-pinées
  (`Requires-Dist` porte les `==x+localver` exacts). Actuellement :
  - `vllm 0.28.0+rocm723` (git `2cf0a6915` — exactement le commit que
    `stack-torch-gfx1151` 0.3.2/0.4.0 tentait de builder)
  - `torch 2.12.0+git6bbd260` — `libtorch_hip.so` est un fat build
    multi-arch, gfx1151/1150/1152/1153 tous présents (vérifié)
  - `triton 3.7.1+gitf0b55c07` — le Triton *matché* de vLLM 0.28
  - cp312 uniquement (l'ancienne chaîne stack-torch buildait du cp313)

  C'est la **même source que celle consommée par
  [`lemonade-sdk/vllm-rocm`](https://github.com/lemonade-sdk/vllm-rocm)**
  pour son bundle gfx1151 (qui passe une qualification 16 tests sur
  silicium gfx1151), juste assemblée à notre sauce (config runtime +
  configs MoE).
- **Runtime ROCm 7.14** : tarball `therock-dist-linux-gfx1151-7.14.0` depuis
  `repo.amd.com` → `/opt/rocm`. **Requis, pas optionnel** : la wheel torch
  n'embarque aucun userspace ROCm et ne déclare aucune dépendance
  `rocm-sdk`, donc `libamdhip64` / `librocblas` / `libhipBLASLt` et le
  `llvm/clang+lld` dont le JIT de Triton (et d'aiter) a besoin **au
  runtime** viennent tous de là. « 7.14.0 » = version du paquet *dist*
  TheRock ; le SDK ROCm/HIP à l'intérieur est ~7.2, cohérent avec le
  `hip 7.2.x` que rapporte la wheel torch.

### Pourquoi l'abandon de stack-torch-gfx1151

Le split en 3 repos (`therock-gfx1151` / `stack-torch-gfx1151` /
`vllm-gfx1151`, mis en place le 2026-08-25) n'a **jamais produit une image
qui fonctionne**. Les 3 dernières tentatives sont toutes mortes dans le
compilo Triton sur gfx1151 : `SyntaxError` codegen Inductor sur 0.3.1, puis
segfault WMMA/LDS dans `OptimizeAMDLDSUsage` sur 0.3.2 et 0.4.0 (bump Triton
`main_perf` → `release/internal/3.8.x`) — voir `think/vllm/DEBUG.md` du repo
`kydah/home`, sessions 2026-08-30 / 08-31. Chaque itération coûtait un
rebuild de plusieurs heures. `stack-torch-gfx1151` est retiré comme ligne
principale ; il reste comme fallback *si* un patch PyTorch/ROCm que personne
ne porte upstream devient réellement nécessaire.

L'ancien build (compilation source de vLLM contre un tarball TheRock complet
dans ce repo, `scripts/install_rocm_sdk.sh` + `scripts/patch_strix.py`
vendorisés depuis [hec-ovi/vllm-awq4-qwen](https://github.com/hec-ovi/vllm-awq4-qwen),
licence Unlicense, voir `LICENSE.upstream`) reste dans l'historique git.

## AITER désactivé par défaut

`VLLM_ROCM_USE_AITER=0` dans l'image. Le gating gfx1151 de vLLM est toujours
cassé upstream (`_aiter_ops.py` teste `on_gfx9()`, `rocm_aiter_fa.py` teste
`on_mi3xx()` — tous deux `False` pour gfx1151) ; le correctif vivait dans les
patches source `aiter-gate-gfx1x.patch` / `aiter-fa-gfx1x-gate.patch` de
`stack-torch-gfx1151`, qu'on **ne peut plus appliquer à une wheel
prébuildée**. La prod tourne déjà en AITER=0. À revisiter quand ces fix de
gating atterrissent upstream, ou en les re-portant comme patches
`site-packages` (au même endroit que le shim `triton_utils` de l'étape 6c du
Dockerfile).

`amd-aiter 0.1.19` est quand même installé (dépendance dure de la wheel
vLLM) ; avec le flag à 0 il n'est simplement pas sur le chemin.

## HIP graphs (`--enforce-eager`)

Historiquement désactivées au runtime — freeze driver documenté (vLLM #32180).
**Ne reproduit plus** sur le tarball TheRock ROCm 7.13 nightly épinglé ici :
capture propre en ~8s, testé le 2026-07-08 sur Qwen3.6-35B-A3B-AWQ. Gain
mesuré **~5x en decode single-stream** (5-6 tok/s → 20-32 tok/s selon la
longueur de génération) — c'était le vrai goulot, pas les kernels MoE.
Retirer `--enforce-eager` par défaut ; ne le remettre que si le freeze
réapparaît sur un tarball ROCm différent (pas encore revérifié sur le modèle
dense 27B ni en usage multi-requêtes concurrentes).

## Versions épinglées (rafraîchir en bloc)

Tout est dans le `Dockerfile` (`ARG`) :

1. `VLLM_WHEELS_COMMIT` (étape 5) — le commit vLLM sous lequel
   `wheels.vllm.ai/rocm/` a publié la série. Pour bumper : lister
   `https://wheels.vllm.ai/rocm/vllm/` (page PEP503), récupérer le nouveau
   `href="../<commit>/vllm-....whl"`, mettre ce `<commit>` ici. Puis
   ré-inspecter la wheel vLLM (`unzip -p vllm-*.whl '*/METADATA' | grep
   Requires-Dist`) pour relever les nouveaux `torch==` / `triton==` /
   `torchvision==` / `torchaudio==` / `amd-aiter==` / `flash-attn==` /
   `amdsmi==` et mettre à jour la liste de fichiers `.whl` de l'étape 5
   (ces pins sont exacts, la CI vLLM les bouge ensemble).
2. `VLLM_TAG` (étape 6) — le tag vLLM le plus proche du commit
   (`0.28.0+rocm723` == `v0.28.0`), pour que `requirements/common.txt` +
   `rocm.txt` récupérés matchent les deps de la wheel.
3. `ROCM_DIST_URL` (étape 2) pointe vers le tarball ROCm officiel sur
   `repo.amd.com/rocm/tarball-multi-arch/` (le dépôt stable AMD, pas un
   bucket CI) — à changer seulement si `stack-torch-gfx1151` bump sa
   propre version ROCm. **Ne pas repasser par `install_rocm_from_artifacts
   .py --run-id` de `ROCm/TheRock`** : testé le 2026-08-25, les artefacts
   bruts par composant liés à un run CI précis semblent avoir une
   rétention très courte (fonctionnait pour `stack-torch-gfx1151` le
   21-22 août, disparu le 25) — confirmé en listant directement le bucket
   S3 `therock-ci-artifacts`. Le canal nightly alternatif
   (`rocm.nightlies.amd.com`/`therock-nightly-tarball`) est encore pire :
   il s'arrête de publier des builds gfx1151 après le 2026-06-08, supplanté
   depuis par le mécanisme par run-id. `repo.amd.com` (le miroir apt/yum
   ROCm) est le seul des trois canaux hébergés par AMD qui semble fait
   pour durer.

## Secrets GitHub Actions requis

- `DOCKERHUB_TOKEN` — token Docker Hub (username = `sebt3`, déduit du repo owner).
- `GITHUB_TOKEN` — fourni automatiquement (push ghcr).

Le build tourne sur `ubuntu-latest` (x86) et ne fait que télécharger des
wheels + un tarball ROCm et `pip install` — aucune compilation, aucun GPU.

## Tuning des kernels MoE (Triton)

Sur un modèle MoE (ex: Qwen3.6-35B-A3B, 256 experts) dont les couches
d'experts tombent sur le fallback générique **"Moe WNA16"** de vLLM (kernel
Triton fused-MoE, correct mais jamais autotuné pour ce GPU tant qu'AITER_MOE
ne couvre pas la couche) : mesuré ~5-6x plus lent en decode qu'un DGX Spark
équivalent sur le même modèle (chemin CUDA/FlashInfer, autotuné au boot).
Fournir un fichier de config Triton tuné pour le shape exact (nombre
d'experts, `intermediate_size`, dtype) comble une bonne partie de cet écart,
sans écrire de nouveau kernel.

⚠️ **vLLM est installé en wheel dans le venv (`/opt/venv/lib/python3.12/
site-packages/vllm/`), pas cloné en source** — pas de `/opt/vllm/`, pas de
`patch_strix.py`. Pour retuner : récupérer `benchmarks/kernels/benchmark_moe.py`
depuis le tag vLLM correspondant (`curl` depuis GitHub, comme les
`requirements/*.txt` à l'étape 6 du `Dockerfile`). Les configs générées se
déposent dans `moe-configs/` de ce repo (l'étape 7 du `Dockerfile` les copie
dans `vllm/model_executor/layers/fused_moe/configs/`).

⚠️ **Nom de fichier des configs MoE** : doit matcher `device_name={get_device_name()}`.
vLLM ≤ 0.24 renvoyait l'archi (`gfx1151`), **0.28 renvoie le nom marketing**
(`AMD Radeon 8060S` → `AMD_Radeon_8060S`). Les fichiers du repo sont nommés
pour 0.28 ; l'étape 7 dépose en plus une copie nommée `gfx1151` pour les
vLLM plus anciens. Un mauvais nom = `WARNING [fused_moe.py] Using default
MoE config. Performance might be sub-optimal!` et ~4x de perte en decode
(mesuré sur rennes, alpha.9 : 6.4 vs ~25 tok/s sur 0.21).

### Recette (à répéter pour chaque nouveau shape MoE — nouveau modèle,
### nouveau `num_experts`/`moe_intermediate_size`, nouveau dtype de quant)

1. **Libérer la mémoire UMA.** Le tuning alloue ses propres tenseurs GPU ; sur
   Strix Halo (RAM unifiée), il n'y a pas de place à côté d'un serveur vLLM
   déjà chargé à `--gpu-memory-utilization 0.88`. Scale à 0 le deployment qui
   sert le modèle (`kubectl scale --replicas=0`) — coupure de service pendant
   tout le tuning, voir point 5.
2. Lancer un pod (ou exec dans un conteneur) sur le même node, même image,
   mêmes volumes (`/dev/dri`, `/dev/kfd`, PVC modèle), mais avec
   `command: ["sleep", "infinity"]` au lieu de la commande serveur.
3. Dans ce conteneur :
   ```bash
   pip install ray   # absent de l'image, seulement requis par le tuner
   # Ray auto-détecte les GPU AMD via amdsmi, mocké par Patch 1.5 -> 0 GPU
   # vu par ray.available_resources(). Forcer explicitement :
   sed -i 's/^    ray.init()$/    ray.init(num_gpus=1)/' \
     /opt/vllm/benchmarks/kernels/benchmark_moe.py
   ```
4. Lancer le tuner (script vLLM natif — supporte nativement l'architecture
   `Qwen3_5MoeForConditionalGeneration` et lit `num_experts`/
   `moe_intermediate_size`/`group_size` depuis le `config.json` du modèle) :
   ```bash
   python /opt/vllm/benchmarks/kernels/benchmark_moe.py \
     --model /models/hub/models--<org>--<model>/snapshots/<sha> \
     --trust-remote-code --tp-size 1 \
     --dtype int4_w4a16 \
     --batch-size 1 8 \
     --tune --save-dir /tmp/moe-tune-out
   ```
   `--dtype` doit correspondre au format de quantification MoE réel :
   `int4_w4a16` pour AWQ/GPTQ 4-bit, `fp8_w8a8` pour FP8, etc.
5. **C'est lent.** Premier run mesuré (E=256, N=512, int4_w4a16,
   `--batch-size 1 8` seulement — 2 shapes) : **~2h11** (7842s). L'espace de
   recherche ROCm complet (`get_rocm_tuning_space`) contient des milliers de
   combinaisons de block-size, et certaines compilent/exécutent en 20-35s au
   lieu de 1-2s (probablement spill registre sur des tailles de bloc mal
   choisies) — le pruning heuristique ROCm ne les filtre pas toutes. Prévoir
   une vraie fenêtre creuse, pas un "vite fait entre deux tâches".
6. Récupérer le(s) JSON généré(s) (`kubectl cp` ou équivalent), les déposer
   dans `moe-configs/` de ce repo (le nom de fichier — `E=...,N=...,
   device_name=gfx1151,dtype=....json` — est généré correctement une fois
   Patch 20 en place ; avant, il faut le renommer à la main).
7. Rebuild + push l'image (le `Dockerfile` copie `moe-configs/*.json` dans
   `vllm/model_executor/layers/fused_moe/configs/` à l'étape 7) — c'est ce
   qui rend le tuning réellement utilisable en prod.
8. Remonter le deployment (`kubectl scale --replicas=1`).

Une fois l'image rebuild avec le JSON baké dedans, **aucune machine gfx1151
n'a besoin de refaire ce tuning** — même silicium, même config optimale.
Il suffit de pull la nouvelle image.

## Config runtime (→ à porter dans le déploiement, Phase 2)

Le build ne fait que produire l'image. La config de lancement vit côté
déploiement. Extraits load-bearing à reprendre :

```
# Flags serve
--attention-backend ROCM_ATTN          # requis pour DFlash non-causal (PR #40176)
--mm-encoder-attn-backend TRITON_ATTN  # TORCH_SDPA => NaN sur images
# PAS de --enforce-eager par défaut — voir section "HIP graphs" ci-dessus
--reasoning-parser qwen3
--tool-call-parser qwen3_coder --enable-auto-tool-choice
--speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":8}'
# PAS de --quantization (auto-détection compressed-tensors => AWQMarlin)

# Env load-bearing (déjà posé par le Dockerfile ENV, étape 8)
VLLM_ROCM_USE_AITER=0   # cf. section "AITER désactivé par défaut"
VLLM_USE_TRITON_AWQ=1
VLLM_DISABLE_COMPILE_CACHE=1
HSA_NO_SCRATCH_RECLAIM=1                # vllm#37151 segfault AWQ load
MIOPEN_FIND_MODE=FAST                   # vllm#37472 hang conv ViT
HSA_OVERRIDE_GFX_VERSION=11.5.1
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
# NE JAMAIS mettre VLLM_LOGGING_LEVEL=DEBUG => decode 20-100x plus lent
```

Le drafter `z-lab/Qwen3.6-27B-DFlash` est **gated** sur HuggingFace (token requis).
