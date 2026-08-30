# vllm-gfx1151

Image d'inférence vLLM OpenAI-compatible pour AMD Strix Halo (**gfx1151 /
RDNA 3.5**), poussée sur Docker Hub + ghcr via GitHub Actions
(`sebt3/vllm-gfx1151`).

**Réécrit le 2026-08-25** : ce repo ne compile plus rien lui-même. C'est un
pur assemblage :

- **Wheels (torch/triton/vllm/AITER/flash-attn)** : viennent du Release
  publié par [`sebt3/stack-torch-gfx1151`](https://github.com/sebt3/stack-torch-gfx1151)
  — build from-source fait *là-bas*, une fois, avec les patches gfx1151
  vendorisés depuis [`bitserv-ai/_gfx115x_`](https://github.com/bitserv-ai/_gfx115x_)
  (gating AITER pour gfx1x, kernels FLA/GDN gfx1151, attention hybride...).
  vLLM actuellement `0.28.1.dev0` (stack-torch-gfx1151 Release `0.3.1`).
- **Runtime ROCm 7.14** : récupéré directement depuis les artefacts CI
  officiels d'AMD (`ROCm/TheRock`, tag `therock-7.14`), pas depuis
  [`sebt3/therock-gfx1151`](https://github.com/sebt3/therock-gfx1151) — ce
  dernier ne patche que le *process* de build de TheRock, pas le
  comportement runtime des libs, donc un build officiel vanilla est
  ABI-identique pour ce qu'on en fait aujourd'hui. À reconnecter le jour où
  `therock-gfx1151` porte un vrai patch comportemental (et aura fini un
  build complet — pas encore le cas au 2026-08-25, cf. son historique CI).

L'ancien build (compilation source de vLLM contre un tarball TheRock complet
dans ce repo, `scripts/install_rocm_sdk.sh` + `scripts/patch_strix.py`
vendorisés depuis [hec-ovi/vllm-awq4-qwen](https://github.com/hec-ovi/vllm-awq4-qwen),
licence Unlicense, voir `LICENSE.upstream`) reste dans l'historique git.
Raison du remplacement : chaque itération payait un rebuild ROCm+vLLM de
plusieurs heures, et ~11.3 GiB de l'image obtenue se sont avérés être du
bloat mort (binaires de test/bench, archives statiques `.a`, RCCL alors que
c'est du single-iGPU — voir `think/vllm/DEBUG.md` dans le repo `kydah/home`,
session 2026-08-25). Le split en 3 repos (`therock-gfx1151` /
`stack-torch-gfx1151` / `vllm-gfx1151`) découple le lourd (rebuild rare,
caché en Release) du léger (cet assemblage, qui ne fait que télécharger et
`pip install`).

## AITER activé (à valider)

`VLLM_ROCM_USE_AITER=1` + les flags granulaires (`_LINEAR`, `_MOE`,
`_RMSNORM`, `_MHA`) sont maintenant le défaut de l'image — **c'est
précisément ce que cette réécriture teste**. L'ancienne image le désactivait
(`VLLM_ROCM_USE_AITER=0`) parce que le gating gfx1151 de vLLM était cassé
(`_aiter_ops.py` teste `on_gfx9()`, `rocm_aiter_fa.py` teste `on_mi3xx()` —
tous deux `False` pour gfx1151 indépendamment du support réel) et le kernel
sampler AITER plantait au premier forward sans ce fix. Les patches
`aiter-gate-gfx1x.patch` / `aiter-fa-gfx1x-gate.patch` de
`stack-torch-gfx1151` visent exactement ce bug de gating. **Pas encore
vérifié sur silicium réel au moment de cette réécriture** — si ça replante,
repasser à `VLLM_ROCM_USE_AITER=0` (les flags granulaires deviennent no-op).

Flash-Attention est maintenant compilée dans le stack (contrairement à
l'ancienne image qui la sautait pour une régression ViT sur gfx1151) — pas
encore revérifié si cette régression est toujours présente sur ce build.

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

Les versions vivent maintenant dans `stack-torch-gfx1151` (`vllm-packages.yaml`
— actuellement vLLM `0.28.1.dev0` (commit `2cf0a6915`, Release `0.3.1`), ROCm
`therock-7.14`, Python `3.13.9`), pas dans ce repo. Pour bumper :

1. Dans `stack-torch-gfx1151` : changer `branch`/`commit` du bloc `vllm:` de
   `vllm-packages.yaml`, re-triager les patches vendorisés depuis
   `bitserv-ai/_gfx115x_` contre la nouvelle version, relancer le build
   (`workflow_dispatch` — conçu comme `therock-gfx1151`, s'attendre à
   plusieurs runs de reprise avant qu'un Release soit réellement publié :
   `Package`/`Publish release` ne se déclenchent que si
   `wheels/vllm-*.whl` existe en fin de run).
2. Ici : bumper `STACK_TORCH_TAG` (build-arg du `Dockerfile`) vers le
   nouveau Release, et `VLLM_TAG` (étape 7) vers le tag vLLM le plus proche
   de ce que `stack-torch-gfx1151` a construit, pour que les
   `requirements/*.txt` récupérés matchent le wheel installé. Les builds
   `0.3.x` sont des `0.28.1.dev0` (entre deux tags upstream) — on garde
   `VLLM_TAG=v0.28.0`, les listes de deps n'ayant pas bougé sur cet écart.
   `0.3.0`→`0.3.1` : uniquement le fix TorchVision (`_cuda_version`
   dupliqué), pas de bump vLLM.
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

Le build tourne sur `ubuntu-latest` (x86) : le kernel gfx1151 est **cross-compilé**
(`--offload-arch=gfx1151`), aucun GPU requis au build.

## Tuning des kernels MoE (Triton)

Sur un modèle MoE (ex: Qwen3.6-35B-A3B, 256 experts) dont les couches
d'experts tombent sur le fallback générique **"Moe WNA16"** de vLLM (kernel
Triton fused-MoE, correct mais jamais autotuné pour ce GPU tant qu'AITER_MOE
ne couvre pas la couche) : mesuré ~5-6x plus lent en decode qu'un DGX Spark
équivalent sur le même modèle (chemin CUDA/FlashInfer, autotuné au boot).
Fournir un fichier de config Triton tuné pour le shape exact (nombre
d'experts, `intermediate_size`, dtype) comble une bonne partie de cet écart,
sans écrire de nouveau kernel.

⚠️ **La recette ci-dessous référence `/opt/vllm/benchmarks/kernels/
benchmark_moe.py` et `Patch 20` de `patch_strix.py` — les deux appartenaient
à l'ancien build source (git history) et n'existent plus dans l'image
réécrite le 2026-08-25** (vLLM est installé en wheel dans le venv, pas
cloné en source ; pas de `patch_strix.py`). `get_device_name()` retourne
déjà `"gfx1151"` nativement sur ce vLLM (`v0.24.0`, plus besoin du patch).
Pour retuner : récupérer `benchmarks/kernels/benchmark_moe.py` depuis le
tag vLLM correspondant (`curl` depuis GitHub, comme les `requirements/*.txt`
à l'étape 6 du `Dockerfile`) plutôt que de chercher un chemin `/opt/vllm/`
qui n'existe plus. Le reste de la recette (points 1-8) reste valide dans
son principe.

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
   `vllm/model_executor/layers/fused_moe/configs/` à l'étape 8c) — c'est ce
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
VLLM_ROCM_USE_AITER=1   # nouveau depuis 2026-08-25, cf. section "AITER activé" — repasser à 0 si crash
VLLM_USE_TRITON_AWQ=1
VLLM_DISABLE_COMPILE_CACHE=1
HSA_NO_SCRATCH_RECLAIM=1                # vllm#37151 segfault AWQ load
MIOPEN_FIND_MODE=FAST                   # vllm#37472 hang conv ViT
HSA_OVERRIDE_GFX_VERSION=11.5.1
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
# NE JAMAIS mettre VLLM_LOGGING_LEVEL=DEBUG => decode 20-100x plus lent
```

Le drafter `z-lab/Qwen3.6-27B-DFlash` est **gated** sur HuggingFace (token requis).
