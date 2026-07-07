# vllm-gfx1151

vLLM compilé depuis les sources contre **TheRock ROCm 7.13 nightly** pour AMD
Strix Halo (**gfx1151 / RDNA 3.5**). Image OpenAI-compatible poussée sur Docker
Hub + ghcr via GitHub Actions (`sebt3/vllm-gfx1151`).

Cible : Qwen3.6-27B AWQ-INT4 (`cyankiwi/Qwen3.6-27B-AWQ-INT4`), decode
single-stream avec DFlash N=8 sur 128 Go UMA. Perf pas encore validée par
mesure — ne pas citer de chiffre tant qu'un run réel n'a pas été observé.

## Origine

Vendorisé depuis [hec-ovi/vllm-awq4-qwen](https://github.com/hec-ovi/vllm-awq4-qwen)
(licence Unlicense / domaine public, voir `LICENSE.upstream`). Trois fichiers
repris **verbatim** :

- `scripts/install_rocm_sdk.sh` — installe le tarball TheRock ROCm (tarball
  **épinglé** au `20260510` pour reproductibilité).
- `scripts/patch_strix.py` — bundle de 18 patches gfx1151 sur vLLM v0.20.0
  (monkey-patch amdsmi, détection arch, overrides AITER, clamp VRAM APU, et
  cherry-picks des PR upstream #40176 / #40898 pour DFlash ROCm).
- `Dockerfile` — build multi-étapes source.

Le kernel HIP expérimental `csrc/awq_mmq_gfx1151` de l'upstream **n'est pas
inclus** : c'est un portage MMQ en cours, live-monté au runtime chez eux, pas
dans le build. Le chemin AWQ rapide vient de `VLLM_USE_TRITON_AWQ=1` + la PR
vLLM #36505 (AWQMarlin → ConchLinearKernel, `conch-triton-kernels`) :
+57 % prefill / +73 % decode sur gfx1151 vs le legacy `ops.awq_gemm`.

## Ce que le build ne fait PAS

- **AITER** : désactivé (`VLLM_ROCM_USE_AITER=0`) — kernels CDNA qui freezent
  gfx1151.
- **Flash-Attention** compilée : sautée — régression ViT sur gfx1151. On passe
  par TRITON_ATTN / AOTriton.
- **HIP graphs** : à désactiver au runtime (`--enforce-eager`) — freeze driver
  documenté (vLLM #32180).

## Versions épinglées (rafraîchir en bloc)

`torch` / `torchvision` / `torchaudio` / `triton` **et** le tarball TheRock sont
tous épinglés au snapshot `rocm7.13.0a20260510`. Pour bumper :

1. `ALLOW_LATEST=1` dans `install_rocm_sdk.sh` pour trouver le dernier tarball.
2. Bumper les 4 wheels + le `PINNED_TARBALL` sur une date commune.
3. Rebuild propre, puis valider (sanity + bench).

Attention à la régression PyTorch #180485 (drop de `HIP_FOUND`) que le Patch 18
compense — voir commentaires dans le `Dockerfile`.

## Secrets GitHub Actions requis

- `DOCKERHUB_TOKEN` — token Docker Hub (username = `sebt3`, déduit du repo owner).
- `GITHUB_TOKEN` — fourni automatiquement (push ghcr).

Le build tourne sur `ubuntu-latest` (x86) : le kernel gfx1151 est **cross-compilé**
(`--offload-arch=gfx1151`), aucun GPU requis au build.

## Tuning des kernels MoE (Triton)

Ce build a été taillé au départ pour un modèle **dense** (Qwen3.6-27B AWQ-INT4)
— aucun des 19 premiers patches ne touche le chemin MoE. Sur un modèle MoE
(ex: Qwen3.6-35B-A3B, 256 experts), les couches d'experts tombent sur le
fallback générique **"Moe WNA16"** de vLLM (kernel Triton fused-MoE, correct
mais jamais autotuné pour ce GPU) : mesuré ~5-6x plus lent en decode qu'un
DGX Spark équivalent sur le même modèle (chemin CUDA/FlashInfer, autotuné au
boot). Fournir un fichier de config Triton tuné pour le shape exact
(nombre d'experts, `intermediate_size`, dtype) comble une bonne partie de cet
écart, sans écrire de nouveau kernel.

**Prérequis : `Patch 20`** (dans `patch_strix.py`) doit être présent — il fixe
`ROCmPlatform.get_device_name()` à retourner la string constante `"gfx1151"`.
Sans ce patch, `amdsmi` étant entièrement mocké (Patch 1.5), cette méthode
renvoie un `MagicMock` dont le `repr()` embarque un `id()` Python — différent
à chaque redémarrage de process. Le nom de fichier de config MoE
(`get_config_file_name()` dans `fused_moe.py`) intègre ce nom tel quel : un
config tuné par un process ne matchera **jamais** celui calculé par le
serveur au runtime sans ce patch. Vérifié empiriquement le 2026-07-07 sur le
premier run de tuning (fichier généré nommé
`device_name=<MagicMock ... id='...'>...json`, inutilisable).

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
--enforce-eager                        # HIP graph capture freeze gfx1151
--reasoning-parser qwen3
--tool-call-parser qwen3_coder --enable-auto-tool-choice
--speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":8}'
# PAS de --quantization (auto-détection compressed-tensors => AWQMarlin)

# Env load-bearing (déjà posé par install_rocm_sdk.sh + Dockerfile ENV)
VLLM_ROCM_USE_AITER=0
VLLM_USE_TRITON_AWQ=1
VLLM_DISABLE_COMPILE_CACHE=1
HSA_NO_SCRATCH_RECLAIM=1                # vllm#37151 segfault AWQ load
MIOPEN_FIND_MODE=FAST                   # vllm#37472 hang conv ViT
HSA_OVERRIDE_GFX_VERSION=11.5.1
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
# NE JAMAIS mettre VLLM_LOGGING_LEVEL=DEBUG => decode 20-100x plus lent
```

Le drafter `z-lab/Qwen3.6-27B-DFlash` est **gated** sur HuggingFace (token requis).
