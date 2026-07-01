# vllm-gfx1151

vLLM compilé depuis les sources contre **TheRock ROCm 7.13 nightly** pour AMD
Strix Halo (**gfx1151 / RDNA 3.5**). Image OpenAI-compatible poussée sur Docker
Hub + ghcr via GitHub Actions (`sebt3/vllm-gfx1151`).

Cible validée : Qwen3.6-27B AWQ-INT4 (`cyankiwi/Qwen3.6-27B-AWQ-INT4`),
~25 tok/s decode single-stream avec DFlash N=8 sur 128 Go UMA.

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

## Config runtime (→ à porter dans le package vynil `vllm`, Phase 2)

Le build ne fait que produire l'image. La config de lancement qui donne les
~25 tok/s vit côté déploiement. Extraits load-bearing à reprendre dans le
package vynil :

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
