# vllm-awq4-qwen source-build image  -  Ubuntu 26.04 + TheRock ROCm 7.14 nightly + vLLM from source.
#
# Builds vLLM (main HEAD, 2026-07-08) against TheRock ROCm 7.14 nightly for
# AMD Strix Halo (gfx1151 / RDNA 3.5). Originally targeted the dense model
# cyankiwi/Qwen3.6-27B-AWQ-INT4 (compressed-tensors W4A16, group_size 32,
# vision tower preserved BF16); also runs the MoE Qwen3.6-35B-A3B checkpoints.
#
# *** 2026-07-08: bumped off the v0.20.0 tag, patches disabled ***
# See step 7 below ("PATCHES-DISABLED") for why and what's left to do.
#
# Why this differed from vllm-qwen (BF16), historically:
#   - vLLM v0.20.0 (released 2026-04-23) was the first stable cut adding
#     gfx1150/1151/1201 device IDs AND PR #36505 which routes AWQ
#     through AWQMarlinLinearMethod -> ConchLinearKernel on ROCm:
#     measured +57% prefill / +73% decode on gfx1151 vs the legacy
#     ops.awq_gemm path. That's now folded into whatever HEAD carries.
#   - Wheel index switched from rocm.prereleases (frozen at 7.12.0rc1)
#     to rocm.nightlies.amd.com/v2-staging/gfx1151/, matched daily set
#     (torch/torchvision/torchaudio/triton), currently pinned to 20260612.
#
# What we DON'T build:
#   - AITER custom build  : SKIPPED. Disabled at runtime via
#                           VLLM_ROCM_USE_AITER=0; the patch bundle also
#                           fences AITER FP8 / MoE / RMSNorm off on gfx1x.
#   - Flash-Attention      : SKIPPED. ROCm/flash-attention main_perf HEAD
#                           is a 2.2-3.7x ViT regression on gfx1151
#                           (Dao-AILab/flash-attention#2392). vLLM's
#                           TRITON_ATTN + AOTriton experimental SDPA is
#                           the recommended path on RDNA 3.5.
#   - bitsandbytes ROCm    : SKIPPED. AWQ via compressed-tensors.
#   - Custom RCCL          : SKIPPED. Single-iGPU, no NCCL.

# All toolchain versions in this Dockerfile are pinned. AMD's
# rocm.nightlies.amd.com index hosts multiple torch series concurrently
# (2.7, 2.8, 2.9, 2.10, 2.11, 2.12.0a0, 2.13.0a0) all rebuilt daily
# against the same +rocm7.13.0a<DATE> snapshot. With `--pre torch` and
# no upper bound, pip's PEP 440 ordering picks 2.13.0a0 - which since
# PyTorch PR #180485 (commit d921fd000eef, merged 2026-05-07) drops
# the legacy uppercase HIP_FOUND variable that vLLM v0.20.0 still
# checks in CMakeLists.txt:147. To refresh the pin, see
# .research/pytorch-180485-hip-found-regression/FINDINGS.md and bump
# all four (torch / torchvision / torchaudio / triton) together to a
# matched daily set, then rebuild.

FROM ubuntu:26.04

# 1. Build + inference-runtime system deps (inference-only  -  no IB, no ffmpeg).
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git \
      build-essential cmake ninja-build \
      aria2 tar xz-utils \
      libatomic1 libnuma-dev libgomp1 libelf1t64 \
      libdrm-dev zlib1g-dev libssl-dev \
      libgoogle-perftools4 \
      procps \
    && rm -rf /var/lib/apt/lists/*

# 2. TheRock ROCm SDK -> /opt/rocm. Pinned to the latest tarball AMD had
# published on rocm.nightlies.amd.com at the time of this bump: 7.14.0a20260612
# (2026-07-08 bump: was 7.13.0a20260510 — see PATCHES-DISABLED note below).
WORKDIR /tmp
ARG ROCM_MAJOR_VER=7
ARG GFX=gfx1151
COPY scripts/install_rocm_sdk.sh /tmp/install_rocm_sdk.sh
RUN chmod +x /tmp/install_rocm_sdk.sh && \
    ROCM_MAJOR_VER=${ROCM_MAJOR_VER} GFX=${GFX} /tmp/install_rocm_sdk.sh && \
    rm /tmp/install_rocm_sdk.sh

# 3. Python 3.12 venv via uv (Ubuntu 26.04 ships 3.14, no python3.12 apt pkg).
# uv pinned to 0.11.12 (last verified-working version against this stack).
COPY --from=ghcr.io/astral-sh/uv:0.11.12 /uv /usr/local/bin/uv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
RUN uv venv /opt/venv --python 3.12 && \
    uv pip install \
      pip==26.1.1 \
      wheel==0.47.0 \
      packaging==26.2 \
      setuptools==79.0.1

# 4. PyTorch + triton from the AMD v2-staging gfx1151 nightly index, all four
# pinned to the matched daily set for 2026-06-12 (2026-07-08 bump from
# 2026-05-10). Triton is listed explicitly so it cannot float independently
# of torch (it is normally a transitive dep). The +rocm7.14.0a20260612
# local-version suffix locks the wheel to this exact daily snapshot.
# NOT YET VERIFIED against Patch 18's HIP_FOUND workaround or any other
# patch in patch_strix.py — see PATCHES-DISABLED note at step 7.
RUN uv pip install --pre \
      torch==2.13.0a0+rocm7.14.0a20260612 \
      torchvision==0.27.0+rocm7.14.0a20260612 \
      torchaudio==2.11.0+rocm7.14.0a20260612 \
      triton==3.7.1+git5d6048aa.rocm7.14.0a20260612 \
      --index-url https://rocm.nightlies.amd.com/v2-staging/gfx1151/ && \
    rm -rf /root/.cache/uv /root/.cache/pip

# 5. Build tool deps for the vLLM native build (all pinned).
# setuptools-rust added 2026-07-08: new build-time requirement on the
# vLLM HEAD pin (v0.20.0 didn't need it) - not declared in vLLM's own
# build-system.requires, so it must be pre-installed for --no-build-isolation.
RUN uv pip install \
      cmake==4.3.2 \
      ninja==1.13.0 \
      numpy==2.4.4 \
      setuptools-scm==10.0.5 \
      scikit-build-core==0.12.2 \
      pybind11==3.0.4 \
      setuptools-rust==1.13.0

# 6. Conch Triton kernels  -  required by vLLM's AWQMarlin-on-ROCm path
# selected via choose_mp_linear_kernel (PR #36505). Without this, AWQ
# falls back to the slow legacy ops.awq_gemm.
RUN uv pip install conch-triton-kernels==1.3

# 7. Clone vLLM. Pinned to vllm-project/vllm main HEAD as of 2026-07-08
# (was the v0.20.0 release tag, 2026-04-23). Bump reason: PR #45413
# (merged 2026-06-15) replaces the per-model streaming reasoning/tool-call
# parsers with a unified O(n)-guaranteed engine. v0.20.0 predates that fix,
# and the old parsers degrade to O(n²) when a single decode step delivers
# multiple tokens (speculative decoding / MTP) - measured ~5x decode
# slowdown on gfx1151 when --reasoning-parser and --enable-auto-tool-choice
# are both active together with MTP. See think/vllm session notes
# (2026-07-08) in the sibling kydah/home repo for the full diagnosis.
# Override via build-arg if you need a different sha.
ARG VLLM_COMMIT=7c67da967f5f9a744fc5f6260918523fc4777417
RUN git clone https://github.com/vllm-project/vllm.git /opt/vllm
WORKDIR /opt/vllm
RUN if [ -n "$VLLM_COMMIT" ]; then \
      echo "Pinning vLLM to ${VLLM_COMMIT}"; \
      git checkout "${VLLM_COMMIT}"; \
    else \
      echo "Tracking vLLM HEAD: $(git rev-parse --short HEAD)"; \
    fi

# PATCHES-DISABLED (2026-07-08): patch_strix.py is copied into the image
# for reference but deliberately NOT run this build. All 19-20 patches
# were written against the v0.20.0 tag; against a ~2.5-month-newer vLLM
# HEAD several will likely fail to apply (file moves, renamed functions/
# classes) and a couple may no longer be needed at all (e.g. Patch 13/14
# DFlash cherry-picks, if upstream merged them by now).
#
# RISK, not just perf: Patch 18 was a hard *build* fix (CMake HIP_FOUND
# not set by newer PyTorch's LoadHIP.cmake, upstream PyTorch PR #180485).
# If vLLM HEAD's own CMakeLists.txt hasn't since adapted to that PyTorch
# change independently, step 8 below (`uv pip install .`) may fail outright
# with "Can't find CUDA or HIP installation" rather than just building an
# unoptimized image. If the GH Actions build fails there, re-enable at
# least Patch 18 (or inline its CMakeLists.txt shim) first before touching
# anything else.
#
# Re-enabling the rest of the patches is a separate triage pass: exec into
# a running pod from THIS image,
# check which of patch_strix.py's edits still apply cleanly against the
# installed vllm source, fix/drop/replace as needed, then flip this back
# to `RUN python /opt/vllm/patch_strix.py` and rebuild.
COPY scripts/patch_strix.py /opt/vllm/patch_strix.py
# RUN python /opt/vllm/patch_strix.py

# 7b. pkg-config  -  required by ROCm 7.13's rocm_smi-config.cmake which
# vLLM's find_package(Torch) → Caffe2 → LoadHIP chain pulls in. Kept as
# a separate RUN here (rather than in step 1) so adding it doesn't
# invalidate the ROCm tarball / torch wheel cache layers above.
RUN apt-get update && apt-get install -y --no-install-recommends pkg-config && \
    rm -rf /var/lib/apt/lists/*

# 8. Build vLLM against TheRock ROCm.
# CC/CXX forced to ROCm clang so the compiled extensions have an ABI
# matching the torch wheels.
ENV ROCM_HOME=/opt/rocm \
    ROCM_PATH=/opt/rocm \
    HIP_PATH=/opt/rocm \
    HIP_PLATFORM=amd \
    CMAKE_PREFIX_PATH=/opt/rocm \
    VLLM_TARGET_DEVICE=rocm \
    PYTORCH_ROCM_ARCH=gfx1151 \
    HIP_ARCHITECTURES=gfx1151 \
    GPU_TARGETS=gfx1151 \
    AMDGPU_TARGETS=gfx1151 \
    MAX_JOBS=4 \
    CC=/opt/rocm/llvm/bin/clang \
    CXX=/opt/rocm/llvm/bin/clang++

RUN export HIP_DEVICE_LIB_PATH=$(find /opt/rocm -type d -name bitcode -print -quit) && \
    echo "Building with bitcode: $HIP_DEVICE_LIB_PATH" && \
    export CMAKE_ARGS="-DROCM_PATH=/opt/rocm -DHIP_PATH=/opt/rocm -DGPU_TARGETS=gfx1151 -DHIP_ARCHITECTURES=gfx1151" && \
    uv pip install --no-build-isolation --no-deps -v . && \
    rm -rf /root/.cache/uv /root/.cache/pip /tmp/*

# 8c. Pre-tuned Triton fused-MoE kernel configs for gfx1151, keyed by
# get_device_name() (Patch 20 stabilizes that to the fixed string
# "gfx1151" - see patch_strix.py). Without these, MoE experts run through
# vLLM's untuned WNA16 Triton fallback (~5-6x slower decode than an
# equivalent CUDA/FlashInfer MoE path measured on a DGX Spark for the
# same Qwen3.6-35B-A3B checkpoint). Add one file per (E, N, dtype) shape
# tuned via benchmarks/kernels/benchmark_moe.py --tune; see README.md
# "Tuning MoE kernel configs" for the exact recipe.
COPY moe-configs/*.json /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/configs/

# 8b. vLLM runtime dependencies.
# vLLM declares deps as `dynamic` in pyproject.toml and loads them from
# requirements/common.txt + requirements/rocm.txt at build time. Built
# with --no-deps above; install the same lists here. The constraint pins
# torch/triton to our v2-staging wheels so uv doesn't replace them with
# vanilla CUDA torch from PyPI when resolving.
RUN TORCH_VER=$(python -c "import torch; print(torch.__version__)") && \
    TRITON_VER=$(python -c "import triton; print(triton.__version__)") && \
    printf "torch==%s\ntriton==%s\n" "$TORCH_VER" "$TRITON_VER" > /tmp/constraints.txt && \
    echo "Pinning runtime deps to torch==$TORCH_VER, triton==$TRITON_VER" && \
    uv pip install --no-build-isolation \
      -r /opt/vllm/requirements/common.txt \
      -r /opt/vllm/requirements/rocm.txt \
      --constraint /tmp/constraints.txt && \
    rm -rf /root/.cache/uv /root/.cache/pip /tmp/constraints.txt

# 9. Runtime env. Mirrors /etc/profile.d/rocm-sdk.sh for non-interactive use.
# These are the env vars our research found load-bearing for AWQ vision
# on gfx1151  -  see .research/flashattention-strix-halo and
# .research/vllm-strix-halo-issues for the per-flag justification.
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib \
    HIP_CLANG_PATH=/opt/rocm/llvm/bin \
    ROCBLAS_USE_HIPBLASLT=1 \
    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
    HIP_FORCE_DEV_KERNARG=1 \
    RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES=1 \
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4 \
    HSA_OVERRIDE_GFX_VERSION=11.5.1 \
    HSA_NO_SCRATCH_RECLAIM=1 \
    MIOPEN_FIND_MODE=FAST \
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    VLLM_ROCM_USE_AITER=0 \
    VLLM_USE_TRITON_AWQ=1 \
    VLLM_DISABLE_COMPILE_CACHE=1

WORKDIR /opt
CMD ["/bin/bash"]
