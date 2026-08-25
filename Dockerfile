# syntax=docker/dockerfile:1
# vllm-gfx1151 runtime image — assembled from prebuilt artifacts, no native
# compilation. Rewritten 2026-08-25 to replace the old from-source
# monolithic build (git history: previous Dockerfile compiled vLLM's own
# native extensions against a full TheRock ROCm SDK tarball inside this
# repo — kept working but every iteration paid a multi-hour ROCm+vLLM
# rebuild, and ~11.3GiB of the resulting image turned out to be dead
# test/bench/static-archive weight never touched at runtime, see
# think/vllm/DEBUG.md in the sibling kydah/home repo, 2026-08-25).
#
# Wheels (torch/triton/vllm/aiter/flash-attn + a few C-ext deps with no
# prebuilt ROCm wheel upstream) come from sebt3/stack-torch-gfx1151 —
# built from source there, once, on Ubuntu 24.04, vLLM v0.24.0, ROCm 7.14,
# with the AITER-gfx1151-gating / FLA-gfx1151 / hybrid-attention patches
# from bitserv-ai/_gfx115x_ applied (see that repo's patches/). This image
# does not build anything — it only assembles.
#
# ROCm 7.14 runtime comes from AMD's stable release repo (repo.amd.com,
# not sebt3/therock-gfx1151): that repo's 3 local patches only fix
# TheRock's own build *process*, not any library's runtime behavior, so a
# vanilla official build is ABI-identical for our purposes — and as of
# 2026-08-25 therock-gfx1151 hasn't finished a full build anyway
# (math-libs, esp. hipBLASLt codegen, needs more resumed CI runs than it's
# had). Revisit once a real behavior-changing patch exists. Also
# deliberately not using ROCm/TheRock's own CI-run artifacts (see step 2
# comment) — those turned out to have very short retention.
#
# AITER is enabled by default here (VLLM_ROCM_USE_AITER=1 + granular
# flags) — this is the thing being tested by this rewrite. The previous
# image shipped VLLM_ROCM_USE_AITER=0 because vLLM's own gfx1151 gating
# was wrong (_aiter_ops.py checks on_gfx9(), rocm_aiter_fa.py checks
# on_mi3xx() — both False for gfx1151 regardless of actual support) and
# the AITER sampler kernel crashed at first forward without a fix for
# that. stack-torch-gfx1151's aiter-gate-gfx1x.patch / aiter-fa-gfx1x-gate
# .patch target exactly that gating bug. Not yet verified end-to-end on
# real hardware as of this rewrite — if AITER still crashes, the fallback
# is setting VLLM_ROCM_USE_AITER=0 (and the granular flags become no-ops).

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

ARG STACK_TORCH_TAG=0.1.0
ARG ROCM_DIST_URL=https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-gfx1151-7.14.0.tar.gz

# 1. Runtime system deps only — nothing compiles in this image, so no
# build-essential/cmake/ninja/pkg-config. Same list as the old image's
# runtime portion, kept for parity (libnuma-dev/libelf1t64/libdrm-dev for
# the HIP runtime, libgoogle-perftools4 for the tcmalloc LD_PRELOAD below).
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git \
      python3 python3-pip python3-venv \
      libatomic1 libnuma-dev libgomp1 libelf1t64 \
      libdrm-dev zlib1g-dev libssl-dev \
      libgoogle-perftools4 \
      procps \
    && rm -rf /var/lib/apt/lists/*

# 2. ROCm 7.14 runtime for gfx1151, from AMD's stable release repo
# (repo.amd.com/rocm/tarball-multi-arch — the actual apt/yum ROCm package
# mirror, not a CI staging bucket). Dated 2026-07-15, matching the
# therock-7.14 GitHub tag exactly.
#
# 2026-08-25: originally tried install_rocm_from_artifacts.py pinned to
# the specific ROCm/TheRock CI run stack-torch-gfx1151 built against
# (29052575219) — worked for them on 08-21/22, gone 3 days later (that
# run's raw per-component artifacts on s3://therock-ci-artifacts had
# vanished, confirmed directly against the bucket; presumably short
# retention on CI-run-scoped artifacts, not meant for later reuse). The
# other AMD-hosted nightly channel (rocm.nightlies.amd.com /
# therock-nightly-tarball) stops publishing gfx1151 builds after
# 2026-06-08 — superseded, not a viable fallback either. repo.amd.com is
# the one AMD-hosted location that's actually meant to be durable.
WORKDIR /tmp
RUN mkdir -p /opt/rocm && \
    curl -fsSL "${ROCM_DIST_URL}" | tar xz -C /opt/rocm

# 3. Strip dev/test/bench cruft that install_rocm_from_artifacts.py still
# pulls in (it filters by component, not by file type within a component).
# Confirmed dead weight for vLLM inference — think/vllm/DEBUG.md
# 2026-08-25 image-bloat finding: keep only what's actually loaded at
# runtime (libamdhip64, rocBLAS/hipBLASLt for GEMM, lib/llvm — Triton's
# JIT needs a working `clang` at runtime, verified via `which clang` on
# the old image). RCCL dropped: single-iGPU, no NCCL (same call the old
# Dockerfile made). Static archives (.a) are never loaded by a
# dynamically-linked process, dead weight by construction.
RUN rm -rf \
      /opt/rocm/bin /opt/rocm/clients /opt/rocm/share /opt/rocm/tests \
      /opt/rocm/lib/*.a \
      /opt/rocm/lib/librccl* \
      /opt/rocm/lib/rdc /opt/rocm/lib/librocprof-sys* /opt/rocm/lib/rocprofiler-systems

# 4. Python 3.13 venv via uv — matches the cpython_version stack-torch-
# gfx1151 built its wheels against (vllm-packages.yaml: "3.13.9"; a cp313
# wheel is ABI-compatible with any 3.13.x). uv pinned to the same
# last-verified-working version as the old image.
COPY --from=ghcr.io/astral-sh/uv:0.11.12 /uv /usr/local/bin/uv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
RUN uv venv /opt/venv --python 3.13 && \
    uv pip install \
      pip==26.1.1 \
      wheel==0.47.0 \
      packaging==26.2 \
      setuptools==79.0.1

# 5. Install the prebuilt wheels from stack-torch-gfx1151's release
# (torch, triton, vllm, amd-aiter, flash-attn, plus asyncpg/sentencepiece/
# zstandard/numpy — C-ext deps with no prebuilt ROCm wheel available
# upstream, bundled there so this step stays fully offline-capable for
# these). --no-deps: they were built --no-deps too; the rest of vLLM's
# deps come from step 6.
ARG STACK_TORCH_URL=https://github.com/sebt3/stack-torch-gfx1151/releases/download/${STACK_TORCH_TAG}/stack-torch-gfx1151-${STACK_TORCH_TAG}.tar.gz
RUN mkdir -p /tmp/wheels && \
    curl -fsSL "${STACK_TORCH_URL}" | tar xz -C /tmp/wheels --strip-components=1 && \
    uv pip install --no-deps /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

# 6. vLLM's own runtime dependencies. vLLM declares deps as `dynamic` in
# pyproject.toml, loaded from requirements/common.txt + requirements/
# rocm.txt at its own build time — the wheel was built --no-deps, so
# install the same lists here, pinned to the exact tag stack-torch-gfx1151
# built (v0.24.0) so the requirement pins match what the wheel actually
# needs. Constraint file pins torch/triton to our from-source builds so
# resolution doesn't silently pull vanilla CUDA torch from PyPI.
ARG VLLM_TAG=v0.24.0
RUN TORCH_VER=$(python -c "from importlib.metadata import version; print(version('torch'))") && \
    TRITON_VER=$(python -c "from importlib.metadata import version; print(version('triton'))") && \
    printf "torch==%s\ntriton==%s\n" "$TORCH_VER" "$TRITON_VER" > /tmp/constraints.txt && \
    echo "Pinning runtime deps to torch==$TORCH_VER, triton==$TRITON_VER" && \
    curl -fsSL -o /tmp/common.txt "https://raw.githubusercontent.com/vllm-project/vllm/${VLLM_TAG}/requirements/common.txt" && \
    curl -fsSL -o /tmp/rocm.txt "https://raw.githubusercontent.com/vllm-project/vllm/${VLLM_TAG}/requirements/rocm.txt" && \
    uv pip install -r /tmp/common.txt -r /tmp/rocm.txt --constraint /tmp/constraints.txt && \
    rm -rf /tmp/common.txt /tmp/rocm.txt /tmp/constraints.txt /root/.cache/uv /root/.cache/pip

# 7. Pre-tuned Triton fused-MoE kernel configs for gfx1151 (unchanged from
# the previous image). Still relevant for any layer AITER_MOE doesn't
# cover — without these vLLM falls back to an untuned default config
# (confirmed on the old image, 2026-08-25: "Using default MoE config.
# Performance might be sub-optimal!"). Add new shapes via
# benchmarks/kernels/benchmark_moe.py --tune; see README.md.
COPY moe-configs/*.json /opt/venv/lib/python3.13/site-packages/vllm/model_executor/layers/fused_moe/configs/

# 8. Runtime env. Mirrors the old image's flag set, with AITER flipped on
# (see header comment) and the AWQMarlin/Conch-specific bits dropped
# (Conch (step 6 of the old Dockerfile) was for a linear-layer AWQ path;
# not reinstated here yet — re-add if a checkpoint needs it and AITER's
# own linear path doesn't cover it).
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
    VLLM_USE_TRITON_AWQ=1 \
    VLLM_DISABLE_COMPILE_CACHE=1 \
    VLLM_ROCM_USE_AITER=1 \
    VLLM_ROCM_USE_AITER_LINEAR=1 \
    VLLM_ROCM_USE_AITER_MOE=1 \
    VLLM_ROCM_USE_AITER_RMSNORM=1 \
    VLLM_ROCM_USE_AITER_MHA=1

WORKDIR /opt
CMD ["/bin/bash"]
