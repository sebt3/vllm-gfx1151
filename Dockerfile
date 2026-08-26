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
# built from source there, once, on Ubuntu 24.04, vLLM v0.28.0, ROCm 7.14,
# with the AITER-gfx1151-gating / FLA-gfx1151 / hybrid-attention patches
# from bitserv-ai/_gfx115x_ applied (see that repo's patches/, re-triaged
# against v0.28.0 on 2026-08-26). This image does not build anything — it
# only assembles.
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

ARG STACK_TORCH_TAG=0.2.0
ARG ROCM_DIST_URL=https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-gfx1151-7.14.0.tar.gz

# 1. Runtime system deps. Originally "nothing compiles in this image, so
# no build-essential" — no longer true: build-essential (gcc/g++/make/
# libc6-dev/libstdc++-dev) added 2026-08-26 because AITER JIT-compiles
# some of its own HIP/CK kernels at first use (see step 9's CC/CXX
# comment for the compiler-discovery half of this). Real-hardware boot,
# VLLM_ROCM_USE_AITER=1: clang (CC/CXX, pointed at the toolchain kept
# for Triton's own JIT) failed with "Could not find standard C++ header
# 'cmath'" / "'cstdlib' file not found" - this image had zero C/C++
# headers installed. We don't use gcc/g++ themselves (CC/CXX still
# point at clang), just the headers build-essential drags in - accepted
# the size for it: perf from AITER actually working matters more than
# shipping this image lean, plenty of headroom on this hardware either
# way (explicit call, 2026-08-26 — see think/vllm/DEBUG.md).
# libnuma-dev/libelf1t64/libdrm-dev for the HIP runtime,
# libgoogle-perftools4 for the tcmalloc LD_PRELOAD below.
# libprotobuf32t64 + libsleef3: NOT part of the ROCm SDK tarball (step 2) —
# direct runtime deps of torch's own .so files (libtorch_cpu.so pulls in
# libprotobuf.so.32, libsleef.so.3). Found by readelf -d across every .so
# in the stack-torch-gfx1151 wheels rather than one crash-and-fix cycle at
# a time (2026-08-25: first boot on real hardware crashed on
# libprotobuf.so.32 alone; checked the rest before the next attempt).
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git \
      python3 python3-pip python3-venv \
      build-essential \
      libatomic1 libnuma-dev libgomp1 libelf1t64 \
      libdrm-dev zlib1g-dev libssl-dev \
      libgoogle-perftools4 libprotobuf32t64 libsleef3 \
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

# 3. Python 3.13 venv via uv — matches the cpython_version stack-torch-
# gfx1151 built its wheels against (vllm-packages.yaml: "3.13.9"; a cp313
# wheel is ABI-compatible with any 3.13.x). uv pinned to the same
# last-verified-working version as the old image. Moved ahead of the
# /opt/rocm cleanup (was step 3, now step 5) because that cleanup strips
# /opt/rocm/share — which step 4 below needs to still be there.
COPY --from=ghcr.io/astral-sh/uv:0.11.12 /uv /usr/local/bin/uv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
RUN uv venv /opt/venv --python 3.13 && \
    uv pip install \
      pip==26.1.1 \
      wheel==0.47.0 \
      packaging==26.2 \
      setuptools==79.0.1

# 4. amdsmi — vLLM's ROCm platform detection imports this unconditionally
# (vllm/platforms/__init__.py) to decide it's even running on ROCm at
# all; without it every platform probe fails and vLLM raises "Failed to
# infer device type" before it gets anywhere near loading a model
# (real-hardware boot, 2026-08-25 — confirmed via `podman run` +
# VLLM_LOGGING_LEVEL=DEBUG: "ROCm platform is not available because: No
# module named 'amdsmi'"). stack-torch-gfx1151's own build-vllm.sh builds
# an amdsmi wheel from TheRock's share/amd_smi, but it didn't make it
# into the 0.1.0 release tarball (that build step apparently failed or
# wasn't reached in whichever resumed run produced it). This repo.amd.com
# ROCm tarball ships the same amd_smi source under share/amd_smi/
# (setup.py + a pure-Python ctypes wrapper around libamd_smi.so, no
# compilation needed) — install it straight from there instead of
# waiting on that wheel to exist.
RUN uv pip install /opt/rocm/share/amd_smi

# 5. Strip dev/test/bench cruft that the ROCm tarball still carries.
# Confirmed dead weight for vLLM inference — think/vllm/DEBUG.md
# 2026-08-25 image-bloat finding: keep only what's actually loaded at
# runtime (libamdhip64, rocBLAS/hipBLASLt for GEMM, lib/llvm — Triton's
# JIT needs a working `clang` at runtime). RCCL dropped: single-iGPU, no
# NCCL. Static archives (.a) are never loaded by a dynamically-linked
# process, dead weight by construction.
#
# 2026-08-25/26: that DEBUG.md finding was against the OLD 26.9GiB image
# (a different, much fatter ROCm tarball where /opt/rocm/bin alone was
# 5.3GiB of test/bench binaries). This repo.amd.com tarball is already
# lean - /opt/rocm/bin totals ~345MB, /opt/rocm/share ~36MB (no
# clients/, no tests/ at all) - and both turned out to hide small,
# genuinely load-bearing files: share/amd_smi (step 4, before this
# runs), and then two AITER runtime deps found one real-hardware crash
# at a time (VLLM_ROCM_USE_AITER=1): hipconfig (needed for HIP version
# detection) and rocminfo (needed for GPU arch detection, "Could not
# find rocminfo in PATH or ROCM_HOME"). Given the whole directories are
# only a few hundred MB combined - cheap in an image already carrying a
# ~24GiB PyTorch/Triton build - keeping bin/ and share/ wholesale and
# only excluding the handful of confirmed multi-MB items nothing here
# uses (profilers, debuggers, integration tests, a vendored MIOpen
# driver) is more robust than continuing to whack-a-mole individual
# files back in as each new crash names one.
RUN rm -rf \
      /opt/rocm/bin/rocshmem_info /opt/rocm/bin/hipify-clang \
      /opt/rocm/bin/rocprof-sys-* /opt/rocm/bin/rocgdb-py* \
      /opt/rocm/bin/rdcd /opt/rocm/bin/rdci \
      /opt/rocm/bin/hipdnn_integration_tests /opt/rocm/bin/MIOpenDriver \
      /opt/rocm/bin/flatc \
      /opt/rocm/share/doc /opt/rocm/share/man \
      /opt/rocm/lib/*.a \
      /opt/rocm/lib/librccl* \
      /opt/rocm/lib/rdc /opt/rocm/lib/librocprof-sys* /opt/rocm/lib/rocprofiler-systems

# 6. Install the prebuilt wheels from stack-torch-gfx1151's release
# (torch, triton, vllm, amd-aiter, flash-attn, plus asyncpg/sentencepiece/
# zstandard/numpy — C-ext deps with no prebuilt ROCm wheel available
# upstream, bundled there so this step stays fully offline-capable for
# these). --no-deps: they were built --no-deps too; the rest of vLLM's
# deps come from step 7.
ARG STACK_TORCH_URL=https://github.com/sebt3/stack-torch-gfx1151/releases/download/${STACK_TORCH_TAG}/stack-torch-gfx1151-${STACK_TORCH_TAG}.tar.gz
RUN mkdir -p /tmp/wheels && \
    curl -fsSL "${STACK_TORCH_URL}" | tar xz -C /tmp/wheels --strip-components=1 && \
    uv pip install --no-deps /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

# 7. vLLM's own runtime dependencies. vLLM declares deps as `dynamic` in
# pyproject.toml, loaded from requirements/common.txt + requirements/
# rocm.txt at its own build time — the wheel was built --no-deps, so
# install the same lists here, pinned to the exact tag stack-torch-gfx1151
# built (v0.28.0) so the requirement pins match what the wheel actually
# needs. Constraint file pins torch/triton to our from-source builds so
# resolution doesn't silently pull vanilla CUDA torch from PyPI.
#
# torchvision gets uninstalled right after: common.txt pulls it in
# transitively (mistral_common[image], timm) but stack-torch-gfx1151
# deliberately doesn't build it from source (git log: "unresolved
# upstream duplicate-symbol bug"), so the copy requirements resolution
# grabs from PyPI is a vanilla build compiled against a different torch -
# crashes at import (torch._C._dispatch_has_kernel_for_dispatch_key
# inside torchvision's own op registration; real-hardware boot,
# 2026-08-25) the moment anything imports `transformers`, which vLLM does
# unconditionally regardless of text_only. Removing it lets transformers'
# normal "torchvision not available" fallback take over instead (it's
# always been an optional dep there) - we run text_only: true everywhere
# anyway, so no vision path is actually lost.
ARG VLLM_TAG=v0.28.0
RUN TORCH_VER=$(python -c "from importlib.metadata import version; print(version('torch'))") && \
    TRITON_VER=$(python -c "from importlib.metadata import version; print(version('triton'))") && \
    printf "torch==%s\ntriton==%s\n" "$TORCH_VER" "$TRITON_VER" > /tmp/constraints.txt && \
    echo "Pinning runtime deps to torch==$TORCH_VER, triton==$TRITON_VER" && \
    curl -fsSL -o /tmp/common.txt "https://raw.githubusercontent.com/vllm-project/vllm/${VLLM_TAG}/requirements/common.txt" && \
    curl -fsSL -o /tmp/rocm.txt "https://raw.githubusercontent.com/vllm-project/vllm/${VLLM_TAG}/requirements/rocm.txt" && \
    uv pip install -r /tmp/common.txt -r /tmp/rocm.txt --constraint /tmp/constraints.txt && \
    uv pip uninstall torchvision && \
    rm -rf /tmp/common.txt /tmp/rocm.txt /tmp/constraints.txt /root/.cache/uv /root/.cache/pip

# 7c. pybind11 — aiter JIT-compiles some of its own kernels at first use
# (see step 9's CC/CXX comment) and that build path needs pybind11 to
# generate the Python bindings; neither vllm's nor aiter's declared
# requirements pull it in as a runtime dep (it's normally a build-time-
# only need, invisible until something actually triggers a JIT build).
# Real-hardware boot, 2026-08-26, VLLM_ROCM_USE_AITER=1: "ModuleNotFoundError:
# No module named 'pybind11'" from inside aiter's ninja-file generation.
RUN uv pip install pybind11

# 7b. vllm/triton_utils/__init__.py unconditionally imports several
# newer Triton symbols when HAS_TRITON is true that our custom-built
# Triton (main_perf @ 0ec280cf) doesn't have - same class of drift as
# stack-torch-gfx1151's triton-knobs-import-fix.patch (triton.knobs,
# also missing from this exact pin). Found two so far by real-hardware
# boot, one crash at a time (2026-08-26): `triton.experimental.gluon`,
# then `triton.language.core._aggregate`. Patched here directly
# (installed site-packages) rather than round-tripping through a
# stack-torch-gfx1151 patch + ~6h rebuild for a few-line fix - fold
# these into that repo's patches/ next time a real rebuild happens
# anyway. Each falls back to the file's own TritonLanguagePlaceholder,
# exactly like its HAS_TRITON=False branch already does for the same
# symbols - if another missing-symbol crash turns up, extend this same
# script rather than guessing all of them up front.
RUN python3 <<'PYEOF'
import pathlib
p = pathlib.Path("/opt/venv/lib/python3.13/site-packages/vllm/triton_utils/__init__.py")
src = p.read_text()

fixes = [
    (
        "    from triton.experimental import gluon\n"
        "    from triton.experimental.gluon import language as gl\n"
        ,
        "    try:\n"
        "        from triton.experimental import gluon\n"
        "        from triton.experimental.gluon import language as gl\n"
        "    except ImportError:\n"
        "        gluon = TritonLanguagePlaceholder()\n"
        "        gl = TritonLanguagePlaceholder()\n"
    ),
    (
        "    from triton.language.core import _aggregate as aggregate  # noqa: E501\n"
        ,
        "    try:\n"
        "        from triton.language.core import _aggregate as aggregate  # noqa: E501\n"
        "    except ImportError:\n"
        "        aggregate = TritonLanguagePlaceholder()\n"
    ),
]
for old, new in fixes:
    assert old in src, f"expected block not found (layout changed?): {old!r}"
    src = src.replace(old, new, 1)
p.write_text(src)
PYEOF

# 8. Pre-tuned Triton fused-MoE kernel configs for gfx1151 (unchanged from
# the previous image). Still relevant for any layer AITER_MOE doesn't
# cover — without these vLLM falls back to an untuned default config
# (confirmed on the old image, 2026-08-25: "Using default MoE config.
# Performance might be sub-optimal!"). Add new shapes via
# benchmarks/kernels/benchmark_moe.py --tune; see README.md.
COPY moe-configs/*.json /opt/venv/lib/python3.13/site-packages/vllm/model_executor/layers/fused_moe/configs/

# 9. Runtime env. Mirrors the old image's flag set, with AITER flipped on
# (see header comment) and the AWQMarlin/Conch-specific bits dropped
# (Conch (step 6 of the old Dockerfile) was for a linear-layer AWQ path;
# not reinstated here yet — re-add if a checkpoint needs it and AITER's
# own linear path doesn't cover it).
#
# ROCM_HOME/ROCM_PATH: aiter's _find_rocm_home() checks these first
# (before falling back to `which hipcc`/default-path guesses) - set
# explicitly rather than relying on the fallback chain working.
# CC/CXX: aiter JIT-compiles some of its own kernels at first use (like
# Triton, but via a plain PyTorch-style C++ extension build - ninja +
# a real compiler), and defaults to bare `cc`/`c++` if unset - neither
# exists in this image (no build-essential, this is assembly-only).
# Real-hardware boot, 2026-08-26, VLLM_ROCM_USE_AITER=1: "TypeError:
# expected str, bytes or os.PathLike object, not NoneType" from
# shutil.which(get_cxx_compiler()) inside aiter's JIT build path.
# Point both at the clang toolchain already kept for Triton's own JIT.
ENV LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib \
    HIP_CLANG_PATH=/opt/rocm/llvm/bin \
    ROCM_HOME=/opt/rocm \
    ROCM_PATH=/opt/rocm \
    CC=/opt/rocm/llvm/bin/clang \
    CXX=/opt/rocm/llvm/bin/clang++ \
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
