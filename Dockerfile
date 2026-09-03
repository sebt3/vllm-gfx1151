# syntax=docker/dockerfile:1
# vllm-gfx1151 runtime image — pure assembly from prebuilt artifacts, no
# native compilation.
#
# Rewritten 2026-09-02: wheel source moved from sebt3/stack-torch-gfx1151
# (from-source build of torch+triton+vllm with vendored bitserv-ai/_gfx115x_
# patches) to vLLM's own published ROCm wheel set at
# https://wheels.vllm.ai/rocm/ . Reason: the from-source chain
# (therock-gfx1151 -> stack-torch-gfx1151 -> this repo) never produced a
# working image — the last three attempts all died in the Triton compiler
# on gfx1151 (SyntaxError codegen on 0.3.1, then a WMMA/LDS segfault in
# OptimizeAMDLDSUsage on 0.3.2/0.4.0 — see think/vllm/DEBUG.md 2026-08-30
# / 08-31), each iteration costing a multi-hour rebuild. stack-torch is
# retired as the main line (kept as a fallback only if a PyTorch/ROCm
# patch nobody upstream carries becomes genuinely necessary).
#
# What wheels.vllm.ai/rocm/ gives us, all built together by vLLM CI and
# mutually ABI-pinned (Requires-Dist carries exact `==x+localver` pins):
#
#   vllm         0.28.0+rocm723   (git 2cf0a6915 — the *exact* commit
#                                  stack-torch 0.3.2/0.4.0 was building;
#                                  this commit already got past model load
#                                  on real hardware, DEBUG.md 2026-08-31)
#   torch        2.12.0+git6bbd260 — libtorch_hip.so is a fat multi-arch
#                                    build, gfx1151/1150/1152/1153 all
#                                    present (verified: strings | grep gfx).
#                                    Reports hip 7.2.53211 / rocm 7.2.3.
#   triton       3.7.1+gitf0b55c07 — vLLM's matched Triton. gfx11 -> wave32
#                                    + -real-true16 handled in
#                                    backends/amd/compiler.py. Whether the
#                                    WMMA-layout segfault from the dead
#                                    main_perf@0ec280cf pin is fixed here
#                                    is THE open question for the first
#                                    real-hardware boot of this image.
#   torchvision  0.27.1+df56172
#   torchaudio   2.11.0+34c52a6    (hard dep of vllm 0.28)
#   amd-aiter    0.1.19
#   flash-attn   2.8.3
#   amdsmi       26.2.2+c2d9476115
#
# This is the same wheel source lemonade-sdk/vllm-rocm consumes for its
# gfx1151 bundle (which passes a 16-test gfx1151 hardware qualification),
# just assembled our way with our runtime config + MoE configs.
#
# ROCm 7.14 runtime still comes from AMD's stable repo (repo.amd.com) as a
# tarball into /opt/rocm — REQUIRED, not optional: the torch wheel bundles
# no ROCm userspace and declares no rocm-sdk dependency, so libamdhip64 /
# librocblas / libhipBLASLt and the llvm/clang+lld that Triton's (and
# aiter's) JIT needs at runtime all come from here. "7.14.0" is TheRock's
# dist-package version; the ROCm/HIP SDK inside is ~7.2, matching the torch
# wheel's reported hip 7.2.x.
#
# AITER is ON since alpha.11. vLLM's upstream gfx1151 AITER gating is wrong
# (_aiter_ops.py / rocm_aiter_fa.py gate on CDNA/MI3xx — False for
# gfx1151); step 6d re-ports stack-torch's gate patches as site-packages
# .py patches (patches/), adding the gfx1x sibling so AITER's
# RDNA3.5-tuned Triton kernels engage. Same step fixes the FLA/GDN
# autotune space (NVIDIA-shaped tiles overflow gfx1151's 64KB LDS). This
# is the fix for the ~4 tok/s decode measured on the plain wheel
# (2026-09-02, same-bandwidth DGX does 20-35 on a bigger model).

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

ARG ROCM_DIST_URL=https://repo.amd.com/rocm/tarball-multi-arch/therock-dist-linux-gfx1151-7.14.0.tar.gz
# wheels.vllm.ai/rocm/ serves a PEP503 index (torch/, vllm/, ...) whose
# links point at a per-build commit dir. Pin the commit so a CI-side
# rebuild of a different vLLM commit can't silently shift versions under us.
ARG VLLM_WHEELS_COMMIT=2cf0a6915ce544dc493a0990f2ea38d81601128a
ARG VLLM_WHEELS_BASE=https://wheels.vllm.ai/rocm/${VLLM_WHEELS_COMMIT}
ARG VLLM_TAG=v0.28.0

# 1. Runtime system deps.
# build-essential: aiter JIT-compiles some of its own HIP/CK kernels at
# first use via a plain PyTorch-style C++ extension build (ninja + a real
# compiler) — it needs C/C++ *headers* to exist even when CC/CXX point at
# clang (real-hardware boot 2026-08-26: clang failed with "Could not find
# standard C++ header 'cmath'"). Kept even though AITER is off by default —
# turning it on shouldn't need an image rebuild. Triton's JIT only needs
# clang/lld from /opt/rocm/llvm, not this.
# libprotobuf32t64 + libsleef3: direct runtime deps of torch's own .so
# (libtorch_cpu.so needs libprotobuf.so.32 + libsleef.so.3), NOT in the
# ROCm tarball. libgoogle-perftools4 for the tcmalloc LD_PRELOAD below.
# libnuma/libelf/libdrm for the HIP runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git \
      python3.12 python3.12-venv python3.12-dev \
      build-essential \
      libatomic1 libnuma-dev libgomp1 libelf1t64 \
      libdrm-dev zlib1g-dev libssl-dev \
      libgoogle-perftools4 libprotobuf32t64 libsleef3 \
      procps \
    && rm -rf /var/lib/apt/lists/*

# 2. ROCm 7.14 runtime for gfx1151, from AMD's stable release repo
# (repo.amd.com/rocm/tarball-multi-arch — the actual apt/yum ROCm package
# mirror, not a CI staging bucket; the other two AMD-hosted channels both
# proved non-durable — see git history / README). Dated 2026-07-15,
# matching the therock-7.14 GitHub tag.
WORKDIR /tmp
RUN mkdir -p /opt/rocm && \
    curl -fsSL "${ROCM_DIST_URL}" | tar xz -C /opt/rocm

# 3. Python 3.12 venv via uv — the wheels.vllm.ai ROCm set is cp312 only
# (was 3.13 with stack-torch, which built its own cp313 wheels). uv pinned
# to the last-verified-working version.
COPY --from=ghcr.io/astral-sh/uv:0.11.12 /uv /usr/local/bin/uv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH
# Cap uv parallelism: unpacking torch (2GB) + ~200 deps concurrently
# peaked past this build host's RAM and OOM-killed the session
# (2026-09-02). Serialised installs + no cache keeps the footprint flat.
ENV UV_CONCURRENT_INSTALLS=1 UV_CONCURRENT_DOWNLOADS=4 UV_NO_CACHE=1
RUN uv venv /opt/venv --python 3.12 && \
    uv pip install \
      pip==26.1.1 \
      wheel==0.47.0 \
      packaging==26.2 \
      setuptools==79.0.1

# 4. Strip dev/test/bench cruft the ROCm tarball still carries. This
# repo.amd.com tarball is already fairly lean (/opt/rocm/bin ~345MB,
# /opt/rocm/share ~36MB) but both hide small load-bearing files — keep
# bin/ and share/ wholesale, only drop the confirmed multi-MB items
# nothing on the vLLM inference path touches. KEEP /opt/rocm/lib/llvm
# (Triton + aiter JIT need a working clang/lld at *runtime*), libamdhip64,
# librocblas, libhipblaslt. Static archives (.a) are never dlopen'd — dead
# weight by construction.
# librccl KEPT (was dropped on the stack-torch image): this torch's
# libtorch_hip.so has a hard NEEDED on librccl.so.1, so `import torch`
# fails without it. The tarball's librccl is only ~4MB and ships its own
# SONAME symlinks — cheap, just leave it.
RUN rm -rf \
      /opt/rocm/bin/rocshmem_info /opt/rocm/bin/hipify-clang \
      /opt/rocm/bin/rocprof-sys-* /opt/rocm/bin/rocgdb-py* \
      /opt/rocm/bin/rdcd /opt/rocm/bin/rdci \
      /opt/rocm/bin/hipdnn_integration_tests /opt/rocm/bin/MIOpenDriver \
      /opt/rocm/bin/flatc \
      /opt/rocm/share/doc /opt/rocm/share/man \
      /opt/rocm/lib/*.a \
      /opt/rocm/lib/rdc /opt/rocm/lib/librocprof-sys* /opt/rocm/lib/rocprofiler-systems

# 5. The prebuilt ROCm wheel set from wheels.vllm.ai/rocm/. Downloaded by
# exact filename from the pinned commit dir (the index's per-package pages
# are PEP503 but link back into ${VLLM_WHEELS_COMMIT}/). --no-deps: the
# rest of vLLM's runtime deps come from step 6, same as the old image did
# with the stack-torch wheels. Order doesn't matter with --no-deps.
RUN mkdir -p /tmp/wheels && cd /tmp/wheels && \
    for w in \
      "vllm-0.28.0%2Brocm723-cp312-cp312-manylinux_2_39_x86_64.whl" \
      "torch-2.12.0%2Bgit6bbd260-cp312-cp312-manylinux_2_39_x86_64.whl" \
      "torchvision-0.27.1%2Bdf56172-cp312-cp312-manylinux_2_39_x86_64.whl" \
      "torchaudio-2.11.0%2B34c52a6-cp312-cp312-manylinux_2_39_x86_64.whl" \
      "triton-3.7.1%2Bgitf0b55c07-cp312-cp312-manylinux_2_35_x86_64.whl" \
      "amd_aiter-0.1.19-cp312-cp312-manylinux_2_39_x86_64.whl" \
      "flash_attn-2.8.3-cp312-cp312-manylinux_2_39_x86_64.whl" \
      "amdsmi-26.2.2%2Bc2d9476115-py3-none-any.whl" \
    ; do \
      echo "fetch ${w}" && \
      curl -fsSL -o "$(python3 -c "import urllib.parse,sys;print(urllib.parse.unquote(sys.argv[1]))" "${w}")" \
        "${VLLM_WHEELS_BASE}/${w}" ; \
    done && \
    uv pip install --no-deps /tmp/wheels/*.whl && \
    cd / && rm -rf /tmp/wheels

# 5b. MPI stub. This torch is built WITH MPI: libtorch_cpu.so /
# libtorch_python.so reference ~178 external MPI symbols (C MPI_*/MPIX_*,
# the deprecated C++ MPI:: bindings, and OpenMPI's ompi_mpi_* datatype
# objects) that must *resolve* at `from torch._C import *` — not merely
# satisfy the libmpi.so.40 / libmpi_cxx.so.40 DT_NEEDED (an empty .so got
# past _load_global_deps but then died on undefined symbol
# _ZN3MPI8Datatype4FreeEv, 2026-09-02 rennes boot). Real libopenmpi3 drags
# a conflicting ROCm 5.x / UCX userspace tail next to /opt/rocm 7.14.
# Instead: enumerate every undefined MPI symbol across torch/lib and emit a
# no-op definition for each into both SONAMEs. vLLM's process group is
# nccl/gloo on a single iGPU — MPI_Init is never called, so no-ops are
# safe. c10d::ProcessGroupMPII symbols are torch-internal (defined in
# libtorch_cpu, U only in libtorch_python) — the ^MPI_/^_ZN3MPI/^ompi_
# anchors exclude them so we don't interpose torch's own code.
RUN set -eu; \
    cd /opt/venv/lib/python3.12/site-packages/torch/lib; \
    nm -D --undefined-only *.so 2>/dev/null \
      | awk '$1=="U"{print $2}' \
      | grep -E '^(MPI_|MPIX_|PMPI_|_ZN3MPI|_ZNK3MPI|ompi_)' \
      | sort -u \
      | awk '{print "void " $0 "(void){}"}' > /tmp/mpi_stub.c; \
    echo "MPI stub: $(grep -c . /tmp/mpi_stub.c) symbols"; \
    for s in libmpi.so.40 libmpi_cxx.so.40; do \
      gcc -x c -shared -fPIC -Wl,-soname,"$s" -o "/usr/local/lib/$s" /tmp/mpi_stub.c; \
    done; \
    ldconfig; \
    rm -f /tmp/mpi_stub.c; \
    LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib \
      python -c "import ctypes, os; ctypes.CDLL('/opt/venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so', mode=os.RTLD_NOW|os.RTLD_GLOBAL); print('libtorch_cpu.so: all symbols resolve (MPI stub complete)')"

# 6. vLLM's own runtime dependencies. vLLM declares deps as `dynamic`,
# loaded from requirements/common.txt + rocm.txt at its build time — the
# wheel was built with those, so install the same lists here, pinned to
# the tag closest to what the wheel was cut from (v0.28.0; the wheel is
# 0.28.0+rocm723 == that release). The constraint file pins
# torch/triton/torchvision/torchaudio to the exact wheels.vllm.ai versions
# so the resolver can't swap our gfx1151-capable builds for vanilla CUDA
# wheels off PyPI (common.txt pulls torchvision transitively via
# mistral_common[image] / timm; rocm.txt drags datasets/peft/etc).
RUN TORCH_VER=$(python -c "from importlib.metadata import version; print(version('torch'))") && \
    TRITON_VER=$(python -c "from importlib.metadata import version; print(version('triton'))") && \
    TVIS_VER=$(python -c "from importlib.metadata import version; print(version('torchvision'))") && \
    TAUD_VER=$(python -c "from importlib.metadata import version; print(version('torchaudio'))") && \
    printf "torch==%s\ntriton==%s\ntorchvision==%s\ntorchaudio==%s\n" \
      "$TORCH_VER" "$TRITON_VER" "$TVIS_VER" "$TAUD_VER" > /tmp/constraints.txt && \
    echo "Pinning: torch==$TORCH_VER triton==$TRITON_VER torchvision==$TVIS_VER torchaudio==$TAUD_VER" && \
    curl -fsSL -o /tmp/common.txt "https://raw.githubusercontent.com/vllm-project/vllm/${VLLM_TAG}/requirements/common.txt" && \
    curl -fsSL -o /tmp/rocm.txt "https://raw.githubusercontent.com/vllm-project/vllm/${VLLM_TAG}/requirements/rocm.txt" && \
    uv pip install -r /tmp/common.txt -r /tmp/rocm.txt --constraint /tmp/constraints.txt && \
    rm -rf /tmp/common.txt /tmp/rocm.txt /tmp/constraints.txt /root/.cache/uv /root/.cache/pip

# 6b. aiter JIT-build deps not pulled by anything until a kernel actually
# JIT-compiles (invisible until then): pybind11 (bindings gen) and
# flydsl==0.2.4 (new aiter 0.1.19 dep — check it's on PyPI). Cheap to have
# present even with AITER off.
RUN uv pip install pybind11 "flydsl==0.2.4"

# 6c. Defensive shim: vllm/triton_utils/__init__.py imports a few Triton
# symbols unconditionally when HAS_TRITON. triton 3.7.1 is vLLM 0.28's
# *matched* Triton so these should all exist (unlike the old main_perf
# 3.0.0 pin where gluon / _aggregate were missing and had to be shimmed) —
# this is now best-effort: patch only the blocks that are present, never
# fail the build. If a real-hardware boot still crashes on a missing
# Triton symbol, add its import block here.
RUN python3 <<'PYEOF'
import pathlib
p = pathlib.Path("/opt/venv/lib/python3.12/site-packages/vllm/triton_utils/__init__.py")
if not p.exists():
    raise SystemExit("triton_utils/__init__.py not found — layout changed, revisit")
src = p.read_text()
fixes = [
    (
        "    from triton.experimental import gluon\n"
        "    from triton.experimental.gluon import language as gl\n",
        "    try:\n"
        "        from triton.experimental import gluon\n"
        "        from triton.experimental.gluon import language as gl\n"
        "    except ImportError:\n"
        "        gluon = TritonLanguagePlaceholder()\n"
        "        gl = TritonLanguagePlaceholder()\n",
    ),
    (
        "    from triton.language.core import _aggregate as aggregate  # noqa: E501\n",
        "    try:\n"
        "        from triton.language.core import _aggregate as aggregate  # noqa: E501\n"
        "    except ImportError:\n"
        "        aggregate = TritonLanguagePlaceholder()\n",
    ),
]
applied = 0
for old, new in fixes:
    if old in src:
        src = src.replace(old, new, 1)
        applied += 1
p.write_text(src)
print(f"triton_utils shim: {applied}/{len(fixes)} blocks patched (0 is fine if 3.7.1 has them)")
PYEOF

# 6d. gfx1151 kernel patches (see patches/README.md). Two families:
#
# FLA/GDN autotune + AITER Triton (alpha.11): vLLM's Triton kernels
# autotune over NVIDIA-shaped config spaces (64/128 tiles, 3-4 stages)
# that overflow gfx1151's 64KB LDS.
#
# RDNA3 native W4A16 HIP kernels (alpha.13): vLLM 0.28 ships a hand-written
# RDNA3 WMMA int4 GEMM + fused-MoE kernel (moe_gptq_gemm_rdna3 et al.,
# compiled for gfx1151 — verified in _rocm_C's fatbin), but both its
# selectors gate on on_gfx1100() only ("Future: add RDNA4, CDNA..." — they
# just never added RDNA3.5). rdna3-{moe,linear}-gfx1151 open those gates to
# on_gfx1151(). Only reachable via the compressed-tensors path with a
# *symmetric* W4A16 checkpoint (e.g. MIRALABS/Ornith-1.5-35B-A3B-W4A16-SYM)
# — not the auto_awq path. This is the real int4 MoE decode kernel; the
# Triton WNA16 path it replaces measured ~7 tok/s on rennes (alpha.12).
#
# Pure .py, applied to site-packages, no rebuild. `--forward` = re-apply is
# a no-op; NO --fuzz — context drift must fail the build loudly.
COPY patches/*.patch /tmp/patches/
RUN cd /opt/venv/lib/python3.12/site-packages && \
    for p in \
      fla-chunk-o-gfx1151 \
      fla-chunk-delta-h-gfx1151 \
      aiter-gate-gfx1x \
      aiter-fa-gfx1x-gate \
      aiter-fusion-skip-duplicates \
      rdna3-moe-gfx1151 \
      rdna3-linear-gfx1151 \
      ct-moe-wna16-intermediate-size-full \
    ; do \
      echo "== applying ${p}" && \
      patch -p1 --forward --no-backup-if-mismatch < "/tmp/patches/${p}.patch" ; \
    done && rm -rf /tmp/patches

# 7. Pre-tuned Triton fused-MoE config for the Qwen3.5-MoE shape
# (E=256, N=512, int4_w4a16). Retuned on the actual rennes gfx1151 with
# this exact vLLM/Triton 3.7.1 (benchmark_moe.py --tune, batch 1/2/4/8,
# ~2h14, 2026-09-03) — the earlier 2026-07 values were tuned against a
# different Triton and measured worse than vLLM's default. AITER has no
# int4 MoE kernel for gfx1151 (its MoE kernels are fp8/int8/bf16), so the
# AWQ experts run this Triton WNA16 path regardless of AITER_MOE.
# Filename must match `device_name={get_device_name()}` — 0.28 returns the
# marketing name "AMD Radeon 8060S"; a gfx1151-named copy is dropped for
# older/other vLLMs.
COPY moe-configs/*.json /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/configs/
RUN cd /opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/configs && \
    for f in *device_name=AMD_Radeon_8060S*.json; do \
      cp -n "$f" "$(echo "$f" | sed 's/AMD_Radeon_8060S/gfx1151/')"; \
    done && ls -1 *device_name=*8060S* *device_name=*gfx1151*

# 7b. Import gate — fail the CI build here, not on the rennes GPU, if the
# native lib graph doesn't load. No GPU at build: torch/vllm import fine
# CPU-side (device init is lazy), so this catches missing .so / unresolved
# symbols / the triton_utils shim regressing, which is where every failed
# assembly so far actually broke.
RUN LD_LIBRARY_PATH=/opt/rocm/lib:/opt/rocm/lib64:/opt/rocm/llvm/lib \
    python -c "import torch; print('torch', torch.__version__, torch.version.hip); \
import triton; print('triton', triton.__version__); \
import vllm; print('vllm', vllm.__version__)"

# 8. Runtime env.
# AITER ON since alpha.11 — the step 6d gate patches make AITER's
# RDNA3.5-tuned Triton kernels (MoE / MHA / RMSNorm / linear) engage on
# gfx1151 instead of vLLM's generic NVIDIA-shaped ones. Granular flags all
# on; drop individually to A/B a path. If AITER crashes at first forward,
# fall back to VLLM_ROCM_USE_AITER=0 (granular flags become no-ops).
# ⚠️ A Deployment that sets VLLM_ROCM_USE_AITER=0 in its own env OVERRIDES
# this — the rennes Deployment did that; it must drop that line for
# alpha.11 to mean anything.
# ROCM_HOME/ROCM_PATH: aiter + Triton _find_rocm_home() checks. CC/CXX:
# aiter's JIT build path defaults to bare cc/c++ which don't exist here —
# point at the clang toolchain kept for Triton's JIT.
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
    VLLM_ROCM_USE_AITER_MOE=1 \
    VLLM_ROCM_USE_AITER_RMSNORM=1 \
    VLLM_ROCM_USE_AITER_MHA=1 \
    VLLM_ROCM_USE_AITER_LINEAR=1

WORKDIR /opt
CMD ["/bin/bash"]
