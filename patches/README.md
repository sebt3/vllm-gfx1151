# gfx1151 kernel patches (site-packages, applied at image build)

Pure-Python patches to the installed `vllm` package, applied by the
Dockerfile with `patch -p1` after the wheels are in place. No rebuild.
Re-authored for vLLM v0.28.0 by `sebt3/stack-torch-gfx1151` (2026-08-25),
vendored here 2026-09-02 after stack-torch was retired as the build path.

vLLM's Triton kernels autotune over NVIDIA-shaped config spaces (LDS, warp
counts, stages) that are wrong for gfx1151 (64 KB LDS, wave32). On rennes
the stock wheel decodes a Qwen3.5-MoE hybrid at ~4 tok/s — a same-bandwidth
DGX does 20-35 on a bigger model. These patches route to AITER's
RDNA3.5-tuned Triton kernels and fix the FLA/GDN autotune space.

| patch | file | what |
|---|---|---|
| `fla-chunk-o-gfx1151` | `third_party/flash_linear_attention/ops/chunk_o.py` | force `BK=BV=32`, `num_stages=2` for AMD — the {64,128}/{2,3,4} space overflows gfx1151 LDS (30/40 layers are GDN) |
| `fla-chunk-delta-h-gfx1151` | `.../ops/chunk_delta_h.py` | cast bf16→f32 before `exp()` — Triton HIP compiler can't infer types for `exp(bf16 - block_ptr_bf16)` |
| `aiter-gate-gfx1x` | `_aiter_ops.py` | add `is_aiter_found_and_supported_on_gfx1x()` sibling (mirrors the RDNA4 one) so `rocm_aiter_ops` registers on gfx1151 — AITER's Triton kernels, not CK |
| `aiter-fa-gfx1x-gate` | `v1/attention/backends/rocm_aiter_fa.py` | `supports_compute_capability` accepts gfx1x alongside CDNA |
| `aiter-fusion-skip-duplicates` | `compilation/passes/fusion/rocm_aiter_fusion.py` | `skip_duplicates=True` on the AITER RMSNorm+quant fusion patterns — without it `register_replacement` raises on duplicate patterns once AITER is on |

Dropped from the stack-torch set:
- `fp8-support-gfx1x` — only widens `supports_fp8`; no FP8 accel on gfx1151, not pursuing FP8
- `aiter-fp4-import-fix` — CDNA fp4 gates, irrelevant here, and `on_gfx950` already exists upstream (no import error)
- `triton-*` compat shims — for the dead main_perf Triton; our triton is 3.7.1, symbols present
- `skip-distributed-single-gpu` — single GPU already works on the stock wheel
- all `llama.cpp` patches (`hybrid-attn-*`, `mtp-*`, `grammar-*`, cmake) — wrong repo

Re-triage against a new wheel: `patch -p1 --dry-run` each against a fresh
extract of `vllm/`. `fla-*` and `aiter-fusion` are line-stable; the
`aiter-gate` / `aiter-fa-gate` context can drift when upstream refactors
the gate functions.
