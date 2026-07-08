"""
Strix Halo (gfx1151) patch bundle for vLLM source builds.

This is the same patch script the sibling `vllm-qwen` repo ships, kept
in sync verbatim. Two reasons it lives here unchanged:

  1. Every patch is gfx1151-driven, not quant-driven. The AWQ-INT4
     model exercises the same RDNA 3.5 code paths as the BF16 model
     (same custom_ops registration, same AITER overrides, same APU
     VRAM clamp), so the patch surface is identical.

  2. AWQ INT4 dispatches through vLLM's compressed-tensors kernel,
     which is itself unaffected by these patches. The DeltaNet linear-
     attention layers are mixed-precision (FP16/BF16 weights kept
     unquantized in the AWQ checkpoint, see README "DeltaNet under
     AWQ" note), so they take the standard linear-attn path the
     compressed-tensors loader already handles.

If a future tool-call / reasoning-parser PR is needed before merging
upstream, add it here as Patch 13/14/15 (cherry-pick of vllm#40783,
#40785, #40787) and rebuild  -  the existing patch numbers stay stable
so cross-repo references don't drift.
"""
import sys
import re
import site
from pathlib import Path

def patch_vllm():
    print("Applying Strix Halo patches to vLLM (ai-notes modernization)...")

    # Patch 1: vllm/platforms/__init__.py (amdsmi monkey patch  -  PROVEN working for 5 months)
    # Comment out real amdsmi imports and replace with pass stubs.
    # The actual amdsmi library doesn't work on Strix Halo APUs in containers.
    p_init = Path('vllm/platforms/__init__.py')
    if p_init.exists():
        txt = p_init.read_text()
        txt = txt.replace('import amdsmi', '# import amdsmi')
        txt = re.sub(r'is_rocm = .*', 'is_rocm = True', txt)
        txt = re.sub(r'if len\(amdsmi\.amdsmi_get_processor_handles\(\)\) > 0:', 'if True:', txt)
        txt = txt.replace('amdsmi.amdsmi_init()', 'pass')
        txt = txt.replace('amdsmi.amdsmi_shut_down()', 'pass')
        p_init.write_text(txt)
        print(" -> Patched vllm/platforms/__init__.py (amdsmi disabled, is_rocm forced True)")

    # Patch 1.5: vllm/platforms/rocm.py (MagicMock amdsmi + force gfx1151)
    # Prepend MagicMock so any remaining amdsmi references in rocm.py silently succeed.
    p_rocm_plat = Path('vllm/platforms/rocm.py')
    if p_rocm_plat.exists():
        txt = p_rocm_plat.read_text()
        # Add MagicMock header if not already present
        if 'sys.modules["amdsmi"] = MagicMock()' not in txt:
            header = 'import sys\nfrom unittest.mock import MagicMock\nsys.modules["amdsmi"] = MagicMock()\n'
            txt = header + txt
        # Force arch detection
        if 'def _get_gcn_arch() -> str:\n    return "gfx1151"' not in txt:
            txt = txt.replace('def _get_gcn_arch() -> str:', 'def _get_gcn_arch() -> str:\n    return "gfx1151"\n\ndef _old_get_gcn_arch() -> str:')
            txt = re.sub(r'device_type = .*', 'device_type = "rocm"', txt)
            txt = re.sub(r'device_name = .*', 'device_name = "gfx1151"', txt)
        p_rocm_plat.write_text(txt)
        print(" -> Patched vllm/platforms/rocm.py (MagicMock amdsmi + forced gfx1151)")

    # Patch 2: _aiter_ops.py (Enable AITER on gfx1x, disable FP8 linear)
    p_aiter = Path('vllm/_aiter_ops.py')
    if p_aiter.exists():
        txt = p_aiter.read_text()

        # Ensure on_gfx1x is available globally for our patches below
        if "from vllm.platforms.rocm import on_gfx1x" not in txt:
            txt = txt.replace("from vllm.platforms import current_platform",
                              "from vllm.platforms import current_platform\nfrom vllm.platforms.rocm import on_gfx1x")

        # Extend is_aiter_found_and_supported
        if "or on_gfx1x()" not in txt:
            txt = txt.replace("import on_mi3xx", "import on_mi3xx, on_gfx1x")
            txt = txt.replace("on_mi3xx()", "(on_mi3xx() or on_gfx1x())")

        # Disable FP8 linear
        if "is_linear_fp8_enabled" in txt:
            txt = re.sub(
                r'(def is_linear_fp8_enabled.*?:\n\s+return) (.*?)\n',
                r'\1 False\n',
                txt, count=1, flags=re.DOTALL
            )

        # Disable AITER RMSNorm on gfx1x (CUDA Graph hang)
        if "is_rmsnorm_enabled" in txt:
            txt = re.sub(
                r'(def is_rmsnorm_enabled.*?:\n\s+return) (cls\._AITER_ENABLED and cls\._RMSNORM_ENABLED)\n',
                r'\1 \2 and not getattr(on_gfx1x, "__call__", lambda: False)()\n',
                txt, count=1, flags=re.DOTALL
            )

        # Disable AITER Fused MoE on gfx1x (due to hundreds of CDNA-specific dpp_mov assembly conflicts)
        if "is_fused_moe_enabled" in txt:
            txt = re.sub(
                r'(def is_fused_moe_enabled.*?:\n\s+return) (cls\._AITER_ENABLED and cls\._FMOE_ENABLED)\n',
                r'\1 \2 and not getattr(on_gfx1x, "__call__", lambda: False)()\n',
                txt, count=1, flags=re.DOTALL
            )

        p_aiter.write_text(txt)
        print(" -> Patched vllm/_aiter_ops.py (gfx1x support, FP8 linear empty, MoE disabled)")

    # Patch 3: rocm_aiter_fa.py
    p_fa = Path('vllm/v1/attention/backends/rocm_aiter_fa.py')
    if p_fa.exists():
        txt = p_fa.read_text()
        if "on_gfx1x" not in txt:
            txt = txt.replace("from vllm.platforms.rocm import on_mi3xx", "from vllm.platforms.rocm import on_mi3xx, on_gfx1x")
            txt = txt.replace("on_mi3xx()", "(on_mi3xx() or on_gfx1x())")
            p_fa.write_text(txt)
            print(" -> Patched vllm/v1/attention/backends/rocm_aiter_fa.py (gfx1x support)")

    # Patch 3.5: unquantized.py (Hard-block AITER MoE forced override on gfx1x)
    p_unquant = Path('vllm/model_executor/layers/fused_moe/oracle/unquantized.py')
    if p_unquant.exists():
        txt = p_unquant.read_text()
        if "from vllm.platforms.rocm import on_gfx1x" not in txt:
            txt = txt.replace(
                'if envs.is_set("VLLM_ROCM_USE_AITER")',
                'from vllm.platforms.rocm import on_gfx1x\n    if envs.is_set("VLLM_ROCM_USE_AITER")'
            )
            txt = txt.replace(
                'if not envs.VLLM_ROCM_USE_AITER or not envs.VLLM_ROCM_USE_AITER_MOE:',
                'if getattr(on_gfx1x, "__call__", lambda: False)() or not envs.VLLM_ROCM_USE_AITER or not envs.VLLM_ROCM_USE_AITER_MOE:'
            )
            p_unquant.write_text(txt)
            print(" -> Patched unquantized.py (Blocked AITER MoE override on gfx1x)")


    # Patch 5: custom_ops RMSNorm block on gfx1x (Full CUDA Graph capture)
    p_rocm = Path('vllm/platforms/rocm.py')
    if p_rocm.exists():
        txt = p_rocm.read_text()

        # Legacy vLLM < 0.19 fallback
        if "if is_aiter_found_and_supported():\n            custom_ops.append(\"+rms_norm\")" in txt:
            txt = txt.replace(
                "if is_aiter_found_and_supported():\n            custom_ops.append(\"+rms_norm\")",
                "if is_aiter_found_and_supported() and not getattr(self, 'on_gfx1x', lambda: False)():\n            custom_ops.append(\"+rms_norm\")"
            )

        # Modern vLLM 0.19+ struct (compilation_config.custom_ops)
        elif "compilation_config.custom_ops.append(\"+rms_norm\")" in txt:
            if "if not getattr(self, \"on_gfx1x\", lambda: False)():" not in txt:
                txt = re.sub(
                    r'(\s+)compilation_config\.custom_ops\.append\("\+rms_norm"\)',
                    r'\1if not getattr(self, "on_gfx1x", lambda: False)():\n\1    compilation_config.custom_ops.append("+rms_norm")',
                    txt
                )

        # Modern vLLM 0.19.2rc1+ IrOpPriorityConfig bypass
        if 'rms_norm = ["aiter"] + default' in txt:
            txt = txt.replace(
                'rms_norm = ["aiter"] + default',
                'rms_norm = ["aiter"] + default if not on_gfx1x() else default'
            )

        p_rocm.write_text(txt)
        print(" -> Patched vllm/platforms/rocm.py (custom_ops & IrOpPriorityConfig rms_norm bypassed on gfx1x)")

    # Patch 6: vllm/compilation/passes/fusion/rocm_aiter_fusion.py (duplicate pattern bypass)
    p_fusion = Path('vllm/compilation/passes/fusion/rocm_aiter_fusion.py')
    if p_fusion.exists():
        txt = p_fusion.read_text()
        if "skip_duplicates=True" not in txt:
            txt = re.sub(
                r"(pm\.register_replacement\s*\((?:(?!\bpm\.register_replacement\b).)*?)pm_pass(\s*[\),])",
                r"\1pm_pass, skip_duplicates=True\2",
                txt, flags=re.DOTALL
            )
            p_fusion.write_text(txt)
            print(" -> Patched vllm/compilation/passes/fusion/rocm_aiter_fusion.py (skip_duplicates)")

    # Patch 7: Triton backend AttrsDescriptor repr
    for sp in site.getsitepackages():
        triton_compiler = Path(sp) / "triton/backends/compiler.py"
        if triton_compiler.exists():
            txt = triton_compiler.read_text()
            if "def __repr__(self):" not in txt:
                txt = txt.replace(
                    "def to_dict(self):",
                    "def __repr__(self):\n        return f'AttrsDescriptor.from_dict({self.to_dict()!r})'\n\n    def to_dict(self):"
                )
                triton_compiler.write_text(txt)
                print(f" -> Patched {triton_compiler} (AttrsDescriptor repr)")

    # Patch 7: aiter JIT path fix  -  aiter builds .so files into ~/.aiter/jit/
    # but importlib.import_module("aiter.jit.<module>") only looks in the
    # installed package directory. Fix by adding the JIT cache to __path__.
    for sp in site.getsitepackages():
        aiter_jit_init = Path(sp) / "aiter/jit/__init__.py"
        if aiter_jit_init.exists():
            txt = aiter_jit_init.read_text()
            if "# PATCHED: JIT cache path" not in txt:
                jit_path_fix = '''
# PATCHED: JIT cache path for Strix Halo
# aiter's JIT compiles .so modules into ~/.aiter/jit/ but importlib looks
# in the installed package directory. Add the JIT cache to __path__.
import os as _os
_jit_cache = _os.path.join(_os.path.expanduser("~"), ".aiter", "jit")
if _os.path.isdir(_jit_cache) and _jit_cache not in __path__:
    __path__.append(_jit_cache)
'''
                txt += jit_path_fix
                aiter_jit_init.write_text(txt)
                print(f" -> Patched {aiter_jit_init} (JIT cache added to __path__)")

    # Patch 8: flash_attn_interface.py  -  make aiter import soft as safety net.
    # If aiter JIT fails for any reason, flash_attn should still load (TRITON_ATTN works).
    # ROCM_ATTN will also work when aiter JIT succeeds (patch 7 fixes the path).
    hard_import_bare = "from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_2 as flash_attn_gpu"

    def _patch_flash_interface(fa_iface):
        txt = fa_iface.read_text()
        if hard_import_bare not in txt or "except (ImportError" in txt:
            return False
        # Detect indentation of the original import line
        m = re.search(r'^( *)' + re.escape(hard_import_bare), txt, re.MULTILINE)
        if not m:
            return False
        indent = m.group(1)
        original_line = indent + hard_import_bare
        soft_import = (
            f"{indent}try:\n"
            f"{indent}    {hard_import_bare}\n"
            f"{indent}except (ImportError, KeyError, ModuleNotFoundError):\n"
            f"{indent}    flash_attn_gpu = None"
        )
        txt = txt.replace(original_line, soft_import)
        fa_iface.write_text(txt)
        print(f" -> Patched {fa_iface} (aiter import made resilient)")
        return True

    for sp in site.getsitepackages():
        for fa_egg in Path(sp).glob("flash_attn*.egg"):
            fa_iface = fa_egg / "flash_attn/flash_attn_interface.py"
            if fa_iface.exists():
                _patch_flash_interface(fa_iface)
        # Also check non-egg installs
        fa_iface = Path(sp) / "flash_attn/flash_attn_interface.py"
        if fa_iface.exists():
            _patch_flash_interface(fa_iface)

    # Patch 9: Allow Triton MoE kernels on gfx11xx (Strix Halo)
    # vLLM recently capped MXFP4 Triton MoE kernels to < (11, 0) which excludes RDNA3.5 (11.x)
    for p_triton in [
        Path('vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py'),
        Path('vllm/model_executor/layers/fused_moe/oracle/mxfp4.py')
    ]:
        if p_triton.exists():
            txt = p_triton.read_text()
            if "cap.minor) < (11, 0)" in txt:
                txt = txt.replace("cap.minor) < (11, 0)", "cap.minor) < (12, 0)")
            if "capability() < (11, 0)" in txt:
                txt = txt.replace("capability() < (11, 0)", "capability() < (12, 0)")
            p_triton.write_text(txt)
            print(f" -> Patched {p_triton} (Triton MoE on gfx11xx)")

    # Patch 10: ROCM-21812 APU VRAM Dynamic Margin Patch
    # Explanation: ROCm nightly builds introduced a 50% APU VRAM clamp to prevent
    # OOM kernel panics on headless hosts. This broke vLLM large model loading.
    # This patch intercepts PyTorch memory bounds and dynamically proxies the
    # real amdgpu hardware GTT limits, minus a strict 8GB OS safety margin.
    # By symmetrically carving the OS margin from the top of the GTT ceiling,
    # vLLM's memory profiler allocates flawlessly while guaranteeing the OS stays alive,
    # regardless of the specific GTT allocation size on the host.
    # Ref: https://github.com/ROCm/rocm-systems/pull/5113
    # TODO: Remove this patch block entirely once PR #5113 merges and is
    # incorporated into the ROCm nightly tarballs used by this toolbox.
    p_rocm_plat = Path('vllm/platforms/rocm.py')
    if p_rocm_plat.exists():
        txt = p_rocm_plat.read_text()
        if "_patched_mem_info" not in txt:
            mem_patch = '''
# --- ROCM-21812 VRAM DYNAMIC PATCH ---
import torch
import glob
import os

try:
    _orig_mem_info = torch.cuda.mem_get_info
    _orig_get_dev_prop = torch.cuda.get_device_properties

    class MockCudaDeviceProperties:
        def __init__(self, prop, override_total):
            self._prop = prop
            self.total_memory = override_total
        def __getattr__(self, name):
            return getattr(self._prop, name)
        def __dir__(self):
            return dir(self._prop)

    def _patched_mem_info(device=None):
        free, total = _orig_mem_info(device)
        try:
            # On APUs, ROCm clamps total to 50% limit. We need the real GTT limits.
            if total < 70 * 1024**3:
                drm_cards = glob.glob('/sys/class/drm/card*/device/mem_info_gtt_total')
                if drm_cards:
                    card_dir = os.path.dirname(drm_cards[0])
                    with open(os.path.join(card_dir, 'mem_info_gtt_total'), 'r') as f:
                        gtt_total = int(f.read().strip())
                    with open(os.path.join(card_dir, 'mem_info_gtt_used'), 'r') as f:
                        gtt_used = int(f.read().strip())

                    # Symmetrically carve 8GB off the TOP of the device perfectly.
                    safe_ceiling = gtt_total - (8 * 1024**3)

                    real_total = safe_ceiling
                    real_free = max(0, safe_ceiling - gtt_used)

                    total = max(total, real_total)
                    free = real_free
        except Exception as e:
            pass
        return int(free), int(total)

    def _patched_get_dev_prop(device=None):
        prop = _orig_get_dev_prop(device)
        free, total = _patched_mem_info(device)
        if hasattr(prop, 'total_memory') and prop.total_memory < total:
            return MockCudaDeviceProperties(prop, total)
        return prop

    torch.cuda.mem_get_info = _patched_mem_info
    torch.cuda.get_device_properties = _patched_get_dev_prop
except Exception:
    pass
# ---------------------------
'''
            txt = mem_patch + txt
            p_rocm_plat.write_text(txt)
            print(" -> Patched vllm/platforms/rocm.py (ROCM-21812 APU VRAM Dynamic Margin)")

    # Patch 11 (local addition): silence hipCtx* deprecation warnings in
    # csrc/cumem_allocator_compat.h. vLLM still uses hipCtxGetCurrent /
    # hipCtxSetCurrent / hipDevicePrimaryCtxRetain for CUDA-compat context
    # management; HIP marked these deprecated but there is no clean
    # replacement for the use case, and upstream vLLM hasn't migrated yet.
    # Suppressing the warning class for that file keeps our build clean.
    p_cumem = Path('csrc/cumem_allocator_compat.h')
    if p_cumem.exists():
        txt = p_cumem.read_text()
        marker = '#pragma clang diagnostic ignored "-Wdeprecated-declarations"'
        if marker not in txt:
            txt = marker + "\n" + txt
            p_cumem.write_text(txt)
            print(" -> Patched csrc/cumem_allocator_compat.h (suppress hipCtx* deprecations)")

    # Patch 12 (local): allow transformers' GGUF parser to accept Qwen 3.5/3.6
    # arch tag. Upstream transformers registers "qwen2", "qwen3", "qwen3_moe"
    # but not "qwen35" (the arch tag Unsloth's Qwen 3.6 GGUFs declare). vLLM
    # has a Qwen3_5ForConditionalGeneration class downstream that handles the
    # actual model correctly; we just need transformers' parser to route the
    # GGUF through as if it were qwen3 so the config loads. Harmless for the
    # AWQ path (no GGUF involved) but kept for parity with the BF16 sibling.
    import site as _site
    for _sp in _site.getsitepackages():
        gguf_utils = Path(_sp) / "transformers/modeling_gguf_pytorch_utils.py"
        if gguf_utils.exists():
            _txt = gguf_utils.read_text()
            _marker = 'elif "minimax-m2" in architecture:'
            _inject = (
                'elif "qwen35" in architecture or "qwen3_5" in architecture:\n'
                '        updated_architecture = "qwen3"\n'
                '    '
            )
            if 'qwen35' not in _txt and _marker in _txt:
                _txt = _txt.replace(_marker, _inject + _marker, 1)
                gguf_utils.write_text(_txt)
                print(f" -> Patched {gguf_utils} (qwen35 -> qwen3 alias for GGUF parser)")

    # Patch 14 (local, reworked 2026-07-08): preserve SWA config fields when
    # extracting a DFlash drafter's HF config (algos.py, 14b below).
    #
    # This used to be a full cherry-pick of vLLM PR #40898 ("Add Sliding
    # Window Attention support to DFlash drafter"), covering:
    #   - qwen3_dflash.py: per-layer sliding_window/layer_type wiring
    #   - dflash.py: per-layer causal metadata override for mixed layers
    #   - gpu_model_runner.py: target_layer_ids +1 shift for dflash
    #
    # Retriaged against vLLM HEAD (7c67da967f, 2026-07-08): upstream has
    # since shipped its own _resolve_layer_attention() in qwen3_dflash.py,
    # which already handles pure-SWA and pure-full-attention DFlash drafters
    # correctly (including per-layer sliding_window threading via
    # per_layer_sliding_window). The gpu_model_runner.py +1 shift is also
    # already native there.
    #
    # What upstream still does NOT support: MIXED sliding+full layer_types
    # in one drafter (e.g. z-lab/Qwen3.6-27B-DFlash: 4x sliding_attention +
    # 1x full_attention interleaved). _resolve_layer_attention() explicitly
    # raises NotImplementedError for that case, citing vllm#40898. Digging
    # in: `causal` IS computed per-layer there but is documented as
    # "currently unused" in DFlashQwen3Attention, and DFlashProposer only
    # exposes a single global `dflash_causal` flag that picks ONE causal/
    # non-causal attention backend for the whole draft model — there's no
    # per-layer causal metadata or multi-KV-cache-group support at the
    # spec-decode proposer level to build on. Making the interleaved case
    # work needs real engine-level work upstream hasn't shipped, not a
    # string patch — punted for now.
    #
    # Not currently blocking: think/vllm defaults speculative_method to
    # `mtp` (native self-draft), not `dflash`. Revisit if/when dflash with
    # a mixed-layer checkpoint becomes the active path again.

    # 14b: algos.py  -  preserve SWA fields when extracting HF config
    p_algos = Path('vllm/transformers_utils/configs/speculators/algos.py')
    if p_algos.exists():
        txt = p_algos.read_text()
        old_target = (
            "    if config_dict.get(\"target_hidden_size\") is not None:\n"
            "        pre_trained_config[\"target_hidden_size\"] = config_dict[\"target_hidden_size\"]\n"
            "\n"
            "    aux_layer_ids = config_dict[\"aux_hidden_state_layer_ids\"]\n"
        )
        new_target = (
            "    if config_dict.get(\"target_hidden_size\") is not None:\n"
            "        pre_trained_config[\"target_hidden_size\"] = config_dict[\"target_hidden_size\"]\n"
            "    for key in (\n"
            "        \"layer_types\",\n"
            "        \"use_sliding_window\",\n"
            "        \"sliding_window\",\n"
            "        \"max_window_layers\",\n"
            "    ):\n"
            "        if key in config_dict:\n"
            "            pre_trained_config[key] = config_dict[key]\n"
            "\n"
            "    aux_layer_ids = config_dict[\"aux_hidden_state_layer_ids\"]\n"
        )
        if old_target in txt and '"layer_types",' not in txt:
            txt = txt.replace(old_target, new_target, 1)
            p_algos.write_text(txt)
            print(" -> Patched vllm/transformers_utils/configs/speculators/algos.py (PR #40898: preserve SWA config)")

    # Patch 16 (local): register the AWQ-INT4 MMQ HIP custom op into vLLM's
    # mixed-precision kernel dispatcher so it's picked ahead of TritonW4A16
    # for the W4A16 g32 path on gfx1151. The .so is built from
    # /workspace/csrc/awq_mmq_gfx1151/ (host-mounted at /root/csrc/) and
    # imports lazily at module-load time.
    #
    # Implementation: append a registration block to the dispatcher's
    # __init__.py. On load the block adds the package dir to sys.path,
    # imports our RocmMmqQ4LinearKernel, and inserts it at position 0 of
    # _POSSIBLE_KERNELS[ROCM]. If the import fails (e.g. .so not built yet),
    # the kernel list is left untouched and TritonW4A16 keeps its slot.
    #
    # apply_weights internally dispatches: M >= 32 (prefill) -> our HIP
    # kernel, M < 32 (decode) -> TritonW4A16 fallback. Both paths use the
    # same layer's weight tensors via the dual-storage process_weights step.
    # See .research/mmq-q4-gfx1151-port/FINDINGS.md.
    p_dispatch = Path('vllm/model_executor/kernels/linear/__init__.py')
    if p_dispatch.exists():
        txt = p_dispatch.read_text()
        if "Patch 16" not in txt:
            injection = (
                "\n\n# --- Patch 16: AWQ-INT4 MMQ HIP custom op for gfx1151 (Strix Halo) ---\n"
                "import sys as _sys\n"
                "import os as _os\n"
                "_AWQ_MMQ_DIR = '/root/csrc/awq_mmq_gfx1151'\n"
                "if _os.path.exists(_AWQ_MMQ_DIR) and _AWQ_MMQ_DIR not in _sys.path:\n"
                "    _sys.path.insert(0, _AWQ_MMQ_DIR)\n"
                "try:\n"
                "    from awq_mmq_gfx1151.vllm_kernel import RocmMmqQ4LinearKernel as _RocmMmqQ4\n"
                "    if _RocmMmqQ4 not in _POSSIBLE_KERNELS.get(PlatformEnum.ROCM, []):\n"
                "        _POSSIBLE_KERNELS[PlatformEnum.ROCM].insert(0, _RocmMmqQ4)\n"
                "        logger.info('Patch 16: RocmMmqQ4LinearKernel registered at _POSSIBLE_KERNELS[ROCM][0]')\n"
                "except Exception as _e:\n"
                "    logger.warning('Patch 16: failed to register RocmMmqQ4LinearKernel: %s', _e)\n"
                "# --- end Patch 16 ---\n"
            )
            txt += injection
            p_dispatch.write_text(txt)
            print(" -> Patched vllm/model_executor/kernels/linear/__init__.py (Patch 16: AWQ-INT4 MMQ HIP)")

    # Patch 18 (local): restore HIP_FOUND for newer PyTorch builds.
    #
    # PyTorch's cmake/public/LoadHIP.cmake used to call
    #   find_package(HIP 1.0 MODULE)
    # which set the cmake variable HIP_FOUND (uppercase) as a side effect.
    # Somewhere between PyTorch v2.10.0 and main, that MODULE-mode finder
    # was replaced with
    #   find_package_and_print_version(hip REQUIRED CONFIG)
    # which only sets PYTORCH_FOUND_HIP and hip_FOUND (lowercase). The
    # uppercase HIP_FOUND is no longer exported.
    #
    # vLLM v0.20.0 CMakeLists.txt:125-148 still gates GPU-language
    # detection on `elseif(HIP_FOUND)`, so against newer torch wheels
    # (e.g. 2.13.0a0+rocm7.13.0a20260510 from rocm.nightlies v2-staging)
    # the build dies with
    #   CMake Error at CMakeLists.txt:147 (message):
    #     Can't find CUDA or HIP installation.
    # Trigger: rocm.nightlies index rolled torch 2.10 -> 2.13 in 14 days.
    #
    # Fix: alias HIP_FOUND from PYTORCH_FOUND_HIP / hip_FOUND right after
    # find_package(Torch REQUIRED) so the legacy uppercase variable is
    # populated regardless of which LoadHIP.cmake variant we got.
    p_cmake = Path('CMakeLists.txt')
    if p_cmake.exists():
        txt = p_cmake.read_text()
        marker = "find_package(Torch REQUIRED)\n"
        shim = (
            "find_package(Torch REQUIRED)\n"
            "\n"
            "# Patch 18 (Strix): newer PyTorch (>=~2.11) LoadHIP.cmake drops the legacy\n"
            "# find_package(HIP MODULE) call and only sets PYTORCH_FOUND_HIP / hip_FOUND.\n"
            "# Re-export the uppercase HIP_FOUND that the rest of this file (and vLLM\n"
            "# v0.20.0 in general) still reads.\n"
            "if(NOT HIP_FOUND AND (PYTORCH_FOUND_HIP OR hip_FOUND))\n"
            "  set(HIP_FOUND TRUE)\n"
            "endif()\n"
        )
        if "Patch 18" not in txt and marker in txt:
            txt = txt.replace(marker, shim, 1)
            p_cmake.write_text(txt)
            print(" -> Patched CMakeLists.txt (Patch 18: alias HIP_FOUND from PYTORCH_FOUND_HIP / hip_FOUND)")

    # Patch 19 (local): fix DFlash speculative-decode dtype crash on gfx1151.
    # combine_hidden_states() feeds fp32 hidden states (GDN/mamba layers run in
    # fp32 for stability) into self.model.fc, whose weights are fp16 (the bf16
    # DFlash checkpoint is cast to model_config.dtype=fp16). F.linear then raises
    #   RuntimeError: expected mat1 and mat2 to have the same dtype (float != Half)
    # at the FIRST real decode step (the spec proposal only fires after the first
    # sampled token). Cast hidden states to the fc weight dtype before the call.
    p_dflash = Path('vllm/model_executor/models/qwen3_dflash.py')
    if p_dflash.exists():
        txt = p_dflash.read_text()
        target = "self.model.fc(hidden_states)"
        fixed = "self.model.fc(hidden_states.to(self.model.fc.weight.dtype))"
        if fixed in txt:
            print(" -> qwen3_dflash.py already carries Patch 19 (fc dtype cast)")
        elif target in txt:
            txt = txt.replace(target, fixed, 1)
            p_dflash.write_text(txt)
            print(" -> Patched vllm/model_executor/models/qwen3_dflash.py (Patch 19: cast hidden_states to fc weight dtype)")
        else:
            print(" !! Patch 19 target not found in qwen3_dflash.py — DFlash dtype fix NOT applied (source drift?)")
    else:
        print(" !! Patch 19: qwen3_dflash.py not found — skipped")

    # Patch 20 (local): stabilize ROCmPlatform.get_device_name() so MoE
    # kernel config filenames are reproducible across process restarts.
    #
    # amdsmi is fully mocked (Patch 1.5: sys.modules["amdsmi"] = MagicMock()),
    # since real amdsmi doesn't work on Strix Halo APUs in containers. The
    # unpatched get_device_name() calls amdsmi_get_gpu_asic_info(handle)
    # ["market_name"], which on a MagicMock returns another MagicMock whose
    # repr() embeds the object's id() - e.g.
    #   <MagicMock name='mock.amdsmi_get_gpu_asic_info().__getitem__().replace()'
    #    id='140212920705120'>
    # That id() is a memory address: different every process start. vLLM's
    # fused-MoE kernel loader (get_config_file_name() in
    # model_executor/layers/fused_moe/fused_moe.py) folds get_device_name()
    # straight into the tuned-config JSON filename, so a config generated by
    # one process (e.g. a one-off tuning run) never matches the filename a
    # different process (the actual server) computes at load time - tuned
    # MoE configs silently fail to load, falling back to untuned defaults.
    # See README.md "Tuning des kernels MoE (Triton)" for the tuning recipe
    # and the first tuned config this unblocks (E=256,N=512,dtype=int4_w4a16,
    # baked into this image's vllm/model_executor/layers/fused_moe/configs/).
    #
    # Fix: return a fixed, human-readable name. Safe because this image
    # targets exactly one SKU (gfx1151 / Strix Halo) - there's no second
    # device to disambiguate.
    p_rocm_devname = Path('vllm/platforms/rocm.py')
    if p_rocm_devname.exists():
        txt = p_rocm_devname.read_text()
        old_get_device_name = (
            "    @classmethod\n"
            "    @with_amdsmi_context\n"
            "    @lru_cache(maxsize=8)\n"
            "    def get_device_name(cls, device_id: int = 0) -> str:\n"
            "        physical_device_id = cls.device_id_to_physical_device_id(device_id)\n"
            "        handle = amdsmi_get_processor_handles()[physical_device_id]\n"
            "        asic_info = amdsmi_get_gpu_asic_info(handle)\n"
            "        asic_info_device_id: str = asic_info[\"device_id\"]\n"
            "        if asic_info_device_id in _ROCM_DEVICE_ID_NAME_MAP:\n"
            "            return _ROCM_DEVICE_ID_NAME_MAP[asic_info_device_id]\n"
            "        return asic_info[\"market_name\"]\n"
        )
        new_get_device_name = (
            "    @classmethod\n"
            "    @lru_cache(maxsize=8)\n"
            "    def get_device_name(cls, device_id: int = 0) -> str:\n"
            "        # Patch 20 (Strix Halo): amdsmi is mocked (Patch 1.5), so the\n"
            "        # real lookup below would return a MagicMock whose repr()\n"
            "        # embeds a per-process id(), breaking MoE tuned-config\n"
            "        # filename matching between runs. Fixed name: single-SKU\n"
            "        # image (gfx1151 only), no disambiguation needed.\n"
            "        return \"gfx1151\"\n"
        )
        if old_get_device_name in txt:
            txt = txt.replace(old_get_device_name, new_get_device_name, 1)
            p_rocm_devname.write_text(txt)
            print(" -> Patched vllm/platforms/rocm.py (Patch 20: stable get_device_name -> 'gfx1151')")
        elif "def get_device_name" in txt and "Patch 20" not in txt:
            print(" !! Patch 20 target not found verbatim in vllm/platforms/rocm.py — get_device_name NOT patched (source drift?)")
        else:
            print(" -> vllm/platforms/rocm.py already carries Patch 20 (or was already patched)")

    # Patch 21 (local, added 2026-07-08): same amdsmi-mock landmine as
    # Patch 20, different call site. get_device_total_memory() tries
    # amdsmi first and falls back to torch.cuda.get_device_properties()
    # only on exception — but amdsmi is fully mocked (Patch 1.5), and
    # MagicMock calls/getattr/getitem never raise, so the amdsmi path
    # "succeeds" and returns a MagicMock instead of an int. That reaches
    # vllm/engine/arg_utils.py's get_batch_defaults(), which crashes the
    # API server at startup with:
    #   TypeError: '>=' not supported between instances of 'MagicMock'
    #   and 'int'
    # (device_memory >= 70 * GiB_bytes). Fix: skip amdsmi entirely and go
    # straight to the torch.cuda fallback, which Patch 10's ROCM-21812
    # VRAM shim already patches to report real APU GTT memory correctly.
    p_rocm_devmem = Path('vllm/platforms/rocm.py')
    if p_rocm_devmem.exists():
        txt = p_rocm_devmem.read_text()
        old_get_device_total_memory = (
            "    def get_device_total_memory(cls, device_id: int = 0) -> int:\n"
            "        # Query total VRAM via amdsmi so we don't initialize a HIP context in\n"
            "        # the calling process. torch.cuda.get_device_properties() creates a\n"
            "        # HIP context, which makes vLLM fall back from `fork` to `spawn` for\n"
            "        # worker processes. Keeping this query context-free preserves `fork`\n"
            "        # where it is otherwise valid (e.g. out-of-tree models registered in\n"
            "        # the parent process).\n"
            "        try:\n"
            "            physical_device_id = cls.device_id_to_physical_device_id(device_id)\n"
            "            return _query_total_memory_from_amdsmi(physical_device_id)\n"
            "        except Exception as e:\n"
            "            logger.debug(\"Failed to get total memory via amdsmi: %s\", e)\n"
            "            logger.warning_once(\n"
            "                \"Failed to get total memory via amdsmi, falling back to \"\n"
            "                \"torch.cuda. This will initialize CUDA.\"\n"
            "            )\n"
            "        return torch.cuda.get_device_properties(device_id).total_memory\n"
        )
        new_get_device_total_memory = (
            "    def get_device_total_memory(cls, device_id: int = 0) -> int:\n"
            "        # Patch 21 (Strix Halo): amdsmi is mocked (Patch 1.5) and never\n"
            "        # raises, so the amdsmi path below would silently return a\n"
            "        # MagicMock instead of int, crashing get_batch_defaults() in\n"
            "        # arg_utils.py. Skip straight to the torch.cuda fallback (Patch 10\n"
            "        # patches it to report real APU GTT memory).\n"
            "        return torch.cuda.get_device_properties(device_id).total_memory\n"
        )
        if old_get_device_total_memory in txt:
            txt = txt.replace(old_get_device_total_memory, new_get_device_total_memory, 1)
            p_rocm_devmem.write_text(txt)
            print(" -> Patched vllm/platforms/rocm.py (Patch 21: get_device_total_memory skips amdsmi, uses torch.cuda fallback directly)")
        elif "def get_device_total_memory" in txt and "Patch 21" not in txt:
            print(" !! Patch 21 target not found verbatim in vllm/platforms/rocm.py — get_device_total_memory NOT patched (source drift?)")
        else:
            print(" -> vllm/platforms/rocm.py already carries Patch 21 (or was already patched)")

    print("Successfully patched vLLM/Environment for Strix Halo.")

if __name__ == "__main__":
    patch_vllm()
