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

    # Patch 22 (local cherry-pick of upstream vLLM PR #45916, not yet merged,
    # ported from gfx12x to gfx1x): split-KV Triton paged-decode kernel.
    #
    # Root cause (confirmed by direct measurement on this image, 2026-07-08):
    # the Triton paged-attention decode fallback (used here because this
    # model's hybrid mamba/GDN layers force a non-power-of-2 block_size=1056,
    # which disqualifies vLLM's hand-written ROCm custom kernel) launches ONE
    # Triton program per (sequence, kv_head) that loops SEQUENTIALLY over the
    # entire KV cache. At batch_size=1 (a single agent session - the normal
    # case for opencode/kydah-code) this uses only num_kv_heads programs,
    # leaving most of the GPU's compute units idle, and decode throughput
    # collapses as context grows: measured 18-20 tok/s at ~150 tokens of
    # context vs 1.9 tok/s at ~50k tokens on this build (before this patch).
    #
    # PR #45916 fixes the identical problem on gfx12x (RDNA4) by adding a
    # split-KV ("flash decoding") variant: the KV range is divided into
    # actual_max_splits chunks, each computed by its own Triton program,
    # then reduced (log-sum-exp merge) by a second kernel. This multiplies
    # the number of programs launched per decode step, filling idle compute
    # units at low batch size / long context. Upstream's own benchmark
    # (Qwen3.5-9B, concurrency=1, gfx1201) showed +47% throughput at 32k
    # context, +36% at 16k, +24% at 8k - the same shape of problem we
    # measured here.
    #
    # The PR gates the new path on on_gfx12x() + head_dim == 256 + bf16/fp16
    # + no alibi/sliding-window/sinks/fp8. This model
    # (Chunity/Qwen3.6-35B-A3B-AutoRound-AWQ-4bit, and the Qwen3.6 family in
    # general) has head_dim == 256 (verified via the HF config text_config)
    # and runs bf16 with none of the excluded features, so it qualifies
    # as-is once the gate is widened. This patch changes only that one gate,
    # on_gfx12x() -> on_gfx1x() (already used throughout this patch bundle
    # for gfx11xx/gfx12xx consumer RDNA parts, defined in
    # vllm/platforms/rocm.py as any(arch in _GCN_ARCH for arch in
    # ["gfx11", "gfx12"])) - the kernel code itself is untouched.
    #
    # Caveats carried over from upstream, not resolved by this port:
    #   - The split-count heuristic (_get_num_splits) targets
    #     2 * multi_processor_count "workgroups", calibrated against gfx12's
    #     WGP/CU reporting quirk (see the comment in _num_splits_heuristic).
    #     Whether ROCm/torch reports CUs or WGPs for multi_processor_count on
    #     gfx1151 specifically is unverified here; if it's off, the kernel
    #     still produces correct results (this only affects how many
    #     programs are launched, not the math), just possibly suboptimal
    #     occupancy - retune empirically if the gain looks smaller than
    #     upstream's numbers.
    #   - Upstream's own author tested only on a single gfx1201 card and
    #     explicitly flagged gpt-oss (head_dim==64) as untested; this port
    #     inherits that limited validation. Needs a real before/after
    #     benchmark on gfx1151 (not just "it builds and runs") before being
    #     trusted for production.
    p_chunked_decode = Path('vllm/v1/attention/ops/chunked_prefill_paged_decode.py')
    if p_chunked_decode.exists():
        txt = p_chunked_decode.read_text()
        if "kernel_paged_attention_2d_splitkv" not in txt:
            applied = False


            OLD_IMPORTS_22 = '''import torch

from vllm import _custom_ops as ops
'''
            NEW_IMPORTS_22 = '''import math

import torch

from vllm import _custom_ops as ops
'''
            OLD_HELPERS_ANCHOR_22 = '''float8_info = torch.finfo(current_platform.fp8_dtype())


def has_native_kv_cache_layout('''
            NEW_HELPERS_ANCHOR_22 = '''float8_info = torch.finfo(current_platform.fp8_dtype())

_MAX_SPLITS = 16
_DEFAULT_COMPUTE_BLOCK_SIZE = 32


# The split-kv kernel has the best performance when the
# compute block size is 32.
def _choose_compute_block_size(physical_block_size: int) -> int:
    """Choose the logical attention tile size inside a physical KV block."""
    for block_size in (32, 16, 8, 4, 2):
        if physical_block_size % block_size == 0:
            return min(block_size, _DEFAULT_COMPUTE_BLOCK_SIZE)
    return 1


def has_native_kv_cache_layout('''
            OLD_CALLSITE_22 = '''        kernel_paged_attention_2d[
            (
                num_seqs,
                num_kv_heads,
            )
        ](
            output_ptr=output,
            query_ptr=query,
            key_cache_ptr=key_cache,
            value_cache_ptr=value_cache,
            sink_ptr=sinks,
            block_tables_ptr=processed_block_table,
            seq_lens_ptr=seq_lens,
            alibi_slopes_ptr=alibi_slopes,
            scale=sm_scale,
            k_scale=k_scale,
            v_scale=v_scale,
            out_scale_inv=1.0 / output_scale if output_scale is not None else 1.0,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            num_queries_per_kv_padded=num_queries_per_kv_padded,
            block_table_stride=processed_block_table.stride(0),
            query_stride_0=query.stride(0),
            query_stride_1=query.stride(1),
            output_stride_0=output.stride(0),
            output_stride_1=output.stride(1),
            BLOCK_SIZE=TRITON_BLOCK_SIZE,
            PHYSICAL_BLOCK_SIZE=real_block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            USE_ALIBI_SLOPES=use_alibi_slopes,
            SLIDING_WINDOW=sliding_window,
            x=key_cache.shape[4],
            stride_k_cache_0=key_cache.stride(0),
            stride_k_cache_1=key_cache.stride(1),
            stride_k_cache_2=key_cache.stride(2),
            stride_k_cache_3=key_cache.stride(3),
            stride_k_cache_4=key_cache.stride(4),
            stride_v_cache_0=value_cache.stride(0),
            stride_v_cache_1=value_cache.stride(1),
            stride_v_cache_2=value_cache.stride(2),
            stride_v_cache_3=value_cache.stride(3),
            filter_by_query_len=True,
            query_start_len_ptr=query_start_loc,
            USE_SINKS=sinks is not None,
            USE_FP8=output_scale is not None,
        )
'''
            NEW_CALLSITE_22 = '''        from vllm.platforms.rocm import on_gfx1x

        # Split kv path (upstream vLLM PR #45916, gfx12x-only) ported to
        # gfx1x (Strix Halo / gfx1151 included) - see Patch 22 docstring.
        use_splitkv_decode = (
            on_gfx1x()
            and query.dtype in (torch.float16, torch.bfloat16)
            and head_size == 256
            and not use_alibi_slopes
            and sliding_window == 0
            and sinks is None
            and output_scale is None
            and "fp8" not in kv_cache_dtype
        )
        if use_splitkv_decode:
            paged_attention_2d_splitkv_decode(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                block_tables=processed_block_table,
                seq_lens=seq_lens,
                scale=sm_scale,
                output=output,
                max_seq_len=max_seq_len,
                max_num_splits=_MAX_SPLITS,
                query_start_loc=query_start_loc,
                filter_by_query_len=True,
            )
        else:
            kernel_paged_attention_2d[
                (
                    num_seqs,
                    num_kv_heads,
                )
            ](
                output_ptr=output,
                query_ptr=query,
                key_cache_ptr=key_cache,
                value_cache_ptr=value_cache,
                sink_ptr=sinks,
                block_tables_ptr=processed_block_table,
                seq_lens_ptr=seq_lens,
                alibi_slopes_ptr=alibi_slopes,
                scale=sm_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                out_scale_inv=1.0 / output_scale if output_scale is not None else 1.0,
                num_query_heads=num_query_heads,
                num_queries_per_kv=num_queries_per_kv,
                num_queries_per_kv_padded=num_queries_per_kv_padded,
                block_table_stride=processed_block_table.stride(0),
                query_stride_0=query.stride(0),
                query_stride_1=query.stride(1),
                output_stride_0=output.stride(0),
                output_stride_1=output.stride(1),
                BLOCK_SIZE=TRITON_BLOCK_SIZE,
                PHYSICAL_BLOCK_SIZE=real_block_size,
                HEAD_SIZE=head_size,
                HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
                USE_ALIBI_SLOPES=use_alibi_slopes,
                SLIDING_WINDOW=sliding_window,
                x=key_cache.shape[4],
                stride_k_cache_0=key_cache.stride(0),
                stride_k_cache_1=key_cache.stride(1),
                stride_k_cache_2=key_cache.stride(2),
                stride_k_cache_3=key_cache.stride(3),
                stride_k_cache_4=key_cache.stride(4),
                stride_v_cache_0=value_cache.stride(0),
                stride_v_cache_1=value_cache.stride(1),
                stride_v_cache_2=value_cache.stride(2),
                stride_v_cache_3=value_cache.stride(3),
                filter_by_query_len=True,
                query_start_len_ptr=query_start_loc,
                USE_SINKS=sinks is not None,
                USE_FP8=output_scale is not None,
            )


@triton.jit
def kernel_paged_attention_2d_splitkv(
    mid_out_ptr,  # [num_seqs, num_query_heads, max_num_splits, head_size]
    mid_lse_ptr,  # [num_seqs, num_query_heads, max_num_splits]
    query_ptr,  # [num_tokens, num_query_heads, head_size]
    key_cache_ptr,  # [num_blks, num_kv_heads, head_size // x, blk_size, x]
    value_cache_ptr,  # [num_blks, num_kv_heads, head_size, blk_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    seq_lens_ptr,  # [num_seqs]
    scale,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    num_queries_per_kv_padded: tl.constexpr,
    block_table_stride: tl.int64,
    query_stride_0: tl.int64,
    query_stride_1: tl.int64,
    mid_out_stride_0: tl.int64,
    mid_out_stride_1: tl.int64,
    mid_out_stride_2: tl.int64,
    mid_lse_stride_0: tl.int64,
    mid_lse_stride_1: tl.int64,
    mid_lse_stride_2: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    PHYSICAL_BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    x: tl.constexpr,
    stride_k_cache_0: tl.int64,
    stride_k_cache_1: tl.int64,
    stride_k_cache_2: tl.int64,
    stride_k_cache_3: tl.int64,
    stride_k_cache_4: tl.int64,
    stride_v_cache_0: tl.int64,
    stride_v_cache_1: tl.int64,
    stride_v_cache_2: tl.int64,
    stride_v_cache_3: tl.int64,
    filter_by_query_len: tl.constexpr,
    query_start_len_ptr,  # [num_seqs+1]
):
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    split_idx = tl.program_id(2)

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
        if cur_batch_query_len > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    num_splits = tl.num_programs(2)

    split_len = cdiv_fn(cdiv_fn(seq_len, num_splits), BLOCK_SIZE) * BLOCK_SIZE
    split_start = split_idx * split_len
    split_end = tl.minimum(split_start + split_len, seq_len)

    query_head_idx = kv_head_idx * num_queries_per_kv + tl.arange(
        0, num_queries_per_kv_padded
    )
    head_mask = query_head_idx < (kv_head_idx + 1) * num_queries_per_kv
    head_mask = head_mask & (query_head_idx < num_query_heads)

    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = offs_d < HEAD_SIZE

    query_offset = (
        cur_batch_in_all_start_index * query_stride_0
        + query_head_idx[:, None] * query_stride_1
    )
    Q = tl.load(
        query_ptr + query_offset + offs_d[None, :],
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    M = tl.full([num_queries_per_kv_padded], float("-inf"), dtype=tl.float32)
    L = tl.zeros([num_queries_per_kv_padded], dtype=tl.float32)
    acc = tl.zeros([num_queries_per_kv_padded, HEAD_SIZE_PADDED], dtype=tl.float32)

    block_table_offset = seq_idx * block_table_stride
    offs_n = tl.arange(0, BLOCK_SIZE)

    for start_n in tl.range(split_start, split_end, BLOCK_SIZE):
        abs_token_idx = start_n + offs_n
        l_block_idx = abs_token_idx // PHYSICAL_BLOCK_SIZE
        p_block_idx = tl.load(block_tables_ptr + block_table_offset + l_block_idx)
        internal_offsets = abs_token_idx % PHYSICAL_BLOCK_SIZE
        token_mask = abs_token_idx < split_end

        # Should use stride_k_cache_4 = 1 and stride_k_cache_3 = x here
        # to make triton compiler happy.
        # However benchmark show the compiler correctly generates
        # 128bit memory access instruction,
        # but there is no obvious performance difference.
        # So we keep the original stride for better readability.
        k_offset = (
            p_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_1
            + (offs_d[:, None] // x) * stride_k_cache_2
            + internal_offsets[None, :] * stride_k_cache_3
            + (offs_d[:, None] % x) * stride_k_cache_4
        )
        K = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & token_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )

        v_offset = (
            p_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_1
            + offs_d[None, :] * stride_v_cache_2
            + internal_offsets[:, None] * stride_v_cache_3
        )
        V = tl.load(
            value_cache_ptr + v_offset,
            mask=token_mask[:, None] & dim_mask[None, :],
            other=0.0,
            eviction_policy="evict_last",
        )

        S = scale * tl.dot(Q, K)
        S = tl.where(head_mask[:, None] & token_mask[None, :], S, float("-inf"))

        m_j = tl.maximum(M, tl.max(S, axis=1))
        p = tl.exp(S - m_j[:, None])
        p = tl.where(m_j[:, None] == float("-inf"), 0.0, p)
        l_j = tl.sum(p, axis=1)

        # Previous partial sums are expressed in exp(x - M); rescale them when
        # the running maximum increases before adding this block.
        alpha = tl.exp(M - m_j)
        alpha = tl.where(float("-inf") == M, 0.0, alpha)
        acc = acc * alpha[:, None]
        L = L * alpha + l_j
        M = m_j
        acc += tl.dot(p.to(V.dtype), V)

    mid_out_offset = (
        seq_idx * mid_out_stride_0
        + query_head_idx[:, None] * mid_out_stride_1
        + split_idx * mid_out_stride_2
        + offs_d[None, :]
    )
    mid_lse_offset = (
        seq_idx * mid_lse_stride_0
        + query_head_idx * mid_lse_stride_1
        + split_idx * mid_lse_stride_2
    )

    has_tokens = split_end > split_start
    out = acc / (L[:, None] + 1e-10)
    lse = M + tl.log(L)

    tl.store(
        mid_out_ptr + mid_out_offset,
        out,
        mask=has_tokens & head_mask[:, None] & dim_mask[None, :],
    )
    tl.store(
        mid_lse_ptr + mid_lse_offset,
        lse,
        mask=has_tokens & head_mask,
    )


@triton.jit
def kernel_paged_attention_2d_splitkv_reduce(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    mid_out_ptr,  # [num_seqs, num_query_heads, max_num_splits, head_size]
    mid_lse_ptr,  # [num_seqs, num_query_heads, max_num_splits]
    seq_lens_ptr,  # [num_seqs]
    output_stride_0: tl.int64,
    output_stride_1: tl.int64,
    mid_out_stride_0: tl.int64,
    mid_out_stride_1: tl.int64,
    mid_out_stride_2: tl.int64,
    mid_lse_stride_0: tl.int64,
    mid_lse_stride_1: tl.int64,
    mid_lse_stride_2: tl.int64,
    BLOCK_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    MAX_NUM_SPLITS: tl.constexpr,
    filter_by_query_len: tl.constexpr,
    query_start_len_ptr,  # [num_seqs+1]
):
    seq_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    if filter_by_query_len:
        cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
        cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
        cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
        if cur_batch_query_len > 1:
            return
    else:
        cur_batch_in_all_start_index = seq_idx

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    split_len = cdiv_fn(cdiv_fn(seq_len, MAX_NUM_SPLITS), BLOCK_SIZE) * BLOCK_SIZE

    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    dim_mask = offs_d < HEAD_SIZE

    M = -float("inf")
    L = 0.0
    acc = tl.zeros([HEAD_SIZE_PADDED], dtype=tl.float32)

    for split_idx in tl.range(0, MAX_NUM_SPLITS, num_stages=2):
        split_start = split_idx * split_len
        split_end = tl.minimum(split_start + split_len, seq_len)

        if split_end > split_start:
            lse = tl.load(
                mid_lse_ptr
                + seq_idx * mid_lse_stride_0
                + query_head_idx * mid_lse_stride_1
                + split_idx * mid_lse_stride_2
            )
            partial = tl.load(
                mid_out_ptr
                + seq_idx * mid_out_stride_0
                + query_head_idx * mid_out_stride_1
                + split_idx * mid_out_stride_2
                + offs_d,
                mask=dim_mask,
                other=0.0,
            )

            m_j = tl.maximum(M, lse)
            alpha = tl.exp(M - m_j)
            beta = tl.exp(lse - m_j)
            acc = acc * alpha + partial * beta
            L = L * alpha + beta
            M = m_j

    out = acc / (L + 1e-10)
    tl.store(
        output_ptr
        + cur_batch_in_all_start_index * output_stride_0
        + query_head_idx * output_stride_1
        + offs_d,
        out,
        mask=dim_mask,
    )


def _num_splits_heuristic(
    batch_nheads_mblocks: int,
    num_sms: int,
    num_n_blocks: int,
    max_splits: int,
) -> int:
    """Choose split count for small-batch decode occupancy.

    Use FlashAttention's wave-efficiency heuristic: pick the smallest eligible
    split whose wave efficiency (n_waves / ceil(n_waves)) is within 85% of the
    maximum achievable.  On gfx12 torch reports WGPs while rocprof reports CUs,
    so target two workgroups per reported processor.
    """
    target_workgroups = 2 * num_sms
    if batch_nheads_mblocks >= 0.8 * target_workgroups:
        return 1

    max_splits = min(max_splits, num_sms, num_n_blocks)
    if max_splits <= 1:
        return 1

    def is_split_eligible(num_splits: int) -> bool:
        return num_splits == 1 or _cdiv(num_n_blocks, num_splits) != _cdiv(
            num_n_blocks, num_splits - 1
        )

    max_efficiency = 0.0
    efficiency = []
    for num_splits in range(1, max_splits + 1):
        if not is_split_eligible(num_splits):
            efficiency.append(0.0)
            continue

        n_waves = batch_nheads_mblocks * num_splits / target_workgroups
        eff = n_waves / math.ceil(n_waves)
        max_efficiency = max(max_efficiency, eff)
        efficiency.append(eff)

    for num_splits, eff in enumerate(efficiency, start=1):
        if is_split_eligible(num_splits) and eff >= 0.85 * max_efficiency:
            return num_splits

    return 1


def _choose_decode_num_splits(
    batch_size: int,
    num_kv_heads: int,
    max_seq_len: int,
    block_size: int,
    max_num_splits: int,
    num_sms: int,
) -> int:
    if max_seq_len <= block_size:
        return 1

    batch_nheads = batch_size * num_kv_heads
    num_n_blocks = _cdiv(max_seq_len, block_size)

    # Not enough KV blocks to keep each split busy on the GPU.
    if num_n_blocks < 2 * num_sms:
        return 1

    max_splits = min(max_num_splits, num_sms, num_n_blocks)
    return _num_splits_heuristic(batch_nheads, num_sms, num_n_blocks, max_splits)


def _get_num_splits(
    batch_size: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    max_seq_len: int,
    max_num_splits: int = _MAX_SPLITS,
    num_sms: int | None = None,
) -> int:
    """Heuristic for decode split-KV.

    Decode means one query token per sequence. max_seq_len is the static
    KV/context length bound used to choose one split count for the whole launch.
    """
    if num_sms is None:
        num_sms = torch.cuda.get_device_properties(
            torch.accelerator.current_device_index()
        ).multi_processor_count

    compute_block_size = _choose_compute_block_size(block_size)

    if head_size <= 64 and max_seq_len < 4096:
        return 1

    # Match FlashAttention's 128-thread split-KV occupancy model. The model is
    # based on compute tiles, not physical cache pages; a 528-token page should
    # behave like 33 smaller 16-token tiles for split-KV scheduling.
    return _choose_decode_num_splits(
        batch_size,
        num_kv_heads,
        max_seq_len,
        compute_block_size,
        max_num_splits,
        num_sms,
    )


def paged_attention_2d_splitkv_decode(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    output: torch.Tensor | None = None,
    actual_max_splits: int | None = None,
    max_seq_len: int | None = None,
    mid_out: torch.Tensor | None = None,
    mid_lse: torch.Tensor | None = None,
    max_num_splits: int = _MAX_SPLITS,
    query_start_loc: torch.Tensor | None = None,
    filter_by_query_len: bool = False,
) -> torch.Tensor:
    """Decode-only paged attention launcher with split-KV.

    query is one token per sequence: [batch, num_query_heads, head_size].
    seq_lens is the current KV/context length per sequence, not query length.
    """
    if output is None:
        output = torch.empty_like(query)

    batch_size = seq_lens.shape[0] if filter_by_query_len else query.shape[0]
    num_query_heads = query.shape[1]
    head_size = query.shape[2]
    num_kv_heads = key_cache.shape[1]
    physical_block_size = key_cache.shape[3]
    block_size = _choose_compute_block_size(physical_block_size)
    if block_size != 32:
        logger.warning_once(
            f"Chosen block size {block_size} may lead to suboptimal performance. "
            f"Consider using block size = 32 if possible."
        )
    x = key_cache.shape[4]
    num_queries_per_kv = num_query_heads // num_kv_heads
    num_queries_per_kv_padded = max(triton.next_power_of_2(num_queries_per_kv), 16)
    head_size_padded = triton.next_power_of_2(head_size)

    if max_seq_len is None:
        max_seq_len = block_tables.shape[1] * physical_block_size

    if actual_max_splits is None:
        actual_max_splits = _get_num_splits(
            batch_size,
            num_kv_heads,
            head_size,
            block_size,
            max_seq_len,
            max_num_splits,
        )

    if actual_max_splits > max_num_splits:
        raise ValueError(
            f"actual_max_splits ({actual_max_splits}) must be <= "
            f"max_num_splits ({max_num_splits})."
        )

    if actual_max_splits == 1:
        kernel_paged_attention_2d[(batch_size, num_kv_heads)](
            output_ptr=output,
            query_ptr=query,
            key_cache_ptr=key_cache,
            value_cache_ptr=value_cache,
            sink_ptr=None,
            block_tables_ptr=block_tables,
            seq_lens_ptr=seq_lens,
            alibi_slopes_ptr=None,
            scale=scale,
            k_scale=1.0,
            v_scale=1.0,
            out_scale_inv=1.0,
            num_query_heads=num_query_heads,
            num_queries_per_kv=num_queries_per_kv,
            num_queries_per_kv_padded=num_queries_per_kv_padded,
            block_table_stride=block_tables.stride(0),
            query_stride_0=query.stride(0),
            query_stride_1=query.stride(1),
            output_stride_0=output.stride(0),
            output_stride_1=output.stride(1),
            BLOCK_SIZE=block_size,
            PHYSICAL_BLOCK_SIZE=physical_block_size,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=head_size_padded,
            USE_ALIBI_SLOPES=False,
            SLIDING_WINDOW=0,
            x=x,
            stride_k_cache_0=key_cache.stride(0),
            stride_k_cache_1=key_cache.stride(1),
            stride_k_cache_2=key_cache.stride(2),
            stride_k_cache_3=key_cache.stride(3),
            stride_k_cache_4=key_cache.stride(4),
            stride_v_cache_0=value_cache.stride(0),
            stride_v_cache_1=value_cache.stride(1),
            stride_v_cache_2=value_cache.stride(2),
            stride_v_cache_3=value_cache.stride(3),
            filter_by_query_len=filter_by_query_len,
            query_start_len_ptr=query_start_loc,
            USE_SINKS=False,
            USE_FP8=False,
        )
        return output

    # In practice the intermediate buffers should be pre-allocated,
    # however the calls to this function is deeply coupled with CUDA FlashAttention,
    # so we allocate them here for simplicity.
    if mid_out is None:
        mid_out = torch.empty(
            (batch_size, num_query_heads, actual_max_splits, head_size),
            device=query.device,
            dtype=torch.float32,
        )
    if mid_lse is None:
        mid_lse = torch.empty(
            (batch_size, num_query_heads, actual_max_splits),
            device=query.device,
            dtype=torch.float32,
        )

    kernel_paged_attention_2d_splitkv[(batch_size, num_kv_heads, actual_max_splits)](
        mid_out,
        mid_lse,
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        num_queries_per_kv_padded=num_queries_per_kv_padded,
        block_table_stride=block_tables.stride(0),
        query_stride_0=query.stride(0),
        query_stride_1=query.stride(1),
        mid_out_stride_0=mid_out.stride(0),
        mid_out_stride_1=mid_out.stride(1),
        mid_out_stride_2=mid_out.stride(2),
        mid_lse_stride_0=mid_lse.stride(0),
        mid_lse_stride_1=mid_lse.stride(1),
        mid_lse_stride_2=mid_lse.stride(2),
        BLOCK_SIZE=block_size,
        PHYSICAL_BLOCK_SIZE=physical_block_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        x=x,
        stride_k_cache_0=key_cache.stride(0),
        stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2),
        stride_k_cache_3=key_cache.stride(3),
        stride_k_cache_4=key_cache.stride(4),
        stride_v_cache_0=value_cache.stride(0),
        stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2),
        stride_v_cache_3=value_cache.stride(3),
        filter_by_query_len=filter_by_query_len,
        query_start_len_ptr=query_start_loc,
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
    )
    kernel_paged_attention_2d_splitkv_reduce[(batch_size, num_query_heads)](
        output,
        mid_out,
        mid_lse,
        seq_lens,
        output.stride(0),
        output.stride(1),
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_lse.stride(0),
        mid_lse.stride(1),
        mid_lse.stride(2),
        BLOCK_SIZE=block_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        MAX_NUM_SPLITS=actual_max_splits,
        filter_by_query_len=filter_by_query_len,
        query_start_len_ptr=query_start_loc,
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
    )
    return output
'''

            old_imports = OLD_IMPORTS_22
            new_imports = NEW_IMPORTS_22
            if old_imports in txt:
                txt = txt.replace(old_imports, new_imports, 1)
                applied = True

            old_helpers_anchor = OLD_HELPERS_ANCHOR_22
            new_helpers_anchor = NEW_HELPERS_ANCHOR_22
            if old_helpers_anchor in txt:
                txt = txt.replace(old_helpers_anchor, new_helpers_anchor, 1)
                applied = True

            old_callsite_22 = OLD_CALLSITE_22
            new_callsite_22 = NEW_CALLSITE_22
            if old_callsite_22 in txt:
                txt = txt.replace(old_callsite_22, new_callsite_22, 1)
                applied = True

            if applied:
                p_chunked_decode.write_text(txt)
                print(" -> Patched vllm/v1/attention/ops/chunked_prefill_paged_decode.py (Patch 22: split-KV decode kernel from PR #45916, gfx12x -> gfx1x)")
            else:
                print(" !! Patch 22 targets not found in chunked_prefill_paged_decode.py - split-KV decode kernel NOT applied (source drift?)")
        else:
            print(" -> chunked_prefill_paged_decode.py already carries Patch 22 (or was already patched)")

    print("Successfully patched vLLM/Environment for Strix Halo.")

if __name__ == "__main__":
    patch_vllm()
