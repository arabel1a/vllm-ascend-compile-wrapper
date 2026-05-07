"""
Drive vllm-ascend's exact torch.compile path on a single user-supplied
nn.Module, no engine boot.

Path traversed (matches NPUModelRunner.execute_model):
    dynamo trace
        -> vllm.compilation.backends.VllmBackend
            -> CompilerManager
                -> AscendCompiler.compile
                    -> fusion_pass_compile        (compile_path=fusion)
                       npugraph_ex_compile        (compile_path=npugraph_ex)

Run:
    python -m debug_compile.main
    python -m debug_compile.main compile_path=npugraph_ex
    python -m debug_compile.main module._target_=debug_compile.ops.MyOp
"""

import logging
from pathlib import Path

import hydra
import torch
import torch_npu  # noqa: F401  -- registers 'npu' device
import vllm_ascend  # noqa: F401  -- registers NPUPlatform via entry point
from omegaconf import DictConfig

from vllm.compilation.backends import VllmBackend
from vllm.config import (
    CompilationConfig,
    ModelConfig,
    CompilationMode,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.platforms import current_platform

from vllm_ascend.ascend_forward_context import set_ascend_forward_context

log = logging.getLogger(__name__)


def setup_output_dir(cfg: DictConfig) -> Path:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, cfg.log_level.upper())
    logging.getLogger().setLevel(level)
    return out


def materialize_inputs(cfg: DictConfig) -> tuple[torch.Tensor, ...]:
    dtype = getattr(torch, cfg.dtype)
    return tuple(
        torch.randn(list(spec.shape), dtype=dtype, device=cfg.device)
        for spec in cfg.inputs
    )


def build_vllm_config(cfg: DictConfig) -> VllmConfig:
    """Smallest VllmConfig that NPUPlatform.check_and_update_config accepts."""
    # NOTE: cudagraph_mode MUST NOT be NONE. NPUPlatform.check_and_update_config
    # rewrites mode -> CompilationMode.NONE if cudagraph_mode == NONE, which
    # silently disables the whole compile path.
    compilation_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        use_inductor_graph_partition=False,
    )
    additional_config = {
        "ascend_compilation_config": dict(cfg.ascend_compilation_config),
        "npugraph_ex_config": {
            "enable": cfg.compile_path == "npugraph_ex",
            "enable_static_kernel": cfg.npugraph_ex_config.enable_static_kernel,
        },
    }
    model_config = ModelConfig(**cfg.model_config)
    vllm_config = VllmConfig(
        compilation_config=compilation_config,
        additional_config=additional_config,
        model_config=model_config,
    )
    # Mirrors what the engine does at startup: lets NPUPlatform mutate the
    # config in place AND initializes the AscendConfig singleton that
    # AscendCompiler.compile reads via get_ascend_config().
    current_platform.check_and_update_config(vllm_config)
    return vllm_config


def register_ascend_ops(vllm_config: VllmConfig) -> None:
    """Mirror NPUWorker.init_device op-registration block.

    Without this, the FX graph references torch ops that don't exist yet:
      - torch.ops._C_ascend.weak_ref_tensor (vllm_ascend_C extension)
      - torch.ops._C_ascend.quantize / fused-norm-quant (atb extensions)
      - the dummy fusion ops the GraphFusionPassManager rewrites into
    """
    from vllm_ascend import ops as _ascend_ops
    from vllm_ascend.utils import (
        AscendDeviceType,
        adapt_patch,
        enable_custom_op,
        get_ascend_device_type,
        register_ascend_customop,
    )
    from torch_npu.op_plugin.atb._atb_ops import _register_atb_extensions

    adapt_patch()
    enable_custom_op()
    _ascend_ops.register_dummy_fusion_op()
    if get_ascend_device_type() != AscendDeviceType.A5:
        _register_atb_extensions()
    register_ascend_customop(vllm_config)


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    setup_output_dir(cfg)
    torch.manual_seed(cfg.seed)

    assert current_platform.device_name == "npu", (
        f"current_platform={current_platform!r}; expected NPUPlatform. "
        "Install vllm-ascend in editable mode so its entry point fires."
    )

    mod = hydra.utils.instantiate(cfg.module).to(cfg.device).eval()
    inputs = materialize_inputs(cfg)

    if cfg.compile_path == "eager":
        with torch.inference_mode():
            out = mod(*inputs)
    else:
        vllm_config = build_vllm_config(cfg)
        register_ascend_ops(vllm_config)
        # ACLGraphWrapper reads batch_descriptor + cudagraph_runtime_mode off
        # the forward context. Use the ascend variant — it adds sp_enabled,
        # moe_comm_type, etc. that vllm_ascend ops/passes read.
        num_tokens = int(inputs[0].shape[0])
        with set_current_vllm_config(vllm_config):
            backend = VllmBackend(vllm_config)
            # dynamic=True so at least one input dim becomes SymInt; otherwise
            # PiecewiseBackend.__call__ trips on `args[self.sym_shape_indices[0]]`
            # (vllm/compilation/piecewise_backend.py:183).
            compiled = torch.compile(mod, backend=backend, fullgraph=True, dynamic=True)
            with torch.inference_mode(), set_ascend_forward_context(
                attn_metadata=None,
                vllm_config=vllm_config,
                num_tokens=num_tokens,
                aclgraph_runtime_mode=CUDAGraphMode.PIECEWISE,
                batch_descriptor=BatchDescriptor(num_tokens=num_tokens),
                model_instance=mod,
            ):
                out = compiled(*inputs)

    log.info("output shape=%s dtype=%s", tuple(out.shape), out.dtype)
    log.info("first row: %s", out.flatten()[:8].tolist())


if __name__ == "__main__":
    main()
