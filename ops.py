"""Operators under test. Swap module._target_ in config.yaml.

For lora ops, use run_lora_test() from main.py to get compile+eager comparison.
"""

import traceback
import torch
from baselines import sort_metadata

# =============================================================================
# Simple op (no external deps) – keep as-is for compile-path smoke testing
# =============================================================================

def silu_mm(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.silu(x) * y


# =============================================================================
# LoRA shrink operators (require triton_lora.generate_lora_metadata + sort_metadata)
# =============================================================================

# ----- V0: @torch.inference_mode() + repeat_interleave --------------------------------

@torch.inference_mode()
def _torch_shrink_one_slice_v0(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    scaling: float,
    token_to_lora_idx: torch.Tensor,
):
    token_to_lora = lora_a_weights[token_to_lora_idx].squeeze(1)
    output_tensor.copy_(scaling * torch.einsum("th,tlh->tl", inputs, token_to_lora))


@torch.inference_mode()
def torch_shrink_v0(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    scaling: float,
) -> None:
    token_to_lora_idx = torch.repeat_interleave(lora_indices_tensor, seq_len_tensor)
    for slice_idx in range(len(lora_a_weights)):
        _torch_shrink_one_slice_v0(
            inputs,
            lora_a_weights[slice_idx],
            output_tensor[slice_idx],
            scaling,
            token_to_lora_idx,
        )


# ----- V1: @torch.no_grad() (inference_mode fails with compile) ------------------------

@torch.no_grad()
def torch_shrink_v1(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    scaling: float,
) -> None:
    output_tensor.zero_()
    seq_len = inputs.shape[0]
    group_starts = torch.zeros(token_nums, device=inputs.device, dtype=torch.long)
    group_starts.scatter_(0, b_seq_start_loc, 1)
    group_id_per_token = torch.cumsum(group_starts, dim=0) - 1
    token_to_lora_idx = lora_indices_tensor[group_id_per_token]
    for slice_idx in range(len(lora_a_weights)):
        token_to_lora = lora_a_weights[slice_idx][token_to_lora_idx].squeeze(1)
        output_tensor[slice_idx].copy_(
            scaling * torch.einsum("th,tlh->tl", inputs, token_to_lora)
        )


# ----- V2: separate _torch_shrink_one_slice_v2 with @torch.no_grad() ------------------

@torch.no_grad()
def _torch_shrink_one_slice_v2(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    scaling: float,
    token_to_lora_idx: torch.Tensor,
):
    token_to_lora = lora_a_weights[token_to_lora_idx].squeeze(1)
    output_tensor.copy_(scaling * torch.einsum("th,tlh->tl", inputs, token_to_lora))


@torch.no_grad()
def torch_shrink_v2(
    inputs: torch.Tensor,
    lora_a_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    b_seq_start_loc: torch.Tensor,
    seq_len_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    batches: int,
    max_seq_length: int,
    token_nums: int,
    scaling: float,
) -> None:
    output_tensor.zero_()
    seq_len = inputs.shape[0]
    group_starts = torch.zeros(token_nums, device=inputs.device, dtype=torch.long)
    group_starts.scatter_(0, b_seq_start_loc, 1)
    group_id_per_token = torch.cumsum(group_starts, dim=0) - 1
    token_to_lora_idx = lora_indices_tensor[group_id_per_token]
    for slice_idx in range(len(lora_a_weights)):
        _torch_shrink_one_slice_v2(
            inputs,
            lora_a_weights[slice_idx],
            output_tensor[slice_idx],
            scaling,
            token_to_lora_idx,
        )


# =============================================================================
# test() – mirrors the original triton_lora test pattern
# Call run_lora_test(fn, rank, hidden_size, num_loras, num_requests) from main.py
# =============================================================================

def run_lora_test(fn, call_fn=None, rank=8, hidden_size=64, num_loras=2, num_requests=3):
    """
    Compare eager vs compiled on the lora shrink op.

    Generates realistic lora metadata via triton_lora.common, then loops over
    (num_tokens, num_slices) combinations and checks that the output tensor
    matches the reference torch_shrink_v0 output.

    Args:
        fn: the raw function under test
        call_fn: if provided (non-eager path), called as
                 call_fn(fn, inputs, lora_a_weights, output_tensor, *sorted_meta)
                 to invoke the compiled function inside the forward context.
                 If None, fn is called directly (eager baseline).
    """
    from triton_lora.common import generate_lora_metadata, production_to_triton_metadata
    from baselines import sort_metadata

    for num_tokens in (1024, 2048, 512):
        for num_slices in (3, 7, 1, 2):
            torch.manual_seed(0)
            tag = f"tokens={num_tokens} loras={num_loras} slices={num_slices} rank={rank}"
            print(tag)

            lora_a_weights = [
                torch.randn(
                    num_loras, rank, hidden_size,
                    dtype=torch.float16, device="npu"
                ).contiguous()
                for _ in range(num_slices)
            ]
            inputs = torch.randn(
                num_tokens, hidden_size,
                dtype=torch.float16, device="npu"
            ).contiguous()

            metadata = generate_lora_metadata(
                num_tokens, num_loras, "npu", num_requests
            )
            sorted_meta = sort_metadata(*metadata)

            # Reference output via eager torch_shrink_v0
            ref_out = torch.ones(
                num_slices, num_tokens, rank,
                dtype=torch.float16, device="npu"
            )
            torch_shrink_v0(inputs, lora_a_weights, ref_out, *sorted_meta, 1.0)

            # Test target output (call_fn handles compile+context, or direct call)
            test_out = torch.zeros(
                num_slices, num_tokens, rank,
                dtype=torch.float16, device="npu"
            )
            if call_fn is None:
                fn(inputs, lora_a_weights, test_out, *sorted_meta, 1.0)
            else:
                call_fn(fn, inputs, lora_a_weights, test_out, *sorted_meta, 1.0)

            diff = (test_out - ref_out).abs().sum()
            print(f"  diff={diff.item()}")
