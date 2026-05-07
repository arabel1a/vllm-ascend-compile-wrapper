"""Operators under test. Add a new nn.Module here, point _target_ at it."""

import torch


class SiluMul(torch.nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(x) * y
