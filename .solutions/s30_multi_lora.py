"""Stage 30 - Multi-LoRA Serving & Batched GEMV (BGMV).

Reference solution.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LoRAAdapter:
    """A low-rank adapter representation."""
    adapter_id: str
    r: int
    alpha: float
    A: torch.Tensor  # (r, d_in)
    B: torch.Tensor  # (d_out, r)

    @property
    def scaling(self) -> float:
        return self.alpha / self.r


class BatchedLoRAManager:
    """Manages adapter registration and applies per-request LoRA deltas."""

    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        self.adapters: Dict[str, LoRAAdapter] = {}

    def register_adapter(self, adapter_id: str, r: int, alpha: float,
                         A: torch.Tensor, B: torch.Tensor):
        """Register a new LoRA adapter."""
        assert A.shape == (r, self.in_features), f"A shape {A.shape} != {(r, self.in_features)}"
        assert B.shape == (self.out_features, r), f"B shape {B.shape} != {(self.out_features, r)}"
        self.adapters[adapter_id] = LoRAAdapter(adapter_id, r, alpha, A, B)

    def forward(self, x: torch.Tensor, W0: torch.Tensor,
                adapter_ids: List[Optional[str]]) -> torch.Tensor:
        """Apply base linear projection and per-sequence LoRA deltas.

        Args:
            x: Input tensor of shape (batch_size, in_features)
            W0: Base weight of shape (out_features, in_features)
            adapter_ids: List of adapter IDs for each sequence in the batch (None = base model)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        # 1. Base model output for entire batch
        y = F.linear(x, W0)

        # 2. Per-sequence LoRA deltas (BGMV logic)
        for i, aid in enumerate(adapter_ids):
            if aid is None:
                continue
            adapter = self.adapters.get(aid)
            if adapter is None:
                continue

            # xi is (1, in_features)
            xi = x[i:i+1]
            # Low-rank delta: (1, r) = xi @ A.T
            h = torch.matmul(xi, adapter.A.t())
            # Output delta: (1, out_features) = h @ B.T * scaling
            delta = torch.matmul(h, adapter.B.t()) * adapter.scaling
            y[i:i+1] += delta

        return y
