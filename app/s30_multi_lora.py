"""Stage 30 - Multi-LoRA Serving & Batched GEMV (BGMV).

`./vc lore 30` for the insight. `./vc test 30` to check yourself.

In production LLM serving, dozens of tenants query different fine-tuned models
derived from the same base model (e.g. Llama-3-8B-Code vs Llama-3-8B-Legal).
Loading separate models would multiply VRAM consumption and shatter batching.

LoRA represents weight updates as low-rank matrices:
    W = W_0 + (alpha / r) * (B @ A)
where A is (r, d_in) and B is (d_out, r) with rank r << min(d_in, d_out).

In a continuous batch, different sequences request different adapters.
This stage implements `BatchedLoRAManager` which computes:
    y_i = x_i @ W_0.T + (alpha_k / r_k) * (x_i @ A_k.T) @ B_k.T
in a unified forward pass without duplicating the base model weights.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import torch
import torch.nn as nn


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
        raise NotImplementedError("stage 30: implement register_adapter")

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
        raise NotImplementedError("stage 30: implement forward")
