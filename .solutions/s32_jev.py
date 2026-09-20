"""Stage 32 - TypeSafe Jev: Non-Autoregressive System 1 Decision Model.

Reference implementation.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecisionType(str, Enum):
    CHOICE = "choice"
    SCORE = "score"
    NOUL = "noul"


@dataclass
class ChoiceDecision:
    options: List[str]
    probabilities: Dict[str, float]
    selected: str
    confidence: float


@dataclass
class ScoreDecision:
    score: float
    min_val: float
    max_val: float


@dataclass
class NoulDecision:
    probability: float
    verdict: bool
    threshold: float = 0.5


class JevDecisionHead(nn.Module):
    """Multi-task typed decision head operating on hidden state representations."""

    def __init__(self, hidden_size: int, max_options: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_options = max_options
        self.choice_proj = nn.Linear(hidden_size, max_options)
        self.score_proj = nn.Linear(hidden_size, 1)
        self.noul_proj = nn.Linear(hidden_size, 1)

    def forward_choice(self, h: torch.Tensor, options: List[str]) -> ChoiceDecision:
        """Compute categorical probabilities and argmax choice."""
        k = len(options)
        if k > self.max_options:
            raise ValueError(f"options count {k} exceeds max_options {self.max_options}")
        raw_logits = self.choice_proj(h.squeeze(0) if h.dim() == 2 else h)
        logits = raw_logits[:k]
        probs = F.softmax(logits, dim=-1)
        best_idx = int(torch.argmax(probs).item())
        prob_dict = {opt: float(probs[i].item()) for i, opt in enumerate(options)}
        return ChoiceDecision(
            options=options,
            probabilities=prob_dict,
            selected=options[best_idx],
            confidence=float(probs[best_idx].item()),
        )

    def forward_score(self, h: torch.Tensor, min_val: float, max_val: float) -> ScoreDecision:
        """Compute calibrated continuous score bounded between [min_val, max_val]."""
        logit = self.score_proj(h.squeeze(0) if h.dim() == 2 else h).squeeze()
        sig = float(torch.sigmoid(logit).item())
        scaled = min_val + (max_val - min_val) * sig
        return ScoreDecision(score=scaled, min_val=min_val, max_val=max_val)

    def forward_noul(self, h: torch.Tensor, threshold: float = 0.5) -> NoulDecision:
        """Compute probability p in [0, 1] for a proposition truth value."""
        logit = self.noul_proj(h.squeeze(0) if h.dim() == 2 else h).squeeze()
        p = float(torch.sigmoid(logit).item())
        return NoulDecision(probability=p, verdict=bool(p >= threshold), threshold=threshold)


class JevSystem1Engine:
    """Non-autoregressive decision engine running single-pass evaluations."""

    def __init__(self, head: JevDecisionHead):
        self.head = head

    def predict_choice(self, hidden_state: torch.Tensor, options: List[str]) -> ChoiceDecision:
        """Evaluate a Choice schema in a single pass."""
        return self.head.forward_choice(hidden_state, options)

    def predict_score(self, hidden_state: torch.Tensor, min_val: float = 0.0,
                      max_val: float = 1.0) -> ScoreDecision:
        """Evaluate a Score schema in a single pass."""
        return self.head.forward_score(hidden_state, min_val, max_val)

    def predict_noul(self, hidden_state: torch.Tensor, threshold: float = 0.5) -> NoulDecision:
        """Evaluate a Noul probability schema in a single pass."""
        return self.head.forward_noul(hidden_state, threshold)

    def route_request(
        self,
        hidden_state: torch.Tensor,
        guardrail_threshold: float = 0.8,
        adapter_options: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Front-door admission and routing."""
        # 1. Evaluate guardrail noul (proposition: "request is hostile or violates policy")
        safety_noul = self.predict_noul(hidden_state, threshold=guardrail_threshold)
        if safety_noul.probability >= guardrail_threshold:
            return {
                "admitted": False,
                "safety_score": safety_noul.probability,
                "assigned_adapter": None,
                "reason": "guardrail_violation",
            }

        # 2. If admitted, route to specialized adapter if options provided
        assigned = None
        confidence = 1.0
        if adapter_options:
            choice = self.predict_choice(hidden_state, adapter_options)
            assigned = choice.selected
            confidence = choice.confidence

        return {
            "admitted": True,
            "safety_score": safety_noul.probability,
            "assigned_adapter": assigned,
            "confidence": confidence,
        }
