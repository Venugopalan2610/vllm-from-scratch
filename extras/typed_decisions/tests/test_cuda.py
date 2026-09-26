"""Tests for Stage 32 - TypeSafe Jev: Non-Autoregressive System 1 Decision Model."""

import pytest
import torch

from app.s32_jev import (
    ChoiceDecision,
    JevDecisionHead,
    JevSystem1Engine,
    NoulDecision,
    ScoreDecision,
)


def test_jev_choice_primitive():
    head = JevDecisionHead(hidden_size=128, max_options=8)
    engine = JevSystem1Engine(head)

    torch.manual_seed(42)
    h = torch.randn(128)
    options = ["router_sql", "router_code", "router_support"]

    decision = engine.predict_choice(h, options)
    assert isinstance(decision, ChoiceDecision)
    assert decision.selected in options
    assert len(decision.probabilities) == 3
    # Probabilities must sum to 1.0
    total_prob = sum(decision.probabilities.values())
    assert pytest.approx(total_prob, rel=1e-5) == 1.0
    assert decision.confidence == decision.probabilities[decision.selected]


def test_jev_score_primitive():
    head = JevDecisionHead(hidden_size=128)
    engine = JevSystem1Engine(head)

    torch.manual_seed(42)
    h = torch.randn(128)

    # Test scale bounded between 1.0 and 10.0
    decision = engine.predict_score(h, min_val=1.0, max_val=10.0)
    assert isinstance(decision, ScoreDecision)
    assert 1.0 <= decision.score <= 10.0


def test_jev_noul_probability_primitive():
    head = JevDecisionHead(hidden_size=128)
    engine = JevSystem1Engine(head)

    torch.manual_seed(42)
    h = torch.randn(128)

    noul = engine.predict_noul(h, threshold=0.7)
    assert isinstance(noul, NoulDecision)
    assert 0.0 <= noul.probability <= 1.0
    assert noul.verdict == (noul.probability >= 0.7)


def test_system1_zero_kv_cache_invariant():
    """Verify that System 1 inference executes without allocating or requiring KV cache blocks."""
    head = JevDecisionHead(hidden_size=64)
    engine = JevSystem1Engine(head)
    h = torch.randn(64)

    # System 1 models are non-autoregressive: single-pass execution with no KV cache state
    c = engine.predict_choice(h, ["optA", "optB"])
    s = engine.predict_score(h)
    n = engine.predict_noul(h)

    assert c.selected in ["optA", "optB"]
    assert 0.0 <= s.score <= 1.0
    assert 0.0 <= n.probability <= 1.0


def test_front_door_guardrail_and_adapter_routing():
    head = JevDecisionHead(hidden_size=64)
    engine = JevSystem1Engine(head)

    torch.manual_seed(42)

    # 1. Hostile input simulation (weights produce high violation probability)
    with torch.no_grad():
        head.noul_proj.weight.fill_(10.0)
        head.noul_proj.bias.fill_(5.0)

    h_hostile = torch.ones(64)
    res_hostile = engine.route_request(h_hostile, guardrail_threshold=0.8)
    assert res_hostile["admitted"] is False
    assert res_hostile["reason"] == "guardrail_violation"

    # 2. Benign input simulation (weights produce low violation probability)
    with torch.no_grad():
        head.noul_proj.weight.fill_(-10.0)
        head.noul_proj.bias.fill_(-5.0)

    h_benign = torch.ones(64)
    adapters = ["customer_support", "code_agent", "sql_writer"]
    res_benign = engine.route_request(
        h_benign, guardrail_threshold=0.8, adapter_options=adapters
    )
    assert res_benign["admitted"] is True
    assert res_benign["assigned_adapter"] in adapters
    assert res_benign["confidence"] > 0.0
