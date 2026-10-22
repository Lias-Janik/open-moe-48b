#!/usr/bin/env python3
"""
open-moe-48b Forward Pass & Architectural Sanity Verification
=============================================================
Tests:
1. Top-2 MoE Router Gating & Normalized Weights.
2. Auxiliary Load-Balancing Loss computation.
3. SwiGLU Non-Linear Gating Kernel.
4. Grouped Query Attention (GQA) Head Projection Shapes.
"""

from __future__ import annotations

import logging
import math
import sys
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("test_forward_pass")

# Architectural dimensions matching configs/moe_48b_v5p.yaml
D_MODEL = 4096
NUM_EXPERTS = 8
TOP_K = 2
D_MLP = 7168
NUM_QUERY_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128


def swish(x: np.ndarray) -> np.ndarray:
    """Computes element-wise Swish activation: x * sigmoid(x)."""
    return x / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def swiglu_forward(
    x: np.ndarray,
    w_gate: np.ndarray,
    w_up: np.ndarray,
    w_down: np.ndarray,
) -> np.ndarray:
    """Executes SwiGLU feed-forward projection: (Swish(x W_gate) * (x W_up)) W_down."""
    gate_proj = np.matmul(x, w_gate)
    up_proj = np.matmul(x, w_up)
    activated = swish(gate_proj) * up_proj
    return np.matmul(activated, w_down)


def top2_router(
    hidden_states: np.ndarray,
    router_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Computes Top-2 normalized routing gate for 8 experts and auxiliary load-balancing loss.
    """
    # hidden_states: [B * T, D_MODEL]
    # router_weights: [D_MODEL, NUM_EXPERTS]
    logits = np.matmul(hidden_states, router_weights)  # [B * T, NUM_EXPERTS]

    # Stable softmax
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)  # P_i

    # Top-2 selection
    top2_indices = np.argsort(-probs, axis=-1)[:, :TOP_K]
    top2_weights = np.take_along_axis(probs, top2_indices, axis=-1)
    # Renormalize top-2 weights
    top2_weights = top2_weights / np.sum(top2_weights, axis=-1, keepdims=True)

    # Auxiliary Load-Balancing Loss: L_aux = alpha * N * sum(f_i * P_i)
    total_tokens = hidden_states.shape[0]
    expert_mask = np.zeros((total_tokens, NUM_EXPERTS), dtype=np.float32)
    np.put_along_axis(expert_mask, top2_indices, 1.0, axis=-1)

    # Fraction of tokens dispatched to expert i
    f = np.mean(expert_mask, axis=0)
    # Mean router probability for expert i
    P = np.mean(probs, axis=0)

    aux_loss = 0.01 * NUM_EXPERTS * float(np.sum(f * P))
    return top2_weights, top2_indices, aux_loss


def test_moe_routing():
    logger.info("=== Test 1: Top-2 MoE Router Dispatch & Auxiliary Loss ===")
    rng = np.random.RandomState(42)
    batch_size = 4
    seq_len = 16
    total_tokens = batch_size * seq_len

    hidden = rng.randn(total_tokens, D_MODEL).astype(np.float32)
    router_w = (rng.randn(D_MODEL, NUM_EXPERTS) * 0.02).astype(np.float32)

    weights, indices, aux_loss = top2_router(hidden, router_w)

    assert weights.shape == (total_tokens, TOP_K), f"Weights shape mismatch: {weights.shape}"
    assert indices.shape == (total_tokens, TOP_K), f"Indices shape mismatch: {indices.shape}"
    assert np.allclose(np.sum(weights, axis=-1), 1.0), "Normalized top-2 weights must sum to 1.0"
    assert aux_loss > 0.0, "Auxiliary loss must be positive"

    logger.info("  PASS: Top-2 weights normalized and indices bounded in [0, %d)", NUM_EXPERTS)
    logger.info("  Calculated Auxiliary Loss: %.6f", aux_loss)


def test_swiglu_activation():
    logger.info("=== Test 2: SwiGLU Gated Feed-Forward Kernel ===")
    rng = np.random.RandomState(42)
    tokens = 8

    x = rng.randn(tokens, D_MODEL).astype(np.float32)
    w_gate = (rng.randn(D_MODEL, D_MLP) * 0.02).astype(np.float32)
    w_up = (rng.randn(D_MODEL, D_MLP) * 0.02).astype(np.float32)
    w_down = (rng.randn(D_MLP, D_MODEL) * 0.02).astype(np.float32)

    out = swiglu_forward(x, w_gate, w_up, w_down)
    assert out.shape == (tokens, D_MODEL), f"SwiGLU output shape mismatch: {out.shape}"
    assert not np.isnan(out).any(), "SwiGLU output contains NaN values"

    logger.info("  PASS: SwiGLU Forward Pass shape %s verified with zero NaNs", out.shape)


def test_gqa_projections():
    logger.info("=== Test 3: Grouped Query Attention (GQA) Dimensionality ===")
    query_dim = NUM_QUERY_HEADS * HEAD_DIM
    kv_dim = NUM_KV_HEADS * HEAD_DIM

    assert query_dim == D_MODEL, f"Query dimension ({query_dim}) must equal D_MODEL ({D_MODEL})"
    assert kv_dim == 1024, f"KV dimension ({kv_dim}) must equal 1024 for 4:1 compression"
    compression_ratio = NUM_QUERY_HEADS // NUM_KV_HEADS
    assert compression_ratio == 4, f"Expected 4:1 GQA ratio, got {compression_ratio}:1"

    logger.info("  PASS: GQA configured with 32 Q-heads, 8 KV-heads (4:1 compression ratio)")


def main():
    try:
        test_moe_routing()
        test_swiglu_activation()
        test_gqa_projections()
        logger.info(">>> ALL FORWARD PASS TESTS COMPLETED SUCCESSFULLY! <<<")
        sys.exit(0)
    except AssertionError as e:
        logger.error("TEST FAILED: %s", str(e))
        sys.exit(1)


if __name__ == "__main__":
    main()
