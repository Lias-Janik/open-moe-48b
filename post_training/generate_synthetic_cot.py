#!/usr/bin/env python3
"""
open-moe-48b Synthetic Chain-of-Thought (CoT) & Reasoning Trajectory Synthesizer
================================================================================
Generates high-fidelity reasoning trajectories embedded within <think> ... </think> tags
to bootstrap verifiable test-time compute for systems engineering and distributed training.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("generate_synthetic_cot")

SYSTEM_TOPICS = [
    {
        "category": "distributed_sharding_mesh",
        "question": "How do you configure 3D ICI parallelism for a 48B MoE model on a TPU v5p-128 slice?",
        "thoughts": [
            "Identify available hardware topology: TPU v5p-128 consists of 128 physical TensorCore chips organized in a 3D torus with low-latency Inter-Chip Interconnect (ICI).",
            "Evaluate model parameters: 48B total sparse parameters, 40 layers, 8 routed experts per layer, Top-2 routing.",
            "Determine sharding axes: Need to shard data batch, expert weights, and sequence length without introducing cross-torus communication bottlenecks.",
            "Select Expert Parallelism (EP): With 8 experts, assigning 1 expert per EP mesh dimension yields ici_expert_parallelism = 8.",
            "Select Data/FSDP Parallelism: Remaining chips = 128 / 8 = 16. Setting ici_fsdp_parallelism = 16 evenly shards the optimizer states and non-expert weights across 16 ranks.",
            "Select Tensor Parallelism (TP): With 16GB HBM per chip and FSDP+EP, single-chip attention computation fits comfortably in memory. Set ici_tensor_parallelism = 1 to eliminate all-reduce overhead in intra-layer attention.",
            "Conclusion: The optimal mesh is [fsdp=16, ep=8, tp=1] with remat_policy='full'.",
        ],
        "answer": (
            "To optimally shard OpenMoE-48B across a TPU v5p-128 slice:\n"
            "1. **Expert Parallelism (EP = 8)**: Pin each of the 8 routed experts to dedicated TPU core sub-meshes, transforming expert dispatch into an efficient all-to-all collective over ICI.\n"
            "2. **Fully Sharded Data Parallel (FSDP = 16)**: Distribute attention weights, embeddings, and AdamW optimizer states across 16 FSDP ranks.\n"
            "3. **Tensor Parallelism (TP = 1)**: Maintain attention layers without intra-head sharding to avoid unnecessary all-reduce latency.\n"
            "This configuration achieves maximum Model FLOPs Utilization (MFU) on TPU v5p hardware."
        ),
    },
    {
        "category": "moe_auxiliary_loss",
        "question": "Why is an auxiliary load balancing loss necessary in Top-2 MoE routing, and how is it derived?",
        "thoughts": [
            "Analyze the failure mode of unconstrained top-k routing: The router gating network tends to gravitate toward a small subset of experts that received early gradient updates.",
            "Identify the positive feedback loop: Preferred experts get more tokens, receive more gradient updates, and become even more dominant, leading to 'expert collapse' where other experts starve.",
            "Consider computational implications: In distributed training, each TPU core computes a fixed capacity of tokens per expert. If tokens are imbalanced, cores with starved experts idle while overloaded experts drop tokens.",
            "Formulate auxiliary loss: Let f_i be the fraction of tokens routed to expert i, and P_i be the average router probability allocated to expert i over the batch.",
            "The auxiliary loss is defined as L_aux = alpha * N * sum_{i=1}^N (f_i * P_i), where alpha = 0.01.",
            "Check optimality: The minimum occurs when both f_i = 1/N and P_i = 1/N for all i, enforcing a uniform distribution of tokens across all experts.",
        ],
        "answer": (
            "In sparse MoE architectures, an auxiliary load-balancing loss prevents **expert collapse**.\n\n"
            "Without regularization, the router gating mechanism concentrates routing decisions onto 1-2 experts, leaving the remaining experts unutilized and causing significant token dropping under fixed buffer capacities.\n\n"
            "The auxiliary loss is computed across $N$ experts as:\n"
            "$$\\mathcal{L}_{\\text{aux}} = \\alpha \\cdot N \\sum_{i=1}^{N} f_i \\cdot P_i$$\n"
            "where $f_i$ is the fraction of batch tokens routed to expert $i$, $P_i$ is the mean gating probability for expert $i$, and $\\alpha = 0.01$ is the penalty weight. This loss penalizes non-uniform assignment and drives the router toward an equal distribution of compute."
        ),
    },
    {
        "category": "swiglu_activation",
        "question": "Derive the mathematical formulation of the SwiGLU activation and explain its memory footprint relative to standard GELU.",
        "thoughts": [
            "Recall definition of Gated Linear Units (GLU): A component with a linear projection multiplied element-wise by a non-linear gate.",
            "Formulate SwiGLU: SwiGLU(x) = (x * W_gate) * Swish(x * W_up), where Swish(z) = z * sigmoid(beta * z).",
            "In LLM transformer FFNs, the output is projected down: FFN_{SwiGLU}(x) = (Swish(x W_gate) \\otimes (x W_up)) W_down.",
            "Compare parameter count: Standard GELU FFN requires 2 weight matrices: W_in (d_model x d_ffn) and W_out (d_ffn x d_model), total 2 * d_model * d_ffn.",
            "SwiGLU requires 3 weight matrices: W_gate, W_up, W_down, total 3 * d_model * d_ffn.",
            "To maintain parameter parity with standard FFN (where d_ffn = 4 * d_model), SwiGLU sets d_ffn = (2/3) * 4 * d_model = 8/3 * d_model.",
            "Memory footprint: During backprop, intermediate activations of both gate and up projections must be stored or recomputed via rematerialization.",
        ],
        "answer": (
            "**SwiGLU** replaces standard non-linear activations with a bilinear gating mechanism:\n"
            "$$\\text{SwiGLU}(x) = \\left( \\text{Swish}(x W_{\\text{gate}}) \\odot (x W_{\\text{up}}) \\right) W_{\\text{down}}$$\n"
            "where $\\text{Swish}(z) = z \\cdot \\sigma(z)$.\n\n"
            "**Memory & Parameter Characteristics:**\n"
            "- Parameter count: Uses 3 projection matrices ($W_{\\text{gate}}, W_{\\text{up}}, W_{\\text{down}}$) rather than 2 in standard GELU.\n"
            "- Dimension scaling: To maintain constant parameter count, $d_{\\text{mlp}}$ is scaled to $\\approx \\frac{8}{3} d_{\\text{model}}$ (7,168 in OpenMoE-48B).\n"
            "- Activation memory: Requires retaining two tensor activations prior to the element-wise Hadamard product, effectively mitigated on TPU v5p using JAX activation rematerialization (`remat_policy='full'`)."
        ),
    },
]


def generate_cot_dataset(
    output_path: Path,
    num_samples: int = 500,
    seed: int = 42,
) -> None:
    """Generates synthetic CoT datasets formatted with open-moe-48b special tokens."""
    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    logger.info("Generating %d synthetic CoT reasoning trajectories...", num_samples)

    for i in range(num_samples):
        topic = rng.choice(SYSTEM_TOPICS)
        # Formulate full reasoning trace
        reasoning_trace = "\n".join(f"{idx+1}. {step}" for idx, step in enumerate(topic["thoughts"]))

        record = {
            "id": f"cot_sample_{i:05d}",
            "messages": [
                {
                    "role": "system",
                    "content": "You are OpenMoE-48B, a specialized distributed systems and AI infrastructure foundation model. Always reason rigorously inside <think> tags before providing precise answers.",
                },
                {
                    "role": "user",
                    "content": topic["question"],
                },
                {
                    "role": "assistant",
                    "reasoning": reasoning_trace,
                    "content": topic["answer"],
                },
            ],
            "category": topic["category"],
        }
        records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Successfully synthesized %d CoT samples -> %s", len(records), output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize high-verifiability Chain-of-Thought (CoT) datasets for OpenMoE-48B."
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/synthetic_cot.jsonl"),
        help="Target output path for JSONL dataset.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of reasoning samples to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_cot_dataset(
        output_path=args.output_file,
        num_samples=args.num_samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
