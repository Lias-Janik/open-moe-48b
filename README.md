# OpenMoE-48B: Engineering Control Plane & Distributed Pre-training

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Framework: JAX / MaxText](https://img.shields.io/badge/Framework-JAX%20%2F%20MaxText-red.svg)](https://github.com/google/maxtext)
[![Data Engine: Google Grain](https://img.shields.io/badge/Ingestion-Google%20Grain-green.svg)](https://github.com/google/grain)
[![Hardware Target: Cloud TPU v5p/v4](https://img.shields.io/badge/Hardware-Google%20Cloud%20TPU-orange.svg)](https://cloud.google.com/tpu)

OpenMoE-48B is a reproduction-ready, open-source foundation model engineering control plane designed for high-efficiency distributed pre-training and alignment on Google Cloud TPU Pods (v4/v5p).

The architecture implements a 48B Sparse Mixture-of-Experts (MoE) routing topology with ~6.2B active parameters per token, specifically engineered for systems engineering, rigorous reasoning, and prompt-injection-resistant tool execution.

---

## Architectural Specifications

| Parameter | Specification | Mathematical / Implementation Detail |
| :--- | :--- | :--- |
| Total Parameters | ~48.0 Billion | 40 layers, d_model = 4096, 8 SwiGLU experts (d_mlp = 7168) |
| Active Parameters | ~6.2 Billion / token | Top-2 routing (2/8 experts) + GQA dense attention + embeddings |
| Context Length | 8,192 tokens | Long-context pre-training with RoPE base frequency theta = 500,000.0 |
| Attention Mechanism | Grouped Query Attention (GQA) | 32 Query Heads, 8 Key/Value Heads (4:1 compression ratio) |
| Vocabulary Size | 128,000 | Custom Byte-Level BPE preserving multi-space indentations |
| Routing Mechanism | Top-2 Load-Balanced Gating | Auxiliary router loss (alpha = 0.01) preventing expert collapse |
| Precision | bfloat16 Native | Mixed-precision matrix multiplications via TPU TensorCores |
| Distributed Mesh | 3D ICI Parallelism | ICI_FSDP = 16, ICI_EP = 8, ICI_TP = 1 |

---

## Empirical Pipeline Benchmarks (Validated Pilot Shard)

The end-to-end data pipeline was empirically benchmarked on a verified sample corpus from FineWeb-Edu (50,000 documents):

- Ingestion Throughput: ~1,142 documents/s (~1.15 million tokens/s sustained on local workstation).
- Pilot Shard Output: shard_00000.bin (6,024 full sequences packed at 8,192 context length).
- Total Packed Tokens: 49,348,608 discrete tokens.
- Storage Representation: Memory-mapped native int32 (4 bytes/token), eliminating runtime HBM type-casting overhead during TPU XLA execution.

---

## Repository Structure

open-moe-48b/
├── configs/
│   └── moe_48b_v5p.yaml          # MaxText YAML config (v5p mesh, MoE sharding, Orbax)
├── tokenizer/
│   ├── train_tokenizer.py        # 128k Byte-Level BPE trainer with code whitespace preservation
│   └── test_tokenizer.py         # Round-trip, discrete special token, and BPT benchmark suite
├── data_pipeline/
│   ├── extract_pilot_data.py     # HTTP streaming extraction for FineWeb-Edu / Stack v2
│   ├── filter_ast.py             # Parallel AST syntax verification & noise heuristics
│   ├── deduplicate_minhash.py    # 64-bit MinHash LSH near-deduplication engine
│   └── pack_to_grain.py          # int32 packing, zero-dump streaming & Grain DataSource
├── post_training/
│   ├── filter_injections.py      # Untrusted payload sanitation and formatting
│   ├── generate_synthetic_cot.py # Synthetic reasoning trajectory and thought trace synthesis
│   ├── format_dataset.py         # Supervised loss masking for instruction datasets
│   └── injection_guard_dpo.py    # Direct Preference Optimization generator for tool isolation
├── infra/
│   ├── setup_tpu.sh              # Multi-host TPU VM bootstrapping and JAX/libtpu setup
│   └── run_training.sh           # Resilient pre-training daemon with Orbax preemption recovery
├── requirements.txt              # Pinned production dependencies
└── README.md                     # Engineering runbook and architecture specification

---

## Discrete Control Tokens & Safety Isolation

The custom 128k BPE vocabulary includes dedicated control tokens for verifiable reasoning traces and strict prompt-injection isolation:

<|endoftext|>          (ID 0)  - Document & sequence boundary delimiter
<|im_start|>           (ID 1)  - ChatML turn start
<|im_end|>             (ID 2)  - ChatML turn termination
<think>                (ID 3)  - Start of test-time compute & reasoning trace
</think>               (ID 4)  - End of reasoning trace
<tool_call>            (ID 5)  - Structured tool call dispatch
</tool_call>           (ID 6)  - Tool call dispatch termination
<tool_response>        (ID 7)  - Raw environment/sandbox execution return
</tool_response>       (ID 8)  - Execution return termination
<untrusted_content>    (ID 9)  - Sandbox boundary for web search & untrusted data
</untrusted_content>   (ID 10) - Untrusted sandbox termination

---

## Quickstart & Reproduction

### 1. Local Environment Setup

git clone https://github.com/Lias-Janik/open-moe-48b.git
cd open-moe-48b

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

### 2. Verify Tokenizer Test Suite

python tokenizer/test_tokenizer.py

### 3. Extract & Pack Data Shards

python data_pipeline/extract_pilot_data.py

python data_pipeline/pack_to_grain.py --input-file data/raw_pilot/pilot_documents.jsonl --tokenizer-path tokenizer_output/tokenizer.json --output-dir data/packed_shards --seq-length 8192

---

## Distributed TPU Pod Launch

To execute on a Cloud TPU v5p slice:

bash infra/setup_tpu.sh

bash infra/run_training.sh --config configs/moe_48b_v5p.yaml --data-bucket gs://<YOUR_GCS_BUCKET>/packed_shards --checkpoint-bucket gs://<YOUR_GCS_BUCKET>/checkpoints

---

## License & Attribution

This project is licensed under the Apache 2.0 License. Distributed pre-training infrastructure and scaling experiments are conducted in coordination with the Google TPU Research Cloud (TRC) program.
