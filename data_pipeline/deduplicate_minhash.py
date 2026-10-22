#!/usr/bin/env python3
"""
open-moe-48b MinHash + LSH Near-Deduplication Engine
===================================================
Production-grade, distributed-ready MinHash Locality-Sensitive Hashing (LSH) deduplicator.
Designed to eliminate code clones, boilerplate duplicates, and near-identical documents
before tokenization and packing to save TPU compute and reduce memorization.

Algorithm:
1. Tokenize document text into 5-gram shingles.
2. Compute K MinHash signatures using 64-bit linear congruential hash functions.
3. Partition K hashes into B bands of R rows each (B * R = K).
4. Group documents with identical band hashes into candidate buckets.
5. Apply Disjoint Set Union (Union-Find) to form equivalence clusters.
6. Retain the canonical document per cluster (longest / highest quality) and discard duplicates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Set, Tuple

try:
    import xxhash

    def hash64(data: bytes) -> int:
        return xxhash.xxh64(data).intdigest()

except ImportError:
    # High-speed fallback using hashlib.sha256 if xxhash is not yet installed
    def hash64(data: bytes) -> int:
        return int.from_bytes(hashlib.sha256(data).digest()[:8], byteorder="big", signed=False)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("deduplicate_minhash")

# Large 64-bit Mersenne Prime (2^61 - 1)
MERSENNE_PRIME_61 = (1 << 61) - 1


class DisjointSet:
    """Disjoint Set Union (DSU) with path compression and rank union."""

    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.rank: dict[str, int] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
            return item
        root = item
        while root != self.parent[root]:
            root = self.parent[root]
        # Path compression
        curr = item
        while curr != root:
            nxt = self.parent[curr]
            self.parent[curr] = root
            curr = nxt
        return root

    def union(self, item1: str, item2: str) -> None:
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            if self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            elif self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1


def generate_hash_coefficients(num_perm: int, seed: int = 42) -> tuple[list[int], list[int]]:
    """Generates independent pseudo-random multiplier (a) and offset (b) pairs for MinHash."""
    import random

    rng = random.Random(seed)
    a_coeffs: list[int] = []
    b_coeffs: list[int] = []
    seen: set[int] = set()

    while len(a_coeffs) < num_perm:
        a = rng.randint(1, MERSENNE_PRIME_61 - 1)
        if a not in seen:
            seen.add(a)
            a_coeffs.append(a)

    seen.clear()
    while len(b_coeffs) < num_perm:
        b = rng.randint(0, MERSENNE_PRIME_61 - 1)
        if b not in seen:
            seen.add(b)
            b_coeffs.append(b)

    return a_coeffs, b_coeffs


def get_shingles(text: str, n_gram: int = 5) -> set[str]:
    """Extracts alphanumeric n-gram shingles normalized for whitespace."""
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < n_gram:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n_gram]) for i in range(len(tokens) - n_gram + 1)}


def compute_minhash_signature(
    shingles: set[str],
    a_coeffs: list[int],
    b_coeffs: list[int],
    num_perm: int,
) -> list[int]:
    """Computes the MinHash signature array for a set of shingles."""
    if not shingles:
        return [0] * num_perm

    shingle_hashes = [hash64(s.encode("utf-8")) & ((1 << 61) - 1) for s in shingles]
    signature: list[int] = []

    for i in range(num_perm):
        a = a_coeffs[i]
        b = b_coeffs[i]
        min_h = min(((a * h + b) % MERSENNE_PRIME_61) for h in shingle_hashes)
        signature.append(min_h)

    return signature


def run_minhash_lsh(
    input_jsonl: Path,
    output_jsonl: Path,
    num_perm: int = 128,
    bands: int = 16,
    n_gram: int = 5,
) -> tuple[int, int]:
    """Runs single-pass streaming MinHash LSH deduplication over a JSONL corpus."""
    rows_per_band = num_perm // bands
    assert (
        bands * rows_per_band == num_perm
    ), f"num_perm ({num_perm}) must be divisible by bands ({bands})."

    approx_threshold = (1.0 / bands) ** (1.0 / rows_per_band)
    logger.info(
        "MinHash Config: num_perm=%d, bands=%d, rows/band=%d (Effective Jaccard Threshold: ~%.2f)",
        num_perm,
        bands,
        rows_per_band,
        approx_threshold,
    )

    a_coeffs, b_coeffs = generate_hash_coefficients(num_perm)
    lsh_buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    doc_registry: dict[str, dict] = {}
    dsu = DisjointSet()

    logger.info("Pass 1: Reading %s, extracting shingles and computing signatures...", input_jsonl)
    total_docs = 0

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            doc_id = str(record.get("id", f"doc_{line_num}"))
            text = record.get("text", "") or record.get("content", "")
            if not text:
                continue

            doc_registry[doc_id] = {
                "record": record,
                "length": len(text),
            }
            total_docs += 1

            shingles = get_shingles(text, n_gram=n_gram)
            signature = compute_minhash_signature(shingles, a_coeffs, b_coeffs, num_perm)

            # Banding
            for band_idx in range(bands):
                start_idx = band_idx * rows_per_band
                end_idx = start_idx + rows_per_band
                band_slice = tuple(signature[start_idx:end_idx])
                bucket_key = (band_idx, hash(band_slice))

                matched_docs = lsh_buckets[bucket_key]
                if matched_docs:
                    # Near duplicate collision detected: unify with existing doc in bucket
                    dsu.union(doc_id, matched_docs[0])
                matched_docs.append(doc_id)

    logger.info("Indexed %d documents into LSH buckets.", total_docs)

    # Pass 2: Identify clusters and select canonical representative (longest document)
    clusters: dict[str, list[str]] = defaultdict(list)
    for doc_id in doc_registry:
        root = dsu.find(doc_id)
        clusters[root].append(doc_id)

    duplicate_docs_count = 0
    unique_docs: list[dict] = []

    for root, cluster_doc_ids in clusters.items():
        if len(cluster_doc_ids) > 1:
            duplicate_docs_count += len(cluster_doc_ids) - 1
            # Sort descending by text length; choose longest as canonical
            cluster_doc_ids.sort(key=lambda d: doc_registry[d]["length"], reverse=True)

        canonical_id = cluster_doc_ids[0]
        unique_docs.append(doc_registry[canonical_id]["record"])

    # Pass 3: Write deduplicated output
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(output_jsonl, "w", encoding="utf-8") as out_f:
        for rec in unique_docs:
            out_f.write(json.dumps(rec) + "\n")

    kept_docs_count = len(unique_docs)
    logger.info("=== Deduplication Complete ===")
    logger.info("  Total Input Documents:   %d", total_docs)
    logger.info("  Canonical Kept:          %d (%.2f%%)", kept_docs_count, 100.0 * kept_docs_count / max(total_docs, 1))
    logger.info("  Duplicates Removed:      %d (%.2f%%)", duplicate_docs_count, 100.0 * duplicate_docs_count / max(total_docs, 1))
    logger.info("  Deduplicated File:       %s", output_jsonl)

    return kept_docs_count, duplicate_docs_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Near-Deduplication Engine via MinHash and LSH for open-moe-48b."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Input JSONL file containing documents with 'text' or 'content' fields.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data_pipeline_output/deduplicated.jsonl"),
        help="Output JSONL path for deduplicated documents.",
    )
    parser.add_argument(
        "--num-perm",
        type=int,
        default=128,
        help="Number of MinHash permutations (default: 128).",
    )
    parser.add_argument(
        "--bands",
        type=int,
        default=16,
        help="Number of LSH bands (default: 16, resulting in 8 rows per band).",
    )
    parser.add_argument(
        "--n-gram",
        type=int,
        default=5,
        help="Shingle word n-gram size (default: 5).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_file.exists():
        logger.error("Input file %s not found.", args.input_file)
        sys.exit(1)

    run_minhash_lsh(
        input_jsonl=args.input_file,
        output_jsonl=args.output_file,
        num_perm=args.num_perm,
        bands=args.bands,
        n_gram=args.n_gram,
    )


if __name__ == "__main__":
    main()
