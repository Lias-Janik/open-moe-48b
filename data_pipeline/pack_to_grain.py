"""
data_pipeline/pack_to_grain.py
Packs tokenized text into fixed-length binary sequence shards (int32)
optimized for zero-copy memory-mapped loading via Google Grain and MaxText.
"""
import argparse
import json
import os
import sys
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Pack tokenized documents into binary Grain shards.")
    parser.add_argument("--input-file", type=str, required=True, help="Path to input JSONL documents")
    parser.add_argument("--tokenizer-path", type=str, required=True, help="Path to trained tokenizer.json")
    parser.add_argument("--output-dir", type=str, default="data/packed_shards", help="Output directory for shards")
    parser.add_argument("--seq-length", type=int, default=8192, help="Context length per sequence (default: 8192)")
    parser.add_argument("--max-seqs-per-shard", type=int, default=10000, help="Maximum 8k sequences per shard file")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.input_file):
        print(f"[ERROR] Input file not found: {args.input_file}")
        sys.exit(1)

    if not os.path.exists(args.tokenizer_path):
        print(f"[ERROR] Tokenizer not found: {args.tokenizer_path}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    eos_token_id = tokenizer.token_to_id("<|endoftext|>")
    if eos_token_id is None:
        eos_token_id = 0

    print(f"Target sequence length: {args.seq_length:,} tokens")
    print(f"Packing from:           {args.input_file}")
    print(f"Output directory:       {args.output_dir}\n")

    token_buffer = []
    shard_idx = 0
    total_packed_sequences = 0
    total_tokens_processed = 0

    shard_tokens = []

    def flush_shard(tokens_list, idx):
        if not tokens_list:
            return
        bin_path = os.path.join(args.output_dir, f"shard_{idx:05d}.bin")
        idx_path = os.path.join(args.output_dir, f"shard_{idx:05d}.idx")

        # int32 accommodates full 128k vocabulary without overflow
        arr = np.array(tokens_list, dtype=np.int32)
        arr.tofile(bin_path)

        num_seqs = len(arr) // args.seq_length
        metadata = {
            "shard_index": idx,
            "seq_length": args.seq_length,
            "num_sequences": num_seqs,
            "total_tokens": int(len(arr)),
            "dtype": "int32",
            "bytes_per_token": 4,
        }
        with open(idx_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        file_size_mb = os.path.getsize(bin_path) / (1024 * 1024)
        print(f"\n[SHARD WRITTEN] {bin_path} | {num_seqs:,} sequences ({len(arr):,} tokens) | {file_size_mb:.2f} MB")

    print("Reading input documents...")
    batch_texts = []
    batch_size = 2000

    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Ingesting & Packing"):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                text = doc.get("text", "")
            except json.JSONDecodeError:
                text = line

            if text:
                batch_texts.append(text)

            if len(batch_texts) >= batch_size:
                encodings = tokenizer.encode_batch(batch_texts)
                for enc in encodings:
                    ids = enc.ids
                    if ids:
                        token_buffer.extend(ids)
                        token_buffer.append(eos_token_id)
                        total_tokens_processed += len(ids) + 1
                batch_texts = []

                while len(token_buffer) >= args.seq_length:
                    seq = token_buffer[:args.seq_length]
                    token_buffer = token_buffer[args.seq_length:]
                    shard_tokens.extend(seq)
                    total_packed_sequences += 1

                    if len(shard_tokens) >= args.max_seqs_per_shard * args.seq_length:
                        flush_shard(shard_tokens, shard_idx)
                        shard_idx += 1
                        shard_tokens = []

    if batch_texts:
        encodings = tokenizer.encode_batch(batch_texts)
        for enc in encodings:
            ids = enc.ids
            if ids:
                token_buffer.extend(ids)
                token_buffer.append(eos_token_id)
                total_tokens_processed += len(ids) + 1

    while len(token_buffer) >= args.seq_length:
        seq = token_buffer[:args.seq_length]
        token_buffer = token_buffer[args.seq_length:]
        shard_tokens.extend(seq)
        total_packed_sequences += 1

        if len(shard_tokens) >= args.max_seqs_per_shard * args.seq_length:
            flush_shard(shard_tokens, shard_idx)
            shard_idx += 1
            shard_tokens = []

    if shard_tokens:
        flush_shard(shard_tokens, shard_idx)
        shard_idx += 1

    print("\n==========================================")
    print(" PACKING PIPELINE COMPLETE ")
    print("==========================================")
    print(f"Total Raw Tokens Processed: {total_tokens_processed:,}")
    print(f"Total 8,192 Sequences:      {total_packed_sequences:,}")
    print(f"Total Packed Shards:        {shard_idx}")
    print(f"Unpacked Tail Tokens:       {len(token_buffer)} (dropped/padded)")


if __name__ == "__main__":
    main()
