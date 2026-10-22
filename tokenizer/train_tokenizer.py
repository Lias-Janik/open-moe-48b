"""
tokenizer/train_tokenizer.py
Trains a production-grade 128k Byte-Level BPE tokenizer on raw or JSONL text.
"""
import argparse
import json
import os
from typing import Iterator
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

VOCAB_SIZE = 128_000
SPECIAL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<untrusted_content>",
    "</untrusted_content>",
]

def batch_iterator(file_path: str, batch_size: int = 1000) -> Iterator[list[str]]:
    batch = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{") and line.endswith("}"):
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                except json.JSONDecodeError:
                    text = line
            else:
                text = line

            if text:
                batch.append(text)

            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch

def build_tokenizer() -> tuple[Tokenizer, trainers.BpeTrainer]:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    return tokenizer, trainer

def main():
    parser = argparse.ArgumentParser(description="Train 128k BPE Tokenizer")
    parser.add_argument("--input-file", type=str, default="data/raw_pilot/pilot_documents.jsonl", help="Path to JSONL/txt data")
    parser.add_argument("--output-dir", type=str, default="tokenizer_output", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer, trainer = build_tokenizer()

    print(f"Training 128k BPE tokenizer from: {args.input_file}")
    tokenizer.train_from_iterator(batch_iterator(args.input_file), trainer=trainer)

    out_path = os.path.join(args.output_dir, "tokenizer.json")
    tokenizer.save(out_path)
    print(f"Tokenizer saved successfully to: {out_path}")

if __name__ == "__main__":
    main()
