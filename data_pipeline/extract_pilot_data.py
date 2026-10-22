"""
data_pipeline/extract_pilot_data.py
Streamt eine reale Stichprobe aus FineWeb-Edu ohne lokalen Download-Overhead.
"""
import json
import os
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_DIR = "data/raw_pilot"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "pilot_documents.jsonl")
NUM_DOCS = 50_000  # ca. 50k Web-/Bildungstexte (~100-150 Mio. Tokens)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Streamt {NUM_DOCS} Dokumente aus FineWeb-Edu (sample-10BT)...")

    # streaming=True liest direkt den HTTP-Stream zeilenweise
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True
    )

    count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in tqdm(dataset, total=NUM_DOCS, desc="Dokumente sichern"):
            record = {
                "id": f"doc_{count}",
                "text": item["text"]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if count >= NUM_DOCS:
                break

    print(f"\nFertig! {count} Dokumente gespeichert in: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
