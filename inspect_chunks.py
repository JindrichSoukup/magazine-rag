"""
Rychlá kontrola ziva_embedding_chunks.jsonl - spustit po build_chunks.py.

    python inspect_chunks.py ziva_embedding_chunks.jsonl
"""
import json
import random
import statistics
import sys
from collections import Counter


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def length_histogram(lengths, bucket=200, max_bucket=3000):
    buckets = Counter()
    for L in lengths:
        b = min((L // bucket) * bucket, max_bucket)
        buckets[b] += 1
    for b in sorted(buckets):
        label = f"{b}-{b+bucket}" if b < max_bucket else f"{max_bucket}+"
        bar = "#" * (buckets[b] // max(1, len(lengths) // 200) or 1)
        print(f"  {label:>10}: {buckets[b]:>6}  {bar}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "ziva_embedding_chunks.jsonl"
    chunks = load(path)
    print(f"Načteno {len(chunks)} chunků z {path}\n")

    # --- 1) unikátnost ID a prázdné texty ---------------------------------
    ids = [c["chunk_id"] for c in chunks]
    dupes = [cid for cid, n in Counter(ids).items() if n > 1]
    empties = [c["chunk_id"] for c in chunks if not c["text"].strip()]
    print(f"Duplicitní chunk_id: {len(dupes)}" + (f" (ukázka: {dupes[:5]})" if dupes else ""))
    print(f"Prázdné texty: {len(empties)}" + (f" (ukázka: {empties[:5]})" if empties else ""))
    print()

    # --- 2) délky textu podle typu -----------------------------------------
    for chunk_type in ("body", "caption"):
        lens = [len(c["text"]) for c in chunks if c["chunk_type"] == chunk_type]
        if not lens:
            continue
        print(f"=== délky '{chunk_type}' (n={len(lens)}) ===")
        print(f"  min={min(lens)} max={max(lens)} "
              f"průměr={statistics.mean(lens):.0f} medián={statistics.median(lens):.0f}")
        length_histogram(lens)
        print()

    # --- 3) rozložení podle roku/čísla -------------------------------------
    by_year = Counter(c["year"] for c in chunks)
    print("=== chunků podle ročníku ===")
    for year in sorted(by_year):
        print(f"  {year}: {by_year[year]}")
    print()

    # --- 4) pár náhodných embedding_text ukázek ----------------------------
    print("=== náhodné ukázky embedding_text (3x body, 2x caption) ===\n")
    random.seed(42)
    body = [c for c in chunks if c["chunk_type"] == "body"]
    captions = [c for c in chunks if c["chunk_type"] == "caption"]
    for c in random.sample(body, min(3, len(body))):
        print("-" * 70)
        print(c["embedding_text"][:500])
        print("...")
    for c in random.sample(captions, min(2, len(captions))):
        print("-" * 70)
        print(c["embedding_text"][:500])
        print("...")


if __name__ == "__main__":
    main()