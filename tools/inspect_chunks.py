"""A quick sanity check of chunks.jsonl - run it after build_chunks.py.

    python -m tools.inspect_chunks output/chunks.jsonl
"""
import json
import random
import statistics
import sys
from collections import Counter

from magrag.console import setup_console


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
    setup_console()
    path = sys.argv[1] if len(sys.argv) > 1 else "output/chunks.jsonl"
    chunks = load(path)
    print(f"Loaded {len(chunks)} chunks from {path}\n")

    # --- 1) unique IDs and empty texts ------------------------------------
    ids = [c["chunk_id"] for c in chunks]
    dupes = [cid for cid, n in Counter(ids).items() if n > 1]
    empties = [c["chunk_id"] for c in chunks if not c["text"].strip()]
    print(f"Duplicate chunk_ids: {len(dupes)}"
          + (f" (sample: {dupes[:5]})" if dupes else ""))
    print(f"Empty texts: {len(empties)}"
          + (f" (sample: {empties[:5]})" if empties else ""))
    print()

    # --- 2) text lengths by type ------------------------------------------
    for chunk_type in ("body", "caption"):
        lens = [len(c["text"]) for c in chunks if c["chunk_type"] == chunk_type]
        if not lens:
            continue
        print(f"=== {chunk_type!r} lengths (n={len(lens)}) ===")
        print(f"  min={min(lens)} max={max(lens)} "
              f"mean={statistics.mean(lens):.0f} "
              f"median={statistics.median(lens):.0f}")
        length_histogram(lens)
        print()

    # --- 3) distribution by year ------------------------------------------
    by_year = Counter(c["year"] for c in chunks)
    print("=== chunks per year ===")
    for year in sorted(by_year):
        print(f"  {year}: {by_year[year]}")
    print()

    # --- 4) a few random embedding_text samples ---------------------------
    print("=== random embedding_text samples (3 body, 2 caption) ===\n")
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
