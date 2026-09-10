"""Fill a Chroma collection with the chunks and their ALREADY COMPUTED
embeddings.

We supply the vectors ourselves - see the note on keeping embedding an
interchangeable function - so Chroma never learns which model or provider
produced them.

CARE when running this repeatedly over the same --db-dir/--collection:
  - The collection is filled through upsert(), not add(). add() on an
    existing ID silently does NOTHING: no error, but nothing is
    overwritten either (verified). After regenerating the corpus you
    would otherwise keep getting the old results even though the chunks
    and embeddings on disk are already new. upsert() overwrites an
    existing ID correctly.
  - upsert() does not, however, cope with a change in the STRUCTURE of
    the chunks: different article boundaries mean a different number of
    chunks per article, so some old chunk_ids are never generated again.
    Such orphaned records would hang around in the collection forever
    and keep coming back in results. Use --overwrite in that case, so
    the collection is dropped and rebuilt from scratch.

Usage:
    python -m magazine_rag.build_chroma \\
        --chunks output/chunks.jsonl \\
        --vectors output/ziva_embeddings__intfloat__multilingual-e5-base.npy \\
        --ids output/ziva_embeddings__intfloat__multilingual-e5-base_ids.json \\
        --db-dir ./chroma_db --collection ziva --overwrite
"""
import argparse
import json
from pathlib import Path

import chromadb
import numpy as np

from magazine_rag.console import setup_console

BATCH_SIZE = 4000  # Chroma caps the size of a single add()/upsert() call


def load_chunks(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def sanitize_metadata(chunk: dict) -> dict:
    """Chroma accepts only str/int/float/bool in metadata - no None, no
    nested structures - so everything is coerced to a safe type here.

    year and issue are deliberately converted to int (the source data
    keeps them as strings) so that they can later be filtered
    numerically, e.g. {"year": {"$gt": 2020}}.
    """
    return {
        "chunk_type": chunk["chunk_type"],
        "chunk_index": chunk["chunk_index"],
        "year": int(chunk["year"]),
        "issue": int(chunk["issue"]),
        "article_id": chunk["article_id"],
        "title": chunk["title"] or "",
        "author": chunk["author"] or "",
        "page_start": chunk["page_start"],
        "page_end": chunk["page_end"],
    }


def main():
    setup_console()
    ap = argparse.ArgumentParser(
        description="load chunks and their vectors into Chroma")
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--db-dir", required=True,
                    help="where Chroma stores its data on disk")
    ap.add_argument("--collection", default="ziva")
    ap.add_argument("--overwrite", action="store_true",
                    help="drop the existing collection and recreate it from "
                         "scratch (use when the structure of the chunks or "
                         "articles changed, not just their content)")
    args = ap.parse_args()

    chunks = load_chunks(args.chunks)
    vectors = np.load(args.vectors)
    ids = json.loads(Path(args.ids).read_text(encoding="utf-8"))

    assert len(chunks) == vectors.shape[0] == len(ids), (
        f"counts disagree: chunks={len(chunks)} vectors={vectors.shape[0]} "
        f"ids={len(ids)}")
    assert all(c["chunk_id"] == i for c, i in zip(chunks, ids)), \
        "the order of IDs in chunks and in the vectors/ids file does not match"

    client = chromadb.PersistentClient(path=args.db_dir)

    existing = [c.name for c in client.list_collections()]
    if args.collection in existing and args.overwrite:
        client.delete_collection(args.collection)
        print(f"Dropped the old collection {args.collection!r} (--overwrite).")
        existing.remove(args.collection)

    if args.collection in existing:
        coll = client.get_collection(args.collection)
        print(f"Collection {args.collection!r} already exists ({coll.count()} "
              f"items) - chunks will be added or overwritten by ID through "
              f"upsert(). But if the STRUCTURE of the chunks has changed "
              f"since last time (different article boundaries), this will "
              f"not clear the orphaned old records - rerun with --overwrite.")
    else:
        coll = client.create_collection(
            name=args.collection,
            # Our vectors are already normalised, but this makes the
            # returned "distance" directly readable as cosine distance
            # (1 - cosine similarity) rather than L2.
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Created a new collection {args.collection!r}.")

    n = len(chunks)
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        coll.upsert(
            ids=[c["chunk_id"] for c in chunks[start:end]],
            embeddings=vectors[start:end].tolist(),
            documents=[c["text"] for c in chunks[start:end]],
            metadatas=[sanitize_metadata(c) for c in chunks[start:end]],
        )
        print(f"  upserted {end}/{n}")

    print(f"\nDone. Collection {args.collection!r} holds {coll.count()} "
          f"items (stored in {args.db_dir}).")


if __name__ == "__main__":
    main()
