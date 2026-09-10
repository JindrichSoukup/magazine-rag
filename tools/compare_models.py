"""Compare candidate embedding models on a handful of questions whose
right answer you KNOW - a specific article_id, or at least a substring of
the title that ought to show up among the top-k results.

How it works: for each model, embed every chunk plus the test queries,
find the top-k nearest chunks (plain numpy; no Chroma or FAISS needed for
a comparison this small) and check whether at least one chunk from the
expected article is among them.

Edit TEST_QUERIES below for your own archive. Five to ten are enough, but
you have to know yourself which article should come out as the answer -
which is precisely why this cannot be automated or downloaded from
anywhere.

Usage:
    python -m tools.compare_models --input output/chunks.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from magrag.console import setup_console
from magrag.embed import embed

# Adapt these to your own archive and your knowledge of its content.
# "expect_title_substring" is a substring that should appear in the title
# of the article whose chunk you expect among the top-k results.
#
# The queries below are in Czech because the corpus they were written for
# is Czech: a retrieval test has to be in the language of the documents.
# They are test data, not prose to translate.
TEST_QUERIES = [
    {"query": "Co je mixotrofie u rostlin?",
     "expect_title_substring": "houbami"},
    {"query": "Jak vzniká Huntingtonova choroba?",
     "expect_title_substring": "Huntingtonova choroba"},
    {"query": "Jaké zvířecí modely se používají pro výzkum neurodegenerativních chorob?",
     "expect_title_substring": "Huntingtonova choroba"},
    {"query": "Proč se ve městech vyskytuje víc druhů rostlin než ve volné krajině?",
     "expect_title_substring": "velkoměst"},
    {"query": "Jak fungují mravenci jako legionářský druh?",
     "expect_title_substring": "fylogeneze a evoluce mravenců"},
]

CANDIDATE_MODELS = [
    "intfloat/multilingual-e5-small",
    "intfloat/multilingual-e5-base",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    # "BAAI/bge-m3",  # large model, slow download - uncomment when wanted
]

TOP_K = 5


def load_chunks(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate_model(model_name: str, chunks, chunk_vectors):
    hits = 0
    rows = []
    for tq in TEST_QUERIES:
        q_vec = embed([tq["query"]], model_name=model_name, is_query=True,
                      show_progress=False)[0]
        scores = chunk_vectors @ q_vec  # normalised vectors -> cosine similarity
        top_idx = np.argsort(-scores)[:TOP_K]

        found = any(tq["expect_title_substring"].lower() in chunks[i]["title"].lower()
                    for i in top_idx)
        hits += found
        rows.append({
            "query": tq["query"],
            "found": found,
            "top_titles": [chunks[i]["title"] for i in top_idx],
        })
    return hits, rows


def main():
    setup_console()
    ap = argparse.ArgumentParser(
        description="compare candidate embedding models on known-answer queries")
    ap.add_argument("--input", required=True, help="chunks.jsonl")
    ap.add_argument("--verbose", action="store_true",
                    help="also print the top-k results for each query and model")
    args = ap.parse_args()

    chunks = load_chunks(Path(args.input))
    print(f"Loaded {len(chunks)} chunks, {len(TEST_QUERIES)} test queries, "
          f"{len(CANDIDATE_MODELS)} models to compare.\n")

    texts = [c["embedding_text"] for c in chunks]
    results = {}

    for model_name in CANDIDATE_MODELS:
        print(f"=== {model_name} ===")
        t0 = time.time()
        chunk_vectors = embed(texts, model_name=model_name, is_query=False,
                              show_progress=False)
        hits, rows = evaluate_model(model_name, chunks, chunk_vectors)
        dt = time.time() - t0
        results[model_name] = hits
        print(f"  hit {hits}/{len(TEST_QUERIES)} queries ({dt:.1f}s for "
              f"{len(chunks)} chunks)")
        if args.verbose:
            for r in rows:
                mark = "OK  " if r["found"] else "MISS"
                print(f"    {mark} {r['query']!r}")
                for t in r["top_titles"]:
                    print(f"        - {t[:60]}")
        print()

    print("=== Summary ===")
    for model_name, hits in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {hits}/{len(TEST_QUERIES)}  {model_name}")


if __name__ == "__main__":
    main()
