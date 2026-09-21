"""Interactive querying of a Chroma collection.

The model and the collection are loaded ONCE at start-up, then you ask
repeatedly inside the same process - no reloading the model for every
question, which is what running the script per query would cost.

Usage:
    python -m tools.query_chroma --db-dir ./chroma_db --collection ziva \\
        --model intfloat/multilingual-e5-base

    # then keep asking at the prompt until "exit", "quit" or an empty line

A one-off query straight from the command line, without entering the
loop, still goes through --query, for quick scripting and testing:
    python -m tools.query_chroma --db-dir ./chroma_db --collection ziva \\
        --model intfloat/multilingual-e5-base --query "..."
"""
import argparse

import chromadb

from magazine_rag.console import setup_console
from magazine_rag.embed import embed


def run_query(coll, model_name, query_text, top_k, year=None, chunk_type=None):
    query_vector = embed([query_text], model_name=model_name, is_query=True,
                         show_progress=False)[0]

    conditions = []
    if year is not None:
        conditions.append({"year": year})
    if chunk_type is not None:
        conditions.append({"chunk_type": chunk_type})
    where = None
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    result = coll.query(query_embeddings=[query_vector.tolist()],
                        n_results=top_k, where=where)

    print(f"\nQuery: {query_text!r}")
    if where:
        print(f"Filter: {where}")
    print()

    if not result["ids"][0]:
        print("(no results)\n")
        return

    for i, (doc, meta, dist, cid) in enumerate(zip(
            result["documents"][0], result["metadatas"][0],
            result["distances"][0], result["ids"][0]), 1):
        # No year for a magazine that numbers its issues continuously
        issue = (f"{meta['year']}/{meta['issue']}" if "year" in meta
                 else meta["issue"])
        print(f"{i}. [cosine distance {dist:.3f}, lower = more similar] "
              f"{meta['title']} ({issue}, "
              f"PDF pp. {meta['page_start']}-{meta['page_end']})")
        print(f"   authors: {meta['author'] or 'unknown'} | "
              f"type: {meta['chunk_type']} | id: {cid}")
        print(f"   {doc[:200]}...")
        print()


def main():
    setup_console()
    ap = argparse.ArgumentParser(
        description="query a Chroma collection interactively")
    ap.add_argument("--db-dir", required=True)
    ap.add_argument("--collection", default="ziva")
    ap.add_argument("--model", required=True,
                    help="MUST be the same model the chunk embeddings were "
                         "computed with")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--year", type=int, default=None,
                    help="optional metadata filter - this year only")
    ap.add_argument("--chunk-type", choices=["body", "caption"], default=None,
                    help="optional filter - article bodies only, or figure "
                         "captions only")
    ap.add_argument("--query", default=None,
                    help="a one-off query; without it an interactive loop runs")
    args = ap.parse_args()

    print("Connecting to the collection and loading the model (once) ...")
    client = chromadb.PersistentClient(path=args.db_dir)
    coll = client.get_collection(args.collection)
    # Warm the model up right away, so the first query in the interactive
    # loop does not pay the loading cost the later ones avoid.
    embed(["warm-up query"], model_name=args.model, is_query=True,
          show_progress=False)
    print(f"Ready ({coll.count()} items in collection "
          f"{args.collection!r}).\n")

    if args.query is not None:
        run_query(coll, args.model, args.query, args.top_k, args.year,
                  args.chunk_type)
        return

    print("Enter a query (an empty line, 'exit' or 'quit' ends the session):")
    while True:
        try:
            query_text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query_text or query_text.lower() in ("exit", "quit"):
            break
        run_query(coll, args.model, query_text, args.top_k, args.year,
                  args.chunk_type)

    print("Finished.")


if __name__ == "__main__":
    main()
