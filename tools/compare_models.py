"""
Porovnání kandidátních embedding modelů na pár otázek, u kterých ZNÁTE
správnou odpověď (konkrétní article_id nebo aspoň podřetězec v title, co
by se měl objevit mezi top-k výsledky).

Princip: pro každý model spočítej embeddingy všech chunků + embeddingy
testovacích dotazů, najdi top-k nejbližších chunků (obyčejný numpy,
žádná Chroma/FAISS potřeba pro tak malé srovnání) a zkontroluj, jestli je
mezi nimi aspoň jeden chunk z očekávaného článku.

Testovací dotazy si upravte v TEST_QUERIES níž - potřebujete jich jen
5-10, ale musíte sám/sama vědět, který článek/chunk by měl vyjít jako
odpověď (proto to nejde automatizovat/stáhnout odjinud).

Použití:
    python compare_models.py --input output/ziva_embedding_chunks.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from magrag.embed import embed

from magrag.console import setup_console

# Upravte podle vlastního archivu a znalosti obsahu! "expect_title_substring"
# je podřetězec, který by se měl objevit v title článku, jehož chunk čekáte
# mezi top-k výsledky.
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
    # "BAAI/bge-m3",  # velký model / pomalé stažení - odkomentujte, až budete chtít
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
        scores = chunk_vectors @ q_vec  # normalizované vektory -> kosinová podobnost
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="ziva_embedding_chunks.jsonl")
    ap.add_argument("--verbose", action="store_true",
                     help="vypiš i top-k výsledky pro každý dotaz/model")
    args = ap.parse_args()

    chunks = load_chunks(Path(args.input))
    print(f"Načteno {len(chunks)} chunků, {len(TEST_QUERIES)} testovacích dotazů, "
          f"{len(CANDIDATE_MODELS)} modelů k porovnání.\n")

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
        print(f"  trefeno {hits}/{len(TEST_QUERIES)} dotazů (čas: {dt:.1f}s "
              f"na {len(chunks)} chunků)")
        if args.verbose:
            for r in rows:
                mark = "✓" if r["found"] else "✗"
                print(f"    {mark} {r['query']!r}")
                for t in r["top_titles"]:
                    print(f"        - {t[:60]}")
        print()

    print("=== Souhrn ===")
    for model_name, hits in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {hits}/{len(TEST_QUERIES)}  {model_name}")


if __name__ == "__main__":
    main()
