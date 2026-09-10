"""
Interaktivní dotazování do Chroma kolekce - model i kolekce se načtou
JEDNOU při startu, pak se ptáte opakovaně ve stejném procesu (žádné
opakované načítání modelu při každém dotazu, jak by to dělalo spouštění
skriptu pro každý dotaz zvlášť).

Použití:
    python query_chroma.py --db-dir ./chroma_db --collection ziva \
        --model intfloat/multilingual-e5-base

    # pak se v promptu ptejte, dokud nenapíšete "exit"/"quit"/prázdný řádek

Jednorázový dotaz z příkazové řádky (bez vstupu do smyčky) jde pořád přes
--query, pro rychlé skriptování/testování:
    python query_chroma.py --db-dir ./chroma_db --collection ziva \
        --model intfloat/multilingual-e5-base --query "..."
"""
import argparse

import chromadb

from magrag.embed import embed


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

    result = coll.query(query_embeddings=[query_vector.tolist()], n_results=top_k, where=where)

    print(f"\nDotaz: {query_text!r}")
    if where:
        print(f"Filtr: {where}")
    print()

    if not result["ids"][0]:
        print("(žádné výsledky)\n")
        return

    for i, (doc, meta, dist, cid) in enumerate(zip(
            result["documents"][0], result["metadatas"][0],
            result["distances"][0], result["ids"][0]), 1):
        print(f"{i}. [kos. vzdálenost {dist:.3f}, nižší = podobnější] "
              f"{meta['title']} ({meta['year']}/{meta['issue']}, "
              f"str. {meta['page_start']}-{meta['page_end']})")
        print(f"   autoři: {meta['author'] or 'neuvedeno'} | typ: {meta['chunk_type']} | id: {cid}")
        print(f"   {doc[:200]}...")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-dir", required=True)
    ap.add_argument("--collection", default="ziva")
    ap.add_argument("--model", required=True,
                     help="MUSÍ být stejný model, jakým se počítaly embeddingy chunků")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--year", type=int, default=None,
                     help="volitelný filtr na metadata - jen tento ročník")
    ap.add_argument("--chunk-type", choices=["body", "caption"], default=None,
                     help="volitelný filtr - jen tělo článků, nebo jen popisky obrázků")
    ap.add_argument("--query", default=None,
                     help="jednorázový dotaz (bez tohohle se spustí interaktivní smyčka)")
    args = ap.parse_args()

    print("Připojuji se ke kolekci a nahrávám model (jen jednou) ...")
    client = chromadb.PersistentClient(path=args.db_dir)
    coll = client.get_collection(args.collection)
    # "zahřátí" modelu hned na začátku, ať první dotaz v interaktivní
    # smyčce neplatí načítací cenu navíc oproti dalším
    embed(["zahřívací dotaz"], model_name=args.model, is_query=True, show_progress=False)
    print(f"Připraveno ({coll.count()} položek v kolekci '{args.collection}').\n")

    if args.query is not None:
        run_query(coll, args.model, args.query, args.top_k, args.year, args.chunk_type)
        return

    print("Zadejte dotaz (prázdný řádek nebo 'exit'/'quit' pro ukončení):")
    while True:
        try:
            query_text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query_text or query_text.lower() in ("exit", "quit"):
            break
        run_query(coll, args.model, query_text, args.top_k, args.year, args.chunk_type)

    print("Ukončeno.")


if __name__ == "__main__":
    main()
