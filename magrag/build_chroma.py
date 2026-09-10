"""
Naplň Chroma kolekci hotovými chunky a JIŽ SPOČÍTANÝMI embeddingy - vektory
dodáváme sami (viz diskuze o "embedding jako vyměnitelná funkce"), takže se
Chroma vůbec nedozví, jaký model/provider embeddingy spočítal.

POZOR na opakované spouštění nad stejnou --db-dir/--collection:
  - Kolekce se plní přes upsert() (ne add()) - add() na existující ID mlčky
    NIC neudělá (žádná chyba, ale ani se nic nepřepíše - ověřeno), takže
    po přegenerování korpusu byste jinak dostávali staré výsledky, i když
    embeddingy i chunky na disku jsou už nové. upsert() existující ID
    correctně přepíše.
  - upsert() ale neřeší situaci, kdy se ZMĚNÍ STRUKTURA chunků (jiné hranice
    článků -> jiný počet chunků na článek -> některá stará chunk_id se už
    vůbec nevygenerují) - takové osiřelé záznamy by v kolekci zůstaly
    viset navždy a pořád by se vracely ve výsledcích. V tom případě použijte
    --overwrite, ať se kolekce nejdřív smaže a založí načisto.

Použití:
    python build_chroma.py \
        --chunks output/ziva_embedding_chunks.jsonl \
        --vectors output/ziva_embeddings__intfloat__multilingual-e5-base.npy \
        --ids output/ziva_embeddings__intfloat__multilingual-e5-base_ids.json \
        --db-dir ./chroma_db --collection ziva --overwrite
"""
import argparse
import json
from pathlib import Path

import chromadb
import numpy as np

from magrag.console import setup_console

BATCH_SIZE = 4000  # Chroma má interní limit na velikost jednoho add()/upsert() volání


def load_chunks(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def sanitize_metadata(chunk: dict) -> dict:
    """Chroma přijímá v metadatech jen str/int/float/bool (žádné None,
    žádné vnořené struktury) - tady se to sjednotí na bezpečné typy.
    year/issue schválně převádíme na int (necháváme je ve zdrojových datech
    jako string), ať jde později filtrovat i číselně ("year": {"$gt": 2020})."""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--db-dir", required=True, help="kam Chroma uloží svá data na disk")
    ap.add_argument("--collection", default="ziva")
    ap.add_argument("--overwrite", action="store_true",
                     help="smaž existující kolekci a založ ji znovu od nuly "
                          "(použijte, když se změnila struktura chunků/článků, "
                          "ne jen jejich obsah)")
    args = ap.parse_args()

    chunks = load_chunks(args.chunks)
    vectors = np.load(args.vectors)
    ids = json.loads(Path(args.ids).read_text(encoding="utf-8"))

    assert len(chunks) == vectors.shape[0] == len(ids), (
        f"počty nesedí: chunks={len(chunks)} vectors={vectors.shape[0]} ids={len(ids)}")
    assert all(c["chunk_id"] == i for c, i in zip(chunks, ids)), \
        "pořadí ID v chunks a ve vectors/ids souboru nesedí"

    client = chromadb.PersistentClient(path=args.db_dir)

    existing = [c.name for c in client.list_collections()]
    if args.collection in existing and args.overwrite:
        client.delete_collection(args.collection)
        print(f"Smazána stará kolekce '{args.collection}' (--overwrite).")
        existing.remove(args.collection)

    if args.collection in existing:
        coll = client.get_collection(args.collection)
        print(f"Kolekce '{args.collection}' už existuje ({coll.count()} položek) - "
              f"chunky se přes upsert() doplní/přepíšou podle ID. Pokud se ale "
              f"od minule změnila STRUKTURA chunků (jiné hranice článků), takhle "
              f"se osiřelých starých záznamů nezbavíte - spusťte znovu s --overwrite.")
    else:
        coll = client.create_collection(
            name=args.collection,
            # naše vektory jsou už normalizované, ale takhle jsou vrácené
            # "distance" hodnoty rovnou interpretovatelné jako kosinová
            # vzdálenost (1 - kosinová podobnost), ne L2
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Založena nová kolekce '{args.collection}'.")

    n = len(chunks)
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        coll.upsert(
            ids=[c["chunk_id"] for c in chunks[start:end]],
            embeddings=vectors[start:end].tolist(),
            documents=[c["text"] for c in chunks[start:end]],
            metadatas=[sanitize_metadata(c) for c in chunks[start:end]],
        )
        print(f"  upsertnuto {end}/{n}")

    print(f"\nHotovo. Kolekce '{args.collection}' obsahuje {coll.count()} "
          f"položek (uloženo v {args.db_dir}).")


if __name__ == "__main__":
    main()

