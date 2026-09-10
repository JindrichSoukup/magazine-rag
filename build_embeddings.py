"""
Spočítej embeddingy pro všechny chunky v ziva_embedding_chunks.jsonl a ulož
je na disk - dva soubory vedle sebe, propojené POŘADÍM (řádek N v .jsonl
odpovídá řádku N v .npy):

    ziva_embeddings__<model>.npy       - pole (n_chunks, dim), float32
    ziva_embeddings__<model>_ids.json  - list chunk_id ve stejném pořadí

Embedduje se pole "embedding_text" (s metadata hlavičkou), ne holé "text" -
viz diskuze o "contextual chunking".

PRŮBĚŽNÉ UKLÁDÁNÍ (checkpointing)
----------------------------------
Na velkém archivu (desítky tisíc chunků) může výpočet na CPU trvat hodiny.
Aby se dal běh bezpečně přerušit (Ctrl+C, pád, vypnutí počítače) a pak
navázat tam, kde skončil, se místo "spočítej vše -> ulož najednou" dělá
takhle:

  1. Vektory se průběžně zapisují do dočasného souboru <model>.raw přes
     np.memmap - to je pole na disku, do kterého lze zapisovat po kouskách
     bez nutnosti mít celý výsledek v paměti, a hlavně nezávisle na tom,
     jestli proces doběhne až do konce.
  2. Po každých --checkpoint-every chunků (výchozí 500) se do
     <model>_progress.json zapíše, kolik je hotovo.
  3. Když skript spustíte znovu se stejným --output-dir/--model, nejdřív
     zkontroluje, jestli tam už rozdělaný běh je - a pokud ano, pokračuje
     od posledního checkpointu místo od začátku.
  4. Až je hotovo úplně vše, .raw soubor se převede na normální .npy
     (přesně formát, jaký čekává zbytek pipeline/Chroma) a dočasné soubory
     (.raw, _progress.json) se smažou.

POZOR: checkpoint je platný jen pro STEJNÝ vstupní .jsonl (kontroluje se
jen počet řádků, ne obsah) - pokud mezi přerušením a pokračováním vstupní
soubor změníte (jiné pořadí/jiný obsah chunků), index by se rozjel. Necháno
takhle kvůli jednoduchosti - běžně vstupní soubor mezi jedním "sedněte
a spočítejte embeddingy" během neměníte.

Použití:
    python build_embeddings.py --input output/ziva_embedding_chunks.jsonl \\
        --output-dir output --model intfloat/multilingual-e5-base

    # po přerušení stačí spustit úplně stejný příkaz znovu - naváže samo
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from embed import embed, embedding_dim, DEFAULT_MODEL, DEFAULT_BATCH_SIZE

DEFAULT_CHECKPOINT_EVERY = 500


def load_chunks(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def paths_for(out_dir: Path, model_name: str):
    """Jedno místo, kde se skládají všechna jména souborů - ať se výsledný
    .npy/_ids.json, dočasný .raw i _progress.json vždycky shodují."""
    slug = model_name.replace("/", "__")
    base = out_dir / f"ziva_embeddings__{slug}"
    return {
        "npy": base.with_suffix(".npy"),
        "ids": Path(f"{base}_ids.json"),
        "raw": base.with_suffix(".raw"),
        "progress": Path(f"{base}_progress.json"),
    }


def format_eta(seconds: float) -> str:
    if seconds == float("inf") or seconds != seconds:  # inf nebo NaN
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="ziva_embedding_chunks.jsonl")
    ap.add_argument("--output-dir", required=True, help="kam uložit .npy a _ids.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                     help="dávka pro samotný model.encode() (výpočetní jednotka)")
    ap.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY,
                     help="po kolika chuncích průběžně uložit postup na disk")
    args = ap.parse_args()

    chunks = load_chunks(Path(args.input))
    n = len(chunks)
    print(f"Načteno {n} chunků z {args.input}")

    dim = embedding_dim(args.model)
    print(f"Model: {args.model} (dim={dim})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = paths_for(out_dir, args.model)

    if p["npy"].exists():
        print(f"{p['npy']} už existuje (hotovo z dřívějška) - nic nedělám. "
              f"Smažte ho, pokud chcete přepočítat znovu.")
        return

    texts = [c["embedding_text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]

    # --- najdi/založ checkpoint ------------------------------------------
    if p["progress"].exists() and p["raw"].exists():
        progress = json.loads(p["progress"].read_text(encoding="utf-8"))
        if progress.get("model") != args.model or progress.get("total") != n:
            raise SystemExit(
                f"Rozjetý checkpoint v {p['raw'].name} neodpovídá aktuálnímu "
                f"vstupu/modelu (byl pro model={progress.get('model')!r}, "
                f"total={progress.get('total')}) - smažte {p['raw'].name} "
                f"a {p['progress'].name}, nebo použijte jiné --output-dir."
            )
        done = progress["done"]
        print(f"Navazuji na předchozí běh: {done}/{n} chunků už hotovo.")
    else:
        done = 0
        mm_init = np.memmap(p["raw"], dtype="float32", mode="w+", shape=(n, dim))
        mm_init.flush()
        del mm_init
        p["ids"].write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
        p["progress"].write_text(
            json.dumps({"model": args.model, "total": n, "done": 0}), encoding="utf-8")

    mm = np.memmap(p["raw"], dtype="float32", mode="r+", shape=(n, dim))

    # --- hlavní smyčka: embeduj po checkpoint_every kouskách -------------
    idx = done
    t0 = time.time()
    try:
        while idx < n:
            end = min(idx + args.checkpoint_every, n)
            vectors = embed(texts[idx:end], model_name=args.model, is_query=False,
                             batch_size=args.batch_size, show_progress=False)
            mm[idx:end] = vectors
            mm.flush()
            idx = end
            p["progress"].write_text(
                json.dumps({"model": args.model, "total": n, "done": idx}),
                encoding="utf-8")

            elapsed = time.time() - t0
            rate = (idx - done) / elapsed if elapsed > 0 else 0
            eta = (n - idx) / rate if rate > 0 else float("inf")
            print(f"  {idx}/{n} hotovo ({rate:.2f} chunků/s, "
                  f"zbývá odhadem {format_eta(eta)})")
    except KeyboardInterrupt:
        print(f"\nPřerušeno na {idx}/{n}. Checkpoint uložen - spusťte stejný "
              f"příkaz znovu, naváže se odsud.")
        return

    # --- hotovo: převeď dočasný .raw na normální .npy --------------------
    final = np.array(mm)  # načte se do paměti - pro naše velikosti v pohodě
    del mm
    np.save(p["npy"], final)
    p["raw"].unlink()
    p["progress"].unlink()

    print(f"Hotovo. Vektory: {p['npy']}  (shape={final.shape}, dtype={final.dtype})")
    print(f"ID (stejné pořadí): {p['ids']}")


if __name__ == "__main__":
    main()
