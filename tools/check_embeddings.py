"""
Statistický (ne jen anekdotický) sanity check hotových embeddingů:
porovná DISTRIBUCI kosinové podobnosti přes mnoho párů chunků ze STEJNÉHO
článku s distribucí přes mnoho párů z RŮZNÝCH článků - místo srovnání
jednoho vybraného páru, jak to bylo poprvé.

Kvůli anisotropii embedding prostoru a sdílené metadata hlavičce (viz
diskuze) je docela možné, že obě distribuce budou ležet vysoko a nebudou
"na první pohled" jasně oddělené - proto se počítá i podíl párů ze stejného
článku nad mediánem párů z různých článků, což je robustnější míra než
pouhé porovnání průměrů.

Použití:
    python check_embeddings.py \
        --chunks output/ziva_embedding_chunks.jsonl \
        --vectors output/ziva_embeddings__intfloat__multilingual-e5-base.npy \
        --ids output/ziva_embeddings__intfloat__multilingual-e5-base_ids.json
"""
import argparse
import json
import random
from collections import defaultdict

import numpy as np

from magrag.console import setup_console


def load_chunks(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def describe(name, values):
    values = np.array(values)
    print(f"  {name}: n={len(values)}  průměr={values.mean():.4f}  "
          f"medián={np.median(values):.4f}  sd={values.std():.4f}  "
          f"min={values.min():.4f}  max={values.max():.4f}")


def main():
    setup_console()
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--n-pairs", type=int, default=2000,
                     help="kolik párů od každého druhu (stejný/různý článek) vzorkovat")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    chunks = load_chunks(args.chunks)
    vectors = np.load(args.vectors)
    ids = json.loads(open(args.ids, encoding="utf-8").read())

    assert len(chunks) == vectors.shape[0] == len(ids), \
        f"počty nesedí: chunks={len(chunks)} vectors={vectors.shape[0]} ids={len(ids)}"
    assert all(c["chunk_id"] == i for c, i in zip(chunks, ids)), \
        "pořadí ID v chunks a ve vectors/ids souboru nesedí"

    n = len(chunks)
    article_to_indices = defaultdict(list)
    for idx, c in enumerate(chunks):
        article_to_indices[c["article_id"]].append(idx)

    # --- páry ze STEJNÉHO článku (jen články s >=2 chunky) -----------------
    eligible_articles = [aid for aid, idxs in article_to_indices.items() if len(idxs) >= 2]
    if not eligible_articles:
        raise SystemExit("Žádný článek nemá 2+ chunky - nelze srovnávat.")

    same_pairs = []
    for _ in range(args.n_pairs):
        aid = random.choice(eligible_articles)
        i, j = random.sample(article_to_indices[aid], 2)
        same_pairs.append((i, j))

    # --- páry z RŮZNÝCH článků - skutečně náhodně losované, ne jeden pár --
    diff_pairs = []
    while len(diff_pairs) < args.n_pairs:
        i, j = random.randrange(n), random.randrange(n)
        if chunks[i]["article_id"] != chunks[j]["article_id"]:
            diff_pairs.append((i, j))

    same_sims = [float(vectors[i] @ vectors[j]) for i, j in same_pairs]
    diff_sims = [float(vectors[i] @ vectors[j]) for i, j in diff_pairs]

    print(f"Vzorkováno {len(same_pairs)} párů ze stejného článku a "
          f"{len(diff_pairs)} párů z různých článků (n_pairs={args.n_pairs}, "
          f"seed={args.seed}).\n")
    describe("stejný článek", same_sims)
    describe("různé články ", diff_sims)

    # Robustnější míra než porovnání průměrů: kolik % párů "stejný článek"
    # leží nad mediánem párů "různé články". Kdyby embeddingy obsah vůbec
    # nerozlišovaly, čekali bychom ~50 % (čistá náhoda). Čím blíž 100 %,
    # tím lépe model odlišuje související od nesouvisejícího obsahu -
    # nezávisle na tom, jak vysoko/nízko leží absolutní čísla podobnosti.
    diff_median = float(np.median(diff_sims))
    pct_above = 100 * float(np.mean(np.array(same_sims) > diff_median))
    print(f"\n% párů ze stejného článku nad mediánem párů z různých článků: "
          f"{pct_above:.1f}%")
    print("(50 % by znamenalo, že embeddingy vůbec nerozlišují obsah; "
          "čím blíž 100 %, tím lépe.)")


if __name__ == "__main__":
    main()
