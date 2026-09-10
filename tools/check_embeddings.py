"""A statistical - not merely anecdotal - sanity check of the finished
embeddings.

It compares the DISTRIBUTION of cosine similarity across many pairs of
chunks from the SAME article with the distribution across many pairs from
DIFFERENT articles, rather than comparing one hand-picked pair, which is
what the first version did.

Because embedding spaces are anisotropic and every chunk shares a
metadata header, both distributions may well sit high and not look
"obviously separated". So the script also reports the share of
same-article pairs above the median of different-article pairs, which is
a more robust measure than comparing means: it says how well the model
*ranks* related content above unrelated content, regardless of where the
absolute similarity numbers happen to lie.

Usage:
    python -m tools.check_embeddings \\
        --chunks output/chunks.jsonl \\
        --vectors output/ziva_embeddings__intfloat__multilingual-e5-base.npy \\
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
    print(f"  {name}: n={len(values)}  mean={values.mean():.4f}  "
          f"median={np.median(values):.4f}  sd={values.std():.4f}  "
          f"min={values.min():.4f}  max={values.max():.4f}")


def main():
    setup_console()
    ap = argparse.ArgumentParser(
        description="statistical sanity check of the embeddings")
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--ids", required=True)
    ap.add_argument("--n-pairs", type=int, default=2000,
                    help="how many pairs of each kind (same/different "
                         "article) to sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    chunks = load_chunks(args.chunks)
    vectors = np.load(args.vectors)
    ids = json.loads(open(args.ids, encoding="utf-8").read())

    assert len(chunks) == vectors.shape[0] == len(ids), \
        (f"counts disagree: chunks={len(chunks)} vectors={vectors.shape[0]} "
         f"ids={len(ids)}")
    assert all(c["chunk_id"] == i for c, i in zip(chunks, ids)), \
        "the order of IDs in chunks and in the vectors/ids file does not match"

    n = len(chunks)
    article_to_indices = defaultdict(list)
    for idx, c in enumerate(chunks):
        article_to_indices[c["article_id"]].append(idx)

    # --- pairs from the SAME article (articles with 2+ chunks only) ------
    eligible_articles = [aid for aid, idxs in article_to_indices.items()
                         if len(idxs) >= 2]
    if not eligible_articles:
        raise SystemExit("No article has 2+ chunks - nothing to compare.")

    same_pairs = []
    for _ in range(args.n_pairs):
        aid = random.choice(eligible_articles)
        i, j = random.sample(article_to_indices[aid], 2)
        same_pairs.append((i, j))

    # --- pairs from DIFFERENT articles - genuinely sampled at random -----
    diff_pairs = []
    while len(diff_pairs) < args.n_pairs:
        i, j = random.randrange(n), random.randrange(n)
        if chunks[i]["article_id"] != chunks[j]["article_id"]:
            diff_pairs.append((i, j))

    same_sims = [float(vectors[i] @ vectors[j]) for i, j in same_pairs]
    diff_sims = [float(vectors[i] @ vectors[j]) for i, j in diff_pairs]

    print(f"Sampled {len(same_pairs)} same-article pairs and "
          f"{len(diff_pairs)} different-article pairs (n_pairs={args.n_pairs}, "
          f"seed={args.seed}).\n")
    describe("same article     ", same_sims)
    describe("different article", diff_sims)

    # A more robust measure than comparing means: what share of
    # same-article pairs sits above the median of different-article
    # pairs. If the embeddings told content apart not at all, we would
    # expect about 50%, pure chance. The closer to 100%, the better the
    # model separates related from unrelated content - independently of
    # how high or low the absolute similarity numbers are.
    diff_median = float(np.median(diff_sims))
    pct_above = 100 * float(np.mean(np.array(same_sims) > diff_median))
    print(f"\nShare of same-article pairs above the median of "
          f"different-article pairs: {pct_above:.1f}%")
    print("(50% would mean the embeddings do not distinguish content at "
          "all; the closer to 100%, the better.)")


if __name__ == "__main__":
    main()
