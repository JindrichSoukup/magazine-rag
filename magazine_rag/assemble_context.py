"""Assemble context from the results of a Chroma query.

Rather than returning the individual, short chunks separately:

  1. Take the top_N hits (10 by default, not just 5 - a wider window is
     needed to decide whether one article is "dominant").
  2. Group the hits by article_id.
  3. An article with at least PROMOTE_THRESHOLD hits inside top_N (3 by
     default) is PROMOTED to the whole article, its entire full_text from
     the corpus: a strong signal that the query is aimed at that article
     as a whole.
  4. The other articles (one or two hits) get "window expansion": each
     hit is padded with WINDOW neighbouring chunks either side, by
     chunk_index within the same article, and overlapping windows are
     merged into one piece.
  5. Every resulting piece of context carries a citation (magazine,
     year, issue, article, authors, pages) so the LLM can attribute what
     it says.

Usable as a library (see answer.py, which calls it) and on its own:
    python -m magazine_rag.assemble_context --db-dir ./chroma_db --collection ziva \\
        --model intfloat/multilingual-e5-base \\
        --chunks output/chunks.jsonl \\
        --corpus output/corpus.json \\
        --query "How does Huntington's disease arise?"
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import chromadb

from magazine_rag import profiles
from magazine_rag.console import setup_console
from magazine_rag.embed import embed

try:
    import tiktoken
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _TOKENIZER = None  # fall back to a rough character estimate, see below

DEFAULT_TOP_N = 10
DEFAULT_WINDOW = 1
DEFAULT_PROMOTE_THRESHOLD = 3

# Characters per token when tiktoken is not installed. Czech tokenises
# worse than English because of its diacritics: measured on a sample of
# this corpus the ratio was about 2.3 characters per token, not the usual
# 4. The estimate is deliberately pessimistic - overestimating the
# context is safer than being surprised by a rejected request.
CHARS_PER_TOKEN_FALLBACK = 2.3


def count_tokens(text: str) -> int:
    """Token count via tiktoken (cl100k_base, the same order of magnitude
    as the Claude and GPT tokenizers for European languages), or a rough
    character estimate when tiktoken is not installed
    (`pip install tiktoken`)."""
    if _TOKENIZER is not None:
        return len(_TOKENIZER.encode(text))
    return round(len(text) / CHARS_PER_TOKEN_FALLBACK)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_body_sequences(chunks):
    """article_id -> a sorted list of (chunk_index, page_start, page_end,
    text) for chunk_type=="body" ONLY. Window expansion happens within
    this sequence, not across figure captions, which have their own
    numbering."""
    by_article = defaultdict(list)
    for c in chunks:
        if c["chunk_type"] != "body":
            continue
        by_article[c["article_id"]].append(
            (c["chunk_index"], c["page_start"], c["page_end"], c["text"]))
    for seq in by_article.values():
        seq.sort(key=lambda t: t[0])
    return by_article


def build_corpus_index(corpus):
    return {a["article_id"]: a for a in corpus}


def merge_spans(spans):
    """spans: a list of (lo, hi, best_distance). Merge overlapping and
    adjacent intervals, keeping the smallest (best) distance of each
    group."""
    spans = sorted(spans, key=lambda s: s[0])
    merged = []
    for lo, hi, dist in spans:
        if merged and lo <= merged[-1][1] + 1:
            prev_lo, prev_hi, prev_dist = merged[-1]
            merged[-1] = (prev_lo, max(prev_hi, hi), min(prev_dist, dist))
        else:
            merged.append((lo, hi, dist))
    return merged


def format_citation(meta, profile, page_start=None, page_end=None):
    """Render one source citation. The wording and its language come from
    the profile, because they follow the corpus rather than the code."""
    ps = page_start if page_start is not None else meta["page_start"]
    pe = page_end if page_end is not None else meta["page_end"]
    pages = (profile.page_single_label.format(page=ps) if ps == pe
             else profile.page_range_label.format(start=ps, end=pe))
    return profile.citation_template.format(
        journal=profile.journal_name,
        year=meta.get("year", ""),
        issue=meta.get("issue", ""),
        title=meta.get("title", ""),
        author=meta.get("author") or profile.unknown_author_label,
        pages=pages,
    )


def assemble_context(hits, corpus_index, body_sequences, profile,
                     window=DEFAULT_WINDOW,
                     promote_threshold=DEFAULT_PROMOTE_THRESHOLD):
    """hits: a list of {chunk_id, meta, distance, text} from Chroma (the
    top_N). Returns a list of context blocks ordered most relevant first,
    each with a "citation" and a "text"."""
    by_article = defaultdict(list)
    for h in hits:
        by_article[h["meta"]["article_id"]].append(h)

    blocks = []
    for article_id, article_hits in by_article.items():
        meta0 = article_hits[0]["meta"]
        best_dist = min(h["distance"] for h in article_hits)

        if len(article_hits) >= promote_threshold:
            # PROMOTED: the whole article from the corpus, not its chunks
            article = corpus_index.get(article_id)
            if article is None:
                continue  # should not happen, but better not to crash
            blocks.append({
                "mode": "full_article",
                "n_hits": len(article_hits),
                "citation": format_citation(
                    meta0, profile,
                    article["pdf_page_start"], article["pdf_page_end"]),
                "text": article["full_text"],
                "best_distance": best_dist,
            })
            continue

        # NOT PROMOTED: window expansion around each hit, overlaps merged
        spans = [(h["meta"]["chunk_index"] - window,
                  h["meta"]["chunk_index"] + window,
                  h["distance"]) for h in article_hits]
        seq = body_sequences.get(article_id, [])
        for lo, hi, dist in merge_spans(spans):
            pieces = [(idx, ps, pe, t) for idx, ps, pe, t in seq if lo <= idx <= hi]
            if not pieces:
                continue
            text = "\n\n".join(t for _, _, _, t in pieces)
            page_start = min(ps for _, ps, _, _ in pieces)
            page_end = max(pe for _, _, pe, _ in pieces)
            blocks.append({
                "mode": "window",
                "n_hits": len(article_hits),
                "citation": format_citation(meta0, profile, page_start, page_end),
                "text": text,
                "best_distance": dist,
            })

    blocks.sort(key=lambda b: b["best_distance"])
    return blocks


def run_search(coll, model_name, query_text, top_n):
    query_vector = embed([query_text], model_name=model_name, is_query=True,
                         show_progress=False)[0]
    result = coll.query(query_embeddings=[query_vector.tolist()], n_results=top_n)
    hits = []
    for cid, meta, dist, doc in zip(result["ids"][0], result["metadatas"][0],
                                    result["distances"][0], result["documents"][0]):
        hits.append({"chunk_id": cid, "meta": meta, "distance": dist, "text": doc})
    return hits


def main():
    setup_console()
    ap = argparse.ArgumentParser(
        description="assemble LLM context from a Chroma query")
    ap.add_argument("--db-dir", required=True)
    ap.add_argument("--collection", default="ziva")
    ap.add_argument("--model", required=True, help="embedding model")
    ap.add_argument("--chunks", required=True, help="chunks.jsonl")
    ap.add_argument("--corpus", required=True, help="corpus.json")
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                    help="how many neighbouring chunks to attach either side "
                         "of a hit (default 1 - the chunks are large enough "
                         "that +-2 made the context needlessly huge)")
    ap.add_argument("--promote-threshold", type=int,
                    default=DEFAULT_PROMOTE_THRESHOLD)
    ap.add_argument("--query", default=None,
                    help="a one-off query; without it an interactive loop runs")
    ap.add_argument("--preview-chars", type=int, default=0,
                    help="truncate the printed text to this many characters "
                         "(0 = print it all, exactly what would go to the LLM)")
    profiles.add_profile_argument(ap)
    args = ap.parse_args()

    profile = profiles.get(args.profile)

    print("Loading chunks, corpus and model, and connecting to the "
          "collection (once) ...")
    chunks = load_jsonl(args.chunks)
    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    body_sequences = build_body_sequences(chunks)
    corpus_index = build_corpus_index(corpus)

    client = chromadb.PersistentClient(path=args.db_dir)
    coll = client.get_collection(args.collection)
    embed(["warm-up query"], model_name=args.model, is_query=True,
          show_progress=False)
    print(f"Ready ({coll.count()} chunks, {len(corpus)} articles in the corpus).\n")

    def handle(query_text):
        hits = run_search(coll, args.model, query_text, args.top_n)
        blocks = assemble_context(hits, corpus_index, body_sequences, profile,
                                  args.window, args.promote_threshold)
        print(f"\nQuery: {query_text!r}  (top_n={args.top_n}, "
              f"window={args.window}, "
              f"promote_threshold={args.promote_threshold})\n")
        total_tokens = 0
        for i, b in enumerate(blocks, 1):
            tag = "FULL ARTICLE" if b["mode"] == "full_article" else "excerpt"
            n_tokens = count_tokens(b["citation"] + "\n" + b["text"])
            total_tokens += n_tokens
            print(f"--- Source {i} [{tag}, {b['n_hits']} hits in top_n, "
                  f"best distance {b['best_distance']:.3f}, "
                  f"{len(b['text'])} chars, ~{n_tokens} tokens] ---")
            print(b["citation"])
            print()
            if args.preview_chars > 0 and len(b["text"]) > args.preview_chars:
                print(b["text"][:args.preview_chars] + "...")
            else:
                print(b["text"])  # the whole text, exactly what goes to the LLM
            print()
        query_tokens = count_tokens(query_text)
        estimate_note = "" if _TOKENIZER else (
            ", ESTIMATED from characters - install tiktoken for a real count")
        print(f"=== About {total_tokens} tokens of context + about "
              f"{query_tokens} tokens of query = about "
              f"{total_tokens + query_tokens} tokens (excluding the system "
              f"prompt{estimate_note}) ===")

    if args.query is not None:
        handle(args.query)
        return

    print("Enter a query (an empty line, 'exit' or 'quit' ends the session):")
    while True:
        try:
            query_text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query_text or query_text.lower() in ("exit", "quit"):
            break
        handle(query_text)

    print("Finished.")


if __name__ == "__main__":
    main()
