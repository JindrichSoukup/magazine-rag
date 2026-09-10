"""
Sestavení kontextu z výsledků dotazu do Chroma - namísto vracení
jednotlivých (krátkých) chunků odděleně:

  1. top_N zásahů (výchozí 10, ne jen top_5 - potřebujeme širší okno na
     rozhodnutí, jestli je jeden článek "dominantní").
  2. Zásahy se seskupí podle article_id.
  3. Článek s >= PROMOTE_THRESHOLD zásahy v top_N (výchozí 3) se POVÝŠÍ na
     celý článek (jeho celé full_text z korpusu) - silný signál, že dotaz
     cílí přímo na tenhle článek jako celek.
  4. Ostatní články (1-2 zásahy) dostanou "window expansion": ke každému
     zásahu se přibalí +-WINDOW sousedních chunků (podle chunk_index ve
     stejném článku), překrývající se okna se sloučí do jednoho kusu.
  5. Každý výsledný kus kontextu nese citaci (časopis/ročník/číslo/článek/
     autoři/strany), aby LLM mohlo v odpovědi citovat zdroj.

Použití jako knihovna (viz query_chroma.py, kde se to volá) i samostatně:
    python assemble_context.py --db-dir ./chroma_db --collection ziva \
        --model intfloat/multilingual-e5-base \
        --chunks output/ziva_embedding_chunks.jsonl \
        --corpus output/ziva_corpus.json \
        --query "Jak vzniká Huntingtonova choroba?"
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import chromadb

try:
    import tiktoken
    _TOKENIZER = tiktoken.get_encoding("cl100k_base")
except ImportError:
    _TOKENIZER = None  # spadneme na hrubý odhad podle znaků (viz count_tokens níž)


def count_tokens(text: str) -> int:
    """Počet tokenů - přes tiktoken (cl100k_base, stejný řád velikosti jako
    Claude/GPT tokenizery pro evropské jazyky), nebo hrubý znakový odhad,
    když tiktoken není nainstalovaný (`pip install tiktoken`). Čeština kvůli
    diakritice tokenizuje hůř než třeba angličtina - poměr real. naměřený
    na ukázce z korpusu byl ~2,3 znaku/token, ne obvyklých ~4."""
    if _TOKENIZER is not None:
        return len(_TOKENIZER.encode(text))
    return round(len(text) / 2.3)  # hrubý odhad specificky pro češtinu

from magrag.embed import embed

DEFAULT_TOP_N = 10
DEFAULT_WINDOW = 1
DEFAULT_PROMOTE_THRESHOLD = 3


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_body_sequences(chunks):
    """article_id -> seřazený seznam (chunk_index, page_start, page_end, text)
    JEN pro chunk_type=="body" - window expansion se dělá v rámci téhle
    posloupnosti, ne přes popisky obrázků (ty mají vlastní číslování)."""
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
    """spans: list of (lo, hi, best_distance). Slij překrývající se/sousedící
    intervaly, ponech nejmenší (=nejlepší) distance z každé skupiny."""
    spans = sorted(spans, key=lambda s: s[0])
    merged = []
    for lo, hi, dist in spans:
        if merged and lo <= merged[-1][1] + 1:
            prev_lo, prev_hi, prev_dist = merged[-1]
            merged[-1] = (prev_lo, max(prev_hi, hi), min(prev_dist, dist))
        else:
            merged.append((lo, hi, dist))
    return merged


def format_citation(meta, page_start=None, page_end=None):
    ps = page_start if page_start is not None else meta["page_start"]
    pe = page_end if page_end is not None else meta["page_end"]
    pages = f"str. {ps}" if ps == pe else f"str. {ps}-{pe}"
    return (f"{meta['title']} (Živa {meta['year']}/{meta['issue']}, {pages}), "
            f"autoři: {meta['author'] or 'neuvedeno'}")


def assemble_context(hits, corpus_index, body_sequences,
                      window=DEFAULT_WINDOW, promote_threshold=DEFAULT_PROMOTE_THRESHOLD):
    """hits: list of dict {chunk_id, meta, distance, text} - výsledky
    z Chroma (top_N). Vrať list bloků kontextu seřazených od
    nejrelevantnějšího, každý s "citation" a "text"."""
    by_article = defaultdict(list)
    for h in hits:
        by_article[h["meta"]["article_id"]].append(h)

    blocks = []
    for article_id, article_hits in by_article.items():
        meta0 = article_hits[0]["meta"]
        best_dist = min(h["distance"] for h in article_hits)

        if len(article_hits) >= promote_threshold:
            # POVÝŠENO: celý článek z korpusu, ne jen jeho chunky
            article = corpus_index.get(article_id)
            if article is None:
                continue  # nemělo by nastat, ale radši nespadnout
            blocks.append({
                "mode": "full_article",
                "n_hits": len(article_hits),
                "citation": format_citation(
                    meta0, article["pdf_page_start"], article["pdf_page_end"]),
                "text": article["full_text"],
                "best_distance": best_dist,
            })
            continue

        # NEPOVÝŠENO: window expansion kolem každého zásahu, sloučit překryvy
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
                "citation": format_citation(meta0, page_start, page_end),
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-dir", required=True)
    ap.add_argument("--collection", default="ziva")
    ap.add_argument("--model", required=True)
    ap.add_argument("--chunks", required=True, help="ziva_embedding_chunks.jsonl")
    ap.add_argument("--corpus", required=True, help="ziva_corpus.json")
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW,
                     help="kolik sousedních chunků přibalit na každou stranu zásahu "
                          "(výchozí 1 - chunky jsou dost velké, +-2 dělalo zbytečně "
                          "obří kontext)")
    ap.add_argument("--promote-threshold", type=int, default=DEFAULT_PROMOTE_THRESHOLD)
    ap.add_argument("--query", default=None,
                     help="jednorázový dotaz (bez tohohle se spustí interaktivní smyčka)")
    ap.add_argument("--preview-chars", type=int, default=0,
                     help="zkrátit výpis textu na tolik znaků (0 = vypsat celý "
                          "text, přesně to, co by šlo do LLM)")
    args = ap.parse_args()

    print("Nahrávám chunky/korpus/model a připojuji se ke kolekci (jen jednou) ...")
    chunks = load_jsonl(args.chunks)
    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    body_sequences = build_body_sequences(chunks)
    corpus_index = build_corpus_index(corpus)

    client = chromadb.PersistentClient(path=args.db_dir)
    coll = client.get_collection(args.collection)
    embed(["zahřívací dotaz"], model_name=args.model, is_query=True, show_progress=False)
    print(f"Připraveno ({coll.count()} chunků, {len(corpus)} článků v korpusu).\n")

    def handle(query_text):
        hits = run_search(coll, args.model, query_text, args.top_n)
        blocks = assemble_context(hits, corpus_index, body_sequences,
                                   args.window, args.promote_threshold)
        print(f"\nDotaz: {query_text!r}  (top_n={args.top_n}, window={args.window}, "
              f"promote_threshold={args.promote_threshold})\n")
        total_tokens = 0
        for i, b in enumerate(blocks, 1):
            tag = "CELÝ ČLÁNEK" if b["mode"] == "full_article" else "výřez"
            n_tokens = count_tokens(b["citation"] + "\n" + b["text"])
            total_tokens += n_tokens
            print(f"--- Zdroj {i} [{tag}, {b['n_hits']} zásahů v top_n, "
                  f"nejlepší vzdálenost {b['best_distance']:.3f}, "
                  f"{len(b['text'])} znaků, ~{n_tokens} tokenů] ---")
            print(b["citation"])
            print()
            if args.preview_chars > 0 and len(b["text"]) > args.preview_chars:
                print(b["text"][:args.preview_chars] + "...")
            else:
                print(b["text"])  # celý text - přesně to, co by šlo do LLM
            print()
        query_tokens = count_tokens(query_text)
        print(f"=== Celkem ~{total_tokens} tokenů kontextu + ~{query_tokens} "
              f"tokenů dotazu = ~{total_tokens + query_tokens} tokenů "
              f"(bez systémové instrukce{'' if _TOKENIZER else ', ODHAD podle znaků - nainstalujte tiktoken pro přesnější číslo'}) ===")

    if args.query is not None:
        handle(args.query)
        return

    print("Zadejte dotaz (prázdný řádek nebo 'exit'/'quit' pro ukončení):")
    while True:
        try:
            query_text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query_text or query_text.lower() in ("exit", "quit"):
            break
        handle(query_text)

    print("Ukončeno.")


if __name__ == "__main__":
    main()
