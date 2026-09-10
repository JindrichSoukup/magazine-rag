"""
Stage 5: nasekej ziva_corpus.json na chunky vhodné pro embedding.

Na rozdíl od "chunks" v articles.json (= jednotlivé bloky, extrémně
nesourodá délka od "a" po pár set znaků) tenhle skript chunkuje
article["full_text_paragraphs"]/["captions_paragraphs"] - čistý, souvislý
text článku (bez popisků/annotací u full_text) rozdělený po odstavcích, KAŽDÝ
SE SVOJÍ STRÁNKOU - na kusy s rozumně konzistentní velikostí (cílově
~1200 znaků, s malým překryvem mezi chunky, ať se u hranice chunku
neztratí kontext).

Proč zrovna odstavce se stránkami, a ne holý string (full_text)? Protože
teprve TADY, na úrovni chunkování pro RAG, se rozhoduje, jestli se mají
zahodit obálkové stránky na začátku/konci PDF (viz filter_cover_pages) -
a k tomu je potřeba znát stránku KAŽDÉHO odstavce, ne jen to, kde článek
podle TOC začíná a končí. Dřívější stránky pipeline (extract_blocks.py,
assign_articles.py) žádné stránky nezahazují a nic neví o tom, co je
"obálka" - to je čistě rozhodnutí na úrovni "jak dělám RAG", ne "jak
digitalizuju PDF".

Ke každému chunku se přidá:
  - strukturovaná metadata (year, issue, title, author, ...) jako
    samostatná pole - pro filtrování a citaci zdroje
  - "text" - čistý text chunku (na zobrazení / předání LLM jako kontext)
  - "embedding_text" - text s metadata hlavičkou navíc ("Časopis: Živa
    Ročník: ... Článek: ... Autoři: ... Text: ...") - tohle se posílá do
    embedding modelu, protože osamocený chunk bez kontextu ("Mixotrofové
    představují...") embedding modelu i LLM říká míň než s hlavičkou.
  - unikátní "chunk_id" ("2014-6-1-chunk-0")
  - "page_start"/"page_end" - PDF stránky, ze kterých chunk reálně pochází
    (ne stránka celého článku - u dlouhého článku by to u pozdějších
    chunků bylo zavádějící)

Použití:
    python build_chunks.py --input output/ziva_corpus.json --output ziva_embedding_chunks.jsonl
"""
import argparse
import json
import re
from pathlib import Path

from magrag import profiles
from magrag.console import setup_console

TARGET_CHARS = 1200
# Poznámka k limitu 512 tokenů standardních BERT/XLM-R modelů (celá E5
# rodina): řeší se to teď v embed.py (enforce_max_length) tak, že se ořízne
# jen ten konkrétní text, co limit přesáhne - ne že by se kvůli vzácným
# výjimkám preventivně zmenšovaly VŠECHNY chunky. Naměřeno na reálném
# vzorku: TARGET_CHARS=1200 dávalo nejdelší embedding_text (chunk + metadata
# hlavička) 1544 znaků == 510 tokenů u multilingual-e5-base - těsně pod
# hranicí, ale u větších titulů/autorů v hlavičce se občas přesáhne.
OVERLAP_CHARS = 150
MIN_CHUNK_CHARS = 200  # kratší poslední zbytek radši připoj k předchozímu chunku

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ])")


def filter_cover_pages(paragraphs, total_pdf_pages, skip_first, skip_last):
    """Vynech odstavce z prvních/posledních N stránek PDF (obálka, inzerce
    příštího čísla apod.) - přesně tady, na úrovni chunkování pro RAG, se
    má tahle volba dít (ne dřív v pipeline). Nejčastější dopad: poslední
    článek v čísle mívá pdf_page_end až do úplně poslední strany PDF,
    protože po něm už nic dalšího není v obsahu - a tahle poslední strana
    bývá samostatná obálková fotka, co s článkem obsahově nesouvisí."""
    if not paragraphs or not total_pdf_pages:
        return paragraphs
    lo, hi = skip_first, total_pdf_pages - skip_last
    return [p for p in paragraphs if lo < p["page"] <= hi]


def split_oversized_paragraph(page: int, text: str, limit: int):
    """Když je jeden odstavec sám o sobě delší než limit (dlouhá citace,
    ale taky třeba tabulka/výčet dat BEZ jediné tečky - viz níž), rozsekej
    ho nejdřív po větách. Pokud ani jedna "věta" (== celý text, když není
    žádná interpunkce k rozdělení) není kratší než limit, tvrdě ji rozsekej
    po slovech - jinak by v chunku zůstal jeden obří kus přesahující limit
    třeba 4x (přesně tenhle případ nastal u bloku s tabulkou dat bez teček:
    "Kuno Pepino Bebe ... září leden únor ..."). Stránka se u všech
    výsledných kusů zachová stejná jako u původního odstavce."""
    if len(text) <= limit:
        return [(page, text)]

    sentences = SENTENCE_SPLIT_RE.split(text)
    out, current = [], ""
    for s in sentences:
        if len(s) > limit:
            if current:
                out.append(current.strip())
                current = ""
            words = s.split()
            piece = ""
            for w in words:
                if piece and len(piece) + len(w) + 1 > limit:
                    out.append(piece.strip())
                    piece = w
                else:
                    piece = f"{piece} {w}".strip()
            if piece:
                current = piece
            continue
        if current and len(current) + len(s) > limit:
            out.append(current.strip())
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        out.append(current.strip())
    return [(page, t) for t in out]


def _finalize_chunk(items):
    """items: list of (page, text) -> {"page_start", "page_end", "text"}."""
    pages = [p for p, _ in items]
    text = "\n\n".join(t for _, t in items)
    return {"page_start": min(pages), "page_end": max(pages), "text": text}


def chunk_paragraphs(paragraphs, target_chars=TARGET_CHARS, overlap_chars=OVERLAP_CHARS):
    """paragraphs: list of {"page": int, "text": str} (např. article's
    full_text_paragraphs, případně už zbavené obálkových stránek).
    Vrať list {"page_start", "page_end", "text"}."""
    expanded = []
    for p in paragraphs:
        expanded.extend(split_oversized_paragraph(p["page"], p["text"], target_chars))

    chunks = []
    current = []  # list of (page, text)
    current_len = 0
    for page, text in expanded:
        if current and current_len + len(text) + 2 > target_chars:
            chunks.append(_finalize_chunk(current))
            # překryv: vezmi z konce právě uzavřeného chunku tolik
            # odstavců, kolik se vejde do overlap_chars
            overlap, olen = [], 0
            for prev_page, prev_text in reversed(current):
                if olen + len(prev_text) > overlap_chars:
                    break
                overlap.insert(0, (prev_page, prev_text))
                olen += len(prev_text)
            current, current_len = overlap.copy(), olen
        current.append((page, text))
        current_len += len(text) + 2

    if current:
        last = _finalize_chunk(current)
        if chunks and len(last["text"]) < MIN_CHUNK_CHARS:
            # kratší zbytek (typicky jen překryv) radši připoj k předchozímu
            chunks[-1]["text"] += "\n\n" + last["text"]
            chunks[-1]["page_end"] = last["page_end"]
        else:
            chunks.append(last)
    return chunks


def build_embedding_text(article: dict, chunk_text: str, profile) -> str:
    """Text, který jde do embedding modelu: chunk s hlavičkou o tom, odkud
    pochází. Šablonu i její jazyk určuje profil zdroje."""
    return profile.chunk_header_template.format(
        journal=profile.journal_name,
        year=article["year"],
        issue=article["issue"],
        title=article["title"],
        author=article["author"] or profile.unknown_author_label,
        text=chunk_text,
    )


def build_chunks_for_article(article: dict, profile, skip_first=None,
                             skip_last=None):
    if skip_first is None:
        skip_first = profile.skip_first_pages
    if skip_last is None:
        skip_last = profile.skip_last_pages
    out = []
    total_pages = article.get("total_pdf_pages")

    body_paragraphs = filter_cover_pages(
        article.get("full_text_paragraphs", []), total_pages, skip_first, skip_last)
    for i, ch in enumerate(chunk_paragraphs(body_paragraphs)):
        out.append({
            "chunk_id": f"{article['article_id']}-chunk-{i}",
            "chunk_type": "body",
            "chunk_index": i,
            "year": article["year"],
            "issue": article["issue"],
            "article_id": article["article_id"],
            "title": article["title"],
            "author": article["author"],
            "page_start": ch["page_start"],
            "page_end": ch["page_end"],
            "text": ch["text"],
            "embedding_text": build_embedding_text(article, ch["text"], profile),
        })

    caption_paragraphs = filter_cover_pages(
        article.get("captions_paragraphs", []), total_pages, skip_first, skip_last)
    for i, ch in enumerate(chunk_paragraphs(caption_paragraphs)):
        out.append({
            "chunk_id": f"{article['article_id']}-caption-{i}",
            "chunk_type": "caption",
            "chunk_index": i,
            "year": article["year"],
            "issue": article["issue"],
            "article_id": article["article_id"],
            "title": article["title"],
            "author": article["author"],
            "page_start": ch["page_start"],
            "page_end": ch["page_end"],
            "text": ch["text"],
            "embedding_text": build_embedding_text(article, ch["text"], profile),
        })
    return out


def main():
    setup_console()
    ap = argparse.ArgumentParser(description="nasekej korpus na chunky pro embedding")
    ap.add_argument("--input", required=True, help="corpus.json (z run_all.py)")
    ap.add_argument("--output", required=True, help="výstupní .jsonl")
    ap.add_argument("--skip-first", type=int, default=None,
                     help="kolik stránek na začátku PDF čísla ignorovat (obálka); "
                          "výchozí hodnota je v profilu")
    ap.add_argument("--skip-last", type=int, default=None,
                     help="kolik stránek na konci PDF čísla ignorovat (zadní obálka); "
                          "výchozí hodnota je v profilu")
    profiles.add_profile_argument(ap)
    args = ap.parse_args()

    profile = profiles.get(args.profile)
    articles = json.loads(Path(args.input).read_text(encoding="utf-8"))

    all_chunks = []
    for article in articles:
        if not article.get("full_text", "").strip():
            continue
        all_chunks.extend(build_chunks_for_article(
            article, profile, args.skip_first, args.skip_last))

    with open(args.output, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    n_body = sum(1 for c in all_chunks if c["chunk_type"] == "body")
    n_caption = sum(1 for c in all_chunks if c["chunk_type"] == "caption")
    print(f"{len(articles)} článků -> {len(all_chunks)} chunků "
          f"({n_body} body, {n_caption} caption) -> {args.output}")


if __name__ == "__main__":
    main()
