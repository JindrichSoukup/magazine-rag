"""Stage 5: cut corpus.json into chunks suitable for embedding.

Unlike the "chunks" in articles.json - individual blocks, wildly
inconsistent in length from one character to a few hundred - this script
chunks `article["full_text_paragraphs"]` and `["captions_paragraphs"]`:
clean, continuous article text (with captions and annotations kept out of
the body) split into paragraphs, EACH WITH ITS OWN PAGE, into pieces of
reasonably consistent size. The target is ~1200 characters with a small
overlap between chunks so that context is not lost at a chunk boundary.

Why paragraphs with pages rather than a plain string? Because it is HERE,
at the level of chunking for RAG, that the decision is made whether to
drop the cover pages at the start and end of the PDF (see
filter_cover_pages) - and that needs the page of EVERY paragraph, not
just where the article starts and ends according to the contents. The
earlier stages (extract_blocks.py, assign_articles.py) discard no pages
and know nothing about covers: that is purely a "how do I do RAG"
decision, not a "how do I digitise a PDF" one.

Each chunk gets:
  - structured metadata (year, issue, title, author, ...) as separate
    fields, for filtering and for citing the source
  - "text" - the plain chunk text, for display and for handing to the LLM
    as context
  - "embedding_text" - the text with a metadata header prepended
    ("Magazine: ... Year: ... Article: ... Authors: ... Text: ..."). This
    is what goes to the embedding model, because a lone chunk without
    context ("Mixotrophs represent...") tells both the embedding model
    and the LLM less than one with a header.
  - a unique "chunk_id" ("2014-6-1-chunk-0")
  - "page_start"/"page_end" - the PDF pages the chunk really came from,
    not the pages of the whole article, which for a long article would be
    misleading on the later chunks

Usage:
    python -m magazine_rag.build_chunks --input output/corpus.json \\
        --output output/chunks.jsonl
"""
import argparse
import json
import re
from pathlib import Path

from magazine_rag import profiles
from magazine_rag.console import setup_console

TARGET_CHARS = 1200
# On the 512-token limit of standard BERT/XLM-R models (the whole E5
# family): it is handled in embed.py (enforce_max_length) by truncating
# only the specific text that exceeds it, rather than shrinking ALL
# chunks pre-emptively because of a rare exception. Measured on a real
# sample: TARGET_CHARS=1200 produced a longest embedding_text (chunk plus
# metadata header) of 1544 characters == 510 tokens under
# multilingual-e5-base - just under the line, but a longer title or
# author list in the header does occasionally push it over.
OVERLAP_CHARS = 150
MIN_CHUNK_CHARS = 200  # a shorter final remainder is appended to the previous chunk

# Czech capitals are in the class deliberately: this splits sentences in
# the source language of the corpus, it is not prose.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ])")


def filter_cover_pages(paragraphs, total_pdf_pages, skip_first, skip_last):
    """Drop paragraphs from the first and last N pages of the PDF - the
    cover, the advertisement for the next issue and so on.

    This is exactly where that choice belongs, at the level of chunking
    for RAG, and not earlier in the pipeline. The commonest effect: the
    last article in an issue tends to have a pdf_page_end running all the
    way to the final PDF page, because nothing follows it in the
    contents - and that final page is usually a standalone cover photo
    with no relation to the article.
    """
    if not paragraphs or not total_pdf_pages:
        return paragraphs
    lo, hi = skip_first, total_pdf_pages - skip_last
    return [p for p in paragraphs if lo < p["page"] <= hi]


def split_oversized_paragraph(page: int, text: str, limit: int):
    """Split a paragraph that is longer than the limit on its own.

    That happens with a long quotation, but also with a table or a data
    listing containing NOT ONE full stop. Sentences are tried first. If
    even a single "sentence" - which is the entire text when there is no
    punctuation to split on - is still over the limit, it is split hard
    on word boundaries; otherwise one enormous piece would stay in the
    chunk at perhaps four times the limit. That case really occurred, in
    a block of table data with no full stops. All resulting pieces keep
    the page of the original paragraph.
    """
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
    """items: a list of (page, text) -> {"page_start", "page_end", "text"}."""
    pages = [p for p, _ in items]
    text = "\n\n".join(t for _, t in items)
    return {"page_start": min(pages), "page_end": max(pages), "text": text}


def chunk_paragraphs(paragraphs, target_chars=TARGET_CHARS,
                     overlap_chars=OVERLAP_CHARS):
    """paragraphs: a list of {"page": int, "text": str} - an article's
    full_text_paragraphs, optionally already stripped of cover pages.
    Returns a list of {"page_start", "page_end", "text"}."""
    expanded = []
    for p in paragraphs:
        expanded.extend(split_oversized_paragraph(p["page"], p["text"], target_chars))

    chunks = []
    current = []  # list of (page, text)
    current_len = 0
    for page, text in expanded:
        if current and current_len + len(text) + 2 > target_chars:
            chunks.append(_finalize_chunk(current))
            # overlap: take as many paragraphs off the end of the chunk
            # just closed as fit into overlap_chars
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
            # a short remainder, typically just the overlap, is better
            # appended to the previous chunk than left standing alone
            chunks[-1]["text"] += "\n\n" + last["text"]
            chunks[-1]["page_end"] = last["page_end"]
        else:
            chunks.append(last)
    return chunks


def build_embedding_text(article: dict, chunk_text: str, profile) -> str:
    """The text that goes to the embedding model: the chunk with a header
    saying where it came from. Both the template and its language come
    from the source profile."""
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
    ap = argparse.ArgumentParser(
        description="cut the corpus into chunks for embedding")
    ap.add_argument("--input", required=True, help="corpus.json (from run_all.py)")
    ap.add_argument("--output", required=True, help="output .jsonl")
    ap.add_argument("--skip-first", type=int, default=None,
                    help="how many pages at the start of an issue to ignore "
                         "(the cover); the default comes from the profile")
    ap.add_argument("--skip-last", type=int, default=None,
                    help="how many pages at the end of an issue to ignore "
                         "(the back cover); the default comes from the profile")
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
    print(f"{len(articles)} articles -> {len(all_chunks)} chunks "
          f"({n_body} body, {n_caption} caption) -> {args.output}")


if __name__ == "__main__":
    main()
