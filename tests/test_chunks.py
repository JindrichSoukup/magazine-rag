"""Chunking for embedding, and joining text across a line break.

Both are generic parts of the pipeline and do not depend on a specific
magazine. The tests describe mainly the edge cases that turned out to be
bugs on a real archive, because those are the ones a later refactor
breaks most easily.

The Czech strings below are test data: the hyphenation and sentence
rules being tested are Czech ones, so the fixtures have to be Czech.
"""
import pytest

from magrag.build_chunks import (
    build_chunks_for_article,
    build_embedding_text,
    chunk_paragraphs,
    filter_cover_pages,
    split_oversized_paragraph,
)
from magrag.extract_blocks import smart_join
from magrag.profiles import get

ZIVA = get("ziva")
MAGPI = get("magpi")


# --- gluing back words broken by a hyphen ----------------------------------

def test_hyphen_break_is_glued_back():
    assert smart_join(["dlouhodo-", "bé řešení"]) == "dlouhodobé řešení"


def test_hyphen_break_survives_stray_spaces_from_justification():
    assert smart_join(["nerovnoměr - ", "ný"]) == "nerovnoměrný"


def test_en_dash_is_not_a_line_break():
    """In Czech typography an en dash is a genuine break, not a wrap
    artefact - gluing "1990 –" to "2000" would corrupt the text."""
    assert smart_join(["v letech 1990 –", "2000"]) == "v letech 1990 – 2000"


def test_hyphen_before_a_capital_is_a_real_hyphen():
    """The continuation of a broken word starts with a lowercase letter.
    A capital means a proper noun or a heading, not a continuation."""
    assert smart_join(["Rakousko-", "Uhersko"]) == "Rakousko- Uhersko"


def test_smart_join_skips_empty_parts():
    assert smart_join(["a", "", "   ", "b"]) == "a b"


# --- trimming the covers ---------------------------------------------------

def _paras(*pages):
    return [{"page": p, "text": f"paragraph on page {p}"} for p in pages]


def test_cover_pages_are_dropped_from_both_ends():
    kept = filter_cover_pages(_paras(1, 2, 3, 10, 19, 20), total_pdf_pages=20,
                              skip_first=2, skip_last=2)
    assert [p["page"] for p in kept] == [3, 10]


def test_cover_filter_is_a_no_op_without_a_known_page_count():
    """Without the total page count there is no way to tell which pages
    are the last ones - in that case dropping nothing is safer than
    guessing."""
    paragraphs = _paras(1, 2, 3)
    assert filter_cover_pages(paragraphs, None, 2, 2) == paragraphs


def test_cover_filter_respects_the_profile_defaults():
    """How many pages are cover is a property of the magazine, not of
    the pipeline."""
    assert ZIVA.skip_first_pages == 2
    assert MAGPI.skip_first_pages == 2


# --- splitting oversized paragraphs ----------------------------------------

def test_oversized_paragraph_is_split_on_sentences():
    text = " ".join(["Toto je věta číslo %d." % i for i in range(60)])
    parts = split_oversized_paragraph(5, text, limit=300)
    assert len(parts) > 1
    assert all(page == 5 for page, _ in parts)
    joined = "".join(t for _, t in parts).replace(" ", "")
    assert joined == text.replace(" ", ""), "splitting must not lose text"


def test_paragraph_without_any_punctuation_still_gets_split():
    """A real bug: a listing of table data has not one full stop, so it
    used to be returned whole - up to four times the limit - and then
    overflowed the embedding model's 512-token limit."""
    text = " ".join(str(i) for i in range(500))
    parts = split_oversized_paragraph(1, text, limit=200)
    assert len(parts) > 1
    assert all(len(t) <= 400 for _, t in parts)


def test_short_paragraph_is_left_alone():
    parts = split_oversized_paragraph(1, "A short sentence.", limit=1000)
    assert parts == [(1, "A short sentence.")]


# --- assembling chunks -----------------------------------------------------

def test_chunks_carry_the_page_range_they_came_from():
    paragraphs = [{"page": p, "text": "x" * 700} for p in (4, 5, 6)]
    chunks = chunk_paragraphs(paragraphs, target_chars=1200, overlap_chars=100)
    assert chunks
    assert chunks[0]["page_start"] == 4
    assert chunks[-1]["page_end"] == 6


def test_chunking_keeps_all_the_text():
    paragraphs = [{"page": 1, "text": f"Paragraph {i}. " * 20} for i in range(10)]
    chunks = chunk_paragraphs(paragraphs, target_chars=600, overlap_chars=0)
    joined = " ".join(c["text"] for c in chunks)
    for i in range(10):
        assert f"Paragraph {i}." in joined


# --- the metadata header (contextual chunking) -----------------------------

ARTICLE = {
    "article_id": "2014-6-0", "year": "2014", "issue": "6",
    "title": "Article title", "author": "Jan Novák",
    "total_pdf_pages": 20,
    "full_text_paragraphs": [{"page": 5, "text": "Article body. " * 30}],
    "captions_paragraphs": [{"page": 5, "text": "1 Figure caption."}],
    "full_text": "Article body.",
}


def test_embedding_text_header_follows_the_profile_language():
    """Živa's corpus is Czech, so its header is Czech; The MagPi's is
    English. The header's language follows the corpus, not the code."""
    cs = build_embedding_text(ARTICLE, "text", ZIVA)
    en = build_embedding_text(ARTICLE, "text", MAGPI)
    assert cs.startswith("Časopis: Živa")
    assert en.startswith("Magazine: The MagPi")
    assert cs.endswith("Text: text") and en.endswith("Text: text")


def test_missing_author_uses_the_profile_label():
    article = dict(ARTICLE, author=None)
    assert "neuvedeno" in build_embedding_text(article, "t", ZIVA)
    assert "unknown" in build_embedding_text(article, "t", MAGPI)


def test_body_and_caption_chunks_are_kept_apart():
    """A figure caption sitting mid-column would otherwise cut a
    sentence in half in the running text - hence its own chunk type and
    its own id."""
    chunks = build_chunks_for_article(ARTICLE, ZIVA)
    types = {c["chunk_type"] for c in chunks}
    assert types == {"body", "caption"}
    assert all(c["chunk_id"].startswith("2014-6-0-") for c in chunks)


def test_chunk_ids_are_unique():
    chunks = build_chunks_for_article(ARTICLE, ZIVA)
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))


def test_explicit_skip_arguments_override_the_profile():
    article = dict(ARTICLE, full_text_paragraphs=[{"page": 1, "text": "cover"}])
    assert build_chunks_for_article(article, ZIVA) == \
        [c for c in build_chunks_for_article(article, ZIVA)
         if c["chunk_type"] != "body"]
    assert any(c["chunk_type"] == "body"
               for c in build_chunks_for_article(article, ZIVA, skip_first=0))
