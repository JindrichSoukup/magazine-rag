"""Assembling the LLM context from search hits, mainly the handling of
figure captions, which are kept apart from the body text in their own
chunks with their own numbering."""
from magazine_rag.assemble_context import (
    assemble_context,
    build_body_sequences,
    caption_span,
)
from magazine_rag.profiles import get

ZIVA = get("ziva")


def chunk(i, page, text, type_="body", article="a"):
    return {"article_id": article, "chunk_type": type_, "chunk_index": i,
            "page_start": page, "page_end": page, "text": text}


def hit(c, distance=0.1):
    meta = {k: c[k] for k in ("article_id", "chunk_type", "chunk_index",
                              "page_start", "page_end")}
    meta.update(year=2014, issue=6, title="T", author="A")
    return {"chunk_id": f"{c['chunk_type']}-{c['chunk_index']}", "meta": meta,
            "distance": distance, "text": c["text"]}


BODY = [chunk(0, 1, "body0"), chunk(1, 1, "body1"), chunk(2, 2, "body2"),
        chunk(3, 3, "body3"), chunk(4, 3, "body4"), chunk(5, 4, "body5")]
SEQ = build_body_sequences(BODY)


def test_caption_hit_gets_the_body_on_its_own_page_not_by_its_index():
    """caption-0 on page 3 is about body3/body4, not body chunks 0-1."""
    cap = chunk(0, 3, "Obr. 1 Samec v toku", "caption")
    blocks = assemble_context([hit(cap)], {}, SEQ, ZIVA)
    assert len(blocks) == 1
    text = blocks[0]["text"]
    assert "body0" not in text and "body1" not in text
    assert "body3" in text and "body4" in text
    assert text.endswith("Popisky obrázků:\n\nObr. 1 Samec v toku")


def test_caption_on_a_page_without_body_text_still_goes_in():
    cap = chunk(0, 9, "Obr. 2 Celostránková fotografie", "caption")
    blocks = assemble_context([hit(cap)], {}, SEQ, ZIVA)
    assert [b["text"] for b in blocks] == [
        "Popisky obrázků:\n\nObr. 2 Celostránková fotografie"]
    assert "9" in blocks[0]["citation"]


def test_caption_window_is_capped_like_a_body_window():
    many = build_body_sequences([chunk(i, 1, f"b{i}") for i in range(10)])
    lo, hi = caption_span({"page_start": 1, "page_end": 1}, many["a"], 1)
    assert hi - lo == 2


def test_caption_and_body_hit_in_one_window_are_merged():
    cap = chunk(0, 3, "Obr. 1", "caption")
    blocks = assemble_context([hit(BODY[3], 0.2), hit(cap, 0.1)], {}, SEQ, ZIVA,
                              promote_threshold=3)
    assert len(blocks) == 1
    assert blocks[0]["text"].count("Obr. 1") == 1
    assert blocks[0]["best_distance"] == 0.1


def test_promoted_article_carries_its_captions():
    corpus = {"a": {"article_id": "a", "pdf_page_start": 1, "pdf_page_end": 4,
                    "full_text": "Celý text.", "captions_text": "Obr. 1 X"}}
    hits = [hit(BODY[0]), hit(BODY[2]), hit(BODY[5])]
    blocks = assemble_context(hits, corpus, SEQ, ZIVA)
    assert blocks[0]["mode"] == "full_article"
    assert blocks[0]["text"] == "Celý text.\n\nPopisky obrázků:\n\nObr. 1 X"
