"""Stage 1: parse one issue's PDF into text blocks -> blocks.json

How it works:

  1. fitz (PyMuPDF) splits every page into "raw" blocks according to how
     the PDF is internally assembled (typically one InDesign text frame).
  2. A single column's text frame sometimes falls apart into TWO raw
     blocks even though visually it is one unbroken flow of text (a
     subsection heading immediately followed by its paragraph). The tell
     is that both blocks share almost the same x range - they are in the
     same column - and the gap between them is minimal, a few points,
     much smaller than the gap before a genuinely new visual block or
     heading. Such adjacent raw blocks are merged into one logical block.
  3. Images (raw block type 1) do not go into the output, but their IDs
     are "reserved" - which is why text blocks have gaps in their
     numbering in the JSON. That is harmless; it simply reflects the
     order in which the PDF blocks really followed each other.
  4. Every resulting block gets a type (title/heading/other/body/caption/
     annotation). This is the only step tied to a specific typesetting
     and it is entirely factored out into the source profile - see
     `magrag/profiles/` and `magrag/typography.py`.

Usage:
    python -m magrag.extract_blocks --profile ziva input.pdf blocks.json
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

from magrag import profiles
from magrag.console import setup_console
from magrag.typography import (
    DocumentStats,
    classify_block,
    looks_like_caption_lead,
)

# --- tuning constants -------------------------------------------------------
COLUMN_X_TOLERANCE = 4.0   # how far x0/x1 of two blocks may differ and still
                           # count as "the same column"
MERGE_Y_GAP_MAX = 6.0      # max gap (pt) between the end of one block and the
                           # start of the next for them to still be merged

# Justified text breaks words at the end of a line with a plain hyphen,
# e.g. "dlouhodo-" / "bé" - sometimes with a stray space around the hyphen
# thrown in by the justification: "nerovnoměr - ný". We want them glued
# back into "dlouhodobé"/"nerovnoměrný". This must not touch the en dash
# "–" (U+2013), which Czech typography uses for genuine breaks
# ("1990 – 2000"); only the plain ASCII hyphen is a line-wrap artefact.
HYPHEN_BREAK_RE = re.compile(r"(\w)\s*-\s*$")


def looks_like_word_break(before: str, after: str) -> bool:
    m = HYPHEN_BREAK_RE.search(before)
    if not m:
        return False
    after = after.lstrip()
    # A real line wrap continues the SAME word, so the next fragment
    # starts with a lowercase letter (a new sentence, heading or proper
    # noun would not).
    return bool(after) and after[0].isalpha() and after[0].islower()


def smart_join(parts):
    """Join text fragments - lines within a block, or blocks being merged -
    into one string, gluing back words split by a line-wrap hyphen instead
    of just inserting a space."""
    out = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if out and looks_like_word_break(out, part):
            out = HYPHEN_BREAK_RE.sub(r"\1", out)  # drop " - "/"-", keep the letter
            out += part
        elif out:
            out += " " + part
        else:
            out = part
    return out


def spans_of_lines(lines):
    """Return a list of (text, font, size) for every span in these lines."""
    out = []
    for line in lines:
        for span in line["spans"]:
            out.append((span["text"], span["font"], span["size"]))
    return out


def text_of_lines(lines):
    """Assemble the text of these lines, gluing back words split by a
    line-wrap hyphen (see smart_join / HYPHEN_BREAK_RE above)."""
    parts = []
    for line in lines:
        line_text = "".join(span["text"] for span in line["spans"])
        if line_text.strip():
            parts.append(line_text)
    return smart_join(parts)


def split_lines_by_style(lines, size_tol=3.0):
    """Split the lines of ONE raw fitz block into groups by font SIZE (not
    weight - see styles_compatible below).

    fitz can merge visually close but semantically UNRELATED text into one
    block: typically the end of one article or review immediately followed
    by the heading of the next. Found for real: "Kontaktní adresy autorů"
    at 15pt merged with the tail of the preceding review at ~9pt into a
    single fitz block, before our own between-block merging even got a
    look in. Without splitting, the longer fragment would outvote the
    dominant size (see dominant_font_and_size) and the heading would
    vanish under the "body" classification.
    """
    groups = []
    current = []
    current_size = None
    for line in lines:
        text = "".join(s["text"] for s in line["spans"]).strip()
        if not text:
            if current:
                current.append(line)
            continue
        _, size = dominant_font_and_size(
            [(s["text"], s["font"], s["size"]) for s in line["spans"]])
        if current_size is None:
            current = [line]
            current_size = size
        elif abs(size - current_size) <= size_tol:
            current.append(line)
        else:
            groups.append(current)
            current = [line]
            current_size = size
    if current:
        groups.append(current)
    return groups


def dominant_font_and_size(spans):
    """font = the most common family by character count, so that a short
    bold subheading does not outvote the font of a whole paragraph;
    size = the size of the FIRST span, because headings tend to run a
    fraction larger than body text and that detail is what the original
    JSON records (see the comment in assign_articles.py)."""
    char_counts = Counter()
    for text, font, size in spans:
        char_counts[font] += len(text)
    font = char_counts.most_common(1)[0][0] if char_counts else ""
    size = spans[0][2] if spans else 0.0
    return font, size


def same_column(b1, b2):
    x0a, _, x1a, _ = b1["bbox"]
    x0b, _, x1b, _ = b2["bbox"]
    return (abs(x0a - x0b) <= COLUMN_X_TOLERANCE and
            abs(x1a - x1b) <= COLUMN_X_TOLERANCE * 3)  # x1 varies more (justification)


def styles_compatible(prev_entry, new_spans, size_tol=3.0):
    """Refuse to merge two blocks whose font SIZE differs markedly.

    Typically the end of one article or review and the heading of the one
    right after it - found for real: "Kontaktní adresy autorů" at size 15
    merged with the tail of the preceding review at ~9. Weight itself is
    deliberately NOT compared: figure captions routinely carry a bold
    figure number next to normal text of the SAME size, and it is one and
    the same caption, not two blocks - checking weight alone would tear
    that apart needlessly and wrongly.
    """
    _, prev_size = dominant_font_and_size(prev_entry["spans"])
    _, new_size = dominant_font_and_size(new_spans)
    return abs(prev_size - new_size) <= size_tol


def merge_page_entries(page):
    """Raw fitz blocks of one page -> logical blocks (see points 2 and 3
    in the module docstring). Returns a list of
    `{"kind": "text"|"image", ...}` entries, keeping the slots where
    images were so that block numbering stays faithful."""
    raw = page.get_text("dict")["blocks"]
    merged = []
    prev_text_entry = None

    for raw_block in raw:
        if raw_block["type"] != 0:  # image or drawing -> just reserve a slot
            merged.append({"kind": "image"})
            prev_text_entry = None  # never merge across an image
            continue

        # fitz can merge stylistically unrelated text into one raw block
        # (see styles_compatible above), so split it into internally
        # homogeneous groups of lines first and treat EACH as its own
        # unit, merging with the neighbouring block included.
        for lines in split_lines_by_style(raw_block["lines"]):
            spans = spans_of_lines(lines)
            text = text_of_lines(lines)
            if not text.strip():
                merged.append({"kind": "image"})  # empty text frame
                prev_text_entry = None
                continue

            bbox = [
                min(l["bbox"][0] for l in lines),
                min(l["bbox"][1] for l in lines),
                max(l["bbox"][2] for l in lines),
                max(l["bbox"][3] for l in lines),
            ]
            entry = {"kind": "text", "bbox": bbox, "text": text, "spans": spans}

            if (prev_text_entry is not None
                    and same_column(prev_text_entry, entry)
                    and bbox[1] - prev_text_entry["bbox"][3] <= MERGE_Y_GAP_MAX
                    and styles_compatible(prev_text_entry, spans)):
                # merge into the previous block instead of starting a new one
                prev_text_entry["text"] = smart_join([prev_text_entry["text"], text])
                prev_text_entry["spans"].extend(spans)
                prev_text_entry["bbox"][2] = max(prev_text_entry["bbox"][2], bbox[2])
                prev_text_entry["bbox"][3] = bbox[3]
            else:
                merged.append(entry)
                prev_text_entry = entry

    return merged


def blocks_from_entries(entries, page_num, profile, stats):
    out = []
    for block_id, entry in enumerate(entries):
        if entry["kind"] != "text":
            continue
        font, size = dominant_font_and_size(entry["spans"])
        btype = classify_block(profile, entry["text"], font, size, stats)

        if btype == "body" and looks_like_caption_lead(entry["text"]):
            btype = "caption"  # a figure caption set in the body typeface

        out.append({
            "page": page_num,
            "block_id": block_id,
            "type": btype,
            "text": entry["text"].strip(),
            "font": font,
            "font_size": round(size, 2),
            "bbox": [round(v, 1) for v in entry["bbox"]],
        })
    return out


def extract_pdf(pdf_path, profile):
    """PDF -> a list of blocks. Under an adaptive profile the document is
    walked twice: once for the typography statistics, once for the
    classification itself."""
    doc = fitz.open(pdf_path)
    # 1-indexed, as in the original JSON
    pages = [(i, merge_page_entries(page)) for i, page in enumerate(doc, start=1)]

    stats = None
    if profile.adaptive:
        stats = DocumentStats.from_spans(
            span
            for _, entries in pages
            for entry in entries if entry["kind"] == "text"
            for span in entry["spans"]
        )

    blocks = []
    for page_num, entries in pages:
        blocks.extend(blocks_from_entries(entries, page_num, profile, stats))
    return blocks


def main():
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pdf", help="input PDF of a single issue")
    ap.add_argument("out", help="output blocks.json")
    profiles.add_profile_argument(ap)
    args = ap.parse_args()

    profile = profiles.get(args.profile)
    blocks = extract_pdf(args.pdf, profile)
    Path(args.out).write_text(
        json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = Counter(b["type"] for b in blocks)
    print(f"{args.pdf}: {len(blocks)} blocks -> {args.out}")
    print("  " + ", ".join(f"{t}={n}" for t, n in counts.most_common()))


if __name__ == "__main__":
    main()
