"""Stage 0: pull the issue's contents (who / what / on which printed page)
off the contents page -> toc.json

How it works: the printed page number is set in a different weight from
the rest of the line (MeliorCE Bold in Živa) and acts as the separator
between entries; inside one entry, the COLOUR of the first span separates
the title from the author. Both are properties of a specific typesetting,
so both come from the source profile.

If several items are printed under one page number, separated by a
semicolon ("24. ročník...; Zaujalo nás: ..."), they are split into
separate TOC records sharing a printed_page - assign_articles.py then
tells them apart by their own headings in the text (see
find_heading_positions there).

    python -m magrag.create_toc --profile ziva input.pdf toc.json
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import fitz

from magrag import profiles
from magrag.console import setup_console
from magrag.typography import font_style_key

# How many leading pages are searched when the profile does not say where
# the contents are (empty toc_page_indices) - see find_toc_pages().
TOC_SEARCH_PAGES = 8
TOC_MIN_ENTRIES = 5
# How dense another page must be relative to the best one to also count
# as part of the contents (they are usually spread across a spread).
TOC_PEER_RATIO = 0.4
# The same for "number style + title style" pairs inside the contents:
# how frequent a pair must be relative to the most frequent one to count
# as real entries rather than a decorative callout or a section heading.
TOC_STYLE_PEER_RATIO = 0.4


def strip_control_chars(text: str) -> str:
    """Replace control characters with a space.

    They arrive from decorative glyphs set in a symbol font - a real case
    (The MagPi): the bullet before every contents entry comes out of the
    PDF as U+0007. It has no business in an article title and it breaks
    the comparison between a contents title and the heading found in the
    body of the issue (texts_match in assign_articles). This compares
    characters rather than using a regex with a range: a range is hard to
    read and easy to get wrong.
    """
    return "".join(" " if ch < " " or ch == "\x7f" else ch for ch in text)


def clean(text):
    return re.sub(r"\s+", " ", strip_control_chars(text)).strip()


def remove_footer(text, profile):
    """Truncate a title or author from the point where the imprint begins.

    The imprint is often set as a continuation of the last contents line,
    so without this it ends up inside the last entry's title.
    """
    for marker in profile.toc_drop_markers:
        if marker in text:
            text = text.split(marker)[0]
    return text.strip(" ,;")


def is_roman(text):
    # Case-insensitive: a typesetting slip occasionally drops a lowercase
    # letter into the middle of a Roman numeral ("CXLVIiI") - see the same
    # fix in build_page_map.py.
    return bool(re.fullmatch(r"[IVXLCDM]+", text.strip(), re.IGNORECASE))


def is_page_number(text):
    text = text.strip()
    return text.isdigit() or is_roman(text)


# An abbreviated page range, typical of small back-matter items that fit
# in under two pages: "XXXI–II" (= XXXI to XXXII), or in Arabic "285-6"
# (= 285 to 286). Without this, is_page_number() does not match such a
# token at all (it contains a hyphen or en dash), so it is not recognised
# as the start of a new contents record and its title is wrongly glued to
# the previous one.
RANGE_RE = re.compile(r"^([IVXLCDM]+|\d+)\s*[-–]\s*([IVXLCDM]+|\d+)$", re.IGNORECASE)


def page_span_value(text):
    """Return the real printed-page value for a token that looks like a
    page number, abbreviated ranges included (see RANGE_RE above).

    Only the START of a range is returned - that is all the pipeline needs
    downstream, because an article's end is derived from the start of the
    NEXT contents record, not from the end of its own range.

    The text is cleaned first: in MagPi 150 an entry's number arrives with
    a tab and a decorative bullet (U+0007) glued into the same span.
    Without cleaning, such a number is not recognised at all and the whole
    contents entry is lost - without a trace, because nothing crashes.
    """
    text = clean(text)
    if is_page_number(text):
        return text.upper() if is_roman(text) else text
    m = RANGE_RE.match(text)
    if m:
        value = m.group(1)
        return value.upper() if is_roman(value) else value
    return None


def is_page_span(span, profile):
    """Is this span a printed page number, i.e. the start of a new entry?

    It is not enough that the text looks like a number - titles contain
    numbers routinely ("24. ročník", "Rok 1968"). What decides is the
    font weight reserved for page numbers in the contents (see
    toc_page_number_prefixes in the profile).
    """
    txt = span["text"].strip()
    return (page_span_value(txt) is not None
            and profile.is_toc_page_number_font(span["font"]))


def split_title_author(spans):
    spans = [s for s in spans if s["text"].strip()]
    if not spans:
        return None, None
    title_color = spans[0]["color"]
    title, author, author_started = [], [], False
    for span in spans:
        txt = span["text"].strip()
        if not txt:
            continue
        if not author_started and span["color"] == title_color:
            title.append(txt)
        else:
            author_started = True
            author.append(txt)
    return clean(" ".join(title)), (clean(" ".join(author)) if author else None)


def count_toc_entries(page, profile):
    """How many "entry starts" - page numbers in the reserved weight -
    this page carries. Used only to detect which page the contents are on."""
    n = 0
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", ()):
            for span in line["spans"]:
                if span["text"].strip() and is_page_span(span, profile):
                    n += 1
    return n


def find_toc_pages(doc, profile):
    """Find the pages holding the issue's contents when the profile does
    not pin them down.

    For a known magazine the contents are always on the same physical page
    and there is nothing to guess (`toc_page_indices` in the profile). For
    an unknown one nobody knows in advance, so the front of the issue is
    searched for the densest cluster of page numbers - which is what a
    contents page is by definition.

    **There is more than one page.** The contents are usually spread over
    a double page, and in a thicker issue across three pages separated by
    advertising (in reality: The MagPi has its contents on pages 5, 6 and
    8). Taking only the best one loses two thirds of the issue - and it
    does not register as an error, because the rest of the pipeline
    dutifully processes whatever it was given.

    Two thresholds: the absolute `TOC_MIN_ENTRIES` tells the contents
    apart from a page where a few numbers happened to meet, and the
    relative `TOC_PEER_RATIO` against the best page picks up its
    counterparts but not a page with one stray number.
    """
    counts = {i: count_toc_entries(doc[i], profile)
              for i in range(min(TOC_SEARCH_PAGES, doc.page_count))}
    best = max(counts.values(), default=0)
    if best < TOC_MIN_ENTRIES:
        return ()
    threshold = max(TOC_MIN_ENTRIES, best * TOC_PEER_RATIO)
    return tuple(i for i in sorted(counts) if counts[i] >= threshold)


def _starts_entry(span, txt, style, profile, number_styles):
    """Is this span a page number that begins a new contents entry?

    Two routes depending on the profile: either the font written into the
    profile decides (Živa), or the style derived from the page itself
    (see detect_entry_styles).
    """
    if page_span_value(txt) is None:
        return False
    if number_styles is not None:
        return style in number_styles
    return profile.is_toc_page_number_font(span["font"])


def collect_spans(doc, page_indices):
    """Every non-empty span from the given pages, in typesetting order."""
    out = []
    for page_index in page_indices:
        for block in doc[page_index].get_text("dict")["blocks"]:
            for line in block.get("lines", ()):
                for span in line["spans"]:
                    if span["text"].strip():
                        out.append(span)
    return out


def detect_entry_styles(spans, profile):
    """Derive from the contents page what an entry's number looks like and
    what its text looks like.

    Why this cannot be written into the profile the way it is for Živa:
    The MagPi redesigned between issues 150 and 152. Fonts changed
    (Rajdhani/RobotoSlab to Roboto*), so did sizes and the number format
    ("22" to "032"). One set of hard-coded values would hold for part of
    the archive only and, worse, would quietly produce nonsense on the
    rest instead of failing.

    What does hold across both designs: **a contents page carries three
    different kinds of number and only one of them is an entry.** Besides
    the real page numbers there are decorative callouts, reversed out in
    white in slightly larger type, and the number of the contents page
    itself in the running footer. They are told apart by the fact that
    the real entries are the most numerous - the contents are a list of
    them, whereas there are only a handful of callouts.

    Returns `(a set of number styles, a set of text styles)`, where a
    style is `(font family, size)`. Text styles are sought separately
    because they are usually different from the number style (in MagPi
    150 the number is Rajdhani and the title RobotoSlab), and they are
    taken from the first span after each accepted number, which is always
    the title. That is what filters out section headings, the imprint and
    the captions under decorative callouts.
    """
    pairs = _number_text_pairs(spans)
    if not pairs:
        return None, None

    # The most frequent pair is by definition the real one: the contents
    # are a list, so the same "number + title" pair repeats for every
    # entry. There can be several equally valid pairs, though - Živa uses
    # two sizes of number (9 and 10pt) and both are real - so every
    # sufficiently frequent one is taken. The gap on real data is
    # comfortable: in Živa the second real pair sits at 79% of the first,
    # in The MagPi the first decorative callout at 14%.
    best = pairs.most_common(1)[0][1]
    threshold = best * TOC_STYLE_PEER_RATIO
    accepted = [(num, text) for (num, text), n in pairs.items() if n >= threshold]
    return {num for num, _ in accepted}, {text for _, text in accepted}


def _number_text_pairs(spans):
    """Count (number style, style of the text right after it) pairs.

    The text right after a number is always the entry's title, which is
    why styles are found through this pair rather than through the
    frequency of styles on their own. Taking just the most frequent
    number style would lose half the entries in a magazine that uses two
    sizes of number in its contents (Živa).
    """
    pairs = Counter()
    pending = None
    for span in spans:
        style = font_style_key(span["font"], span["size"])
        if page_span_value(span["text"]) is not None:
            pending = style          # waiting for its title
        elif pending is not None:
            pairs[(pending, style)] += 1
            pending = None
    return pairs


def build_toc(pdf_path, profile, toc_page_indices=None):
    """toc_page_indices are zero-indexed PDF pages carrying the printed
    contents. When not given they come from the profile; when the profile
    does not pin them either, detection is attempted (see find_toc_pages)."""
    doc = fitz.open(pdf_path)
    if toc_page_indices is None:
        toc_page_indices = profile.toc_page_indices or find_toc_pages(doc, profile)

    spans = collect_spans(doc, toc_page_indices)

    number_styles = text_styles = None
    if profile.toc_adaptive_styles:
        number_styles, text_styles = detect_entry_styles(spans, profile)

    items_raw, current = [], None
    for span in spans:
        txt = clean(span["text"])
        if not txt:
            continue
        style = font_style_key(span["font"], span["size"])

        if _starts_entry(span, txt, style, profile, number_styles):
            if current:
                items_raw.append(current)
            current = {"page": page_span_value(txt), "spans": []}
            continue

        if current is None:
            continue
        if text_styles is not None and style not in text_styles:
            continue  # section heading, imprint, caption of a callout
        if any(m in txt for m in profile.toc_skip_span_markers):
            continue
        current["spans"].append({
            "text": txt, "color": span["color"],
            "font": span["font"], "size": span["size"],
        })
    if current:
        items_raw.append(current)

    toc = []
    for item in items_raw:
        if profile.toc_has_authors:
            title, author = split_title_author(item["spans"])
        else:
            title = clean(" ".join(s["text"] for s in item["spans"]))
            author = None
        if title:
            # A magazine sometimes prints several short items on one page
            # under ONE shared page number in the contents, separated by
            # a semicolon in the title ("24. ročník...; Zaujalo nás:
            # ..."). Split them into separate records sharing a
            # printed_page - assign_articles.py then tells them apart
            # inside that shared page by their own headings (see
            # find_heading_positions).
            for part in title.split(";"):
                part = part.strip()
                if part:
                    toc.append({
                        "printed_page": item["page"],
                        "title": remove_footer(part, profile),
                        "author": remove_footer(author, profile) if author else None,
                    })
    return toc


def main():
    setup_console()
    ap = argparse.ArgumentParser(
        description="pull the issue's contents out of a PDF")
    ap.add_argument("pdf", help="input PDF of a single issue")
    ap.add_argument("out", help="output toc.json")
    ap.add_argument("--toc-page", type=int, action="append", dest="toc_pages",
                    help="zero-indexed PDF page holding the contents; may be "
                         "given more than once. Without it the profile is "
                         "used, or failing that automatic detection.")
    profiles.add_profile_argument(ap)
    args = ap.parse_args()

    profile = profiles.get(args.profile)
    toc = build_toc(args.pdf, profile,
                    tuple(args.toc_pages) if args.toc_pages else None)
    Path(args.out).write_text(
        json.dumps(toc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.pdf}: {len(toc)} contents entries -> {args.out}")


if __name__ == "__main__":
    main()
