"""Source profile - everything that makes one magazine differ from another.

The pipeline splits into two parts that behave very differently when you
move it to another magazine:

* **The generic part** (mapping printed pages to PDF pages, splitting
  articles by Y coordinate, rejoining hyphenated words, chunking,
  embedding, the vector store, retrieval) does not depend on the
  typesetting at all - it works with the structure
  `{page, type, font, bbox, text}` and nothing else.

* **The typography-specific part** is wired to one magazine completely:
  what the fonts are called, which point size means a title, which page
  carries the contents, what the running footer looks like.

This module is the boundary between them. Typography-specific decisions
live here as **data**, not as an `if` in the middle of a parser, so adding
a magazine means writing one `SourceProfile` instead of touching five
scripts.

Do not write a new profile from an armchair - run `tools/inspect_fonts.py`,
which prints which fonts and sizes a given PDF actually uses and in what
volume. The procedure is in the README under "Adding a new magazine".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass(frozen=True)
class SizeRule:
    """One "font size -> block type" rule within a single font family.

    `size_min`/`size_max` are both **inclusive**. Rules are evaluated in
    order and the first match wins, so with overlapping ranges it is the
    order that decides, not the width of the interval.

    `bold=None` means "weight does not matter".
    """
    block_type: str
    size_min: float = 0.0
    size_max: float = float("inf")
    bold: Optional[bool] = None

    def matches(self, size: float, is_bold: bool) -> bool:
        if self.bold is not None and self.bold != is_bold:
            return False
        return self.size_min <= size <= self.size_max


@dataclass(frozen=True)
class RelativeRule:
    """An adaptive-profile rule: font size **relative** to the document's
    body text (1.0 = the same size as the article text).

    `same_family=True` means "the same font family as body text", `False`
    "a different family", `None` "does not matter". The family is the font
    name without the subset prefix and weight, see
    `typography.font_family()`.
    """
    block_type: str
    ratio_min: float = 0.0
    ratio_max: float = float("inf")
    bold: Optional[bool] = None
    same_family: Optional[bool] = None

    def matches(self, ratio: float, is_bold: bool, same_family: bool) -> bool:
        if self.bold is not None and self.bold != is_bold:
            return False
        if self.same_family is not None and self.same_family != same_family:
            return False
        return self.ratio_min <= ratio <= self.ratio_max


@dataclass(frozen=True)
class FontFamily:
    """A group of fonts sharing one set of rules.

    Families are tried in order and the first whose prefix matches the
    font name wins. When no rule inside the winning family matches, **that
    family's** `fallback` applies - it does not continue into the next
    family. That matters: in a serif setting an unknown size is almost
    certainly body text, whereas in a sans setting it is almost certainly
    a figure caption.
    """
    name: str
    prefixes: tuple[str, ...]
    rules: tuple[SizeRule, ...] = ()
    fallback: str = "body"
    detect_annotations: bool = False

    def matches_font(self, font: str) -> bool:
        return any(font.startswith(p) for p in self.prefixes)


@dataclass(frozen=True)
class SourceProfile:
    """A complete description of one magazine for the whole pipeline."""

    # --- identity -----------------------------------------------------------
    key: str
    journal_name: str
    language: str = "cs"

    # --- input files --------------------------------------------------------
    # How the PDFs are named, and where the year and issue sit in the name.
    # Needs at most two groups: (year, issue). run_all.py names the output
    # directories from them.
    filename_pattern: str = r"(\d{4})-(\d+)\.pdf$"

    # --- contents page ------------------------------------------------------
    # Zero-indexed PDF pages carrying the printed contents of the issue.
    toc_page_indices: tuple[int, ...] = (2,)
    # The font of a page number in the contents (font-name prefix plus a
    # weight requirement). `toc_page_number_bold=None` means "any weight".
    toc_page_number_prefixes: tuple[str, ...] = ()
    toc_page_number_bold: Optional[bool] = None

    # Derive the styles of contents entries from the page itself rather
    # than from the settings above. Turn this on for a magazine whose font
    # names are unknown, or change over the archive (The MagPi redesigned
    # between issues 150 and 152, changing fonts and the page-number
    # format alike). See create_toc.detect_entry_styles() for details.
    toc_adaptive_styles: bool = False

    # Does the contents page list authors? Živa does (and separates them
    # by colour), The MagPi does not. When it does not, splitting the
    # title is pointless - it would produce part of the title, or the
    # section name, as the author.
    toc_has_authors: bool = True
    # Strings that appear in the contents as imprint or copyright. `drop`
    # truncates the title from that point on (the rest of the line is
    # imprint glued onto the last entry); `skip_span` discards the whole
    # span (a standalone imprint line between entries). Two separate lists
    # deliberately: truncation must also work on strings that never stand
    # as a span of their own.
    toc_drop_markers: tuple[str, ...] = ()
    toc_skip_span_markers: tuple[str, ...] = ()

    # --- running header / footer -------------------------------------------
    # How the footer is recognised. Two strategies, because one is not
    # enough:
    #
    #   "keyword"  - by content: the footer carries the magazine's name or
    #                its website ("živa 6/2014", "ziva.avcr.cz"). Precise,
    #                but somebody has to find out what the footer says for
    #                each magazine.
    #   "position" - by position: small text near the top or bottom edge
    #                of the page whose content is essentially just a
    #                number. Works on an unknown magazine with no
    #                calibration, at the cost of the occasional false
    #                positive (a number in the corner of a chart).
    #   "none"     - the magazine has no running footer; printed-page
    #                mapping is not used.
    footer_detection: str = "keyword"
    # A regex that recognises the footer in arbitrary text (extract_blocks
    # never classifies such a block as a heading).
    footer_pattern: str = ""
    # A stricter test for build_page_map/assign_articles: the footer is
    # always in the same small sans face, otherwise it would be confused
    # with body text.
    footer_font_prefixes: tuple[str, ...] = ()
    footer_max_size: float = 9.0
    footer_keywords: tuple[str, ...] = ()
    # Tokens inside the footer that look like a number but are not one.
    footer_skip_tokens: tuple[str, ...] = ()
    # For "position" only: how high or low on the page the footer is
    # sought, as a fraction of page height, and how many words it may have.
    footer_zone: float = 0.90
    header_zone: float = 0.08
    footer_max_tokens: int = 5

    # --- boilerplate repeated on every page ---------------------------------
    # Each inner tuple is an AND: a block is boilerplate when it contains
    # ALL of its strings. The outer tuple is an OR.
    junk_marker_sets: tuple[tuple[str, ...], ...] = ()

    # --- typography ---------------------------------------------------------
    families: tuple[FontFamily, ...] = ()
    default_block_type: str = "body"

    # Adaptive mode: sizes are not taken absolutely from
    # `families[*].rules` but relative to the document's body text (see
    # profiles/adaptive.py). `families` then merely carries the fallbacks:
    # [0] = the body-text family, [1] = everything else.
    adaptive: bool = False
    relative_rules: tuple[RelativeRule, ...] = ()

    # --- chunking -----------------------------------------------------------
    # How many pages at the start and end of an issue are covers and ads,
    # i.e. content that should not enter the RAG corpus.
    skip_first_pages: int = 2
    skip_last_pages: int = 2

    # The header baked into the text that goes to the embedding model
    # ("contextual chunking"). A lone chunk without context tells both the
    # model and the LLM less, so the article it came from is prepended.
    # The header's language follows the corpus, not the pipeline - which
    # is why it lives in the profile.
    chunk_header_template: str = (
        "Magazine: {journal}\n"
        "Year: {year}\n"
        "Issue: {issue}\n"
        "Article: {title}\n"
        "Authors: {author}\n"
        "Text: {text}"
    )
    unknown_author_label: str = "unknown"

    # --- answering ----------------------------------------------------------
    # The citation printed above each block of retrieved context, and
    # handed to the LLM so it can attribute what it says. Like the chunk
    # header, its language follows the corpus rather than the code.
    # The pages are PDF pages, not printed page numbers (see
    # assemble_context.format_citation), so the labels say so.
    citation_template: str = (
        "{title} ({journal} {year}/{issue}, {pages}), authors: {author}")
    page_single_label: str = "PDF p. {page}"
    page_range_label: str = "PDF pp. {start}-{end}"
    # Heading of the figure captions appended to a context block (the
    # body text and the captions are kept apart until then, see
    # assemble_context).
    captions_label: str = "Figure captions:"

    system_prompt_file: str = ""

    # --- derived (cache of compiled regexes) --------------------------------
    _compiled: dict = field(default_factory=dict, repr=False, compare=False)

    # ------------------------------------------------------------------ API
    @property
    def filename_re(self) -> re.Pattern:
        return self._cached("filename", self.filename_pattern, re.IGNORECASE)

    @property
    def footer_re(self) -> Optional[re.Pattern]:
        if not self.footer_pattern:
            return None
        return self._cached("footer", self.footer_pattern, re.IGNORECASE)

    def _cached(self, key: str, pattern: str, flags: int = 0) -> re.Pattern:
        if key not in self._compiled:
            self._compiled[key] = re.compile(pattern, flags)
        return self._compiled[key]

    def is_footer_text(self, text: str) -> bool:
        """Does this text look like a running header/footer? (content only)"""
        rx = self.footer_re
        return bool(rx.search(text)) if rx else False

    def is_footer_block(self, font: str, size: float, text: str,
                        bbox=None, page_height: float = 0.0) -> bool:
        """A stricter test than `is_footer_text`: besides the content it
        weighs the font, the size and (under the "position" strategy) the
        place on the page.

        Used by build_page_map, which pulls the page number out of the
        footer, and by assign_articles, which throws the footer out of the
        article text. The font check matters under "keyword": the keyword
        itself legitimately occurs in body text too ("the magazine Živa is
        published..."), and that would throw the page numbering off.
        """
        if self.footer_detection == "none":
            return False
        if size >= self.footer_max_size:
            return False
        if self.footer_font_prefixes and not any(
                font.startswith(p) for p in self.footer_font_prefixes):
            return False

        if self.footer_detection == "position":
            return self._in_footer_zone(bbox, page_height) and \
                len(text.split()) <= self.footer_max_tokens

        if not self.footer_keywords:
            # "keyword" with no keyword at all would declare every small
            # piece of text on the page a footer - worse than detecting
            # nothing.
            return False
        return self._has_footer_keyword(text)

    def _in_footer_zone(self, bbox, page_height: float) -> bool:
        if not bbox or not page_height:
            return False
        y0, y1 = bbox[1], bbox[3]
        return y0 >= page_height * self.footer_zone or \
            y1 <= page_height * self.header_zone

    def _has_footer_keyword(self, text: str) -> bool:
        if not self.footer_keywords:
            return True
        low = text.lower()
        return any(k.lower() in low for k in self.footer_keywords)

    def is_junk_text(self, text: str) -> bool:
        return any(all(m in text for m in markers)
                   for markers in self.junk_marker_sets)

    def family_for(self, font: str) -> Optional[FontFamily]:
        for fam in self.families:
            if fam.matches_font(font):
                return fam
        return None

    def is_toc_page_number_font(self, font: str) -> bool:
        if self.toc_page_number_prefixes and not any(
                font.startswith(p) for p in self.toc_page_number_prefixes):
            return False
        if self.toc_page_number_bold is None:
            return True
        return ("Bold" in font) == self.toc_page_number_bold

    def system_prompt(self) -> str:
        if not self.system_prompt_file:
            raise ValueError(
                f"profile {self.key!r} has no system_prompt_file set")
        return (PROMPTS_DIR / self.system_prompt_file).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Profile registry
# --------------------------------------------------------------------------

def _registry() -> dict[str, SourceProfile]:
    from . import adaptive, magpi, ziva
    return {p.key: p for p in (ziva.PROFILE, magpi.PROFILE, adaptive.PROFILE)}


def available() -> list[str]:
    return sorted(_registry())


def get(key: str) -> SourceProfile:
    reg = _registry()
    if key not in reg:
        raise KeyError(
            f"unknown profile {key!r}; available: {', '.join(sorted(reg))}")
    return reg[key]


def add_profile_argument(parser, default: str = "ziva"):
    """The shared `--profile` switch for every pipeline entry point."""
    parser.add_argument(
        "--profile", default=default, choices=available(),
        help=f"source magazine profile (default: {default})")
    return parser
