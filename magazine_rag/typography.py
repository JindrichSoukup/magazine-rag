"""Classifying a text block by type (title/heading/other/body/caption/
annotation) from its typography.

This is the only place in the pipeline that decides "what is that piece of
text, actually". Everything downstream works with the resulting type, not
with fonts. There are two routes to the decision and both go through
`classify_block()`:

* **a hand-written profile** - absolute rules ("MeliorCE Bold at 18pt and
  above is a title"): accurate, but valid for one specific magazine only;
* **an adaptive profile** - rules relative to the document's most
  voluminous typeface ("1.7x larger than body text is a title"): portable
  to an unknown magazine with no calibration.

See `profiles/ziva.py` and `profiles/adaptive.py`.

Note on the Czech literals below: they are *data about Czech text*, not
prose. They describe how the source magazine is typeset and must not be
translated.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

# --- editorial furniture stuck straight onto a figure or diagram ----------
# This is NOT a figure caption (running text explaining what can be seen).
# It is a graphic element the editor uses to label the panels of a
# composite figure (a, b, c, ...), to number a reference (1, 2, 3, ...) or
# to give a scale bar ("1 cm", "0,2 mm", "1 000 μm"). Only unambiguous
# cases are caught - see the limitation in is_diagram_annotation() below.
SCALE_BAR_RE = re.compile(
    r"^([\d.,]+\s*(mm|cm|km|μm|µm|nm|m)\s*)+$", re.IGNORECASE)
LEGEND_WORDS = {"do", "nad", "pod", "až"}  # Czech connectives in scale legends
UNIT_LABEL_RE = re.compile(r"^\[[^\[\]]{1,6}\]$")  # "[°C]", "[%]", "[m]"

# Real figure captions are often set in the SAME typeface as body text, so
# font-and-size classification cannot tell them apart from the article. The
# giveaway is that they almost always open with a reference to the figure
# number: "1 a 2 Nejnápadnějším příznakem...", "3 Schéma normálního...".
# (I tried combining this with a check that the block sits next to a figure
# in the PDF, but at article boundaries a caption is often far from "its"
# figure in raw block order - so this relies on the shape of the text plus
# a minimum length, the latter to avoid catching short numbered headings
# such as "1. Introduction".)
CAPTION_LEAD_RE = re.compile(
    r"^\d+(\s*(a|až|,|-|–)\s*\d+)*\s+[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]")
CAPTION_LEAD_MIN_CHARS = 20

# A PDF font name arrives as "ABCDEF+MeliorCE-Bold" (six-letter subset
# prefix + family + weight) or "Arial,BoldItalic". The family is what is
# left after stripping both.
SUBSET_PREFIX_RE = re.compile(r"^[A-Z]{6}\+")
# After the hyphen there may still be an optical size
# ("RobotoSerif-20ptRegular") before the weight itself. Without allowing
# for it, "RobotoSerif-20ptRegular" and "RobotoSerif-Italic" come out as
# two different families even though they are the same text in a different
# weight - and an italic run inside a contents entry would then be dropped
# as a foreign style.
STYLE_SUFFIX_RE = re.compile(
    r"[-,](?:\d+pt)?"
    r"(?:Bold|Italic|Oblique|Light|Medium|Regular|Roman|Semibold|SemiBold"
    r"|Black|Thin|ExtraBold|Condensed)+.*$",
    re.IGNORECASE)


def font_style_key(font: str, size: float):
    """Font family plus size rounded to half-points.

    The unit in which "is this the same kind of text?" gets compared.
    Rounding is necessary because a PDF routinely sets the same text as
    8.5 and 8.502, which would otherwise split one style into several.
    """
    return font_family(font), round(size * 2) / 2


def font_family(font: str) -> str:
    """"ABCDEF+MeliorCE-BoldItalic" -> "MeliorCE": no subset prefix, no weight."""
    name = SUBSET_PREFIX_RE.sub("", font or "")
    return STYLE_SUFFIX_RE.sub("", name)


def is_bold_font(font: str) -> bool:
    return "bold" in (font or "").lower()


def _is_number_token(tok: str) -> bool:
    return bool(re.fullmatch(r"-?\d+[.,]?\d*", tok))


def _is_range_token(tok: str) -> bool:
    """"3–4", "10-11" - two numbers joined by a hyphen or en dash, typical
    of the legend of a colour scale on a map or chart."""
    return bool(re.fullmatch(r"-?\d+[.,]?\d*[–-]-?\d+[.,]?\d*", tok))


def is_diagram_annotation(text: str) -> bool:
    """True for clearly recognisable panel labels and scale bars.

    It does not catch the shorter anatomical or technical fragments
    scattered around composite figures ("substantia", "nigra", "karyotyp",
    "gen IT15"): those cannot be told from a real caption by the shape of
    the text alone, that would need the position relative to a specific
    figure, which this module does not know.

    The key property: the WHOLE text is numbers, ranges and a couple of
    connectives. No normal paragraph is that clean.
    """
    t = text.strip()
    if not t:
        return False
    if SCALE_BAR_RE.match(t) or UNIT_LABEL_RE.match(t):
        return True
    tokens = t.split()
    if not tokens:
        return False
    # a run of bare numbers: "1 2 3 4 5", "35 30 25 20 15 10 5 0"
    # (a chart axis, panel numbering)
    if all(re.fullmatch(r"-?\d+[.,]?\d*\.?", tok) for tok in tokens):
        return True
    # a colour-scale legend: "pod -150 -100 až -50 0 až 50 100 až 150 nad 150",
    # "do 3 3–4 4–5 5–6 ... nad 12"
    if all(_is_number_token(tok) or _is_range_token(tok)
           or tok.lower() in LEGEND_WORDS for tok in tokens):
        return True
    # a run of single-character labels, optionally with a full stop:
    # "a b c d e f", "1. 2. 3."
    if all(len(tok.rstrip(".")) == 1 for tok in tokens):
        return True
    return False


# --------------------------------------------------------------------------
# Document statistics for the adaptive profile
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DocumentStats:
    """The document's reference typography, derived from the document itself.

    `body_size`/`body_family` is the combination that sets the most
    CHARACTERS in the issue. Weighting by characters rather than by block
    count is the crux: a page carries many titles and captions but little
    of their text, so by block count a caption could "win" and shift the
    whole relative scale.
    """
    body_size: float
    body_family: str

    @classmethod
    def from_spans(cls, spans) -> "DocumentStats":
        """spans: an iterable of (text, font, size) from the whole document."""
        weights: Counter = Counter()
        for text, font, size in spans:
            n = len(text.strip())
            if n:
                # Round the size to half-points: a PDF routinely sets the
                # same text as 9.0 and 9.000001, which would shatter the
                # histogram into dozens of near-identical classes.
                weights[(font_family(font), round(size * 2) / 2)] += n
        if not weights:
            return cls(body_size=10.0, body_family="")
        (family, size), _ = weights.most_common(1)[0]
        return cls(body_size=size or 10.0, body_family=family)


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

def classify_block(profile, text: str, font: str, size: float,
                   stats: "DocumentStats | None" = None) -> str:
    """Return the block type according to the profile.

    `stats` is required only for an adaptive profile; a hand-written
    profile ignores it.
    """
    # A running header or footer is never classified as a heading even
    # though it is set in bold. It is removed from the text later, by
    # assign_articles using its own stricter test; here it merely must not
    # leak into the headings.
    if profile.is_footer_text(text):
        return "body"

    if profile.adaptive:
        block_type, family = _classify_adaptive(profile, font, size, stats)
    else:
        block_type, family = _classify_absolute(profile, font, size)

    if block_type is not None:
        return block_type

    # No rule matched - the family's fallback decides.
    if family is None:
        return profile.default_block_type
    if family.detect_annotations and is_diagram_annotation(text):
        return "annotation"
    return family.fallback


def _classify_absolute(profile, font: str, size: float):
    family = profile.family_for(font)
    if family is None:
        return profile.default_block_type, None
    is_bold = is_bold_font(font)
    for rule in family.rules:
        if rule.matches(size, is_bold):
            return rule.block_type, family
    return None, family


def _classify_adaptive(profile, font: str, size: float, stats):
    if stats is None:
        raise ValueError(
            "an adaptive profile needs DocumentStats - call extract_pdf(), "
            "not classify_block() directly")
    same_family = font_family(font) == stats.body_family
    ratio = size / stats.body_size if stats.body_size else 1.0
    is_bold = is_bold_font(font)
    for rule in profile.relative_rules:
        if rule.matches(ratio, is_bold, same_family):
            return rule.block_type, _adaptive_family(profile, same_family)
    return None, _adaptive_family(profile, same_family)


def _adaptive_family(profile, same_family: bool):
    """In an adaptive profile `families` carries nothing but the fallbacks:
    [0] the body-text family, [1] everything else."""
    if not profile.families:
        return None
    return profile.families[0] if same_family else profile.families[-1]


def looks_like_caption_lead(text: str) -> bool:
    """A figure caption set in the body typeface, recognised by the figure
    reference opening the block (see CAPTION_LEAD_RE)."""
    stripped = text.strip()
    return (len(stripped) > CAPTION_LEAD_MIN_CHARS
            and bool(CAPTION_LEAD_RE.match(stripped)))
