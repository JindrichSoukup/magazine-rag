"""Block classification: hand-written profile vs. adaptive.

These tests deliberately use no text from a real magazine - they work on
synthetic blocks where the expected outcome is clear by construction.
Živa is under copyright and The MagPi is not needed here: what is being
tested is the rule, not the content.
"""
import pytest

from magazine_rag.profiles import get
from magazine_rag.typography import (
    DocumentStats,
    classify_block,
    font_family,
    is_diagram_annotation,
)

ZIVA = get("ziva")
ADAPTIVE = get("adaptive")


# --- taking a font name apart ----------------------------------------------

@pytest.mark.parametrize("font,expected", [
    ("MeliorCE", "MeliorCE"),
    ("MeliorCE-Bold", "MeliorCE"),
    ("ABCDEF+MeliorCE-BoldItalic", "MeliorCE"),
    ("Arial,Bold", "Arial"),
    ("HelveticaCE-Bold", "HelveticaCE"),
    ("", ""),
])
def test_font_family_strips_subset_prefix_and_style(font, expected):
    assert font_family(font) == expected


# --- the hand-written profile (Živa) ---------------------------------------

@pytest.mark.parametrize("font,size,expected", [
    ("MeliorCE-Bold", 21.0, "title"),      # article title
    ("MeliorCE-Bold", 18.0, "title"),      # exactly on the boundary
    ("MeliorCE-Bold", 15.0, "heading"),    # chapter heading
    ("MeliorCE", 13.0, "heading"),         # subheading in the back matter
    ("MeliorCE", 12.0, "other"),           # the author line
    ("MeliorCE", 9.0, "body"),             # body text
    ("HelveticaCE", 7.0, "caption"),       # figure caption
    ("HelveticaCE-Bold", 40.0, "title"),   # the big number on the cover
    ("HelveticaCE-Bold", 11.0, "other"),   # a cover blurb
    ("ZapfDingbats", 9.0, "body"),         # unknown family -> fallback
])
def test_ziva_classification(font, size, expected):
    assert classify_block(ZIVA, "Some neutral text.", font, size) == expected


def test_ziva_footer_never_becomes_heading():
    """The footer is set in bold and could pass as a heading, so the
    profile catches it before any size rule gets to it."""
    assert classify_block(ZIVA, "ziva.avcr.cz 262 živa 6/2014",
                          "MeliorCE-Bold", 21.0) == "body"


def test_ziva_unknown_size_falls_back_within_its_own_family():
    """An unknown size in the serif family is text, in the sans family a
    caption - the fallback must not "fall through" into the other
    family."""
    assert classify_block(ZIVA, "text", "MeliorCE", 6.0) == "body"
    assert classify_block(ZIVA, "text", "HelveticaCE", 6.0) == "caption"


# --- editorial furniture on figures ----------------------------------------

@pytest.mark.parametrize("text", [
    "1 cm",
    "0,2 mm 1 mm 2 mm",
    "[°C]",
    "35 30 25 20 15 10 5 0",
    "a b c d e f",
    "do 3 3–4 4–5 nad 12",
])
def test_diagram_annotation_detected(text):
    assert is_diagram_annotation(text)


@pytest.mark.parametrize("text", [
    "A view of the sample from the side.",
    "substantia nigra",
    "",
])
def test_diagram_annotation_not_overreaching(text):
    assert not is_diagram_annotation(text)


def test_annotation_only_in_families_that_ask_for_it():
    """The caption family may be overridden by an annotation; the text
    family may not - otherwise a paragraph opening with a run of numbers
    would turn into an 'annotation'."""
    assert classify_block(ZIVA, "1 2 3 4", "HelveticaCE", 7.0) == "annotation"
    assert classify_block(ZIVA, "1 2 3 4", "MeliorCE", 9.0) == "body"


# --- the adaptive profile --------------------------------------------------

def _stats(body_family="BodyFont", body_size=10.0):
    return DocumentStats(body_size=body_size, body_family=body_family)


def test_document_stats_weighs_by_characters_not_by_span_count():
    """A page carries many titles but little of their text. Counting
    spans instead of characters could let a caption decide the reference
    size."""
    spans = [("Heading", "TitleFont", 24.0)] * 20 + \
            [("x" * 200, "BodyFont", 10.0)] * 3
    stats = DocumentStats.from_spans(spans)
    assert stats.body_family == "BodyFont"
    assert stats.body_size == 10.0


def test_document_stats_buckets_near_identical_sizes():
    """A PDF routinely sets the same text as 9.0 and 9.02 - without
    rounding, the histogram would shatter into dozens of near-identical
    classes."""
    spans = [("x" * 100, "BodyFont", 9.0), ("y" * 100, "BodyFont", 9.02)]
    assert DocumentStats.from_spans(spans).body_size == 9.0


def test_document_stats_on_empty_document_does_not_divide_by_zero():
    stats = DocumentStats.from_spans([])
    assert stats.body_size > 0


@pytest.mark.parametrize("size,bold,expected", [
    (20.0, True, "title"),      # 2.0x the text
    (17.0, False, "title"),     # 1.7x the text, on the boundary
    (13.0, True, "heading"),    # 1.3x the text
    (10.0, True, "heading"),    # bold at text size = a subheading
    (10.0, False, "body"),      # body text
    (7.0, False, "body"),       # smaller, but still the text family
])
def test_adaptive_classification_is_relative_to_body_size(size, bold, expected):
    font = "BodyFont-Bold" if bold else "BodyFont"
    assert classify_block(ADAPTIVE, "text", font, size, _stats()) == expected


def test_adaptive_scales_with_the_document():
    """The same rule has to hold for a magazine set a third larger -
    that is the whole point of relative rules."""
    small = classify_block(ADAPTIVE, "text", "BodyFont-Bold", 20.0,
                           _stats(body_size=10.0))
    large = classify_block(ADAPTIVE, "text", "BodyFont-Bold", 30.0,
                           _stats(body_size=15.0))
    assert small == large == "title"


def test_adaptive_other_family_falls_back_to_caption():
    assert classify_block(ADAPTIVE, "text", "OtherFont", 8.0, _stats()) == "caption"


def test_adaptive_without_stats_fails_loudly():
    """Silently classifying everything as 'body' would be exactly the
    kind of bug that only shows up three stages later in the corpus."""
    with pytest.raises(ValueError):
        classify_block(ADAPTIVE, "text", "BodyFont", 10.0)


# --- optical size inside a font name ---------------------------------------

def test_optical_size_in_the_font_name_is_not_part_of_the_family():
    """"RobotoSerif-20ptRegular" and "RobotoSerif-Italic" are the same
    text in a different weight. If they came out as two families, an
    italic run inside a contents title would be dropped as a foreign
    style."""
    assert font_family("RobotoSerif-20ptRegular") == "RobotoSerif"
    assert font_family("RobotoSerif-20ptRegular") == font_family("RobotoSerif-Italic")


def test_similar_family_names_stay_apart():
    """Roboto, RobotoSerif, RobotoSlab and RobotoMono are four different
    families - in The MagPi they are what distinguishes a contents entry
    from a section heading and from the footer."""
    families = {font_family(f) for f in
                ("Roboto-Black", "RobotoSerif-Italic", "RobotoSlab-Light",
                 "RobotoMono-Light", "RobotoCondensed-Light")}
    assert len(families) == 5


def test_font_style_key_buckets_to_half_points():
    from magazine_rag.typography import font_style_key
    assert font_style_key("Roboto-Bold", 8.51) == font_style_key("Roboto", 8.49)
