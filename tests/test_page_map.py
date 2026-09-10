"""Mapping printed page numbers onto PDF pages.

This is the trickiest generic part of the pipeline: a single issue
routinely alternates between Arabic and Roman numbering, and each such
run has a different offset from the PDF pages. These tests describe
exactly the cases that turned out to be real bugs on a real archive.
"""
import pytest

from magrag.build_page_map import (
    build_page_map,
    extend_runs_into_gaps,
    extract_label,
    find_runs,
    int_to_label,
    int_to_roman,
    label_to_int,
    normalize_label,
    roman_to_int,
)
from magrag.profiles import get

ZIVA = get("ziva")
MAGPI = get("magpi")


# --- Roman numerals --------------------------------------------------------

@pytest.mark.parametrize("roman,value", [
    ("I", 1), ("IV", 4), ("XL", 40), ("CXXXIII", 133), ("MCMXCIV", 1994),
])
def test_roman_round_trip(roman, value):
    assert roman_to_int(roman) == value
    assert int_to_roman(value) == roman


def test_roman_is_case_insensitive():
    """A typesetting slip: a real issue contained "CXLVIiI" with a
    lowercase i. Without normalising, the whole page was assigned to no
    run at all."""
    scheme, value = label_to_int("CXLVIiI")
    assert scheme == "roman"
    assert value == roman_to_int("CXLVIII")


def test_single_letter_roman_is_a_valid_page_label():
    """Single-letter numerals ("I", "V", "X") used to be discarded as
    ambiguous. That was wrong - by the time we look at a token, the
    whole line is already confirmed to be a footer."""
    assert label_to_int("V") == ("roman", 5)


def test_arabic_beats_roman_for_pure_digits():
    assert label_to_int("262") == ("arabic", 262)
    assert int_to_label("arabic", 262) == "262"


# --- pulling the label out of a footer -------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("ziva.avcr.cz 262 živa 6/2014", "262"),
    ("živa 6/2014 263 ziva.avcr.cz", "263"),
    ("živa 6/2014 CXXXIII ziva.avcr.cz", "CXXXIII"),
])
def test_extract_label_from_running_footer(text, expected):
    assert extract_label(text, ZIVA) == expected


def test_extract_label_ignores_issue_over_year_token():
    """"6/2014" is the issue over the year, never a page number - taking
    it would give the whole numbering run a nonsensical offset."""
    assert extract_label("živa 6/2014 ziva.avcr.cz", ZIVA) is None


def test_extract_label_requires_the_line_to_be_a_footer():
    assert extract_label("In the year 262 BC something happened...", ZIVA) is None


# --- contiguous numbering runs ---------------------------------------------

def test_find_runs_splits_on_changed_offset():
    """A real case: main articles in Arabic, the supplement in Roman,
    then Arabic again - so "Arabic numbering" is not one set of numbers
    with one offset, but two separate runs."""
    detections = [
        (4, "arabic", 262), (5, "arabic", 263),      # offset -258
        (26, "roman", 133), (27, "roman", 134),      # offset -107
        (58, "arabic", 285), (59, "arabic", 286),    # offset -227
    ]
    runs = find_runs(detections)
    assert [r["offset"] for r in runs] == [-258, -107, -227]
    assert [(r["value_min"], r["value_max"]) for r in runs] == \
        [(262, 263), (133, 134), (285, 286)]


def test_find_runs_bridges_a_page_without_a_footer():
    """A full-page photo has no footer. As long as the detections either
    side agree on the offset, it is still one run."""
    detections = [(4, "arabic", 262), (6, "arabic", 264)]
    runs = find_runs(detections)
    assert len(runs) == 1
    assert (runs[0]["value_min"], runs[0]["value_max"]) == (262, 264)


def test_extend_runs_fills_missing_edge_pages():
    runs = [{"scheme": "arabic", "offset": -258, "value_min": 262, "value_max": 270}]
    extended = extend_runs_into_gaps(runs, total_pdf_pages=20, max_extend=5)
    # forwards it can only go 3 (PDF page 1 is the floor); backwards it
    # hits the max_extend cap
    assert extended[0]["value_min"] == 259
    assert extended[0]["value_max"] == 275


def test_extend_runs_never_steals_a_page_from_a_neighbour():
    """Two adjacent runs must not both claim the same PDF page - that
    was a real finding, where the Roman supplement pulled a page away
    from the Arabic run."""
    runs = [
        {"scheme": "arabic", "offset": 0, "value_min": 1, "value_max": 5},
        {"scheme": "roman", "offset": 5, "value_min": 1, "value_max": 5},
    ]
    extended = extend_runs_into_gaps(runs, total_pdf_pages=10, max_extend=5)
    pages = []
    for r in extended:
        pages += [v + r["offset"] for v in range(r["value_min"], r["value_max"] + 1)]
    assert len(pages) == len(set(pages)), "the runs stole a page from each other"


# --- the whole mapping -----------------------------------------------------

def _footer(page, text, y=800.0):
    return {"page": page, "block_id": 0, "type": "body", "text": text,
            "font": "HelveticaCE-Bold", "font_size": 8.0,
            "bbox": [50.0, y, 200.0, y + 8]}


def test_build_page_map_resolves_labels_to_pdf_pages():
    blocks = [_footer(p, f"ziva.avcr.cz {258 + p} živa 6/2014") for p in range(4, 10)]
    result = build_page_map(blocks, ZIVA)
    assert result["label_to_page"]["262"] == 4
    assert result["label_to_page"]["267"] == 9


def test_position_strategy_finds_a_bare_page_number():
    """The MagPi's footer holds only a number, so there is no content to
    latch onto - position near the bottom edge has to decide."""
    blocks = [
        {"page": p, "block_id": 0, "type": "body", "text": str(p),
         "font": "SomeSans", "font_size": 8.0,
         "bbox": [50.0, 780.0, 70.0, 790.0]}
        for p in range(3, 9)
    ]
    result = build_page_map(blocks, MAGPI)
    assert result["label_to_page"]["5"] == 5


def test_position_strategy_ignores_a_number_in_the_middle_of_the_page():
    """A number in the corner of a chart is not a page number - position
    is what decides."""
    blocks = [
        {"page": 3, "block_id": 0, "type": "body", "text": "42",
         "font": "SomeSans", "font_size": 8.0,
         "bbox": [50.0, 400.0, 70.0, 410.0]},
        {"page": 3, "block_id": 1, "type": "body", "text": "ordinary page text",
         "font": "SomeSans", "font_size": 8.0,
         "bbox": [50.0, 780.0, 300.0, 790.0]},
    ]
    assert build_page_map(blocks, MAGPI)["page_to_label"] == {}


def test_a_long_run_of_roman_letters_is_not_a_page_number():
    """Without an upper length bound, any long enough run of the letters
    I/V/X/L/C/D/M passes as a Roman numeral - and under position-based
    detection such a string really does turn up."""
    assert label_to_int("x" * 50) == (None, None)


# --- normalising how a label is written ------------------------------------

def test_normalize_label_strips_leading_zeros():
    """A real case (The MagPi): the contents give "032", the running
    footer just "32". Without normalising, the article maps to no page
    at all even though the numbering was detected flawlessly - and
    nothing crashes in the process."""
    assert normalize_label("032") == "32"
    assert normalize_label("32") == "32"


def test_normalize_label_uppercases_roman():
    assert normalize_label("cxxxiii") == "CXXXIII"


def test_normalize_label_leaves_unknown_shapes_alone():
    """Whatever cannot be recognised must not be silently discarded."""
    assert normalize_label("A-12") == "A-12"
