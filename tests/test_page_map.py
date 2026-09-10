"""Mapování tištěných čísel stránek na stránky PDF.

Tohle je nejzrádnější obecná část pipeline: v jednom čísle časopisu se
běžně střídají arabské a římské číslování a každý takový úsek má jiný
posun vůči stránkám PDF. Testy popisují právě ty případy, které se na
reálném archivu ukázaly jako skutečné chyby.
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
    roman_to_int,
)
from magrag.profiles import get

ZIVA = get("ziva")
MAGPI = get("magpi")


# --- římské číslice --------------------------------------------------------

@pytest.mark.parametrize("roman,value", [
    ("I", 1), ("IV", 4), ("XL", 40), ("CXXXIII", 133), ("MCMXCIV", 1994),
])
def test_roman_round_trip(roman, value):
    assert roman_to_int(roman) == value
    assert int_to_roman(value) == roman


def test_roman_is_case_insensitive():
    """Sazečský šotek: v reálném čísle se objevilo "CXLVIiI" s malým i.
    Bez normalizace se celá stránka nepřiřadila k žádnému úseku."""
    scheme, value = label_to_int("CXLVIiI")
    assert scheme == "roman"
    assert value == roman_to_int("CXLVIII")


def test_single_letter_roman_is_a_valid_page_label():
    """Dřív se jednopísmenné číslice ("I", "V", "X") zahazovaly jako
    nejednoznačné. Byla to chyba - v okamžiku, kdy se dívám na token,
    už mám ověřené, že celý řádek je patička."""
    assert label_to_int("V") == ("roman", 5)


def test_arabic_beats_roman_for_pure_digits():
    assert label_to_int("262") == ("arabic", 262)
    assert int_to_label("arabic", 262) == "262"


# --- vytažení popisku z patičky -------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("ziva.avcr.cz 262 živa 6/2014", "262"),
    ("živa 6/2014 263 ziva.avcr.cz", "263"),
    ("živa 6/2014 CXXXIII ziva.avcr.cz", "CXXXIII"),
])
def test_extract_label_from_running_footer(text, expected):
    assert extract_label(text, ZIVA) == expected


def test_extract_label_ignores_issue_over_year_token():
    """"6/2014" je číslo/ročník, nikdy ne číslo stránky - kdyby se vzalo,
    celý úsek číslování by dostal nesmyslný posun."""
    assert extract_label("živa 6/2014 ziva.avcr.cz", ZIVA) is None


def test_extract_label_requires_the_line_to_be_a_footer():
    assert extract_label("V roce 262 př. n. l. se stalo...", ZIVA) is None


# --- souvislé úseky číslování ---------------------------------------------

def test_find_runs_splits_on_changed_offset():
    """Reálný případ: hlavní články arabsky, příloha římsky, pak arabsky
    znovu - "arabské číslování" tedy nejsou jedny čísla s jedním posunem,
    ale dva různé úseky."""
    detections = [
        (4, "arabic", 262), (5, "arabic", 263),      # posun -258
        (26, "roman", 133), (27, "roman", 134),      # posun -107
        (58, "arabic", 285), (59, "arabic", 286),    # posun -227
    ]
    runs = find_runs(detections)
    assert [r["offset"] for r in runs] == [-258, -107, -227]
    assert [(r["value_min"], r["value_max"]) for r in runs] == \
        [(262, 263), (133, 134), (285, 286)]


def test_find_runs_bridges_a_page_without_a_footer():
    """Celostránková fotka nemá patičku. Dokud se detekce před ní a za ní
    shodnou na posunu, je to pořád jeden úsek."""
    detections = [(4, "arabic", 262), (6, "arabic", 264)]
    runs = find_runs(detections)
    assert len(runs) == 1
    assert (runs[0]["value_min"], runs[0]["value_max"]) == (262, 264)


def test_extend_runs_fills_missing_edge_pages():
    runs = [{"scheme": "arabic", "offset": -258, "value_min": 262, "value_max": 270}]
    extended = extend_runs_into_gaps(runs, total_pdf_pages=20, max_extend=5)
    # dopředu se dá jen o 3 (PDF stránka 1 je dno), dozadu narazí na strop
    assert extended[0]["value_min"] == 259
    assert extended[0]["value_max"] == 275


def test_extend_runs_never_steals_a_page_from_a_neighbour():
    """Dva sousední úseky si nesmí nárokovat tutéž stránku PDF - to byl
    reálný nález, kdy římská příloha přetáhla stránku arabskému úseku."""
    runs = [
        {"scheme": "arabic", "offset": 0, "value_min": 1, "value_max": 5},
        {"scheme": "roman", "offset": 5, "value_min": 1, "value_max": 5},
    ]
    extended = extend_runs_into_gaps(runs, total_pdf_pages=10, max_extend=5)
    pages = []
    for r in extended:
        pages += [v + r["offset"] for v in range(r["value_min"], r["value_max"] + 1)]
    assert len(pages) == len(set(pages)), "úseky si ukradly stránku"


# --- celé mapování ---------------------------------------------------------

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
    """MagPi má v patičce jen číslo, na obsah se chytit nedá - musí
    rozhodnout poloha u dolního okraje stránky."""
    blocks = [
        {"page": p, "block_id": 0, "type": "body", "text": str(p),
         "font": "SomeSans", "font_size": 8.0,
         "bbox": [50.0, 780.0, 70.0, 790.0]}
        for p in range(3, 9)
    ]
    result = build_page_map(blocks, MAGPI)
    assert result["label_to_page"]["5"] == 5


def test_position_strategy_ignores_a_number_in_the_middle_of_the_page():
    """Číslo v rohu grafu není číslo stránky - o tom rozhoduje poloha."""
    blocks = [
        {"page": 3, "block_id": 0, "type": "body", "text": "42",
         "font": "SomeSans", "font_size": 8.0,
         "bbox": [50.0, 400.0, 70.0, 410.0]},
        {"page": 3, "block_id": 1, "type": "body", "text": "běžný text stránky",
         "font": "SomeSans", "font_size": 8.0,
         "bbox": [50.0, 780.0, 300.0, 790.0]},
    ]
    assert build_page_map(blocks, MAGPI)["page_to_label"] == {}


def test_a_long_run_of_roman_letters_is_not_a_page_number():
    """Bez horní meze délky projde jako římská číslice každý dost dlouhý
    shluk písmen I/V/X/L/C/D/M - a u detekce podle polohy se takový
    řetězec reálně objeví."""
    assert label_to_int("x" * 50) == (None, None)


# --- sjednocení zápisu popisku --------------------------------------------

def test_normalize_label_strips_leading_zeros():
    """Reálný případ (MagPi): obsah čísla uvádí "032", běžící patička jen
    "32". Bez sjednocení se článek nenamapuje na žádnou stránku, i když je
    číslování detekované úplně bez chyby - a nespadne přitom nic."""
    from magrag.build_page_map import normalize_label
    assert normalize_label("032") == "32"
    assert normalize_label("32") == "32"


def test_normalize_label_uppercases_roman():
    from magrag.build_page_map import normalize_label
    assert normalize_label("cxxxiii") == "CXXXIII"


def test_normalize_label_leaves_unknown_shapes_alone():
    """Co se nepodaří rozpoznat, se nesmí tiše zahodit."""
    from magrag.build_page_map import normalize_label
    assert normalize_label("A-12") == "A-12"
