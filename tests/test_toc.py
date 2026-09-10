"""Vytažení obsahu čísla ze stránky s obsahem."""
from magrag.create_toc import (
    TOC_MIN_ENTRIES,
    clean,
    find_toc_pages,
    page_span_value,
    strip_control_chars,
)


# --- čištění textu ---------------------------------------------------------

def test_control_characters_are_stripped():
    """Ozdobná odrážka před položkou obsahu vyjde z PDF jako U+0007
    (reálný případ: MagPi). V titulku nemá co dělat a rozbíjí porovnání
    titulku z obsahu s nadpisem v těle čísla."""
    assert clean("\x07CNC water cooling") == "CNC water cooling"


def test_ordinary_whitespace_still_collapses():
    assert clean("  dva   řádky\n textu ") == "dva řádky textu"


def test_strip_control_chars_keeps_accented_letters():
    assert strip_control_chars("Živa – čeština") == "Živa – čeština"


# --- čísla stránek v obsahu ------------------------------------------------

def test_zero_padded_page_number_is_recognised():
    assert page_span_value("032") == "032"


def test_abbreviated_range_yields_its_first_page():
    """"XXXI–II" znamená XXXI až XXXII; pipeline dál potřebuje jen začátek,
    konec článku se dopočítá z následujícího záznamu v obsahu."""
    assert page_span_value("XXXI–II") == "XXXI"
    assert page_span_value("285-6") == "285"


def test_plain_words_are_not_page_numbers():
    assert page_span_value("Contents") is None


# --- detekce stránek s obsahem --------------------------------------------

class FakeDoc:
    """Dokument, u kterého si sami řekneme, kolik "začátků položky" je na
    které stránce - detekce se testuje bez PDF."""

    def __init__(self, scores):
        self.scores = scores
        self.page_count = len(scores)

    def __getitem__(self, i):
        return self.scores[i]


def _find(scores, monkeypatch):
    import magrag.create_toc as m
    monkeypatch.setattr(m, "count_toc_entries", lambda page, profile: page)
    return find_toc_pages(FakeDoc(scores), profile=None)


def test_toc_spanning_a_spread_is_found_whole(monkeypatch):
    """Obsah bývá rozložený přes dvoustranu, u tlustšího čísla i přes tři
    stránky proložené inzercí. Vzít jen tu nejlepší znamená přijít o dvě
    třetiny čísla - a nepozná se to, protože zbytek pipeline poslušně
    zpracuje to, co dostal."""
    assert _find([1, 0, 2, 0, 12, 20, 1, 13], monkeypatch) == (4, 5, 7)


def test_lone_stray_number_page_is_not_mistaken_for_the_toc(monkeypatch):
    assert _find([0, 0, 1, 0, 18, 2, 0, 1], monkeypatch) == (4,)


def test_no_toc_at_all_returns_nothing(monkeypatch):
    """Když obsah není nikde, je lepší nevrátit nic než ukázat na náhodnou
    stránku - volající to pozná a může zafixovat --toc-page ručně."""
    assert _find([1, 2, 1, 0, 3, 0, 0, 2], monkeypatch) == ()
    assert TOC_MIN_ENTRIES > 3
