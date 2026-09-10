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


# --- odvození stylů položek ze stránky s obsahem ---------------------------

def _span(text, font, size, color=0):
    return {"text": text, "font": font, "size": size, "color": color}


def _entries(number_font, number_size, text_font, text_size, n, start=10):
    """n položek obsahu vysázených zadaným stylem."""
    out = []
    for i in range(n):
        out.append(_span(str(start + i * 2), number_font, number_size))
        out.append(_span(f"Titulek {i}", text_font, text_size))
    return out


def test_two_equally_valid_number_styles_are_both_kept():
    """Živa má v obsahu dvě velikosti čísel (9 a 10 b) a obě jsou pravé.
    Vzít jen tu nejčastější znamená ztratit polovinu položek - což se při
    vývoji taky stalo."""
    from magrag.create_toc import detect_entry_styles
    spans = (_entries("MeliorCE-Bold", 9.0, "MeliorCE", 9.5, 19)
             + _entries("MeliorCE-Bold", 10.0, "MeliorCE", 10.0, 15, start=200))
    numbers, texts = detect_entry_styles(spans, profile=None)
    assert numbers == {("MeliorCE", 9.0), ("MeliorCE", 10.0)}
    assert texts == {("MeliorCE", 9.5), ("MeliorCE", 10.0)}


def test_decorative_callout_numbers_are_rejected():
    """MagPi má na stránce s obsahem ozdobné upoutávky: velké bílé číslo
    s kratším popiskem. Vypadají jako položka, ale vedou na jiný druh
    textu a je jich řádově míň."""
    from magrag.create_toc import detect_entry_styles
    spans = (_entries("RobotoSerif-20ptRegular", 8.5, "RobotoSerif-20ptRegular", 8.5, 32)
             + _entries("Roboto-Bold", 12.0, "Roboto-Black", 12.0, 3, start=90))
    numbers, texts = detect_entry_styles(spans, profile=None)
    assert numbers == {("RobotoSerif", 8.5)}
    assert texts == {("RobotoSerif", 8.5)}


def test_number_and_title_may_use_different_fonts():
    """U MagPi 150 je číslo Rajdhani a titulek RobotoSlab - styl textu se
    proto hledá zvlášť, ne jako 'stejná rodina jako číslo'."""
    from magrag.create_toc import detect_entry_styles
    spans = _entries("Rajdhani-Bold", 14.0, "RobotoSlab-Light", 11.0, 22)
    numbers, texts = detect_entry_styles(spans, profile=None)
    assert numbers == {("Rajdhani", 14.0)}
    assert texts == {("RobotoSlab", 11.0)}


def test_page_without_any_numbers_yields_no_styles():
    from magrag.create_toc import detect_entry_styles
    spans = [_span("jen text", "Whatever", 10.0)]
    assert detect_entry_styles(spans, profile=None) == (None, None)


def test_page_number_survives_a_tab_and_a_decorative_glyph():
    """U MagPi 150 přichází číslo položky slepené s tabulátorem a odrážkou
    do jednoho spanu. Bez pročištění se položka ztratí beze stopy."""
    assert page_span_value("10 \t \x07") == "10"
