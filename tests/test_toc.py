"""Extracting an issue's contents from the contents page."""
from magazine_rag.create_toc import (
    TOC_MIN_ENTRIES,
    clean,
    find_toc_pages,
    page_span_value,
    strip_control_chars,
)


# --- cleaning the text -----------------------------------------------------

def test_control_characters_are_stripped():
    """The decorative bullet before a contents entry comes out of the PDF
    as U+0007 (a real case: The MagPi). It has no business in a title and
    it breaks the comparison with the heading in the body of the issue."""
    assert clean("\x07CNC water cooling") == "CNC water cooling"


def test_ordinary_whitespace_still_collapses():
    assert clean("  two   lines\n of text ") == "two lines of text"


def test_strip_control_chars_keeps_accented_letters():
    assert strip_control_chars("Živa – čeština") == "Živa – čeština"


# --- page numbers in the contents ------------------------------------------

def test_zero_padded_page_number_is_recognised():
    assert page_span_value("032") == "032"


def test_abbreviated_range_yields_its_first_page():
    """"XXXI–II" means XXXI to XXXII; downstream the pipeline needs only
    the start, since an article's end is derived from the next contents
    record."""
    assert page_span_value("XXXI–II") == "XXXI"
    assert page_span_value("285-6") == "285"


def test_plain_words_are_not_page_numbers():
    assert page_span_value("Contents") is None


# --- detecting the contents pages ------------------------------------------

class FakeDoc:
    """A document where we simply declare how many "entry starts" each
    page has, so detection can be tested without a PDF."""

    def __init__(self, scores):
        self.scores = scores
        self.page_count = len(scores)

    def __getitem__(self, i):
        return self.scores[i]


def _find(scores, monkeypatch):
    import magazine_rag.create_toc as m
    monkeypatch.setattr(m, "count_toc_entries", lambda page, profile: page)
    return find_toc_pages(FakeDoc(scores), profile=None)


def test_toc_spanning_a_spread_is_found_whole(monkeypatch):
    """The contents are usually spread over a double page, and in a
    thicker issue across three pages separated by advertising. Taking
    only the best one loses two thirds of the issue - and it does not
    register, because the rest of the pipeline dutifully processes
    whatever it was given."""
    assert _find([1, 0, 2, 0, 12, 20, 1, 13], monkeypatch) == (4, 5, 7)


def test_lone_stray_number_page_is_not_mistaken_for_the_toc(monkeypatch):
    assert _find([0, 0, 1, 0, 18, 2, 0, 1], monkeypatch) == (4,)


def test_no_toc_at_all_returns_nothing(monkeypatch):
    """When the contents are nowhere, returning nothing beats pointing at
    a random page - the caller can tell, and can pin --toc-page by
    hand."""
    assert _find([1, 2, 1, 0, 3, 0, 0, 2], monkeypatch) == ()
    assert TOC_MIN_ENTRIES > 3


# --- deriving entry styles from the contents page --------------------------

def _span(text, font, size, color=0):
    return {"text": text, "font": font, "size": size, "color": color}


def _entries(number_font, number_size, text_font, text_size, n, start=10):
    """n contents entries set in the given style."""
    out = []
    for i in range(n):
        out.append(_span(str(start + i * 2), number_font, number_size))
        out.append(_span(f"Title {i}", text_font, text_size))
    return out


def test_two_equally_valid_number_styles_are_both_kept():
    """Živa uses two sizes of page number in its contents (9 and 10pt)
    and both are real. Taking only the most frequent one loses half the
    entries - which is exactly what happened during development."""
    from magazine_rag.create_toc import detect_entry_styles
    spans = (_entries("MeliorCE-Bold", 9.0, "MeliorCE", 9.5, 19)
             + _entries("MeliorCE-Bold", 10.0, "MeliorCE", 10.0, 15, start=200))
    numbers, texts = detect_entry_styles(spans, profile=None)
    assert numbers == {("MeliorCE", 9.0), ("MeliorCE", 10.0)}
    assert texts == {("MeliorCE", 9.5), ("MeliorCE", 10.0)}


def test_decorative_callout_numbers_are_rejected():
    """The MagPi's contents page carries decorative callouts: a large
    white number with a short caption. They look like an entry, but they
    lead into a different kind of text and there are far fewer of
    them."""
    from magazine_rag.create_toc import detect_entry_styles
    spans = (_entries("RobotoSerif-20ptRegular", 8.5,
                      "RobotoSerif-20ptRegular", 8.5, 32)
             + _entries("Roboto-Bold", 12.0, "Roboto-Black", 12.0, 3, start=90))
    numbers, texts = detect_entry_styles(spans, profile=None)
    assert numbers == {("RobotoSerif", 8.5)}
    assert texts == {("RobotoSerif", 8.5)}


def test_number_and_title_may_use_different_fonts():
    """In MagPi 150 the number is Rajdhani and the title RobotoSlab, so
    the text style is sought separately rather than as "the same family
    as the number"."""
    from magazine_rag.create_toc import detect_entry_styles
    spans = _entries("Rajdhani-Bold", 14.0, "RobotoSlab-Light", 11.0, 22)
    numbers, texts = detect_entry_styles(spans, profile=None)
    assert numbers == {("Rajdhani", 14.0)}
    assert texts == {("RobotoSlab", 11.0)}


def test_page_without_any_numbers_yields_no_styles():
    from magazine_rag.create_toc import detect_entry_styles
    spans = [_span("just text", "Whatever", 10.0)]
    assert detect_entry_styles(spans, profile=None) == (None, None)


def test_page_number_survives_a_tab_and_a_decorative_glyph():
    """In MagPi 150 an entry's number arrives glued to a tab and a bullet
    inside one span. Without cleaning, the entry is lost without a
    trace."""
    assert page_span_value("10 \t \x07") == "10"
