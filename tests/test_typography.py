"""Klasifikace bloků: ruční profil vs. adaptivní.

Testy schválně nepoužívají žádný text ze skutečného časopisu - pracují se
syntetickými bloky, u kterých je z konstrukce jasné, co má vyjít. Živa je
autorsky chráněná a MagPi tady není potřeba; testuje se pravidlo, ne obsah.
"""
import pytest

from magrag.profiles import get
from magrag.typography import (
    DocumentStats,
    classify_block,
    font_family,
    is_diagram_annotation,
)

ZIVA = get("ziva")
ADAPTIVE = get("adaptive")


# --- rozklad jména fontu ---------------------------------------------------

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


# --- ruční profil (Živa) ---------------------------------------------------

@pytest.mark.parametrize("font,size,expected", [
    ("MeliorCE-Bold", 21.0, "title"),      # titulek článku
    ("MeliorCE-Bold", 18.0, "title"),      # přesně na hranici
    ("MeliorCE-Bold", 15.0, "heading"),    # nadpis kapitoly
    ("MeliorCE", 13.0, "heading"),         # podnadpis v zadní části čísla
    ("MeliorCE", 12.0, "other"),           # řádek s autorem
    ("MeliorCE", 9.0, "body"),             # běžný text
    ("HelveticaCE", 7.0, "caption"),       # popisek obrázku
    ("HelveticaCE-Bold", 40.0, "title"),   # velké číslo na obálce
    ("HelveticaCE-Bold", 11.0, "other"),   # titulek na obálce
    ("ZapfDingbats", 9.0, "body"),         # neznámá rodina -> fallback
])
def test_ziva_classification(font, size, expected):
    assert classify_block(ZIVA, "Nějaký neutrální text.", font, size) == expected


def test_ziva_footer_never_becomes_heading():
    """Patička je vysázená tučně a mohla by projít jako nadpis - profil ji
    proto odchytí dřív, než se na ni dostane pravidlo podle velikosti."""
    assert classify_block(ZIVA, "ziva.avcr.cz 262 živa 6/2014",
                          "MeliorCE-Bold", 21.0) == "body"


def test_ziva_unknown_size_falls_back_within_its_own_family():
    """Neznámá velikost v patkové rodině je text, v bezpatkové popisek -
    fallback se nesmí "propadnout" do druhé rodiny."""
    assert classify_block(ZIVA, "text", "MeliorCE", 6.0) == "body"
    assert classify_block(ZIVA, "text", "HelveticaCE", 6.0) == "caption"


# --- editorská hantýrka na obrázcích ---------------------------------------

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
    "Pohled na vzorek z boku.",
    "substantia nigra",
    "",
])
def test_diagram_annotation_not_overreaching(text):
    assert not is_diagram_annotation(text)


def test_annotation_only_in_families_that_ask_for_it():
    """Popiskovou rodinu smí anotace přebít; textovou ne - jinak by se
    z odstavce začínajícího výčtem čísel stala 'annotation'."""
    assert classify_block(ZIVA, "1 2 3 4", "HelveticaCE", 7.0) == "annotation"
    assert classify_block(ZIVA, "1 2 3 4", "MeliorCE", 9.0) == "body"


# --- adaptivní profil ------------------------------------------------------

def _stats(body_family="BodyFont", body_size=10.0):
    return DocumentStats(body_size=body_size, body_family=body_family)


def test_document_stats_weighs_by_characters_not_by_span_count():
    """Titulků je na stránce hodně kusů, ale málo textu. Kdyby se počítaly
    spany místo znaků, referenční velikost by mohl určit popisek."""
    spans = [("Nadpis", "TitleFont", 24.0)] * 20 + \
            [("x" * 200, "BodyFont", 10.0)] * 3
    stats = DocumentStats.from_spans(spans)
    assert stats.body_family == "BodyFont"
    assert stats.body_size == 10.0


def test_document_stats_buckets_near_identical_sizes():
    """PDF běžně vysází tentýž text jako 9.0 i 9.02 - bez zaokrouhlení by
    se histogram rozpadl na desítky skoro shodných tříd."""
    spans = [("x" * 100, "BodyFont", 9.0), ("y" * 100, "BodyFont", 9.02)]
    assert DocumentStats.from_spans(spans).body_size == 9.0


def test_document_stats_on_empty_document_does_not_divide_by_zero():
    stats = DocumentStats.from_spans([])
    assert stats.body_size > 0


@pytest.mark.parametrize("size,bold,expected", [
    (20.0, True, "title"),      # 2,0x text
    (17.0, False, "title"),     # 1,7x text, na hranici
    (13.0, True, "heading"),    # 1,3x text
    (10.0, True, "heading"),    # tučné ve velikosti textu = podnadpis
    (10.0, False, "body"),      # běžný text
    (7.0, False, "body"),       # menší, ale pořád rodina textu
])
def test_adaptive_classification_is_relative_to_body_size(size, bold, expected):
    font = "BodyFont-Bold" if bold else "BodyFont"
    assert classify_block(ADAPTIVE, "text", font, size, _stats()) == expected


def test_adaptive_scales_with_the_document():
    """Totéž pravidlo musí platit i pro časopis sázený o třetinu větším
    písmem - to je celý smysl relativních pravidel."""
    small = classify_block(ADAPTIVE, "text", "BodyFont-Bold", 20.0, _stats(body_size=10.0))
    large = classify_block(ADAPTIVE, "text", "BodyFont-Bold", 30.0, _stats(body_size=15.0))
    assert small == large == "title"


def test_adaptive_other_family_falls_back_to_caption():
    assert classify_block(ADAPTIVE, "text", "OtherFont", 8.0, _stats()) == "caption"


def test_adaptive_without_stats_fails_loudly():
    """Tiše klasifikovat všechno jako 'body' by byla přesně ta chyba, co se
    v korpusu pozná až za tři fáze."""
    with pytest.raises(ValueError):
        classify_block(ADAPTIVE, "text", "BodyFont", 10.0)


# --- optická velikost ve jménu fontu ---------------------------------------

def test_optical_size_in_the_font_name_is_not_part_of_the_family():
    """"RobotoSerif-20ptRegular" a "RobotoSerif-Italic" je tentýž text
    v jiném řezu. Kdyby vyšly jako dvě rodiny, kurzívou vysázená část
    titulku v obsahu by se zahodila jako cizí styl."""
    assert font_family("RobotoSerif-20ptRegular") == "RobotoSerif"
    assert font_family("RobotoSerif-20ptRegular") == font_family("RobotoSerif-Italic")


def test_similar_family_names_stay_apart():
    """Roboto, RobotoSerif, RobotoSlab a RobotoMono jsou čtyři různé
    rodiny - u MagPi je podle nich vidět rozdíl mezi položkou obsahu,
    názvem rubriky a patičkou."""
    families = {font_family(f) for f in
                ("Roboto-Black", "RobotoSerif-Italic", "RobotoSlab-Light",
                 "RobotoMono-Light", "RobotoCondensed-Light")}
    assert len(families) == 5


def test_font_style_key_buckets_to_half_points():
    from magrag.typography import font_style_key
    assert font_style_key("Roboto-Bold", 8.51) == font_style_key("Roboto", 8.49)
