"""Klasifikace textového bloku na typ (title/heading/other/body/caption/
annotation) podle sazby.

Je to jediné místo v pipeline, kde se rozhoduje "co ten kus textu vlastně
je". Všechno ostatní už pracuje s výsledným typem, ne s fonty. Existují dvě
cesty, jak k rozhodnutí dojít, a obě chodí přes `classify_block()`:

* **ruční profil** - absolutní pravidla ("MeliorCE Bold od 18 bodů výš je
  titulek"), přesné, ale platí jen pro jeden konkrétní časopis;
* **adaptivní profil** - relativní pravidla vůči nejobjemnějšímu písmu
  v dokumentu ("1,7× větší než běžný text je titulek"), přenositelné
  na neznámý časopis bez kalibrace.

Viz `profiles/ziva.py` a `profiles/adaptive.py`.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

# --- "editorská hantýrka" nalepená přímo na obrázek/schéma ------------------
# Tohle NENÍ popisek obrázku (souvislý text vysvětlující, co je na obrázku
# vidět) - je to grafický prvek, kterým editor rozlišuje panely composite
# obrázku (a, b, c, ...), číslo/písmeno odkazu (1, 2, 3, ...) nebo udává
# měřítko (scale bar: "1 cm", "0,2 mm", "1 000 μm"). Zachytáváme jen
# jednoznačné případy - viz omezení v is_diagram_annotation() níž.
SCALE_BAR_RE = re.compile(
    r"^([\d.,]+\s*(mm|cm|km|μm|µm|nm|m)\s*)+$", re.IGNORECASE)
LEGEND_WORDS = {"do", "nad", "pod", "až"}  # české spojky v legendě škály
UNIT_LABEL_RE = re.compile(r"^\[[^\[\]]{1,6}\]$")  # "[°C]", "[%]", "[m]"

# Skutečné popisky obrázků bývají v časopisech sazeny STEJNÝM písmem jako
# běžný text, takže je klasifikace podle fontu/velikosti nerozezná od těla
# článku. Poznat je jde podle toho, že skoro vždy začínají odkazem na číslo
# obrázku hned na začátku bloku: "1 a 2 Nejnápadnějším příznakem...",
# "3 Schéma normálního...", "9 a 10 ...". (Zkoušel jsem tohle kombinovat
# ještě s kontrolou, že blok sedí v PDF hned vedle obrázku, ale u přechodů
# mezi články bývá popisek v surovém pořadí bloků docela daleko od "svého"
# obrázku - proto se spoléhá jen na tvar textu + minimální délku, aby to
# nechytlo krátké číslované nadpisy typu "1. Úvod".)
CAPTION_LEAD_RE = re.compile(
    r"^\d+(\s*(a|až|,|-|–)\s*\d+)*\s+[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]")
CAPTION_LEAD_MIN_CHARS = 20

# Jméno fontu v PDF chodí jako "ABCDEF+MeliorCE-Bold" (šestipísmenný subset
# prefix + rodina + řez) nebo "Arial,BoldItalic". Rodina je to, co zbude.
SUBSET_PREFIX_RE = re.compile(r"^[A-Z]{6}\+")
# Za pomlčkou může stát ještě optická velikost ("RobotoSerif-20ptRegular"),
# než přijde vlastní řez. Bez ní by "RobotoSerif-20ptRegular" a
# "RobotoSerif-Italic" vyšly jako dvě různé rodiny, i když jde o tentýž
# text v jiném řezu - a v obsahu čísla by se pak kurzívou vysázená část
# titulku zahodila jako cizí styl.
STYLE_SUFFIX_RE = re.compile(
    r"[-,](?:\d+pt)?"
    r"(?:Bold|Italic|Oblique|Light|Medium|Regular|Roman|Semibold|SemiBold"
    r"|Black|Thin|ExtraBold|Condensed)+.*$",
    re.IGNORECASE)


def font_style_key(font: str, size: float):
    """Rodina písma + velikost zaokrouhlená na půlbody.

    Jednotka, ve které se porovnává "je tohle tentýž druh textu?".
    Zaokrouhlení je nutné, protože PDF běžně vysází tentýž text jako 8.5
    i 8.502 a bez něj by se jeden styl rozpadl na několik.
    """
    return font_family(font), round(size * 2) / 2


def font_family(font: str) -> str:
    """"ABCDEF+MeliorCE-BoldItalic" -> "MeliorCE". Bez subset prefixu a řezu."""
    name = SUBSET_PREFIX_RE.sub("", font or "")
    return STYLE_SUFFIX_RE.sub("", name)


def is_bold_font(font: str) -> bool:
    return "bold" in (font or "").lower()


def _is_number_token(tok: str) -> bool:
    return bool(re.fullmatch(r"-?\d+[.,]?\d*", tok))


def _is_range_token(tok: str) -> bool:
    """"3–4", "10-11" - rozsah dvou čísel spojených pomlčkou/dlouhou
    pomlčkou, typické pro legendu barevné škály na mapě/grafu."""
    return bool(re.fullmatch(r"-?\d+[.,]?\d*[–-]-?\d+[.,]?\d*", tok))


def is_diagram_annotation(text: str) -> bool:
    """Vrať True pro jasně rozpoznatelné panelové značky/měřítka. Nezachytí
    to kratší anatomické/technické útržky rozeseté kolem composite obrázků
    (např. "substantia", "nigra", "karyotyp", "gen IT15") - ty od skutečného
    popisku nejde spolehlivě odlišit jen podle tvaru textu, chtělo by to
    znát polohu vůči konkrétnímu obrázku, což tenhle modul nezjišťuje.
    Klíčové: v CELÉM textu jde jen o čísla/rozsahy/pár spojek - žádný
    normální odstavec takhle "čistý" nebude."""
    t = text.strip()
    if not t:
        return False
    if SCALE_BAR_RE.match(t) or UNIT_LABEL_RE.match(t):
        return True
    tokens = t.split()
    if not tokens:
        return False
    # řada holých čísel: "1 2 3 4 5", "35 30 25 20 15 10 5 0" (osa grafu,
    # číslování panelů)
    if all(re.fullmatch(r"-?\d+[.,]?\d*\.?", tok) for tok in tokens):
        return True
    # legenda barevné škály: "pod -150 -100 až -50 0 až 50 100 až 150 nad 150",
    # "do 3 3–4 4–5 5–6 ... nad 12"
    if all(_is_number_token(tok) or _is_range_token(tok)
           or tok.lower() in LEGEND_WORDS for tok in tokens):
        return True
    # řada jednopísmenných/jednociferných značek, klidně s tečkou:
    # "a b c d e f", "1. 2. 3."
    if all(len(tok.rstrip(".")) == 1 for tok in tokens):
        return True
    return False


# --------------------------------------------------------------------------
# Statistika dokumentu pro adaptivní profil
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DocumentStats:
    """Referenční sazba dokumentu, odvozená z něj samotného.

    `body_size`/`body_family` je kombinace, kterou je vysázeno nejvíc ZNAKŮ
    v celém čísle. Vážení podle znaků, ne podle počtu bloků, je tu klíčové:
    titulků a popisků je na stránce hodně kusů, ale málo textu - podle počtu
    bloků by mohl "vyhrát" popisek obrázku a celá relativní škála by se
    posunula.
    """
    body_size: float
    body_family: str

    @classmethod
    def from_spans(cls, spans) -> "DocumentStats":
        """spans: iterable (text, font, size) z celého dokumentu."""
        weights: Counter = Counter()
        for text, font, size in spans:
            n = len(text.strip())
            if n:
                # velikost zaokrouhlujeme na půlbody: PDF běžně vysází
                # tentýž text jako 9.0 i 9.000001 a to by rozdrobilo
                # histogram na desítky skoro shodných tříd
                weights[(font_family(font), round(size * 2) / 2)] += n
        if not weights:
            return cls(body_size=10.0, body_family="")
        (family, size), _ = weights.most_common(1)[0]
        return cls(body_size=size or 10.0, body_family=family)


# --------------------------------------------------------------------------
# Klasifikace
# --------------------------------------------------------------------------

def classify_block(profile, text: str, font: str, size: float,
                   stats: "DocumentStats | None" = None) -> str:
    """Vrať typ bloku podle profilu.

    `stats` je povinné jen pro adaptivní profil; ruční profil ho ignoruje.
    """
    # Běžící hlavička/patička se nikdy neklasifikuje jako nadpis, i když má
    # tučné písmo - z textu ji stejně vyhazuje až assign_articles podle
    # vlastního (přísnějšího) testu, tady jen nesmí prosáknout do nadpisů.
    if profile.is_footer_text(text):
        return "body"

    if profile.adaptive:
        block_type, family = _classify_adaptive(profile, font, size, stats)
    else:
        block_type, family = _classify_absolute(profile, font, size)

    if block_type is not None:
        return block_type

    # Žádné pravidlo nesedlo - rozhoduje fallback rodiny.
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
            "adaptivní profil potřebuje DocumentStats - volej extract_pdf(), "
            "ne classify_block() napřímo")
    same_family = font_family(font) == stats.body_family
    ratio = size / stats.body_size if stats.body_size else 1.0
    is_bold = is_bold_font(font)
    for rule in profile.relative_rules:
        if rule.matches(ratio, is_bold, same_family):
            return rule.block_type, _adaptive_family(profile, same_family)
    return None, _adaptive_family(profile, same_family)


def _adaptive_family(profile, same_family: bool):
    """U adaptivního profilu nese `families` jen fallbacky: [0] rodina
    běžného textu, [1] všechno ostatní."""
    if not profile.families:
        return None
    return profile.families[0] if same_family else profile.families[-1]


def looks_like_caption_lead(text: str) -> bool:
    """Popisek obrázku sázený běžným písmem textu - pozná se podle odkazu
    na číslo obrázku na začátku bloku (viz CAPTION_LEAD_RE)."""
    stripped = text.strip()
    return (len(stripped) > CAPTION_LEAD_MIN_CHARS
            and bool(CAPTION_LEAD_RE.match(stripped)))
