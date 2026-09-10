"""Profil zdroje - všechno, čím se jeden časopis liší od jiného.

Pipeline se dělí na dvě části, které se chovají úplně jinak, když ji
přenesete na jiný časopis:

* **Obecná část** (mapování tištěných stránek na PDF stránky, dělení článků
  podle Y-souřadnice, spojování slov zalomených pomlčkou, chunking,
  embedding, vektorové úložiště, retrieval) na konkrétní sazbě nezávisí
  vůbec - pracuje už jen se strukturou `{page, type, font, bbox, text}`.

* **Sazbě specifická část** je naopak navázaná na jeden konkrétní časopis
  úplně natvrdo: jak se jmenují fonty, jaká velikost písma znamená titulek,
  na které straně je obsah čísla, jak vypadá běžící patička.

Tenhle modul je hranice mezi nimi. Sazbě specifická rozhodnutí jsou tady
jako **data**, ne jako `if` uprostřed parseru - přidat další časopis pak
znamená napsat jeden `SourceProfile`, ne sáhnout do pěti skriptů.

Nový profil se nepíše od stolu - použijte `tools/inspect_fonts.py`, který
z libovolného PDF vypíše, jaké fonty a velikosti se v něm reálně vyskytují
a v jakém množství. Postup je v README v sekci "Přidání nového časopisu".
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@dataclass(frozen=True)
class SizeRule:
    """Jedno pravidlo "velikost písma -> typ bloku" uvnitř jedné rodiny fontů.

    `size_min`/`size_max` jsou obě **včetně**. Pravidla se vyhodnocují
    v pořadí a vyhrává první, které sedne - u překrývajících se rozsahů
    tedy rozhoduje pořadí, ne šířka intervalu.

    `bold=None` znamená "na tučnosti nezáleží".
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
    """Pravidlo adaptivního profilu - velikost písma **relativně** k běžnému
    textu dokumentu (1,0 = stejně velké jako text článku).

    `same_family=True` znamená "stejná rodina písma jako běžný text",
    `False` "jiná rodina", `None` "nezáleží". Rodina je jméno fontu bez
    subset prefixu a bez řezu, viz `typography.font_family()`.
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
    """Skupina fontů se společnou sadou pravidel.

    Rodiny se zkoušejí v pořadí a vyhrává první, jejíž některý prefix sedne
    na jméno fontu. Když žádné pravidlo uvnitř vyhrané rodiny nesedne,
    použije se `fallback` **té rodiny** - nepokračuje se do rodiny další.
    To je podstatné: v patkové sazbě je neznámá velikost skoro jistě běžný
    text, kdežto v bezpatkové skoro jistě popisek obrázku.
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
    """Kompletní popis jednoho časopisu pro celou pipeline."""

    # --- identita -----------------------------------------------------------
    key: str
    journal_name: str
    language: str = "cs"

    # --- vstupní soubory ----------------------------------------------------
    # Pojmenování PDF, ze kterého se vytáhne ročník a číslo. Musí mít dvě
    # skupiny: (rok, číslo). run_all.py podle toho pojmenuje výstupní složky.
    filename_pattern: str = r"(\d{4})-(\d+)\.pdf$"

    # --- obsah čísla --------------------------------------------------------
    # 0-indexované PDF stránky, na kterých je natištěný obsah čísla.
    toc_page_indices: tuple[int, ...] = (2,)
    # Font čísla stránky v obsahu (prefix jména fontu + požadavek na tučnost).
    toc_page_number_prefixes: tuple[str, ...] = ()
    toc_page_number_bold: bool = True
    # Řetězce, které se v obsahu objevují jako tiráž/copyright. `drop`
    # ořízne titulek od výskytu dál (zbytek řádku je tiráž nalepená na
    # poslední položku), `skip_span` zahodí celý span (samostatný řádek
    # tiráže mezi položkami). Jsou to dva různé seznamy schválně: ořezávat
    # se musí i podle řetězců, které jako samostatný span nikdy nestojí.
    toc_drop_markers: tuple[str, ...] = ()
    toc_skip_span_markers: tuple[str, ...] = ()

    # --- běžící hlavička/patička -------------------------------------------
    # Jak se patička pozná. Dvě strategie, protože jedna nestačí:
    #
    #   "keyword"  - podle obsahu: patička obsahuje název časopisu nebo jeho
    #                web ("živa 6/2014", "ziva.avcr.cz"). Přesné, ale musí
    #                se pro každý časopis zjistit, co v patičce stojí.
    #   "position" - podle polohy: malý text u horního/dolního okraje
    #                stránky, jehož obsah je v podstatě jen číslo. Funguje
    #                na neznámém časopise bez kalibrace, za cenu občasného
    #                falešného poplachu (číslo v rohu grafu).
    #   "none"     - časopis běžící patičku nemá; mapování tištěných stránek
    #                se nepoužije.
    footer_detection: str = "keyword"
    # Regulární výraz, kterým se patička pozná v libovolném textu
    # (extract_blocks: takový blok se nikdy neklasifikuje jako nadpis).
    footer_pattern: str = ""
    # Přesnější test pro build_page_map/assign_articles: patička je vždy
    # týmž malým bezpatkovým písmem, jinak by se pletla s běžným textem.
    footer_font_prefixes: tuple[str, ...] = ()
    footer_max_size: float = 9.0
    footer_keywords: tuple[str, ...] = ()
    # Tokeny uvnitř patičky, které vypadají jako číslo, ale nejsou jím.
    footer_skip_tokens: tuple[str, ...] = ()
    # Jen pro "position": jak vysoko/nízko na stránce se patička hledá,
    # jako podíl výšky stránky, a kolik slov smí mít.
    footer_zone: float = 0.90
    header_zone: float = 0.08
    footer_max_tokens: int = 5

    # --- opakující se balast na každé stránce -------------------------------
    # Každá vnitřní n-tice je AND: blok je balast, když obsahuje VŠECHNY
    # její řetězce. Vnější tice je OR.
    junk_marker_sets: tuple[tuple[str, ...], ...] = ()

    # --- typografie ---------------------------------------------------------
    families: tuple[FontFamily, ...] = ()
    default_block_type: str = "body"

    # Adaptivní režim: velikosti se neberou absolutně z `families[*].rules`,
    # ale relativně k běžnému textu dokumentu (viz profiles/adaptive.py).
    # `families` pak slouží jen jako nositel fallbacků: [0] = rodina textu,
    # [1] = všechno ostatní.
    adaptive: bool = False
    relative_rules: tuple[RelativeRule, ...] = ()

    # --- chunking -----------------------------------------------------------
    # Kolik stránek na začátku/konci čísla je obálka a inzerce, tedy obsah,
    # který nemá jít do RAG.
    skip_first_pages: int = 2
    skip_last_pages: int = 2

    # Hlavička zapečená do textu, který jde do embedding modelu ("contextual
    # chunking"). Osamocený chunk bez kontextu říká modelu i LLM míň, proto
    # se před text lepí, ze kterého článku pochází. Jazyk hlavičky se řídí
    # jazykem korpusu, ne jazykem pipeline - proto je to v profilu.
    chunk_header_template: str = (
        "Časopis: {journal}\n"
        "Ročník: {year}\n"
        "Číslo: {issue}\n"
        "Článek: {title}\n"
        "Autoři: {author}\n"
        "Text: {text}"
    )
    unknown_author_label: str = "neuvedeno"

    # --- odpovídání ---------------------------------------------------------
    system_prompt_file: str = ""

    # --- odvozené (cache zkompilovaných regexů) -----------------------------
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
        """Vypadá text jako běžící hlavička/patička? (jen podle obsahu)"""
        rx = self.footer_re
        return bool(rx.search(text)) if rx else False

    def is_footer_block(self, font: str, size: float, text: str,
                        bbox=None, page_height: float = 0.0) -> bool:
        """Přísnější test než `is_footer_text`, který kromě obsahu bere
        v úvahu i font, velikost a (u strategie "position") polohu na stránce.

        Používá ho build_page_map (vytahuje z patičky číslo stránky)
        a assign_articles (patičku z textu článku vyhazuje). Kontrola fontu
        je u strategie "keyword" podstatná: samotné klíčové slovo se
        legitimně vyskytuje i v běžném textu ("časopis Živa vychází..."),
        a to by číslování stránek rozhodilo.
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
            # "keyword" bez jediného klíčového slova by prohlásil za patičku
            # každý malý text na stránce - to je horší než nedetekovat nic.
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
        if not self.toc_page_number_prefixes:
            return True
        if not any(font.startswith(p) for p in self.toc_page_number_prefixes):
            return False
        return ("Bold" in font) == self.toc_page_number_bold

    def system_prompt(self) -> str:
        if not self.system_prompt_file:
            raise ValueError(
                f"profil {self.key!r} nemá nastavený system_prompt_file")
        return (PROMPTS_DIR / self.system_prompt_file).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Registr profilů
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
            f"neznámý profil {key!r}; dostupné: {', '.join(sorted(reg))}")
    return reg[key]


def add_profile_argument(parser, default: str = "ziva"):
    """Sjednocený `--profile` přepínač pro všechny vstupní body pipeline."""
    parser.add_argument(
        "--profile", default=default, choices=available(),
        help=f"profil zdrojového časopisu (výchozí: {default})")
    return parser
