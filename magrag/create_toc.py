"""
Fáze 0: vytáhni obsah čísla (kdo/co/na jaké tištěné stránce) ze stránky
s obsahem -> toc.json

Logika: číslo tištěné stránky je v obsahu vysázené jiným řezem než zbytek
řádku (u Živy MeliorCE Bold) a slouží jako oddělovač položek; uvnitř jedné
položky odděluje titulek od autora BARVA prvního spanu. Obojí je vlastnost
konkrétní sazby, takže obojí přichází z profilu zdroje.

Pokud je pod jedním číslem stránky natištěno víc položek oddělených
středníkem ("24. ročník...; Zaujalo nás: ..."), rozdělí se na samostatné
záznamy TOC se stejnou printed_page - assign_articles.py je pak podle
vlastních nadpisů v textu rozliší (viz find_heading_positions tam).

    python -m magrag.create_toc --profile ziva vstup.pdf toc.json
"""
import argparse
import json
import re
from pathlib import Path

import fitz

from magrag import profiles
from magrag.console import setup_console

# Kolik prvních stránek se prohledá, když profil neurčuje, kde obsah čísla
# je (prázdné toc_page_indices) - viz find_toc_pages().
TOC_SEARCH_PAGES = 8
TOC_MIN_ENTRIES = 5


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def remove_footer(text, profile):
    """Ořízni titulek/autora od místa, kde na něj navazuje tiráž.

    Tiráž bývá vysázená jako pokračování posledního řádku obsahu, takže se
    bez tohohle připlete do titulku poslední položky.
    """
    for marker in profile.toc_drop_markers:
        if marker in text:
            text = text.split(marker)[0]
    return text.strip(" ,;")


def is_roman(text):
    # case-insensitive: sazečský šotek občas vloudí malé písmeno doprostřed
    # římské číslice ("CXLVIiI") - viz stejná oprava v build_page_map.py
    return bool(re.fullmatch(r"[IVXLCDM]+", text.strip(), re.IGNORECASE))


def is_page_number(text):
    text = text.strip()
    return text.isdigit() or is_roman(text)


# Zkrácený rozsah stránek, typicky u drobných zadních položek, které se
# vejdou na necelé dvě stránky: "XXXI–II" (= XXXI až XXXII) nebo arabsky
# "285-6" (= 285 až 286). Bez tohohle is_page_number() na takový token
# vůbec nesedne (obsahuje pomlčku/dlouhou pomlčku), takže se nepozná jako
# začátek nového záznamu v obsahu a jeho titulek se mylně přilepí k
# předchozímu záznamu.
RANGE_RE = re.compile(r"^([IVXLCDM]+|\d+)\s*[-–]\s*([IVXLCDM]+|\d+)$", re.IGNORECASE)


def page_span_value(text):
    """Vrať skutečnou hodnotu tištěné stránky pro token, který vypadá jako
    číslo stránky - i pro zkrácený rozsah (viz RANGE_RE výše). Vrací JEN
    počáteční stránku rozsahu - to je vše, co pipeline dál potřebuje
    (konec článku se stejně dopočítává podle začátku NÁSLEDUJÍCÍHO
    záznamu v obsahu, ne podle konce vlastního rozsahu)."""
    text = text.strip()
    if is_page_number(text):
        return text.upper() if is_roman(text) else text
    m = RANGE_RE.match(text)
    if m:
        value = m.group(1)
        return value.upper() if is_roman(value) else value
    return None


def is_page_span(span, profile):
    """Je tenhle span číslo tištěné stránky, tedy začátek nové položky?

    Nestačí, že text vypadá jako číslo - v titulcích jsou čísla běžně
    ("24. ročník", "Rok 1968"). Rozhoduje až řez písma, který je pro čísla
    stránek v obsahu vyhrazený (viz toc_page_number_prefixes v profilu).
    """
    txt = span["text"].strip()
    return (page_span_value(txt) is not None
            and profile.is_toc_page_number_font(span["font"]))


def split_title_author(spans):
    spans = [s for s in spans if s["text"].strip()]
    if not spans:
        return None, None
    title_color = spans[0]["color"]
    title, author, author_started = [], [], False
    for span in spans:
        txt = span["text"].strip()
        if not txt:
            continue
        if not author_started and span["color"] == title_color:
            title.append(txt)
        else:
            author_started = True
            author.append(txt)
    return clean(" ".join(title)), (clean(" ".join(author)) if author else None)


def count_toc_entries(page, profile):
    """Kolik "začátků položky" (= čísel stránek ve vyhrazeném řezu) je na
    téhle stránce. Slouží jen k detekci, na které stránce obsah je."""
    n = 0
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", ()):
            for span in line["spans"]:
                if span["text"].strip() and is_page_span(span, profile):
                    n += 1
    return n


def find_toc_pages(doc, profile):
    """Najdi stránku s obsahem čísla, když ji profil neurčuje napevno.

    U známého časopisu je obsah vždy na stejné fyzické stránce a hádat se
    nemá co (`toc_page_indices` v profilu). U neznámého to nikdo předem
    neví, takže se prohledá začátek čísla a vyhraje stránka s nejvíc
    položkami - obsah je z definice nejhustší nakupení čísel stránek
    v celém čísle. Prahem `TOC_MIN_ENTRIES` se odliší skutečný obsah od
    stránky, kde se pár čísel sešlo náhodou.
    """
    best_page, best_count = None, 0
    for i in range(min(TOC_SEARCH_PAGES, doc.page_count)):
        count = count_toc_entries(doc[i], profile)
        if count > best_count:
            best_page, best_count = i, count
    if best_page is None or best_count < TOC_MIN_ENTRIES:
        return ()
    return (best_page,)


def build_toc(pdf_path, profile, toc_page_indices=None):
    """toc_page_indices jsou 0-indexovaná čísla stránek PDF, na kterých je
    natištěný obsah čísla. Když se nepředají, vezmou se z profilu; když je
    ani profil neurčuje, zkusí se detekovat (viz find_toc_pages)."""
    doc = fitz.open(pdf_path)
    if toc_page_indices is None:
        toc_page_indices = profile.toc_page_indices or find_toc_pages(doc, profile)

    items_raw, current = [], None

    for page_index in toc_page_indices:
        page = doc[page_index]
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    txt = span["text"].strip()
                    if not txt:
                        continue
                    if is_page_span(span, profile):
                        if current:
                            items_raw.append(current)
                        current = {"page": page_span_value(txt), "spans": []}
                    elif current:
                        if not any(m in txt for m in profile.toc_skip_span_markers):
                            current["spans"].append({
                                "text": txt, "color": span["color"],
                                "font": span["font"], "size": span["size"],
                            })
    if current:
        items_raw.append(current)

    toc = []
    for item in items_raw:
        title, author = split_title_author(item["spans"])
        if title:
            # Časopis občas natiskne víc krátkých položek na jedné stránce
            # pod JEDNÍM společným číslem stránky v obsahu, oddělené
            # středníkem v titulku ("24. ročník...; Zaujalo nás: ...").
            # Rozdělíme to na samostatné záznamy se stejnou printed_page -
            # assign_articles.py je pak uvnitř té sdílené stránky rozliší
            # podle vlastních nadpisů v textu (viz find_heading_positions).
            for part in title.split(";"):
                part = part.strip()
                if part:
                    toc.append({
                        "printed_page": item["page"],
                        "title": remove_footer(part, profile),
                        "author": remove_footer(author, profile) if author else None,
                    })
    return toc


def main():
    setup_console()
    ap = argparse.ArgumentParser(description="vytáhni obsah čísla z PDF")
    ap.add_argument("pdf", help="vstupní PDF jednoho čísla")
    ap.add_argument("out", help="výstupní toc.json")
    ap.add_argument("--toc-page", type=int, action="append", dest="toc_pages",
                    help="0-indexovaná stránka PDF s obsahem čísla; lze zadat "
                         "vícekrát. Bez tohohle se použije profil, případně "
                         "automatická detekce.")
    profiles.add_profile_argument(ap)
    args = ap.parse_args()

    profile = profiles.get(args.profile)
    toc = build_toc(args.pdf, profile,
                    tuple(args.toc_pages) if args.toc_pages else None)
    Path(args.out).write_text(
        json.dumps(toc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.pdf}: {len(toc)} položek obsahu -> {args.out}")


if __name__ == "__main__":
    main()
