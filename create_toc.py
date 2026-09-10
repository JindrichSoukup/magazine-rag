"""
Stage 0: vytáhni obsah čísla (kdo/co/na jaké tištěné stránce) ze stránky
s obsahem -> ziva_toc.json

Oproti původní verzi je logika stejná (barva prvního spanu = titulek,
zbytek = autor; MeliorCE Bold = číslo tištěné stránky), jen zabalená do
funkce build_toc(), aby ji šlo volat i z run_all.py pro víc čísel najednou.
Používá se z příkazové řádky stejně jako dřív.

Nově navíc: pokud je pod jedním číslem stránky v obsahu natištěno víc
položek oddělených středníkem ("24. ročník...; Zaujalo nás: ..."), rozdělí
se na samostatné záznamy TOC se stejnou printed_page - assign_articles.py
je pak podle vlastních nadpisů v textu rozliší (viz find_heading_positions
v assign_articles.py).

    python create_toc.py ziva-2014-6.pdf ziva_toc.json
"""
import json
import re
import sys
from pathlib import Path

import fitz


def clean(text):
    return re.sub(r"\s+", " ", text).strip()


def remove_footer(text):
    bad = [
        "© Nakladatelství Academia",
        "SSČ AV ČR",
        "Přetisk článků",
        "http://ziva.avcr.cz",
        "www.ziva.avcr.cz",
    ]
    for b in bad:
        if b in text:
            text = text.split(b)[0]
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


def is_page_span(span):
    txt = span["text"].strip()
    return (page_span_value(txt) is not None
            and span["font"].startswith("MeliorCE")
            and "Bold" in span["font"])


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


def build_toc(pdf_path, toc_page_indices=(2,)):
    """toc_page_indices jsou 0-indexované čísla stránek PDF, na kterých je
    natištěný obsah čísla (u Živy to bývá vždy 3. fyzická stránka -> index 2,
    ale u jiných čísel to může sedět jinak, proto je to parametr)."""
    doc = fitz.open(pdf_path)
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
                    if is_page_span(span):
                        if current:
                            items_raw.append(current)
                        current = {"page": page_span_value(txt), "spans": []}
                    elif current:
                        if ("ziva.avcr.cz" not in txt
                                and "© Nakladatelství Academia" not in txt
                                and "Přetisk článků" not in txt):
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
            # Živa občas natiskne víc krátkých položek na jedné stránce pod
            # JEDNÍM společným číslem stránky v obsahu, oddělené středníkem
            # v titulku ("24. ročník...; Zaujalo nás: ..."). Rozdělíme to
            # na samostatné záznamy se stejnou printed_page - assign_articles.py
            # je pak uvnitř té sdílené stránky rozliší podle vlastních
            # nadpisů v textu (viz find_heading_positions tam).
            for part in title.split(";"):
                part = part.strip()
                if part:
                    toc.append({
                        "printed_page": item["page"],
                        "title": remove_footer(part),
                        "author": remove_footer(author) if author else None,
                    })
    return toc


def main():
    if len(sys.argv) != 3:
        print("použití: python create_toc.py vstup.pdf vystup_toc.json")
        sys.exit(1)
    pdf_path, out_path = sys.argv[1], sys.argv[2]
    toc = build_toc(pdf_path)
    Path(out_path).write_text(json.dumps(toc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{pdf_path}: {len(toc)} položek obsahu -> {out_path}")


if __name__ == "__main__":
    main()
