"""Fáze 1: rozparsuj jedno PDF čísla časopisu na textové bloky -> blocks.json

Princip:

  1. fitz (PyMuPDF) rozdělí každou stránku na "raw" bloky podle toho, jak je
     PDF interně poskládané (typicky = jeden textový rámeček v InDesignu).
  2. Textový rámeček jednoho sloupce se ale někdy v PDF rozpadne na DVA raw
     bloky, i když vizuálně jde o jeden nepřerušený tok textu (nadpis
     podkapitoly hned navazující na odstavec). Poznáme to podle toho, že oba
     bloky mají skoro stejný x-rozsah (jsou ve stejném sloupci) a mezera
     mezi nimi je minimální (pár bodů) - o dost menší, než bývá mezera před
     opravdovým novým vizuálním blokem/nadpisem. Takové sousední raw bloky
     proto sléváme do jednoho logického bloku.
  3. Obrázky (raw blok typu 1) do výstupu nedáváme, ale ID pro ně
     "rezervujeme" - proto mají textové bloky v JSONu mezery v číslování
     (je to neškodné, jen to odpovídá tomu, jak PDF bloky reálně šly za
     sebou na stránce).
  4. Každému výslednému bloku přiřadíme typ (title/heading/other/body/
     caption/annotation). Tohle je jediný krok navázaný na konkrétní sazbu
     a je celý vytažený do profilu zdroje - viz `magrag/profiles/`
     a `magrag/typography.py`.

Použití:
    python -m magrag.extract_blocks --profile ziva vstup.pdf blocks.json
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

from magrag import profiles
from magrag.console import setup_console
from magrag.typography import (
    DocumentStats,
    classify_block,
    looks_like_caption_lead,
)

# --- ladicí konstanty -------------------------------------------------------
COLUMN_X_TOLERANCE = 4.0   # o kolik se smí lišit x0/x1 dvou bloků, aby se
                           # pořád považovaly za "stejný sloupec"
MERGE_Y_GAP_MAX = 6.0      # max. mezera (pt) mezi konci/začátky bloků,
                           # aby se ještě slévaly do jednoho

# Zarovnaný text láme slova na konci řádku obyčejnou pomlčkou, např.
# "dlouhodo-" / "bé" (občas i s mezerou navíc kolem pomlčky, jak to vyjde
# ze zarovnání: "nerovnoměr - ný"). Chceme je slepit zpátky na
# "dlouhodobé"/"nerovnoměrný". Nesmí se to dotknout dlouhé pomlčky "–"
# (U+2013), kterou česká typografie používá pro skutečné předěly
# ("1990 – 2000") - jen obyčejný ASCII spojovník je artefakt zalomení.
HYPHEN_BREAK_RE = re.compile(r"(\w)\s*-\s*$")


def looks_like_word_break(before: str, after: str) -> bool:
    m = HYPHEN_BREAK_RE.search(before)
    if not m:
        return False
    after = after.lstrip()
    # a real line-wrap continues the SAME word, so the next fragment starts
    # with a lowercase letter (a new sentence/heading/proper noun wouldn't)
    return bool(after) and after[0].isalpha() and after[0].islower()


def smart_join(parts):
    """Join text fragments (lines within a block, or blocks being merged)
    into one string, gluing back words that got split by a line-wrap
    hyphen instead of just inserting a space."""
    out = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if out and looks_like_word_break(out, part):
            out = HYPHEN_BREAK_RE.sub(r"\1", out)  # drop " - "/"-", keep the letter
            out += part
        elif out:
            out += " " + part
        else:
            out = part
    return out


def spans_of_lines(lines):
    """Vrať list (text, font, size) pro všechny spany v zadaných řádcích."""
    out = []
    for line in lines:
        for span in line["spans"]:
            out.append((span["text"], span["font"], span["size"]))
    return out


def text_of_lines(lines):
    """Slož text ze zadaných řádků, a přitom slep slova rozdělená
    zalomením řádku přes pomlčku (viz smart_join / HYPHEN_BREAK_RE výše)."""
    parts = []
    for line in lines:
        line_text = "".join(span["text"] for span in line["spans"])
        if line_text.strip():
            parts.append(line_text)
    return smart_join(parts)


def split_lines_by_style(lines, size_tol=3.0):
    """Rozděl řádky JEDNOHO syrového fitz bloku do skupin podle VELIKOSTI
    písma (ne tučnosti - viz styles_compatible níž). fitz sám dokáže do
    jednoho bloku spojit vizuálně blízký, ale obsahově NESOUVISEJÍCÍ text
    - typicky konec jednoho článku/recenze hned navazující na nadpis
    dalšího (reálně nalezeno: "Kontaktní adresy autorů" @ vel. 15 slité
    s koncem předchozí recenze @ vel. ~9 do jednoho fitz bloku, ještě než
    se k tomu dostane naše vlastní sloučovací logika mezi bloky). Bez
    rozdělení by delší/kratší fragment "přehlasoval" dominantní velikost
    (viz dominant_font_and_size) a nadpis by zmizel pod klasifikací
    "body"."""
    groups = []
    current = []
    current_size = None
    for line in lines:
        text = "".join(s["text"] for s in line["spans"]).strip()
        if not text:
            if current:
                current.append(line)
            continue
        _, size = dominant_font_and_size(
            [(s["text"], s["font"], s["size"]) for s in line["spans"]])
        if current_size is None:
            current = [line]
            current_size = size
        elif abs(size - current_size) <= size_tol:
            current.append(line)
        else:
            groups.append(current)
            current = [line]
            current_size = size
    if current:
        groups.append(current)
    return groups


def dominant_font_and_size(spans):
    """font = nejčastější rodina písma podle počtu znaků (aby krátký tučný
    podnadpis nepřebil font celého odstavce); size = velikost PRVNÍHO spanu
    (nadpisy bývají o chlup větší než tělo textu, a právě tahle drobnost se
    v původním JSONu objevuje - viz komentář v assign_articles.py)."""
    char_counts = Counter()
    for text, font, size in spans:
        char_counts[font] += len(text)
    font = char_counts.most_common(1)[0][0] if char_counts else ""
    size = spans[0][2] if spans else 0.0
    return font, size


def same_column(b1, b2):
    x0a, _, x1a, _ = b1["bbox"]
    x0b, _, x1b, _ = b2["bbox"]
    return (abs(x0a - x0b) <= COLUMN_X_TOLERANCE and
            abs(x1a - x1b) <= COLUMN_X_TOLERANCE * 3)  # x1 kolísá víc (zarovnání)


def styles_compatible(prev_entry, new_spans, size_tol=3.0):
    """Nedovol sloučení dvou bloků s výrazně odlišnou VELIKOSTÍ písma
    (typicky konec jednoho článku/recenze a nadpis hned následujícího -
    reálně nalezeno: "Kontaktní adresy autorů" @ vel. 15 slité s koncem
    předchozí recenze @ vel. ~9). Záměrně se NEPOROVNÁVÁ tučnost samotná -
    popisky obrázků běžně mají tučné číslo obrázku vedle normálního textu
    STEJNÉ velikosti (např. "1 Popis obrázku...") a jde o jeden a ten samý
    popisek, ne dva různé bloky - kontrola jen podle tučnosti by tohle
    zbytečně (a nesprávně) roztrhala."""
    _, prev_size = dominant_font_and_size(prev_entry["spans"])
    _, new_size = dominant_font_and_size(new_spans)
    return abs(prev_size - new_size) <= size_tol


def merge_page_entries(page):
    """Syrové fitz bloky jedné stránky -> logické bloky (viz body 2 a 3
    v hlavičce modulu). Vrací list položek `{"kind": "text"|"image", ...}`,
    kde místa po obrázcích zůstávají zachovaná kvůli číslování bloků."""
    raw = page.get_text("dict")["blocks"]
    merged = []
    prev_text_entry = None

    for raw_block in raw:
        if raw_block["type"] != 0:  # obrázek/kresba -> jen rezervuj místo
            merged.append({"kind": "image"})
            prev_text_entry = None  # přes obrázek se nikdy neslévá
            continue

        # fitz sám dokáže do jednoho syrového bloku spojit stylisticky
        # nesourodý text (viz styles_compatible výše) - rozdělíme ho proto
        # nejdřív na vnitřně stejnorodé skupiny řádků a KAŽDOU zpracujeme
        # jako svou vlastní jednotku (včetně sloučení se sousedním blokem)
        for lines in split_lines_by_style(raw_block["lines"]):
            spans = spans_of_lines(lines)
            text = text_of_lines(lines)
            if not text.strip():
                merged.append({"kind": "image"})  # prázdný textový rámeček
                prev_text_entry = None
                continue

            bbox = [
                min(l["bbox"][0] for l in lines),
                min(l["bbox"][1] for l in lines),
                max(l["bbox"][2] for l in lines),
                max(l["bbox"][3] for l in lines),
            ]
            entry = {"kind": "text", "bbox": bbox, "text": text, "spans": spans}

            if (prev_text_entry is not None
                    and same_column(prev_text_entry, entry)
                    and bbox[1] - prev_text_entry["bbox"][3] <= MERGE_Y_GAP_MAX
                    and styles_compatible(prev_text_entry, spans)):
                # slij do předchozího bloku místo založení nového
                prev_text_entry["text"] = smart_join([prev_text_entry["text"], text])
                prev_text_entry["spans"].extend(spans)
                prev_text_entry["bbox"][2] = max(prev_text_entry["bbox"][2], bbox[2])
                prev_text_entry["bbox"][3] = bbox[3]
            else:
                merged.append(entry)
                prev_text_entry = entry

    return merged


def blocks_from_entries(entries, page_num, profile, stats):
    out = []
    for block_id, entry in enumerate(entries):
        if entry["kind"] != "text":
            continue
        font, size = dominant_font_and_size(entry["spans"])
        btype = classify_block(profile, entry["text"], font, size, stats)

        if btype == "body" and looks_like_caption_lead(entry["text"]):
            btype = "caption"  # popisek obrázku sazený v běžném písmu textu

        out.append({
            "page": page_num,
            "block_id": block_id,
            "type": btype,
            "text": entry["text"].strip(),
            "font": font,
            "font_size": round(size, 2),
            "bbox": [round(v, 1) for v in entry["bbox"]],
        })
    return out


def extract_pdf(pdf_path, profile):
    """PDF -> list bloků. U adaptivního profilu proběhne dokument dvakrát:
    poprvé kvůli statistice sazby, podruhé kvůli vlastní klasifikaci."""
    doc = fitz.open(pdf_path)
    # 1-indexováno, jako v původním JSONu
    pages = [(i, merge_page_entries(page)) for i, page in enumerate(doc, start=1)]

    stats = None
    if profile.adaptive:
        stats = DocumentStats.from_spans(
            span
            for _, entries in pages
            for entry in entries if entry["kind"] == "text"
            for span in entry["spans"]
        )

    blocks = []
    for page_num, entries in pages:
        blocks.extend(blocks_from_entries(entries, page_num, profile, stats))
    return blocks


def main():
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pdf", help="vstupní PDF jednoho čísla")
    ap.add_argument("out", help="výstupní blocks.json")
    profiles.add_profile_argument(ap)
    args = ap.parse_args()

    profile = profiles.get(args.profile)
    blocks = extract_pdf(args.pdf, profile)
    Path(args.out).write_text(
        json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8")

    counts = Counter(b["type"] for b in blocks)
    print(f"{args.pdf}: {len(blocks)} bloků -> {args.out}")
    print("  " + ", ".join(f"{t}={n}" for t, n in counts.most_common()))


if __name__ == "__main__":
    main()
