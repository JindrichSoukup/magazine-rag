"""
Stage 1: rozparsuj jedno PDF čísla Živy na textové bloky -> ziva_blocks.json

Tohle je skript, který dřív dělal ziva_blocks.json (rekonstruovaný zpětně
z výstupu, protože originál se ztratil). Princip:

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
     caption/annotation) podle použitého fontu, velikosti písma a (jen pro
     annotation) tvaru textu - to je čistě heuristika ušitá na layout Živy,
     uvidíte v classify() níž. "annotation" jsou panelové značky/měřítka
     nalepené přímo na obrázek (a/b/c, 1/2/3, "1 cm") - NE popisky obrázku.

Použití:
    python extract_blocks.py ziva-2014-6.pdf ziva_blocks.json
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

# --- ladicí konstanty -------------------------------------------------------
COLUMN_X_TOLERANCE = 4.0   # o kolik se smí lišit x0/x1 dvou bloků, aby se
                           # pořád považovaly za "stejný sloupec"
MERGE_Y_GAP_MAX = 6.0      # max. mezera (pt) mezi konci/začátky bloků,
                           # aby se ještě slévaly do jednoho

FOOTER_RE = re.compile(r"živa\s+\d/\d{4}|ziva\.avcr\.cz", re.IGNORECASE)

# --- "editorská hantýrka" nalepená přímo na obrázek/schéma ------------------
# Tohle NENÍ popisek obrázku (souvislý text vysvětlující, co je na obrázku
# vidět) - je to grafický prvek, kterým editor rozlišuje panely composite
# obrázku (a, b, c, ...), číslo/písmeno odkazu (1, 2, 3, ...) nebo udává
# měřítko (scale bar: "1 cm", "0,2 mm", "1 000 μm"). Zachytáváme jen
# jednoznačné případy - viz omezení v komentáři níž u ANNOTATION_PATTERNS.
# jedna nebo víc dvojic "číslo (s desetinnou čárkou/tečkou) + jednotka",
# pokrývá i víc měřítek slitých do jednoho textu ("0,2 mm 1 mm 2 mm")
SCALE_BAR_RE = re.compile(
    r"^([\d.,]+\s*(mm|cm|km|μm|µm|nm|m)\s*)+$", re.IGNORECASE)
LEGEND_WORDS = {"do", "nad", "pod", "až"}  # české spojky v legendě škály
UNIT_LABEL_RE = re.compile(r"^\[[^\[\]]{1,6}\]$")  # "[°C]", "[%]", "[m]"

# Skutečné popisky obrázků jsou v Živě sazeny STEJNÝM písmem jako běžný
# text (MeliorCE, ne malé bezpatkové), takže je classify() podle
# fontu/velikosti nerozezná od těla článku. Poznat je jde podle toho, že
# skoro vždy začínají odkazem na číslo obrázku hned na začátku bloku:
# "1 a 2 Nejnápadnějším příznakem...", "3 Schéma normálního...",
# "9 a 10 ...". (Zkoušel jsem tohle kombinovat ještě s kontrolou, že blok
# sedí v PDF hned vedle obrázku, ale u přechodů mezi články bývá popisek
# v surovém pořadí bloků docela daleko od "svého" obrázku - proto se
# spoléhá jen na tvar textu + minimální délku, aby to nechytlo krátké
# číslované nadpisy typu "1. Úvod".)
CAPTION_LEAD_RE = re.compile(
    r"^\d+(\s*(a|až|,|-|–)\s*\d+)*\s+[A-ZÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ]")


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
    znát polohu vůči konkrétnímu obrázku, což tenhle skript nezjišťuje.
    Klíčové v CELÉM textu jde ale jen o čísla/rozsahy/pár spojek - žádný
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

# Czech justified text breaks words across lines with a plain hyphen at the
# line end, e.g. "dlouhodo-" / "bé" (sometimes with a stray extra space
# around the hyphen too, from justification: "nerovnoměr - ný"). We want to
# glue these back into "dlouhodobé"/"nerovnoměrný". This must NOT touch the
# en-dash "–" (U+2013), which Czech typography uses for real phrase breaks
# ("1990 – 2000") - only the plain ASCII hyphen "-" is a line-wrap artifact.
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
    písma (ne tučnosti - viz styles_compatible výše). fitz sám dokáže do
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
    v originálním JSONu objevuje - viz komentář v assign_articles.py)."""
    char_counts = Counter()
    for text, font, size in spans:
        char_counts[font] += len(text)
    font = char_counts.most_common(1)[0][0] if char_counts else ""
    size = spans[0][2] if spans else 0.0
    return font, size


def classify(text, font, size, bbox, page_num):
    """Heuristika: typ bloku podle fontu/velikosti. Přizpůsobeno layoutu
    Živy (MeliorCE = hlavní patkové písmo těla textu a titulků,
    HelveticaCE/Arial = bezpatkové - popisky, patičky, obálka)."""
    is_bold = "Bold" in font
    is_melior = font.startswith("MeliorCE")
    is_sans = font.startswith("HelveticaCE") or font.startswith("Arial")

    # patička/hlavička s číslem stránky - necháváme jako "body", protože ji
    # stejně vždycky filtrujeme zvlášť podle FOOTER_RE (viz assign_articles.py)
    if FOOTER_RE.search(text):
        return "body"

    if is_melior:
        if is_bold and size >= 18:
            return "title"
        if is_bold and 13 <= size < 18:
            return "heading"
        # menší podnadpisy v zadní části čísla ("Kontaktní údaje pro
        # předplatitele", "Vědci z Akademie věd ČR oceněni Českou hlavou")
        # bývají přesně velikost 13, ale ne vždy tučně - proto zvlášť
        if not is_bold and size == 13:
            return "heading"
        if not is_bold and 11.5 <= size <= 12.5:
            return "other"          # autor/autoři článku
        return "body"

    if is_sans:
        if size >= 30:
            return "title"          # velké číslo na obálce, "6 /2014"
        if is_bold and 11 <= size <= 13:
            return "other"          # tučné titulky na obálce
        if is_diagram_annotation(text):
            return "annotation"     # panel a/b/c, číslo 1/2/3, měřítko 1 cm...
        return "caption"            # popisky obrázků, copyright, ...

    return "body"  # fallback pro cokoliv neobvyklého (ZapfDingbats apod.)


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


def extract_page(page, page_num):
    raw = page.get_text("dict")["blocks"]
    merged = []          # (kind, payload) kde kind je "text" nebo "image"
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

    out = []
    for block_id, entry in enumerate(merged):
        if entry["kind"] != "text":
            continue
        font, size = dominant_font_and_size(entry["spans"])
        btype = classify(entry["text"], font, size, entry["bbox"], page_num)

        text_stripped = entry["text"].strip()
        if (btype == "body" and len(text_stripped) > 20
                and CAPTION_LEAD_RE.match(text_stripped)):
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


def extract_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    blocks = []
    for i, page in enumerate(doc, start=1):  # 1-indexováno, jako v původním JSONu
        blocks.extend(extract_page(page, i))
    return blocks


def main():
    if len(sys.argv) != 3:
        print("použití: python extract_blocks.py vstup.pdf vystup_blocks.json")
        sys.exit(1)
    pdf_path, out_path = sys.argv[1], sys.argv[2]
    blocks = extract_pdf(pdf_path)
    Path(out_path).write_text(
        json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{pdf_path}: {len(blocks)} bloků -> {out_path}")


if __name__ == "__main__":
    main()
