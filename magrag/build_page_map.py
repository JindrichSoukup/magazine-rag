"""
Fáze 2a: zjisti, která skutečná stránka PDF odpovídá kterému *tištěnému*
číslu stránky (tomu, které je vidět v běžící hlavičce/patičce časopisu,
např. "262", nebo v příloze římsky "CXXXIII").

Proč to potřebujeme:
  toc.json (z předchozí fáze) uvádí u článků TIŠTĚNÉ číslo stránky
  ("262", "266", ..., "CXXXIII", ...), protože to je to, co je natištěné
  v obsahu. Jenže blocks.json indexuje text podle vlastní stránky PDF
  (1, 2, 3, ...). Abychom věděli "článek X začíná na PDF stránce N",
  musíme tištěné popisky přeložit na indexy stránek PDF.

Jak: skoro každá stránka má běžící hlavičku/patičku, která vypadá třeba
    "ziva.avcr.cz 262 živa 6/2014"   nebo   "živa 6/2014 263 ziva.avcr.cz"
  a obsahuje přesně jeden "volný" token, který je buď celý číslo, nebo
  celý římská číslice. Ten si vezmeme. Jak taková patička u konkrétního
  časopisu vypadá (font, velikost, klíčové slovo, tokeny k přeskočení),
  určuje profil zdroje - viz magrag/profiles/.

v3 changes (after real-world testing on other issues):
  - a page-number token that is a SINGLE roman-numeral character ("I", "V",
    "X", "C", ...) used to be rejected as "too ambiguous". That was wrong:
    by the time we look at individual tokens we've already confirmed the
    whole line looks like a running header (it contains "živa"/"ziva"), so
    there's nothing ambiguous left. Fixed.
  - v2 assumed a single constant offset (pdf_page - printed_value) per
    numbering *scheme* (arabic vs roman). Turns out a single issue can have
    the arabic numbering run TWICE with a big jump in between (main
    articles 262-284, then a roman-numbered "kulérová příloha" CXXXIII-
    CLXIV, then arabic resumes at 285) - so "arabic" isn't one offset, it's
    two. v3 instead detects contiguous RUNS (maximal stretches where the
    printed number keeps increasing by exactly the same amount as the pdf
    page number) and fits an offset per run. Missing footers *inside* a run
    (e.g. a full-page photo with no header) are filled in from that run's
    offset; only pages with no run covering them at all stay unresolved.
"""
import argparse
import json
import re
from pathlib import Path

from magrag import profiles
from magrag.console import setup_console

# Horní mez délky není kosmetika: bez ní projde jako "římská číslice"
# jakýkoli dost dlouhý shluk písmen I/V/X/L/C/D/M - a u detekce patičky
# podle polohy (viz profil magpi) se takový řetězec reálně objeví
# (např. slovo poskládané ze samých x). Nejdelší číslo stránky, které
# dává v časopise smysl, má kolem osmi znaků.
ROMAN_RE = re.compile(r"^[IVXLCDM]{1,10}$", re.IGNORECASE)
DIGIT_RE = re.compile(r"^\d{1,4}$")

ROMAN_VALUES = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def roman_to_int(s: str) -> int:
    # Normalizace na velká písmena musí být TADY, ne až u volajícího:
    # bez ní se "CXLVIiI" (sazečský šotek s malým i) nerozbije nahlas,
    # ale tiše se z něj stane 146 místo 148 - tedy špatné číslo stránky,
    # které se pozná až o tři fáze dál jako článek přiřazený jinam.
    s = (s or "").strip().upper()
    i, total = 0, 0
    for value, symbol in ROMAN_VALUES:
        while s[i:i + len(symbol)] == symbol:
            total += value
            i += len(symbol)
    return total


def int_to_roman(n: int) -> str:
    out = []
    for value, symbol in ROMAN_VALUES:
        while n >= value:
            out.append(symbol)
            n -= value
    return "".join(out)


def label_to_int(label: str):
    """'262' -> ('arabic', 262), 'CXXXIII' -> ('roman', 133)."""
    if DIGIT_RE.match(label):
        return "arabic", int(label)
    if ROMAN_RE.match(label):
        return "roman", roman_to_int(label)
    return None, None


def int_to_label(scheme: str, value: int) -> str:
    return str(value) if scheme == "arabic" else int_to_roman(value)


def extract_label(text: str, profile):
    """Vrať token s číslem stránky schovaný v běžící hlavičce/patičce,
    nebo None, když to jako patička nevypadá.

    Pozor: strategie "keyword" tady kontroluje klíčové slovo ZNOVU, i když
    už ho ověřil is_footer_block(). Není to zbytečné - extract_label() se
    volá i samostatně z diagnostických nástrojů, které blok nemají.
    """
    if profile.footer_detection == "keyword" and not profile._has_footer_keyword(text):
        return None
    for tok in text.split():
        tok = tok.strip()
        if tok in profile.footer_skip_tokens:
            continue
        if "/" in tok:  # e.g. "6/2014" - issue/year, never the page number
            continue
        if DIGIT_RE.match(tok):
            return tok
        if ROMAN_RE.match(tok):
            return tok.upper()  # sazečský šotek: občas se vloudí malé písmeno
                                 # ("CXLVIiI") - normalizuj, ať to sedí s
                                 # velkými písmeny všude jinde v pipeline
    return None


def find_runs(detections):
    """detections: sorted list of (pdf_page, scheme, value).
    Returns a list of runs: {"scheme", "offset", "value_min", "value_max"}
    where offset = pdf_page - value is constant across the whole run.

    Note: we only require the offset to match to keep extending a run - NOT
    that the value increases by exactly 1 each time. That's on purpose: if
    a single page in the middle of a run has no footer at all (e.g. a
    full-bleed photo), we still want to bridge across that gap as long as
    the detections before and after it agree on the same offset. Two
    genuinely unrelated runs landing on the exact same offset by chance is
    extremely unlikely in practice for a real magazine's pagination."""
    runs = []
    for pdf_page, scheme, value in detections:
        offset = pdf_page - value
        if (runs and runs[-1]["scheme"] == scheme
                and runs[-1]["offset"] == offset):
            runs[-1]["value_max"] = max(runs[-1]["value_max"], value)
            runs[-1]["value_min"] = min(runs[-1]["value_min"], value)
        else:
            runs.append({"scheme": scheme, "offset": offset,
                         "value_min": value, "value_max": value})
    return runs


def extend_runs_into_gaps(runs, total_pdf_pages, max_extend=5):
    """Poslední/první stránka nějaké číslované sekce občas nemá přímo
    detekovatelnou patičku (jiné formátování konce sekce apod.), takže
    runu chybí přesně 1-2 hodnoty na jednom z konců, i když jeho offset
    je jinak spolehlivě ověřený z mnoha po sobě jdoucích detekcí uvnitř.

    Bezpečně to natáhneme o dalších nejvýš `max_extend` hodnot na KAŽDÉ
    straně runu, dokud nenarazíme na PDF stránku, kterou už "vlastní" JINÝ
    run (ať už stejného, nebo jiného schématu) - to zaručuje, že si dva
    runy nikdy neukradnou stránku jeden druhému. Strop na max_extend je
    záměrně malý (pár stránek) - má to být bezpečná drobná korekce
    okrajové stránky bez patičky, ne neomezené hádání do velké mezery
    (velká mezera = pravděpodobně celý samostatný, nikdy nedetekovaný
    úsek, který bychom takhle jen špatně uhodli).

    Runy se zpracovávají v pořadí podle PDF stránky, takže když si o
    mezeru "řeknou" dva sousední runy zároveň, vyhrává ten, co je blíž
    (zpracuje se dřív) - přesně tak to bylo i v reálném nálezu (CXLVIII
    patřilo římskému runu, ne sousednímu arabskému, protože římský byl
    blíž a natáhl se tam první)."""
    runs = sorted(runs, key=lambda r: r["value_min"] + r["offset"])
    claimed = set()
    for r in runs:
        for v in range(r["value_min"], r["value_max"] + 1):
            claimed.add(v + r["offset"])

    for r in runs:
        for _ in range(max_extend):
            v = r["value_min"] - 1
            if v < 1 or (v + r["offset"]) < 1 or (v + r["offset"]) in claimed:
                break
            claimed.add(v + r["offset"])
            r["value_min"] = v
        for _ in range(max_extend):
            v = r["value_max"] + 1
            if (v + r["offset"]) > total_pdf_pages or (v + r["offset"]) in claimed:
                break
            claimed.add(v + r["offset"])
            r["value_max"] = v
    return runs


def build_page_map(blocks, profile):
    """blocks -> {"page_to_label": {...directly detected...},
                  "label_to_page": {...fully resolved, gap-filled...}}"""
    # Výška stránky se v blocks.json neuvádí, ale spolehlivě se odvodí:
    # největší y-souřadnice napříč celým číslem je spodní okraj sazby.
    # Potřebuje ji jen strategie "position" (patička podle polohy).
    page_height = max((b["bbox"][3] for b in blocks), default=0.0)

    # 1) direct detections, per page
    page_to_label = {}
    for b in blocks:
        if not profile.is_footer_block(b["font"], b["font_size"], b["text"],
                                       b["bbox"], page_height):
            continue
        label = extract_label(b["text"], profile)
        if label:
            page_to_label.setdefault(b["page"], label)

    detections = []
    for pdf_page, label in sorted(page_to_label.items()):
        scheme, value = label_to_int(label)
        if scheme:
            detections.append((pdf_page, scheme, value))

    runs = find_runs(detections)

    # sanity check: a real single-page misdetection would show up as its
    # own length-1 run sandwiched between two much longer runs of the same
    # scheme+offset - not fatal, just flag it so it's easy to notice.
    lone_runs = [r for r in runs if r["value_min"] == r["value_max"]]
    if len(lone_runs) > max(2, len(runs) * 0.2):
        print(f"  [warn] {len(lone_runs)}/{len(runs)} detected page-number "
              f"runs are only 1 page long - footer detection may be flaky "
              f"on this issue's layout")

    # 1b) natáhni každý run o pár stránek na obě strany, pokud tam
    #     nekoliduje s jiným runem (viz extend_runs_into_gaps výše) -------
    total_pdf_pages = max(b["page"] for b in blocks)
    runs = extend_runs_into_gaps(runs, total_pdf_pages)

    # 2) build label_to_page by filling in the full value range of each run
    label_to_page = {}
    for run in runs:
        for value in range(run["value_min"], run["value_max"] + 1):
            label_to_page[int_to_label(run["scheme"], value)] = value + run["offset"]

    return {"page_to_label": page_to_label, "label_to_page": label_to_page,
            "runs": runs}


def main():
    setup_console()
    ap = argparse.ArgumentParser(
        description="mapování tištěných čísel stránek na stránky PDF")
    ap.add_argument("blocks", help="vstupní blocks.json (z extract_blocks)")
    ap.add_argument("out", help="výstupní page_map.json")
    profiles.add_profile_argument(ap)
    args = ap.parse_args()

    profile = profiles.get(args.profile)
    blocks = json.loads(Path(args.blocks).read_text(encoding="utf-8"))
    result = build_page_map(blocks, profile)
    page_to_label, label_to_page = result["page_to_label"], result["label_to_page"]

    Path(args.out).write_text(
        json.dumps({"page_to_label": page_to_label, "label_to_page": label_to_page,
                    "runs": result["runs"]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    n_pages = len({b["page"] for b in blocks})
    print(f"Přímo detekováno {len(page_to_label)} / {n_pages} stránek "
          f"v {len(result['runs'])} souvislých úsecích číslování; "
          f"po dopočítání mezer pokrývá label_to_page {len(label_to_page)} popisků.")
    missing = sorted(set(b["page"] for b in blocks) - set(page_to_label))
    if missing:
        print("Stránky bez přímo detekovaného popisku (obálka, obsah, "
              "celostránkové obrázky...):", missing)


if __name__ == "__main__":
    main()
