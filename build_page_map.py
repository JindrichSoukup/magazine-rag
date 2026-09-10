"""
Stage 2a: figure out which real PDF page corresponds to which *printed* page
number (the one shown in the magazine's own running header/footer, e.g. "262"
or, in the back-matter, roman numerals like "CXXXIII").

Why we need this:
  ziva_toc.json (built earlier) lists articles with their PRINTED page number
  ("262", "266", ..., "CXXXIII", ...) because that's what's written on the
  contents page. But ziva_blocks.json indexes text by the PDF's own page
  number (1, 2, 3, ...). To know "article X starts on PDF page N" we must
  translate printed-page-labels -> pdf page indices.

How: every page (usually) has a running header/footer line looking like
    "ziva.avcr.cz 262 živa 6/2014"   or   "živa 6/2014 263 ziva.avcr.cz"
  which contains exactly one "loose" token that is either an all-digit
  number or an all-roman-numeral string. We grab that token.

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
import json
from pathlib import Path
import re

BLOCKS_PATH = Path("ziva_blocks.json")
OUT_PATH = Path("page_map.json")

ROMAN_RE = re.compile(r"^[IVXLCDM]+$", re.IGNORECASE)
DIGIT_RE = re.compile(r"^\d{1,4}$")

ROMAN_VALUES = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def roman_to_int(s: str) -> int:
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


def extract_label(text: str):
    """Return the page-number token hiding in a running header/footer line,
    or None if this doesn't look like one."""
    if "živa" not in text.lower() and "ziva" not in text.lower():
        return None
    for tok in text.split():
        tok = tok.strip()
        if tok in ("ziva.avcr.cz", "http://ziva.avcr.cz"):
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


def build_page_map(blocks):
    """blocks -> {"page_to_label": {...directly detected...},
                  "label_to_page": {...fully resolved, gap-filled...}}"""
    # 1) direct detections, per page
    page_to_label = {}
    for b in blocks:
        if not b["font"].startswith("HelveticaCE-Bold"):
            continue
        if b["font_size"] >= 9:
            continue
        label = extract_label(b["text"])
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
    blocks = json.loads(BLOCKS_PATH.read_text(encoding="utf-8"))
    result = build_page_map(blocks)
    page_to_label, label_to_page = result["page_to_label"], result["label_to_page"]

    OUT_PATH.write_text(
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
