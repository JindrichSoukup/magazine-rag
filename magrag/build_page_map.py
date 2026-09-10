"""Stage 2a: work out which real PDF page corresponds to which *printed*
page number - the one visible in the magazine's running header or footer,
e.g. "262", or in the supplement in Roman numerals, "CXXXIII".

Why this is needed:
  toc.json from the previous stage records each article's PRINTED page
  number ("262", "266", ..., "CXXXIII", ...), because that is what the
  contents page says. But blocks.json indexes text by the PDF's own page
  number (1, 2, 3, ...). To know that "article X starts on PDF page N",
  the printed labels have to be translated into PDF page indices.

How: almost every page carries a running header or footer that looks
  something like
    "ziva.avcr.cz 262 živa 6/2014"   or   "živa 6/2014 263 ziva.avcr.cz"
  and contains exactly one "loose" token that is either all digits or all
  Roman numerals. That is the one we take. What such a footer looks like
  for a given magazine (font, size, keyword, tokens to skip) comes from
  the source profile - see magrag/profiles/.

Two findings from running this over a real archive:

  - A page-number token that is a SINGLE Roman numeral ("I", "V", "X",
    "C", ...) used to be rejected as "too ambiguous". That was wrong: by
    the time we look at individual tokens we have already confirmed the
    whole line is a running header, so nothing ambiguous is left.
  - An earlier version assumed one constant offset (pdf_page minus
    printed value) per numbering *scheme*, Arabic versus Roman. It turns
    out a single issue can run the Arabic numbering TWICE with a big jump
    in between: main articles 262-284, then a Roman-numbered supplement
    CXXXIII-CLXIV, then Arabic resumes at 285. So "Arabic" is not one
    offset, it is two. Instead we now detect contiguous RUNS - maximal
    stretches where the printed number keeps the same offset from the PDF
    page number - and fit an offset per run. Missing footers *inside* a
    run (a full-page photo with no header) are filled in from that run's
    offset; only pages no run covers at all stay unresolved.
"""
import argparse
import json
import re
from pathlib import Path

from magrag import profiles
from magrag.console import setup_console

# The upper length bound is not cosmetic: without it any long enough run
# of the letters I/V/X/L/C/D/M passes as a "Roman numeral" - and under
# position-based footer detection (see the magpi profile) such a string
# really does turn up, e.g. a word made entirely of x's. The longest page
# number that makes sense in a magazine is about eight characters.
ROMAN_RE = re.compile(r"^[IVXLCDM]{1,10}$", re.IGNORECASE)
DIGIT_RE = re.compile(r"^\d{1,4}$")

ROMAN_VALUES = [
    (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
    (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
    (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"),
]


def roman_to_int(s: str) -> int:
    # Upper-casing has to happen HERE rather than at the call site:
    # without it "CXLVIiI" (a typesetting slip with a lowercase i) does
    # not fail loudly but quietly becomes 146 instead of 148 - a wrong
    # page number that only surfaces three stages later, as an article
    # assigned to the wrong place.
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


def normalize_label(label: str) -> str:
    """Normalise a page label to its canonical form.

    Necessary because the same label arrives from two independent places
    and need not be written identically. A real case (The MagPi): the
    contents give "032" with a leading zero, the running footer on the
    page gives just "32" - and without normalising, the article maps to
    no page at all even though the numbering was detected flawlessly. The
    same reasoning applies to the case of Roman numerals.

    An unrecognised form is returned unchanged, so nothing is lost
    silently.
    """
    scheme, value = label_to_int(str(label).strip())
    return int_to_label(scheme, value) if scheme else str(label).strip()


def extract_label(text: str, profile):
    """Return the page-number token hidden in a running header or footer,
    or None when it does not look like one.

    Note: under the "keyword" strategy the keyword is checked AGAIN here
    even though is_footer_block() already verified it. That is not
    redundant - extract_label() is also called on its own from the
    diagnostic tools, which have no block to hand.
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
            return tok.upper()  # typesetting slip: a lowercase letter
                                # creeps in now and then ("CXLVIiI") -
                                # normalise so it matches the upper case
                                # used everywhere else in the pipeline
    return None


def find_runs(detections):
    """detections: a sorted list of (pdf_page, scheme, value).

    Returns a list of runs: {"scheme", "offset", "value_min", "value_max"}
    where offset = pdf_page - value is constant across the whole run.

    Note: a run keeps extending as long as the OFFSET matches - it is not
    required that the value increase by exactly 1 each time. That is
    deliberate: if a single page in the middle of a run has no footer at
    all (a full-bleed photo), we still want to bridge the gap as long as
    the detections either side agree on the same offset. Two genuinely
    unrelated runs landing on the same offset by chance is vanishingly
    unlikely in a real magazine's pagination.
    """
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
    """Extend each run a little way into the gaps around it.

    The first or last page of a numbered section sometimes has no
    directly detectable footer - the section end is formatted
    differently, and so on - so a run is short by one or two values at
    one end even though its offset is solidly confirmed by many
    consecutive detections inside it.

    We extend it safely by at most `max_extend` further values on EACH
    side, stopping as soon as we reach a PDF page already "owned" by
    ANOTHER run, whether of the same scheme or a different one. That
    guarantees two runs can never steal a page from each other. The cap
    is deliberately small, a few pages: this is meant as a safe minor
    correction for an edge page without a footer, not unbounded guessing
    into a large gap. A large gap is probably an entire separate,
    never-detected section, which we would simply guess wrong.

    Runs are processed in PDF-page order, so when two neighbouring runs
    both lay claim to a gap, the nearer one wins because it is processed
    first. That is exactly what happened in the real case: CXLVIII
    belonged to the Roman run, not the adjacent Arabic one, because the
    Roman run was nearer and extended into it first.
    """
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
    # Page height is not recorded in blocks.json but is reliably derived:
    # the largest y coordinate across the whole issue is the bottom edge
    # of the type area. Only the "position" strategy needs it.
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

    # Sanity check: a genuine single-page misdetection shows up as its own
    # length-1 run sandwiched between two much longer runs of the same
    # scheme and offset. Not fatal, just flagged so it is easy to notice.
    lone_runs = [r for r in runs if r["value_min"] == r["value_max"]]
    if len(lone_runs) > max(2, len(runs) * 0.2):
        print(f"  [warn] {len(lone_runs)}/{len(runs)} detected page-number "
              f"runs are only 1 page long - footer detection may be flaky "
              f"on this issue's layout")

    # 1b) extend each run by a few pages on both sides where it does not
    #     collide with another run (see extend_runs_into_gaps above) -----
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
        description="map printed page numbers onto PDF pages")
    ap.add_argument("blocks", help="input blocks.json (from extract_blocks)")
    ap.add_argument("out", help="output page_map.json")
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
    print(f"Directly detected {len(page_to_label)} / {n_pages} pages across "
          f"{len(result['runs'])} contiguous numbering runs; after filling "
          f"gaps, label_to_page covers {len(label_to_page)} labels.")
    missing = sorted(set(b["page"] for b in blocks) - set(page_to_label))
    if missing:
        print("Pages with no directly detected label (cover, contents, "
              "full-page images...):", missing)


if __name__ == "__main__":
    main()
