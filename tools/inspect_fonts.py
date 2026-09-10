"""Calibrating a new profile: what typesetting is actually in this PDF?

Writing a profile for a new magazine means answering three questions:
which fonts it uses, which point sizes correspond to a title, a heading
and body text, and what its running footer looks like. You cannot guess
those, and opening the PDF in a viewer and eyeballing sizes is not much
better - this script simply prints the three answers out of the PDF.

The output has three parts:

  1. **A typography histogram** - how many characters are set in each
     (font family, size, weight) combination. Weighting by CHARACTERS
     rather than by block count is deliberate: the most voluminous row
     of the histogram is by definition the article's body text and
     serves as the reference size for everything else.
  2. **Suggested rules** - a ready-made block of code to paste into a
     new profile, derived from the histogram. It is not finished truth,
     it is a starting point: read the samples at each size and decide
     what really is a title and what is merely a bold paragraph.
  3. **Footer candidates** - small text near the page edge, and how
     often a bare number appears there. The footer_* settings in the
     profile follow from this.

Usage:
    python -m tools.inspect_fonts path/to/issue.pdf
    python -m tools.inspect_fonts path/to/issue.pdf --samples 5 --pages 20
"""
import argparse
import re
from collections import Counter, defaultdict

import fitz

from magazine_rag.console import setup_console
from magazine_rag.typography import font_family, is_bold_font

# How many of the most voluminous combinations are printed. Beyond this
# there are only stray overflow spans, which no rule would be derived
# from anyway.
TOP_COMBINATIONS = 15

# A bare page number near the edge, Arabic or Roman. How often such a
# number occurs decides which footer-detection strategy to recommend.
PAGE_NUMBER_RE = re.compile(r"\d{1,4}|[IVXLCDMivxlcdm]{1,10}")


def collect(doc, max_pages=None):
    """Walk the document and gather typography statistics per span."""
    by_style = Counter()           # (family, size, bold) -> character count
    samples = defaultdict(list)    # the same -> sample texts
    footer_candidates = Counter()  # text near the page edge -> occurrences
    numeric_pages = set()          # pages with a bare number near the edge

    scanned = doc.page_count if max_pages is None else min(max_pages, doc.page_count)
    for page_index in range(scanned):
        page = doc[page_index]
        page_height = page.rect.height
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", ()):
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue
                    key = (font_family(span["font"]),
                           round(span["size"] * 2) / 2,
                           is_bold_font(span["font"]))
                    by_style[key] += len(text)
                    if len(samples[key]) < 20:
                        samples[key].append(text)

                    # footer: small text in the top or bottom tenth
                    y0, y1 = span["bbox"][1], span["bbox"][3]
                    near_edge = (y0 >= page_height * 0.90
                                 or y1 <= page_height * 0.10)
                    if near_edge and span["size"] < 12 and len(text.split()) <= 6:
                        footer_candidates[text] += 1
                        if PAGE_NUMBER_RE.fullmatch(text):
                            # Count PAGES, not occurrences: the page
                            # number is often set twice, at the top and
                            # the bottom, so summing occurrences would
                            # exceed the page count.
                            numeric_pages.add(page_index)
    return by_style, samples, footer_candidates, numeric_pages, scanned


def print_histogram(by_style, samples, n_samples):
    total = sum(by_style.values()) or 1
    print("=== Typography histogram (by character count) ===\n")
    print(f"{'family':<28} {'size':>6} {'weight':>7} {'chars':>9} {'share':>7}")
    print("-" * 72)
    for (family, size, bold), count in by_style.most_common(TOP_COMBINATIONS):
        print(f"{family[:28]:<28} {size:>6} {'bold' if bold else '':>7} "
              f"{count:>9} {count / total:>6.1%}")
        for sample in samples[(family, size, bold)][:n_samples]:
            print(f"        | {sample[:80]}")
    print()


def suggest_rules(by_style):
    """Derive a suggested set of profile rules from the histogram.

    The reference size is the most voluminous combination. The
    thresholds are the ones in profiles/adaptive.py, converted here into
    absolute points so they can be written as SizeRule.
    """
    if not by_style:
        print("There is no text in this document - is it a scan without OCR?\n"
              "This pipeline needs a born-digital PDF with embedded fonts.")
        return

    (body_family, body_size, _), _ = by_style.most_common(1)[0]
    families = Counter()
    for (family, _, _), count in by_style.items():
        families[family] += count
    other = [f for f, _ in families.most_common() if f != body_family][:3]

    print("=== Suggested profile (check it against the samples above!) ===\n")
    print(f"# reference typesetting: {body_family} @ {body_size} pt")
    print("SERIF = FontFamily(")
    print('    name="text",')
    print(f'    prefixes=("{body_family}",),')
    print("    rules=(")
    print(f'        SizeRule("title", size_min={round(body_size * 1.7, 1)}, bold=True),')
    print(f'        SizeRule("heading", size_min={round(body_size * 1.18, 1)}, '
          f'size_max={round(body_size * 1.7, 1)}, bold=True),')
    print(f'        SizeRule("other", size_min={round(body_size * 1.05, 1)}, '
          f'size_max={round(body_size * 1.18, 1)}, bold=False),')
    print("    ),")
    print('    fallback="body",')
    print(")")
    print("SANS = FontFamily(")
    print('    name="captions",')
    print(f'    prefixes=({", ".join(chr(34) + f + chr(34) for f in other)},),')
    print("    rules=(")
    print(f'        SizeRule("title", size_min={round(body_size * 3, 1)}),')
    print("    ),")
    print('    fallback="caption",')
    print("    detect_annotations=True,")
    print(")\n")
    print("If the samples do not bear these thresholds out, there is no need "
          "to tune them\nby hand - the profile can instead set adaptive=True "
          "and have the reference\nsize derived from the document at runtime "
          "(see profiles/adaptive.py).\n")


def print_footers(footer_candidates, numeric_pages, doc_pages):
    """Print what repeats near the page edge - and, separately, how often
    a bare number stands there.

    Those two have to be counted differently, or the more important one
    is lost: the magazine's name in the footer is the SAME string on
    every page, whereas the page number is DIFFERENT on every page. A
    "does it repeat" filter therefore hides page numbers reliably, even
    though they are exactly what decides between the two detection
    strategies.
    """
    print("=== Running-footer candidates ===\n")

    numeric = len(numeric_pages)
    repeated = [(t, n) for t, n in footer_candidates.most_common(20)
                if n >= max(3, doc_pages * 0.1)
                and not PAGE_NUMBER_RE.fullmatch(t.strip())]

    if repeated:
        print("Text repeating near the page edge:")
        for text, n in repeated:
            print(f"  {n:>4}x  {text[:70]}")
        print()
    print(f"Pages with a bare number near the edge: {numeric} of {doc_pages}\n")

    if numeric >= doc_pages * 0.4:
        print('Recommendation: footer_detection="position" - the page number '
              "is near\nthe edge on most pages and can be found by position.")
        if repeated:
            print('("keyword" would work too, given the repeating text above, '
                  "but position\nis simpler and does not depend on the "
                  "language.)")
    elif repeated:
        print('Recommendation: footer_detection="keyword" - a bare number was '
              "not found\nnear the edge often enough, but the text above does "
              "repeat there. Fill in\nfooter_keywords and footer_pattern from "
              "it.")
    else:
        print("Nothing repeats near the page edge and there are no bare "
              "numbers there.\nThe magazine may have no running footer - then "
              'footer_detection="none",\nand expect articles not to map onto '
              "printed pages.")
    print()


def main():
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pdf")
    ap.add_argument("--pages", type=int, default=None,
                    help="look at the first N pages only (faster)")
    ap.add_argument("--samples", type=int, default=3,
                    help="how many sample texts to print per combination")
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    by_style, samples, footers, numeric_pages, scanned = collect(doc, args.pages)

    partial = f" (first {scanned} examined)" if scanned < doc.page_count else ""
    print(f"\n{args.pdf}: {doc.page_count} pages{partial}\n")
    print_histogram(by_style, samples, args.samples)
    suggest_rules(by_style)
    # The denominator must be the number of pages EXAMINED, not of the
    # whole issue - otherwise the shares are computed against pages the
    # script never looked at, and the recommendation comes out backwards.
    print_footers(footers, numeric_pages, scanned)


if __name__ == "__main__":
    main()
