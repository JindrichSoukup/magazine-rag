"""Diagnostics: print EVERY text span on one specific PDF page.

Use it when the printed page number is plainly visible on the page but
build_page_map did not detect it as a footer - the output makes it
immediately clear whether the problem is the font, the size or the
position.

Usage (pdf_page is 1-indexed, as in blocks.json):
    python -m tools.diag_page_spans path/to/issue.pdf 56
"""
import argparse

import fitz

from magazine_rag.console import setup_console


def main():
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pdf")
    ap.add_argument("page", type=int, help="PDF page number (1-indexed)")
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    page = doc[args.page - 1]
    height = page.rect.height

    print(f"Page {args.page} (zero-indexed {args.page - 1}) of {args.pdf}, "
          f"height {height:.0f} pt\n")
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", ()):
            for span in line["spans"]:
                txt = span["text"].strip()
                if not txt:
                    continue
                y0 = span["bbox"][1]
                # The fraction of page height - exactly what
                # position-based footer detection looks at
                # (footer_zone/header_zone in the profile).
                print(f"  y0={y0:7.1f} ({y0 / height:5.1%})  "
                      f"font={span['font']!r}  size={span['size']:.2f}  "
                      f"text={txt!r}")


if __name__ == "__main__":
    main()
