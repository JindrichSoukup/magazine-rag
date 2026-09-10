"""Diagnostics: print the raw spans on the issue's contents page.

Use it when create_toc splits the contents into a different number of
records than there are items on the page. The output shows each span's
font, weight and colour - exactly what decides where one entry ends and
the next begins.

Usage:
    python -m tools.diag_toc path/to/issue.pdf
    python -m tools.diag_toc path/to/issue.pdf --filter XXX --filter Summary
"""
import argparse

import fitz

from magazine_rag import profiles
from magazine_rag.console import setup_console
from magazine_rag.create_toc import find_toc_pages, is_page_span


def main():
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pdf")
    ap.add_argument("--toc-page", type=int, default=None,
                    help="zero-indexed page holding the contents; without it "
                         "the profile is used, or failing that detection")
    ap.add_argument("--filter", action="append", dest="filters",
                    help="print only spans containing this substring; may be "
                         "given more than once. Without a filter the whole "
                         "contents page is printed.")
    profiles.add_profile_argument(ap)
    args = ap.parse_args()

    profile = profiles.get(args.profile)
    doc = fitz.open(args.pdf)

    if args.toc_page is not None:
        pages = (args.toc_page,)
    else:
        pages = profile.toc_page_indices or find_toc_pages(doc, profile)
    if not pages:
        print("Could not find the contents page - try --toc-page.")
        return

    for page_index in pages:
        print(f"=== PDF page {page_index} (zero-indexed) ===\n")
        for block in doc[page_index].get_text("dict")["blocks"]:
            for line in block.get("lines", ()):
                for span in line["spans"]:
                    txt = span["text"].strip()
                    if not txt:
                        continue
                    if args.filters and not any(f in txt for f in args.filters):
                        continue
                    # An asterisk marks a span the profile treats as a page
                    # number, i.e. as the start of a new contents entry.
                    mark = "*" if is_page_span(span, profile) else " "
                    print(f"{mark} {txt!r:<50} font={span['font']} "
                          f"size={span['size']:.1f} color={span['color']}")


if __name__ == "__main__":
    main()
