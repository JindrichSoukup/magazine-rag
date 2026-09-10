"""Diagnostika: vypiš syrové spany na stránce s obsahem čísla.

Použít, když create_toc rozdělí obsah na jiný počet záznamů, než kolik
je na stránce položek - z výpisu je vidět font, tučnost a barva každého
spanu, tedy přesně to, podle čeho se rozhoduje, kde jedna položka končí
a další začíná.

Použití:
    python -m tools.diag_toc cesta/k/cislu.pdf
    python -m tools.diag_toc cesta/k/cislu.pdf --filter XXX --filter Summary
"""
import argparse

import fitz

from magrag import profiles
from magrag.console import setup_console
from magrag.create_toc import find_toc_pages, is_page_span


def main():
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pdf")
    ap.add_argument("--toc-page", type=int, default=None,
                    help="0-indexovaná stránka s obsahem; bez ní se vezme "
                         "z profilu, případně se detekuje")
    ap.add_argument("--filter", action="append", dest="filters",
                    help="vypiš jen spany obsahující tento řetězec; lze "
                         "zadat vícekrát. Bez filtru se vypíše celý obsah.")
    profiles.add_profile_argument(ap)
    args = ap.parse_args()

    profile = profiles.get(args.profile)
    doc = fitz.open(args.pdf)

    if args.toc_page is not None:
        pages = (args.toc_page,)
    else:
        pages = profile.toc_page_indices or find_toc_pages(doc, profile)
    if not pages:
        print("Stránku s obsahem se nepodařilo najít - zkuste --toc-page.")
        return

    for page_index in pages:
        print(f"=== PDF stránka {page_index} (0-indexováno) ===\n")
        for block in doc[page_index].get_text("dict")["blocks"]:
            for line in block.get("lines", ()):
                for span in line["spans"]:
                    txt = span["text"].strip()
                    if not txt:
                        continue
                    if args.filters and not any(f in txt for f in args.filters):
                        continue
                    # hvězdička = span, který profil bere jako číslo stránky,
                    # tedy jako začátek nové položky obsahu
                    mark = "*" if is_page_span(span, profile) else " "
                    print(f"{mark} {txt!r:<50} font={span['font']} "
                          f"size={span['size']:.1f} color={span['color']}")


if __name__ == "__main__":
    main()
