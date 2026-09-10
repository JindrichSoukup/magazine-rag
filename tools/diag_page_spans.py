"""Diagnostika: vypiš VŠECHNY textové spany na konkrétní stránce PDF.

Použít, když víme, že tištěné číslo stránky je na stránce vidět, ale
build_page_map ho nedetekoval jako patičku - z výpisu je hned poznat,
jestli je problém ve fontu, velikosti, nebo poloze.

Použití (pdf_page je 1-indexováno, stejně jako v blocks.json):
    python -m tools.diag_page_spans cesta/k/cislu.pdf 56
"""
import argparse

import fitz

from magrag.console import setup_console


def main():
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pdf")
    ap.add_argument("page", type=int, help="číslo stránky PDF (1-indexováno)")
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    page = doc[args.page - 1]
    height = page.rect.height

    print(f"Stránka {args.page} (0-indexováno {args.page - 1}) z {args.pdf}, "
          f"výška {height:.0f} b\n")
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", ()):
            for span in line["spans"]:
                txt = span["text"].strip()
                if not txt:
                    continue
                y0 = span["bbox"][1]
                # podíl výšky stránky - přesně to, na co se dívá detekce
                # patičky podle polohy (footer_zone/header_zone v profilu)
                print(f"  y0={y0:7.1f} ({y0 / height:5.1%})  "
                      f"font={span['font']!r}  size={span['size']:.2f}  "
                      f"text={txt!r}")


if __name__ == "__main__":
    main()
