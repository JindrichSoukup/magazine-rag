"""
Diagnostika: vypiš VŠECHNY textové spany na konkrétní PDF stránce (font,
velikost, text) - použít, když víme, že tištěné číslo stránky je na
stránce vidět, ale build_page_map.py ho nedetekoval jako patičku.

Použití (pdf_page je 1-indexováno, stejně jako v ziva_blocks.json):
    python diag_page_spans.py ziva-2024-5.pdf 56
"""
import sys
import fitz

pdf_path, pdf_page = sys.argv[1], int(sys.argv[2])
doc = fitz.open(pdf_path)
page = doc[pdf_page - 1]

print(f"Stránka {pdf_page} (0-indexováno {pdf_page - 1}) z {pdf_path}\n")
for block in page.get_text("dict")["blocks"]:
    if "lines" not in block:
        continue
    for line in block["lines"]:
        for span in line["spans"]:
            txt = span["text"].strip()
            if not txt:
                continue
            print(f"  y0={span['bbox'][1]:.1f}  font={span['font']!r}  "
                  f"size={span['size']:.2f}  text={txt!r}")
