"""
Diagnostika: vypiš syrové spany na stránce s obsahem čísla kolem "XXX",
"XXXI", "XXXI-II", ať vidíme přesně font/tučnost/text a zjistíme, proč
create_toc.py neudělal správný split na 3 záznamy.

Použití: python diag_toc.py ziva-2019-1.pdf
"""
import sys
import fitz

pdf_path = sys.argv[1]
doc = fitz.open(pdf_path)
page = doc[2]  # stránka s obsahem - stejný index jako v create_toc.py

for block in page.get_text("dict")["blocks"]:
    if "lines" not in block:
        continue
    for line in block["lines"]:
        for span in line["spans"]:
            txt = span["text"].strip()
            if not txt:
                continue
            # vypiš jen okolí těch problematických řádků, ať to není
            # zahlcené celým obsahem
            if any(k in txt for k in ("XXX", "Zaujalo", "olympi", "předplat", "Aktuality", "Kontaktní", "Summary")):
                print(repr(txt), "| font:", span["font"], "| size:", round(span["size"], 1),
                      "| color:", span["color"])
