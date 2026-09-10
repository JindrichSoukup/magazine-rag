"""Kalibrace nového profilu: co je v tom PDF vlastně za sazbu?

Napsat profil pro nový časopis znamená odpovědět na tři otázky: jaké fonty
se v něm používají, jaké velikosti písma odpovídají titulku / nadpisu /
běžnému textu, a jak vypadá běžící patička. Hádat se to nedá a otevírat
PDF v prohlížeči a měřit odhadem taky ne - tenhle skript ty tři odpovědi
z PDF prostě vypíše.

Výstup má tři části:

  1. **Histogram sazby** - kolik znaků je vysázeno kterou kombinací
     (rodina písma, velikost, řez). Vážení podle ZNAKŮ, ne podle počtu
     bloků, je záměr: nejobjemnější řádek histogramu je z definice běžný
     text článku a slouží jako referenční velikost pro všechno ostatní.
  2. **Návrh pravidel** - hotový blok kódu k vložení do nového profilu,
     odvozený z histogramu. Není to hotová pravda, je to odrazový můstek:
     projděte si ukázky u každé velikosti a rozhodněte, co je opravdu
     titulek a co jen tučný odstavec.
  3. **Kandidáti na patičku** - malé texty u okraje stránky, které se
     opakují napříč čísly. Podle nich se vyplní footer_* v profilu.

Použití:
    python -m tools.inspect_fonts cesta/k/cislu.pdf
    python -m tools.inspect_fonts cesta/k/cislu.pdf --samples 5 --pages 20
"""
import argparse
from collections import Counter, defaultdict

import fitz

from magrag.console import setup_console
from magrag.typography import font_family, is_bold_font

# Kolik nejobjemnějších kombinací se vypisuje. Nad tuhle hranici už jsou
# jen jednotlivé přeteklé spany, ze kterých se pravidlo stejně neodvozuje.
TOP_COMBINATIONS = 15


def collect(doc, max_pages=None):
    """Projdi dokument a posbírej statistiku sazby po spanech."""
    by_style = Counter()          # (rodina, velikost, tučnost) -> počet znaků
    samples = defaultdict(list)   # totéž -> ukázky textu
    footer_candidates = Counter()  # text u okraje stránky -> na kolika stránkách

    pages = doc.page_count if max_pages is None else min(max_pages, doc.page_count)
    for page_index in range(pages):
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

                    # patička: malý text v horní/dolní desetině stránky
                    y0, y1 = span["bbox"][1], span["bbox"][3]
                    near_edge = (y0 >= page_height * 0.90
                                 or y1 <= page_height * 0.10)
                    if near_edge and span["size"] < 12 and len(text.split()) <= 6:
                        footer_candidates[text] += 1
    return by_style, samples, footer_candidates


def print_histogram(by_style, samples, n_samples):
    total = sum(by_style.values()) or 1
    print("=== Histogram sazby (podle počtu znaků) ===\n")
    print(f"{'rodina':<28} {'vel.':>6} {'řez':>6} {'znaků':>9} {'podíl':>7}")
    print("-" * 72)
    for (family, size, bold), count in by_style.most_common(TOP_COMBINATIONS):
        print(f"{family[:28]:<28} {size:>6} {'bold' if bold else '':>6} "
              f"{count:>9} {count / total:>6.1%}")
        for sample in samples[(family, size, bold)][:n_samples]:
            print(f"        | {sample[:80]}")
    print()


def suggest_rules(by_style):
    """Z histogramu odvoď návrh pravidel do profilu.

    Referenční velikost = nejobjemnější kombinace. Prahy odpovídají těm
    v profiles/adaptive.py, jen se tady rovnou přepočítají na absolutní
    body, aby šly zapsat jako SizeRule.
    """
    if not by_style:
        print("V dokumentu není žádný text - je to sken bez OCR?\n"
              "Tahle pipeline potřebuje born-digital PDF s vloženými fonty.")
        return

    (body_family, body_size, _), _ = by_style.most_common(1)[0]
    families = Counter()
    for (family, _, _), count in by_style.items():
        families[family] += count
    other = [f for f, _ in families.most_common() if f != body_family][:3]

    print("=== Návrh do profilu (zkontrolujte podle ukázek výše!) ===\n")
    print(f"# referenční sazba: {body_family} @ {body_size} b")
    print("SERIF = FontFamily(")
    print(f'    name="text",')
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
    print(f'    name="captions",')
    print(f'    prefixes=({", ".join(chr(34) + f + chr(34) for f in other)},),')
    print("    rules=(")
    print(f'        SizeRule("title", size_min={round(body_size * 3, 1)}),')
    print("    ),")
    print('    fallback="caption",')
    print("    detect_annotations=True,")
    print(")\n")
    print("Když se prahy z ukázek nepotvrdí, není nutné je ladit ručně - "
          "profil může\nmísto toho nastavit adaptive=True a nechat referenční "
          "velikost odvodit\nz dokumentu za běhu (viz profiles/adaptive.py).\n")


def print_footers(footer_candidates, doc_pages):
    print("=== Kandidáti na běžící patičku ===\n")
    repeated = [(t, n) for t, n in footer_candidates.most_common(20)
                if n >= max(3, doc_pages * 0.1)]
    if not repeated:
        print("Nic, co by se u okraje stránky opakovalo. Časopis buď patičku\n"
              "nemá, nebo je v ní jen holé číslo stránky (pak v profilu\n"
              'nastavte footer_detection="position").\n')
        return
    for text, n in repeated:
        print(f"  {n:>4}x  {text[:70]}")
    print('\nJe-li v patičce název časopisu nebo jeho web, nastavte '
          'footer_detection="keyword"\na footer_keywords/footer_pattern podle '
          'výpisu výše. Je-li tam jen číslo,\nnastavte '
          'footer_detection="position".\n')


def main():
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pdf")
    ap.add_argument("--pages", type=int, default=None,
                    help="prohlédnout jen prvních N stránek (rychlejší)")
    ap.add_argument("--samples", type=int, default=3,
                    help="kolik ukázek textu vypsat u každé kombinace")
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    by_style, samples, footers = collect(doc, args.pages)

    print(f"\n{args.pdf}: {doc.page_count} stránek\n")
    print_histogram(by_style, samples, args.samples)
    suggest_rules(by_style)
    print_footers(footers, doc.page_count)


if __name__ == "__main__":
    main()
