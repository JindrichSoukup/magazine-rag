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
import re
from collections import Counter, defaultdict

import fitz

from magrag.console import setup_console
from magrag.typography import font_family, is_bold_font

# Kolik nejobjemnějších kombinací se vypisuje. Nad tuhle hranici už jsou
# jen jednotlivé přeteklé spany, ze kterých se pravidlo stejně neodvozuje.
TOP_COMBINATIONS = 15

# Holé číslo stránky u okraje - arabské i římské. Podle toho, jak často
# takové číslo v čísle je, se vybírá strategie detekce patičky.
PAGE_NUMBER_RE = re.compile(r"\d{1,4}|[IVXLCDMivxlcdm]{1,10}")


def collect(doc, max_pages=None):
    """Projdi dokument a posbírej statistiku sazby po spanech."""
    by_style = Counter()          # (rodina, velikost, tučnost) -> počet znaků
    samples = defaultdict(list)   # totéž -> ukázky textu
    footer_candidates = Counter()  # text u okraje stránky -> kolikrát se objevil
    numeric_pages = set()          # stránky, kde u okraje stojí holé číslo

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

                    # patička: malý text v horní/dolní desetině stránky
                    y0, y1 = span["bbox"][1], span["bbox"][3]
                    near_edge = (y0 >= page_height * 0.90
                                 or y1 <= page_height * 0.10)
                    if near_edge and span["size"] < 12 and len(text.split()) <= 6:
                        footer_candidates[text] += 1
                        if PAGE_NUMBER_RE.fullmatch(text):
                            # počítají se STRÁNKY, ne výskyty: číslo stránky
                            # bývá vysázené dvakrát (nahoře i dole) a součet
                            # výskytů by pak přesáhl počet stránek
                            numeric_pages.add(page_index)
    return by_style, samples, footer_candidates, numeric_pages, scanned


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


def print_footers(footer_candidates, numeric_pages, doc_pages):
    """Vypiš, co se u okraje stránky opakuje - a zvlášť, kolikrát tam stojí
    holé číslo.

    Ty dvě věci se musí počítat každá jinak, jinak se ta důležitější ztratí:
    název časopisu v patičce je na každé stránce **týž řetězec**, kdežto
    číslo stránky je na každé stránce **jiné**. Filtr na "opakuje se" tedy
    čísla stránek spolehlivě schová, i když jsou to přesně ony, podle
    kterých se rozhoduje mezi oběma strategiemi detekce.
    """
    print("=== Kandidáti na běžící patičku ===\n")

    numeric = len(numeric_pages)
    repeated = [(t, n) for t, n in footer_candidates.most_common(20)
                if n >= max(3, doc_pages * 0.1)
                and not PAGE_NUMBER_RE.fullmatch(t.strip())]

    if repeated:
        print("Opakující se text u okraje stránky:")
        for text, n in repeated:
            print(f"  {n:>4}x  {text[:70]}")
        print()
    print(f"Stránek s holým číslem u okraje: {numeric} z {doc_pages}\n")

    if numeric >= doc_pages * 0.4:
        print('Doporučení: footer_detection="position" - číslo stránky je\n'
              "u okraje na většině stránek a dá se najít podle polohy.")
        if repeated:
            print('(Strategie "keyword" by taky šla, viz opakující se text '
                  "výše, ale\npoloha je jednodušší a nezávisí na jazyce.)")
    elif repeated:
        print('Doporučení: footer_detection="keyword" - holé číslo se u '
              "okraje\nnenašlo dost často, ale opakuje se tam text výše. "
              "Podle něj vyplňte\nfooter_keywords a footer_pattern.")
    else:
        print("U okraje stránky se neopakuje nic a holá čísla tam nejsou.\n"
              'Časopis možná běžící patičku nemá - pak footer_detection="none"\n'
              "a počítejte s tím, že se články nenamapují na tištěné stránky.")
    print()


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
    by_style, samples, footers, numeric_pages, scanned = collect(doc, args.pages)

    partial = f" (prohlédnuto prvních {scanned})" if scanned < doc.page_count else ""
    print(f"\n{args.pdf}: {doc.page_count} stránek{partial}\n")
    print_histogram(by_style, samples, args.samples)
    suggest_rules(by_style)
    # Jmenovatelem musí být počet PROHLÉDNUTÝCH stránek, ne celého čísla -
    # jinak se podíly počítají proti stránkám, do kterých se skript vůbec
    # nepodíval, a doporučení vyjde naopak.
    print_footers(footers, numeric_pages, scanned)


if __name__ == "__main__":
    main()
