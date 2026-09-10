"""
Fáze 0: vytáhni obsah čísla (kdo/co/na jaké tištěné stránce) ze stránky
s obsahem -> toc.json

Logika: číslo tištěné stránky je v obsahu vysázené jiným řezem než zbytek
řádku (u Živy MeliorCE Bold) a slouží jako oddělovač položek; uvnitř jedné
položky odděluje titulek od autora BARVA prvního spanu. Obojí je vlastnost
konkrétní sazby, takže obojí přichází z profilu zdroje.

Pokud je pod jedním číslem stránky natištěno víc položek oddělených
středníkem ("24. ročník...; Zaujalo nás: ..."), rozdělí se na samostatné
záznamy TOC se stejnou printed_page - assign_articles.py je pak podle
vlastních nadpisů v textu rozliší (viz find_heading_positions tam).

    python -m magrag.create_toc --profile ziva vstup.pdf toc.json
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import fitz

from magrag import profiles
from magrag.console import setup_console
from magrag.typography import font_style_key

# Kolik prvních stránek se prohledá, když profil neurčuje, kde obsah čísla
# je (prázdné toc_page_indices) - viz find_toc_pages().
TOC_SEARCH_PAGES = 8
TOC_MIN_ENTRIES = 5
# Jak hustá musí být další stránka oproti té nejlepší, aby se taky
# považovala za část obsahu (obsah bývá rozložený přes dvoustranu).
TOC_PEER_RATIO = 0.4
# Totéž pro dvojice "styl čísla + styl titulku" uvnitř obsahu: jak častá
# musí dvojice být oproti nejčastější, aby se brala za skutečné položky,
# a ne za ozdobnou upoutávku nebo název rubriky.
TOC_STYLE_PEER_RATIO = 0.4


def strip_control_chars(text: str) -> str:
    """Nahraď řídicí znaky mezerou.

    Do textu se dostávají z ozdobných glyfů sázených symbolovým fontem -
    reálný případ (MagPi): odrážka před každou položkou obsahu vyjde
    z PDF jako U+0007. V titulku článku nemá co dělat a rozbíjí porovnání
    titulku z obsahu s nadpisem nalezeným v těle čísla (texts_match
    v assign_articles). Řeší se to porovnáním znaků, ne regulárním
    výrazem s rozsahem - ten se špatně čte a snadno se v něm udělá chyba.
    """
    return "".join(" " if ch < " " or ch == "\x7f" else ch for ch in text)


def clean(text):
    return re.sub(r"\s+", " ", strip_control_chars(text)).strip()


def remove_footer(text, profile):
    """Ořízni titulek/autora od místa, kde na něj navazuje tiráž.

    Tiráž bývá vysázená jako pokračování posledního řádku obsahu, takže se
    bez tohohle připlete do titulku poslední položky.
    """
    for marker in profile.toc_drop_markers:
        if marker in text:
            text = text.split(marker)[0]
    return text.strip(" ,;")


def is_roman(text):
    # case-insensitive: sazečský šotek občas vloudí malé písmeno doprostřed
    # římské číslice ("CXLVIiI") - viz stejná oprava v build_page_map.py
    return bool(re.fullmatch(r"[IVXLCDM]+", text.strip(), re.IGNORECASE))


def is_page_number(text):
    text = text.strip()
    return text.isdigit() or is_roman(text)


# Zkrácený rozsah stránek, typicky u drobných zadních položek, které se
# vejdou na necelé dvě stránky: "XXXI–II" (= XXXI až XXXII) nebo arabsky
# "285-6" (= 285 až 286). Bez tohohle is_page_number() na takový token
# vůbec nesedne (obsahuje pomlčku/dlouhou pomlčku), takže se nepozná jako
# začátek nového záznamu v obsahu a jeho titulek se mylně přilepí k
# předchozímu záznamu.
RANGE_RE = re.compile(r"^([IVXLCDM]+|\d+)\s*[-–]\s*([IVXLCDM]+|\d+)$", re.IGNORECASE)


def page_span_value(text):
    """Vrať skutečnou hodnotu tištěné stránky pro token, který vypadá jako
    číslo stránky - i pro zkrácený rozsah (viz RANGE_RE výše). Vrací JEN
    počáteční stránku rozsahu - to je vše, co pipeline dál potřebuje
    (konec článku se stejně dopočítává podle začátku NÁSLEDUJÍCÍHO
    záznamu v obsahu, ne podle konce vlastního rozsahu).

    Text se nejdřív pročistí přes clean(): u MagPi 150 přichází číslo
    položky spolu s tabulátorem a ozdobnou odrážkou (U+0007) nalepenými
    do téhož spanu. Bez pročištění se takové číslo vůbec nepozná a celá
    položka obsahu se ztratí - beze stopy, protože nic nespadne.
    """
    text = clean(text)
    if is_page_number(text):
        return text.upper() if is_roman(text) else text
    m = RANGE_RE.match(text)
    if m:
        value = m.group(1)
        return value.upper() if is_roman(value) else value
    return None


def is_page_span(span, profile):
    """Je tenhle span číslo tištěné stránky, tedy začátek nové položky?

    Nestačí, že text vypadá jako číslo - v titulcích jsou čísla běžně
    ("24. ročník", "Rok 1968"). Rozhoduje až řez písma, který je pro čísla
    stránek v obsahu vyhrazený (viz toc_page_number_prefixes v profilu).
    """
    txt = span["text"].strip()
    return (page_span_value(txt) is not None
            and profile.is_toc_page_number_font(span["font"]))


def split_title_author(spans):
    spans = [s for s in spans if s["text"].strip()]
    if not spans:
        return None, None
    title_color = spans[0]["color"]
    title, author, author_started = [], [], False
    for span in spans:
        txt = span["text"].strip()
        if not txt:
            continue
        if not author_started and span["color"] == title_color:
            title.append(txt)
        else:
            author_started = True
            author.append(txt)
    return clean(" ".join(title)), (clean(" ".join(author)) if author else None)


def count_toc_entries(page, profile):
    """Kolik "začátků položky" (= čísel stránek ve vyhrazeném řezu) je na
    téhle stránce. Slouží jen k detekci, na které stránce obsah je."""
    n = 0
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", ()):
            for span in line["spans"]:
                if span["text"].strip() and is_page_span(span, profile):
                    n += 1
    return n


def find_toc_pages(doc, profile):
    """Najdi stránky s obsahem čísla, když je profil neurčuje napevno.

    U známého časopisu je obsah vždy na stejné fyzické stránce a hádat se
    nemá co (`toc_page_indices` v profilu). U neznámého to nikdo předem
    neví, takže se prohledá začátek čísla a hledá se nejhustší nakupení
    čísel stránek - obsah je takové nakupení z definice.

    **Stránek je víc než jedna.** Obsah bývá rozložený přes dvoustranu,
    u tlustšího čísla i přes tři stránky proložené inzercí (reálně: MagPi
    má obsah na stranách 5, 6 a 8). Vzít jen tu nejlepší znamená přijít
    o dvě třetiny čísla - a nepozná se to jako chyba, protože zbytek
    pipeline poslušně zpracuje to, co dostal.

    Práh je dvojí: absolutní `TOC_MIN_ENTRIES` odliší obsah od stránky,
    kde se pár čísel sešlo náhodou, a relativní `TOC_PEER_RATIO` k té
    nejlepší stránce přibere její protějšky, ale ne stránku s jedním
    zatoulaným číslem.
    """
    counts = {i: count_toc_entries(doc[i], profile)
              for i in range(min(TOC_SEARCH_PAGES, doc.page_count))}
    best = max(counts.values(), default=0)
    if best < TOC_MIN_ENTRIES:
        return ()
    threshold = max(TOC_MIN_ENTRIES, best * TOC_PEER_RATIO)
    return tuple(i for i in sorted(counts) if counts[i] >= threshold)


def _starts_entry(span, txt, style, profile, number_styles):
    """Je tenhle span číslem stránky, kterým začíná nová položka obsahu?

    Dvě cesty podle profilu: buď rozhoduje font zapsaný v profilu (Živa),
    nebo styl odvozený ze stránky samotné (viz detect_entry_styles).
    """
    if page_span_value(txt) is None:
        return False
    if number_styles is not None:
        return style in number_styles
    return profile.is_toc_page_number_font(span["font"])


def collect_spans(doc, page_indices):
    """Všechny neprázdné spany ze zadaných stránek, v pořadí sazby."""
    out = []
    for page_index in page_indices:
        for block in doc[page_index].get_text("dict")["blocks"]:
            for line in block.get("lines", ()):
                for span in line["spans"]:
                    if span["text"].strip():
                        out.append(span)
    return out


def detect_entry_styles(spans, profile):
    """Odvoď ze stránky s obsahem, jak vypadá číslo položky a jak její text.

    Proč to nejde zapsat do profilu jako u Živy: MagPi mezi čísly 150 a 152
    předělal grafiku. Změnily se fonty (Rajdhani/RobotoSlab -> Roboto*),
    velikosti i formát čísel (`22` -> `032`). Jedna sada napevno zadaných
    hodnot by tedy platila jen pro část archivu, a co hůř, na zbytku by
    tiše vyrobila nesmysly místo aby spadla.

    Co ale platí v obou grafikách: **na stránce s obsahem jsou tři různé
    druhy čísel a jen jeden z nich jsou položky.** Vedle skutečných čísel
    stránek tam stojí ozdobné upoutávky vysázené vývratně (bíle, o něco
    větším písmem) a číslo samotné stránky s obsahem v běžící patičce.
    Odlišit je jde tím, že skutečných položek je nejvíc - obsah je jejich
    seznam, kdežto upoutávek je pár.

    Vrací `(množina stylů čísla, množina stylů textu)`, kde styl je
    `(rodina písma, velikost)`. Styly textu se hledají zvlášť, protože
    bývají jiné než styl čísla (u MagPi 150 je číslo Rajdhani a titulek
    RobotoSlab), a berou se z prvního spanu za každým přijatým číslem -
    ten je titulkem vždy. Tím se z položek vyřadí názvy rubrik, tiráž
    a popisky u ozdobných upoutávek.

    Stylů textu je **množina, ne jeden vítěz**: obsah legitimně míchá
    velikosti (Živa má titulky 10 b a autory 9,5 b, MagPi jednu jedinou).
    Vzít jen nejčastější z nich by u Živy zahodilo třetinu položek - což
    se při vývoji taky stalo. Práh je stejný jako u hledání stránek
    s obsahem: styl se počítá, pokud je aspoň `TOC_STYLE_PEER_RATIO`
    toho nejčastějšího.
    """
    pairs = _number_text_pairs(spans)
    if not pairs:
        return None, None

    # Nejčastější dvojice je z definice ta pravá: obsah je seznam, takže
    # se v něm tentýž pár "číslo + titulek" opakuje u každé položky.
    # Rovnocenných párů ale může být víc - Živa má v obsahu dvě velikosti
    # čísel (9 a 10 b) a obě jsou pravé - takže se berou všechny, které
    # jsou dost časté. Rozestup je na reálných datech pohodlný: u Živy má
    # druhý pár 79 % četnosti prvního, u MagPi má první ozdobná upoutávka
    # 14 %.
    best = pairs.most_common(1)[0][1]
    threshold = best * TOC_STYLE_PEER_RATIO
    accepted = [(num, text) for (num, text), n in pairs.items() if n >= threshold]
    return {num for num, _ in accepted}, {text for _, text in accepted}


def _number_text_pairs(spans):
    """Spočítej dvojice (styl čísla, styl textu hned za ním).

    Text hned za číslem je titulkem položky vždy - proto se styly hledají
    přes tuhle dvojici, a ne přes četnost stylů samu o sobě. Kdyby se
    bral jen nejčastější styl čísla, u časopisu se dvěma velikostmi čísel
    v obsahu (Živa) by se polovina položek ztratila.
    """
    pairs = Counter()
    pending = None
    for span in spans:
        style = font_style_key(span["font"], span["size"])
        if page_span_value(span["text"]) is not None:
            pending = style          # čeká na svůj titulek
        elif pending is not None:
            pairs[(pending, style)] += 1
            pending = None
    return pairs


def build_toc(pdf_path, profile, toc_page_indices=None):
    """toc_page_indices jsou 0-indexovaná čísla stránek PDF, na kterých je
    natištěný obsah čísla. Když se nepředají, vezmou se z profilu; když je
    ani profil neurčuje, zkusí se detekovat (viz find_toc_pages)."""
    doc = fitz.open(pdf_path)
    if toc_page_indices is None:
        toc_page_indices = profile.toc_page_indices or find_toc_pages(doc, profile)

    spans = collect_spans(doc, toc_page_indices)

    number_styles = text_styles = None
    if profile.toc_adaptive_styles:
        number_styles, text_styles = detect_entry_styles(spans, profile)

    items_raw, current = [], None
    for span in spans:
        txt = clean(span["text"])
        if not txt:
            continue
        style = font_style_key(span["font"], span["size"])

        if _starts_entry(span, txt, style, profile, number_styles):
            if current:
                items_raw.append(current)
            current = {"page": page_span_value(txt), "spans": []}
            continue

        if current is None:
            continue
        if text_styles is not None and style not in text_styles:
            continue  # název rubriky, tiráž, popisek u ozdobné upoutávky
        if any(m in txt for m in profile.toc_skip_span_markers):
            continue
        current["spans"].append({
            "text": txt, "color": span["color"],
            "font": span["font"], "size": span["size"],
        })
    if current:
        items_raw.append(current)

    toc = []
    for item in items_raw:
        if profile.toc_has_authors:
            title, author = split_title_author(item["spans"])
        else:
            title = clean(" ".join(s["text"] for s in item["spans"]))
            author = None
        if title:
            # Časopis občas natiskne víc krátkých položek na jedné stránce
            # pod JEDNÍM společným číslem stránky v obsahu, oddělené
            # středníkem v titulku ("24. ročník...; Zaujalo nás: ...").
            # Rozdělíme to na samostatné záznamy se stejnou printed_page -
            # assign_articles.py je pak uvnitř té sdílené stránky rozliší
            # podle vlastních nadpisů v textu (viz find_heading_positions).
            for part in title.split(";"):
                part = part.strip()
                if part:
                    toc.append({
                        "printed_page": item["page"],
                        "title": remove_footer(part, profile),
                        "author": remove_footer(author, profile) if author else None,
                    })
    return toc


def main():
    setup_console()
    ap = argparse.ArgumentParser(description="vytáhni obsah čísla z PDF")
    ap.add_argument("pdf", help="vstupní PDF jednoho čísla")
    ap.add_argument("out", help="výstupní toc.json")
    ap.add_argument("--toc-page", type=int, action="append", dest="toc_pages",
                    help="0-indexovaná stránka PDF s obsahem čísla; lze zadat "
                         "vícekrát. Bez tohohle se použije profil, případně "
                         "automatická detekce.")
    profiles.add_profile_argument(ap)
    args = ap.parse_args()

    profile = profiles.get(args.profile)
    toc = build_toc(args.pdf, profile,
                    tuple(args.toc_pages) if args.toc_pages else None)
    Path(args.out).write_text(
        json.dumps(toc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{args.pdf}: {len(toc)} položek obsahu -> {args.out}")


if __name__ == "__main__":
    main()
