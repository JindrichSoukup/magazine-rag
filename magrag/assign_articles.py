"""
Stage 2b: attach every text block to the article it belongs to.

Inputs:
  ziva_blocks.json  - per-page text blocks (from your fitz parser)
  ziva_toc.json     - article list with PRINTED page numbers (from create_toc.py)
  page_map.json     - printed-label <-> pdf-page translation (build_page_map.py)

What it does:
  1. Resolves each TOC entry's printed page ("262", "CXXXIII", ...) to a real
     PDF page number, then sorts the TOC into true reading order (the JSON
     order of ziva_toc.json is NOT the reading order - articles and back
     matter got interleaved during extraction).
  2. Builds page ranges: article N owns every PDF page from its own start
     page up to (but not including) the start page of the next DIFFERENT
     page. Consecutive TOC entries sharing the exact same start page (this
     happens when create_toc.py split one semicolon-joined combined TOC
     entry into several - see create_toc.py) are treated as one GROUP that
     collectively owns that page range, then subdivided among themselves
     by locating each entry's own heading/title block within the group's
     shared content - see find_heading_positions().
  3. On the FIRST page of a group's range, checks whether the previous
     article/group's tail content actually spills onto this same page too
     (common when a short remaining bit of the previous article shares a
     page with this group's own title/author/abstract) - if so, splits
     that page's blocks by a vertical cut point instead of handing the
     whole page to this group. See find_split_y() - this was a real bug
     found by inspecting a mismatched article (Selosse & Roy's mycorrhiza
     article leaking into Vít Latzel's epigenetics article purely because
     they shared page 11).
  4. Tags every block with the article it falls in.
  5. Filters out obvious junk (running headers/footers, the copyright line,
     lone figure-number labels floating on top of images).
  6. Produces a *best-effort* column-aware reading order per article, purely
     so a human can eyeball the result - it is NOT required to be perfect
     for the later RAG step (see note at the bottom of the file).

Output: ziva_articles.json - one entry per article:
  {
    "title": ..., "author": ...,
    "printed_page_start": "262", "pdf_page_start": 4, "pdf_page_end": 7,
    "total_pdf_pages": 84,  # celkový počet stránek PDF čísla (ne článku) -
                            # potřeba v build_chunks.py pro rozpoznání
                            # obálkových stránek na konci PDF
    "chunks": [ {block_id, page, type, text}, ... ],  # in best-effort order
    "quality_flags": [],  # [] = spolehlivě přiřazeno. Jinak některé z:
                          #   "boundary_page_unverified" - hraniční stránka
                          #     se sdílela s předchozím článkem, ale
                          #     find_split_y nenašel řez (vzali jsme celou
                          #     stránku, bez jistoty)
                          #   "merged_fallback" - víc položek z obsahu se
                          #     nepodařilo textově rozlišit vůbec, sloučeny
                          #     do jednoho záznamu (viz "original_titles")
                          #   "heading_not_found_empty" - vlastní nadpis
                          #     položky se v textu nenašel, obsah "utekl"
                          #     k sousední položce (viz její "absorbed_titles")
                          #   "absorbed_unmatched_siblings" - tenhle záznam
                          #     v sobě má i obsah sousední položky, jejíž
                          #     vlastní nadpis se nenašel (viz "absorbed_titles")
    "full_text": "...",              # tělo článku jako jeden string (náhled)
    "full_text_paragraphs": [ {"page": 4, "text": "..."}, ... ],  # totéž,
                            # ale po odstavcích se stránkou - z tohohle
                            # chunkuje build_chunks.py (rozhoduje se tam,
                            # ne tady, jestli/jak vynechat obálku)
    "captions_text": "...", "captions_paragraphs": [ ... ],  # totéž pro
                            # popisky obrázků
  }
"""
import argparse
import json
import re
from pathlib import Path

from magrag import profiles
from magrag.console import setup_console
from magrag.build_page_map import normalize_label
from magrag.extract_blocks import looks_like_word_break, HYPHEN_BREAK_RE

COLUMN_GAP = 40  # pt; x0-gap bigger than this starts a new column cluster
BODY_TYPES = {"title", "heading", "other", "body"}


def join_paragraphs(texts):
    """Spoj texty do jednoho stringu, odstavce odděl prázdným řádkem -
    ALE pokud předchozí kus končí zalomeným slovem (viz
    extract_blocks.looks_like_word_break) a další kus začíná jeho
    pokračováním, sleduje se to bez mezery/prázdného řádku, stejně jako
    uvnitř jedné stránky v extract_blocks.smart_join(). Tohle je potřeba
    navíc tady, protože chunky u sebe mohou být z různých sloupců nebo
    i různých stránek PDF - tam extract_blocks.py nikdy nesléval, takže by
    se jinak do full_textu dostalo třeba "...dlouhodo-\n\nbé roční úhrny"."""
    out = ""
    for text in texts:
        text = text.strip()
        if not text:
            continue
        if out and looks_like_word_break(out, text):
            out = HYPHEN_BREAK_RE.sub(r"\1", out) + text
        elif out:
            out += "\n\n" + text
        else:
            out = text
    return out


def join_paragraphs_with_pages(items):
    """Stejná logika jako join_paragraphs (lepení přes zalomenou pomlčku),
    ale pracuje na dvojicích (page, text) a vrací dvojice (page, text) -
    aby šla stránka dohledat i u výsledného (případně slitého) odstavce.
    Když se dva odstavce sletí kvůli zalomenému slovu, ponechá se stránka
    prvního z nich (slitá slova skoro vždy zůstávají na jedné, nanejvýš
    sousední stránce).

    Tohle je klíčové pro build_chunks.py: teprve tam, na úrovni
    chunkování pro RAG, se rozhoduje, jestli se stránky obálky/zadní
    strany mají zahodit - a k tomu potřebuje znát stránku každého
    odstavce, ne jen jeden hotový string."""
    out = []  # list [page, text], mutovatelné kvůli slévání
    for page, text in items:
        text = text.strip()
        if not text:
            continue
        if out and looks_like_word_break(out[-1][1], text):
            out[-1][1] = HYPHEN_BREAK_RE.sub(r"\1", out[-1][1]) + text
        else:
            out.append([page, text])
    return [(p, t) for p, t in out]


BODY_TYPES = {"title", "heading", "other", "body"}


def build_full_text_and_paragraphs(chunks):
    """Slep JEN tělo článku (title/heading/other/body) do jednoho čitelného
    textu, odstavce odděl prázdným řádkem - a vrať to jak jako jeden
    string (pro rychlý náhled/čtení), tak jako list {"page", "text"}
    (pro build_chunks.py, kde se podle stránky dá filtrovat obálka).

    Popisky obrázků (caption) a panelové značky (annotation) sem záměrně
    NEDÁVÁME. Obrázek/graf často fyzicky sedí uprostřed sloupce, takže text
    před ním a za ním jsou v PDF opravdu dva samostatné bloky (nejde je
    slévat, protože je mezi nimi vizuálně obrázek) - ale naše řazení podle
    (sloupec, y) je pak umístí "vedle sebe" s popiskem obrázku mezi nimi,
    což v souvislém textu vypadá jako přeťatá věta uprostřed. Když popisky
    z full_textu vynecháme, text před a za obrázkem skončí v sousedních
    odstavcích (ne ideální navázání, ale aspoň se nerozseká věta) a popisky
    zůstanou pohromadě ve zvláštním poli captions_text (a v chunks vždy)."""
    items = [(c["page"], c["text"].strip()) for c in chunks
             if c["type"] in BODY_TYPES and c["text"].strip()]
    merged = join_paragraphs_with_pages(items)
    full_text = "\n\n".join(t for _, t in merged)
    paragraphs = [{"page": p, "text": t} for p, t in merged]
    return full_text, paragraphs


def build_captions_text_and_paragraphs(chunks):
    """Popisky obrázků (ne panelové značky/annotation - ty nenesou obsah)
    pohromadě, v pořadí, ve kterém se v článku objevily - stejně jako u
    build_full_text_and_paragraphs vrací string i list se stránkami."""
    items = [(c["page"], c["text"].strip()) for c in chunks
             if c["type"] == "caption" and c["text"].strip()]
    merged = join_paragraphs_with_pages(items)
    captions_text = "\n\n".join(t for _, t in merged)
    paragraphs = [{"page": p, "text": t} for p, t in merged]
    return captions_text, paragraphs


def find_split_y(page_blocks, entry):
    """Najdi Y souřadnici řezu na stránce sdílené s PŘEDCHOZÍM článkem:
    nad ní (menší y) je ještě obsah předchozího článku, od ní (nebo níž)
    začíná TENHLE článek.

    Použije se y0 bloku s autorem (přesnější, protože sedí blíž
    skutečnému přechodu), a když se nenajde (autor chybí/nesedí text),
    spadneme na y0 titulku. Řez se pak porovnává s DOLNÍ hranou (y1)
    každého bloku na stránce, ne s pořadím v (sloupec, y) seznamu -
    protože obsah dvou článků se na jedné stránce může prolínat mezi
    sloupci (typicky nezávisle běžící poslední sloupec), takže "vše před
    titulkem v pořadí čtení" by nestačilo - viz komentář u volajícího
    místa."""
    author_text = (entry.get("author") or "").strip()
    if author_text:
        for b in page_blocks:
            if b["type"] == "other" and texts_match(author_text, b["text"]):
                return b["bbox"][1]
    title_text = entry["title"].strip()
    for b in page_blocks:
        if b["type"] in ("title", "heading") and texts_match(title_text, b["text"]):
            return b["bbox"][1]
    return None  # nenalezeno - nespliťuj (bezpečný default: nic neměň)


COMMON_TOC_PREFIXES = (
    "recenze:", "k výuce:", "upoutávka na knihu:", "zaujalo nás:",
    "biozvěst:",
)


def normalize_for_match(text: str) -> str:
    """Sundej běžné prefixy, co v obsahu bývají, ale v samotném nadpisu
    v textu chybí (v obsahu "Recenze: Název knihy", v textu jen "Autor:
    Název knihy s podtitulem")."""
    t = text.strip().lower()
    for prefix in COMMON_TOC_PREFIXES:
        if t.startswith(prefix):
            return t[len(prefix):].strip()
    return t


def longest_common_substring_len(a: str, b: str) -> int:
    """Délka nejdelšího souvislého společného podřetězce (klasické DP,
    O(len(a)*len(b)) - pro krátké titulky/nadpisy v pohodě)."""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best:
                    best = curr[j]
        prev = curr
    return best


def texts_match(toc_title: str, block_text: str, min_common: int = 15) -> bool:
    """Přesná shoda, podřetězec v libovolném směru, NEBO aspoň min_common
    znaků dlouhý společný souvislý úsek. To poslední je nutné třeba pro
    "Fenomén Velká kotlina 1 - pohled geobotanický a horského ekologa",
    kde v obsahu je JEDEN záznam pokrývající DVA různé nadpisy (dva různí
    autoři, dva různé "pohledy" na stejné téma) - ani jeden nadpis
    neobsahuje celý titulek z obsahu, ale oba sdílejí dost dlouhý
    charakteristický úsek ("Fenomén Velká kotlina 1")."""
    a = normalize_for_match(toc_title)
    b = normalize_for_match(block_text)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    return longest_common_substring_len(a, b) >= min_common


def find_heading_positions(ordered_blocks, entries):
    """Když create_toc.py rozdělilo jeden sloučený TOC záznam (víc položek
    pod jedním číslem stránky, oddělených středníkem - viz create_toc.py)
    na víc `entries` se stejnou pdf_page_start, potřebujeme jejich společný
    blok obsahu (`ordered_blocks`, už seřazený podle stránky a v rámci
    stránky sloupcově) rozdělit mezi ně jednotlivě.

    Nejdřív zkusíme obecnější a robustnější pravidlo: spočítat všechny
    title/heading bloky v `ordered_blocks` - pokud jich je přesně tolik,
    kolik je `entries`, prostě je vezmeme v pořadí jako řezy (rychlá
    zkratka, funguje spolehlivě, když sedí počty).

    Když se počty neshodují, hledáme fuzzy shodu (viz texts_match) pro
    každý záznam zvlášť - to zvládne i recenze (v obsahu zkrácený titulek,
    v textu "Autor: Plný název knihy" - ale plný název knihy je v nadpisu
    OBSAŽENÝ jako podřetězec) i záznamy pokrývající 2 nadpisy najednou
    (nenalezený "extra" nadpis se v cyklu volajícího kódu prostě přičte
    k PŘEDCHOZÍ úspěšně nalezené položce, ne že by se ztratil)."""
    heading_idxs = [i for i, b in enumerate(ordered_blocks) if b["type"] in ("title", "heading")]
    if len(heading_idxs) == len(entries):
        return heading_idxs

    positions = []
    search_from = 0
    for entry in entries:
        title_text = entry["title"].strip()
        found = None
        for idx in range(search_from, len(ordered_blocks)):
            b = ordered_blocks[idx]
            if b["type"] in ("title", "heading") and texts_match(title_text, b["text"]):
                found = idx
                break
        if found is None:
            print(f"  [warn] nenašel jsem vlastní nadpis pro {title_text!r} "
                  f"ve sloučeném záznamu obsahu ({len(heading_idxs)} nadpisů "
                  f"nalezeno na stránce vs. {len(entries)} položek v obsahu, "
                  f"počty nesedí) - dostane prázdný obsah, "
                  f"jeho text pravděpodobně zůstal u sousední položky - "
                  f"zkontrolujte ručně")
        else:
            search_from = found
        positions.append(found)  # None, nebo index
    return positions


def is_junk(b: dict, profile, page_height: float = 0.0) -> bool:
    """Opakující se balast, který není obsahem žádného článku.

    První dva testy jsou vlastností konkrétního časopisu (jak vypadá jeho
    patička a tiráž) a berou se z profilu. Třetí je obecný a platí všude.
    """
    text = b["text"].strip()
    font = b["font"]
    size = b["font_size"]

    # běžící hlavička/patička ("ziva.avcr.cz 262 živa 6/2014" a podobné)
    if profile.is_footer_block(font, size, text, b["bbox"], page_height):
        return True

    # pruh s tiráží/copyrightem opakovaný na každé stránce
    if profile.is_junk_text(text):
        return True

    # lone figure-number markers overlaid on photos/diagrams, e.g. a block
    # that is *just* "3" or "12" sitting in a tiny bounding box
    x0, y0, x1, y1 = b["bbox"]
    if text.isdigit() and (x1 - x0) < 15 and (y1 - y0) < 15:
        return True

    return False


def cluster_columns(blocks_on_page):
    """Return each block tagged with a column index (0, 1, 2, ...) based on
    its x0, so we can sort left-to-right, top-to-bottom within a page."""
    xs = sorted(set(round(b["bbox"][0]) for b in blocks_on_page))
    cluster_of_x = {}
    cluster_idx = 0
    prev = None
    for x in xs:
        if prev is not None and x - prev > COLUMN_GAP:
            cluster_idx += 1
        cluster_of_x[x] = cluster_idx
        prev = x
    tagged = []
    for b in blocks_on_page:
        col = cluster_of_x[round(b["bbox"][0])]
        tagged.append((col, b["bbox"][1], b))  # (column, y0, block)
    return tagged


def assemble_articles(blocks, toc, label_to_page, profile, year=None, issue=None):
    # Viz build_page_map: výška stránky se odvodí z bloků, potřebuje ji
    # detekce patičky podle polohy.
    page_height = max((b["bbox"][3] for b in blocks), default=0.0)

    # 1) resolve + sort TOC into real reading order ------------------------
    resolved = []
    for entry in toc:
        # přes normalize_label, protože obsah čísla a patička nemusí
        # tentýž popisek zapisovat stejně ("032" vs "32") - viz tam
        pdf_page = label_to_page.get(normalize_label(entry["printed_page"]))
        if pdf_page is None:
            print(f"  [warn] could not resolve printed page "
                  f"{entry['printed_page']!r} for {entry['title']!r} - skipping")
            continue
        resolved.append({**entry, "pdf_page_start": pdf_page})
    resolved.sort(key=lambda e: e["pdf_page_start"])

    # 2) seskup po sobě jdoucí záznamy se STEJNOU pdf_page_start do skupin.
    #    Většina skupin bude mít velikost 1 (běžný článek se svou vlastní
    #    stránkou) - víc než 1 nastává, když create_toc.py rozdělilo jeden
    #    sloučený TOC záznam (viz create_toc.py, split podle středníku) ---
    groups = []
    for entry in resolved:
        if groups and groups[-1][0]["pdf_page_start"] == entry["pdf_page_start"]:
            groups[-1].append(entry)
        else:
            groups.append([entry])

    # 3) konec stránkového rozsahu se počítá PER SKUPINA (ne per záznam) -
    #    celá skupina končí tam, kde začíná DALŠÍ skupina (jiná stránka) ---
    last_pdf_page = max(b["page"] for b in blocks)
    for gi, group in enumerate(groups):
        end = (groups[gi + 1][0]["pdf_page_start"] - 1
               if gi + 1 < len(groups) else last_pdf_page)
        for entry in group:
            entry["pdf_page_end"] = end

    # 4) group blocks by page for fast lookup --------------------------------
    blocks_by_page = {}
    for b in blocks:
        blocks_by_page.setdefault(b["page"], []).append(b)

    def ordered_page_blocks(pg):
        page_blocks = [b for b in blocks_by_page.get(pg, [])
                       if not is_junk(b, profile, page_height)]
        return [b for col, y0, b in sorted(cluster_columns(page_blocks), key=lambda t: (t[0], t[1]))]

    # 5) rozděl bloky mezi SKUPINY - a na hraniční stránce (tam, kde podle
    #    TOC začíná NOVÁ skupina) zkontroluj, jestli tam náhodou nekončí
    #    ještě PŘEDCHOZÍ skupina/článek (viz find_split_y výše) ------------
    group_raw_blocks = [[] for _ in groups]
    boundary_unverified = [False] * len(groups)  # viz quality_flags níž

    for gi, group in enumerate(groups):
        first_entry = group[0]
        start_pg, end_pg = first_entry["pdf_page_start"], first_entry["pdf_page_end"]
        for pg in range(start_pg, end_pg + 1):
            page_blocks = ordered_page_blocks(pg)
            if pg == start_pg and gi > 0:
                split_y = find_split_y(page_blocks, first_entry)
                if split_y is not None:
                    for b in page_blocks:
                        # y1 (dolní hrana bloku) <= řez => blok skončil
                        # ještě PŘED začátkem tyhle skupiny - patří
                        # předchozí skupině/článku, bez ohledu na to,
                        # v jakém sloupci leží (obsah se může na stránce
                        # prolínat mezi sloupci, viz komentář výše)
                        if b["bbox"][3] <= split_y:
                            group_raw_blocks[gi - 1].append(b)
                        else:
                            group_raw_blocks[gi].append(b)
                    continue
                else:
                    # nenašli jsme vlastní titulek/autora skupiny na týhle
                    # stránce, abychom se podle něj rozhodli, jestli tam
                    # nekončí ještě předchozí článek - vzali jsme tedy
                    # celou stránku pro TUHLE skupinu, ale bez ověření
                    boundary_unverified[gi] = True
            group_raw_blocks[gi].extend(page_blocks)

    # 6) uvnitř skupiny (pokud má víc než 1 záznam) rozděl bloky mezi
    #    jednotlivé TOC záznamy podle pozice jejich vlastního nadpisu.
    #    Pokud se nenajde ANI JEDEN nadpis pro celou skupinu (numerické
    #    počty neseděly A textová shoda selhala pro všechny položky -
    #    v praxi třeba proto, že středník tady oddělil jen podnázvy JEDNOHO
    #    článku, ne víc samostatných článků, viz diskuze u create_toc.py),
    #    NEROZDĚLUJEME VŮBEC - sloučíme skupinu zpátky do jednoho záznamu
    #    se spojeným titulkem. Jinak by celý obsah skupiny tiše zmizel
    #    (byla to reálná chyba - opraveno po nálezu na čísle 2023/1).
    #
    #    Zároveň si u KAŽDÉHO výsledného záznamu poznamenáme do
    #    "quality_flags", jestli/kde jsme se museli spolehnout na nějaký
    #    hádající/záchranný mechanismus - ať to jde v korpusu i v RAG
    #    dotazech rozeznat od spolehlivě přiřazeného obsahu. ---------------
    final_entries = []  # list (entry, raw_blocks) - může mít MÍŇ položek
                         # než `resolved`, pokud se nějaká skupina sloučila
    for gi, (group, raw) in enumerate(zip(groups, group_raw_blocks)):
        if len(group) == 1:
            entry = dict(group[0])
            entry["quality_flags"] = (["boundary_page_unverified"]
                                       if boundary_unverified[gi] else [])
            final_entries.append((entry, raw))
            continue

        positions = find_heading_positions(raw, group)
        if all(p is None for p in positions):
            combined_title = "; ".join(e["title"] for e in group)
            print(f"  [warn] u žádné položky skupiny na str. "
                  f"{group[0]['pdf_page_start']} se nenašel vlastní nadpis - "
                  f"nerozděluji, slučuji zpět do jednoho záznamu: "
                  f"{combined_title!r}")
            merged = dict(group[0])
            merged["title"] = combined_title
            merged["author"] = next((e["author"] for e in group if e.get("author")), None)
            merged["quality_flags"] = ["merged_fallback"] + (
                ["boundary_page_unverified"] if boundary_unverified[gi] else [])
            merged["original_titles"] = [e["title"] for e in group]
            final_entries.append((merged, raw))
            continue

        # obsah PŘED prvním nalezeným nadpisem (typicky žádný, ale pokud
        # find_split_y výš nerozeznal hranici se skutečně předchozím
        # článkem přesně, něco tam může zůstat) nesmí zmizet beze stopy -
        # přiřaď ho první nalezené položce místo zahození
        first_found_i = next(i for i, p in enumerate(positions) if p is not None)
        for i, (entry, pos) in enumerate(zip(group, positions)):
            entry = dict(entry)
            if pos is None:
                entry["quality_flags"] = ["heading_not_found_empty"]
                final_entries.append((entry, []))  # varování už vypsáno výš
                continue
            lo = 0 if i == first_found_i else pos
            # najdi DALŠÍ nalezenou (ne None) pozici mezi zbylými položkami
            # skupiny - přeskoč případné "díry" po nenalezených sousedech,
            # ať jim nekrademe obsah, co jim nepatří
            next_pos = len(raw)
            absorbed = []
            for j in range(i + 1, len(positions)):
                if positions[j] is not None:
                    next_pos = positions[j]
                    break
                absorbed.append(group[j]["title"])
            flags = []
            if i == first_found_i and boundary_unverified[gi]:
                flags.append("boundary_page_unverified")
            if absorbed:
                flags.append("absorbed_unmatched_siblings")
                entry["absorbed_titles"] = absorbed
            entry["quality_flags"] = flags
            final_entries.append((entry, raw[lo:next_pos]))

    # 7) z rozřazených syrových bloků postav chunks - seřazené po stránkách
    #    a v rámci stránky sloupcově (stejná logika jako dřív) -----------
    articles = []
    for art_idx, (entry, entry_blocks) in enumerate(final_entries):
        by_page = {}
        for b in entry_blocks:
            by_page.setdefault(b["page"], []).append(b)

        chunks = []
        for pg in sorted(by_page):
            for col, y0, b in sorted(cluster_columns(by_page[pg]), key=lambda t: (t[0], t[1])):
                chunks.append({
                    "block_id": b["block_id"],
                    "page": b["page"],
                    "type": b["type"],
                    "text": b["text"],
                })

        if not chunks:
            print(f"  [warn] článek {entry['title']!r} nemá po rozdělení "
                  f"hraničních stránek žádné bloky - zkontrolujte ručně")

        article_id = f"{year}-{issue}-{art_idx}" if year and issue else str(art_idx)
        full_text, full_text_paragraphs = build_full_text_and_paragraphs(chunks)
        captions_text, captions_paragraphs = build_captions_text_and_paragraphs(chunks)
        articles.append({
            "article_id": article_id,
            "year": year,
            "issue": issue,
            "title": entry["title"],
            "author": entry["author"],
            "printed_page_start": entry["printed_page"],
            "pdf_page_start": entry["pdf_page_start"],
            "pdf_page_end": entry["pdf_page_end"],
            "total_pdf_pages": last_pdf_page,  # celkový počet stran PDF
            # (čísla, ne článku) - potřeba v build_chunks.py, aby šlo
            # poznat, které stránky jsou "obálka" na konci PDF
            "quality_flags": entry.get("quality_flags", []),  # [] = spolehlivě
            # přiřazeno; jinak viz seznam možných hodnot v docstringu nahoře
            **({"original_titles": entry["original_titles"]}
               if "original_titles" in entry else {}),
            **({"absorbed_titles": entry["absorbed_titles"]}
               if "absorbed_titles" in entry else {}),
            "chunks": chunks,
            "full_text": full_text,
            "full_text_paragraphs": full_text_paragraphs,  # [{"page","text"}, ...]
            "captions_text": captions_text,
            "captions_paragraphs": captions_paragraphs,    # [{"page","text"}, ...]
        })
    return articles


def main():
    setup_console()
    ap = argparse.ArgumentParser(description="přiřaď bloky textu článkům")
    ap.add_argument("blocks", help="blocks.json (z extract_blocks)")
    ap.add_argument("toc", help="toc.json (z create_toc)")
    ap.add_argument("page_map", help="page_map.json (z build_page_map)")
    ap.add_argument("out", help="výstupní articles.json")
    ap.add_argument("--year")
    ap.add_argument("--issue")
    profiles.add_profile_argument(ap)
    args = ap.parse_args()

    profile = profiles.get(args.profile)
    blocks = json.loads(Path(args.blocks).read_text(encoding="utf-8"))
    toc = json.loads(Path(args.toc).read_text(encoding="utf-8"))
    label_to_page = json.loads(
        Path(args.page_map).read_text(encoding="utf-8"))["label_to_page"]

    articles = assemble_articles(blocks, toc, label_to_page, profile,
                                 year=args.year, issue=args.issue)

    Path(args.out).write_text(
        json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(articles)} článků "
          f"({sum(len(a['chunks']) for a in articles)} bloků) -> {args.out}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# NOTE on reading order and RAG:
#
# The "best-effort" ordering above is good enough to *read* an article and
# sanity-check the pipeline, but it will occasionally misplace a block (e.g.
# a long figure-caption that happens to use body-sized font gets sorted by
# its raw y-position and may show up a paragraph too early/late). That's a
# genuine, hard-to-fully-solve problem in multi-column magazine layout
# parsing.
#
# For the RAG step this barely matters: each block/chunk is already a
# reasonably self-contained paragraph (title+intro or subheading+paragraph),
# so we can embed and retrieve *chunks*, not the perfectly-ordered document.
# A caption that ends up slightly out of place is simply retrieved as its
# own small chunk ("Fig. 3: karyotype scheme...") - which is often exactly
# what you'd want it to be anyway.
# ---------------------------------------------------------------------------
