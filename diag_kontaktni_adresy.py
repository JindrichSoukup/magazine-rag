"""
Diagnostika pro opakovaně selhávající "Kontaktní adresy" napříč čísly -
najde skupinu, co ji obsahuje, a vypíše, jaký nadpis se v textu skutečně
najde (pokud nějaký), ať víme, čemu se má "Kontaktní adresy" textově
přizpůsobit.

Použití:
    python diag_kontaktni_adresy.py output/2019-2/blocks.json output/2019-2/toc.json output/2019-2/page_map.json
"""
import json
import sys

from assign_articles import is_junk, cluster_columns, find_split_y, texts_match


def ordered_page_blocks(blocks_by_page, pg):
    page_blocks = [b for b in blocks_by_page.get(pg, []) if not is_junk(b)]
    return [b for col, y0, b in sorted(cluster_columns(page_blocks), key=lambda t: (t[0], t[1]))]


def main():
    blocks_path, toc_path, page_map_path = sys.argv[1:4]
    blocks = json.loads(open(blocks_path, encoding="utf-8").read())
    toc = json.loads(open(toc_path, encoding="utf-8").read())
    page_map = json.loads(open(page_map_path, encoding="utf-8").read())
    label_to_page = page_map["label_to_page"]

    resolved = []
    for entry in toc:
        pdf_page = label_to_page.get(entry["printed_page"])
        if pdf_page is None:
            continue
        resolved.append({**entry, "pdf_page_start": pdf_page})
    resolved.sort(key=lambda e: e["pdf_page_start"])

    groups = []
    for entry in resolved:
        if groups and groups[-1][0]["pdf_page_start"] == entry["pdf_page_start"]:
            groups[-1].append(entry)
        else:
            groups.append([entry])

    last_pdf_page = max(b["page"] for b in blocks)
    for gi, group in enumerate(groups):
        end = (groups[gi + 1][0]["pdf_page_start"] - 1
               if gi + 1 < len(groups) else last_pdf_page)
        for entry in group:
            entry["pdf_page_end"] = end

    blocks_by_page = {}
    for b in blocks:
        blocks_by_page.setdefault(b["page"], []).append(b)

    target_group = None
    target_gi = None
    for gi, group in enumerate(groups):
        if any("kontaktní adresy" in e["title"].strip().lower() for e in group):
            target_group = group
            target_gi = gi
            break

    if target_group is None:
        print("Nenašel jsem žádnou skupinu s 'Kontaktní adresy' v titulku.")
        return

    first_entry = target_group[0]
    start_pg, end_pg = first_entry["pdf_page_start"], first_entry["pdf_page_end"]

    print(f"Skupina obsahující 'Kontaktní adresy': str. {start_pg}-{end_pg}, "
          f"{len(target_group)} položek:")
    for e in target_group:
        print(f"    - {e['title']!r}")
    print()

    raw = []
    for pg in range(start_pg, end_pg + 1):
        page_blocks = ordered_page_blocks(blocks_by_page, pg)
        if pg == start_pg and target_gi > 0:
            split_y = find_split_y(page_blocks, first_entry)
            if split_y is not None:
                page_blocks = [b for b in page_blocks if b["bbox"][3] > split_y]
        raw.extend(page_blocks)

    headings = [b for b in raw if b["type"] in ("title", "heading")]
    print(f"Všechny title/heading bloky v týhle skupině ({len(headings)} celkem):")
    for b in headings:
        print(f"    str.{b['page']} id={b['block_id']} {b['text'][:70]!r}")
    print()

    # zkus fuzzy shodu "Kontaktní adresy" proti KAŽDÉMU nalezenému nadpisu,
    # ať vidíme, jak blízko/daleko to je
    ka_entry = next(e for e in target_group if "kontaktní adresy" in e["title"].strip().lower())
    print(f"Fuzzy shoda {ka_entry['title']!r} proti každému nadpisu:")
    for b in headings:
        print(f"    match={texts_match(ka_entry['title'], b['text'])!s:5} <- {b['text'][:60]!r}")


if __name__ == "__main__":
    main()
