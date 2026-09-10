"""
Diagnostika pro sloučené TOC skupiny (víc položek v obsahu sdílejících
jednu tištěnou stránku - viz create_toc.py/assign_articles.py).

Pro každou takovou skupinu vypíše CELÝ seřazený obsah jejího stránkového
rozsahu (přesně to, na čem pracuje find_heading_positions), s vyznačenými
title/heading bloky - ať je vidět, kolik jich skutečně je, jak se jejich
text liší od titulků v TOC, a kde přesně se "díra" v očekávání bere,
místo abychom dál hádali naslepo.

Použití (potřebuje soubory z run_all.py/create_toc.py/build_page_map.py
pro jedno konkrétní číslo):
    python diag_groups.py output/2023-1/blocks.json output/2023-1/toc.json output/2023-1/page_map.json
"""
import json
import sys

from magrag.assign_articles import is_junk, cluster_columns, find_split_y, find_heading_positions


def ordered_page_blocks(blocks_by_page, pg):
    page_blocks = [b for b in blocks_by_page.get(pg, []) if not is_junk(b)]
    return [b for col, y0, b in sorted(cluster_columns(page_blocks), key=lambda t: (t[0], t[1]))]


def main():
    if len(sys.argv) != 4:
        print("použití: python diag_groups.py blocks.json toc.json page_map.json")
        sys.exit(1)

    blocks_path, toc_path, page_map_path = sys.argv[1:4]
    blocks = json.loads(open(blocks_path, encoding="utf-8").read())
    toc = json.loads(open(toc_path, encoding="utf-8").read())
    page_map = json.loads(open(page_map_path, encoding="utf-8").read())
    label_to_page = page_map["label_to_page"]

    # --- stejné kroky 1-3 jako v assemble_articles() ----------------------
    resolved = []
    for entry in toc:
        pdf_page = label_to_page.get(entry["printed_page"])
        if pdf_page is None:
            print(f"  [warn] nerozpoznaná stránka {entry['printed_page']!r} "
                  f"pro {entry['title']!r} - přeskakuji")
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

    # --- pro každou vícepoložkovou skupinu vypiš diagnostiku --------------
    multi_groups = [(gi, g) for gi, g in enumerate(groups) if len(g) > 1]
    print(f"Nalezeno {len(multi_groups)} vícepoložkových skupin z {len(groups)} celkem.\n")

    for gi, group in multi_groups:
        first_entry = group[0]
        start_pg, end_pg = first_entry["pdf_page_start"], first_entry["pdf_page_end"]

        print("=" * 78)
        print(f"SKUPINA #{gi}  |  str. {start_pg}-{end_pg}  |  "
              f"{len(group)} položek v TOC:")
        for e in group:
            print(f"    - {e['title']!r}  (autor: {e.get('author')!r})")
        print()

        raw = []
        for pg in range(start_pg, end_pg + 1):
            page_blocks = ordered_page_blocks(blocks_by_page, pg)
            if pg == start_pg and gi > 0:
                split_y = find_split_y(page_blocks, first_entry)
                if split_y is not None:
                    before = [b for b in page_blocks if b["bbox"][3] <= split_y]
                    page_blocks = [b for b in page_blocks if b["bbox"][3] > split_y]
                    print(f"  (na str. {pg} odděleno {len(before)} bloků patřících "
                          f"PŘEDCHOZÍMU článku podle find_split_y)")
                else:
                    print(f"  (na str. {pg}: find_split_y nenašel řez - "
                          f"celá stránka jde do týhle skupiny)")
            raw.extend(page_blocks)

        headings = [b for b in raw if b["type"] in ("title", "heading")]
        print(f"\n  --> {len(headings)} title/heading bloků nalezeno vs. "
              f"{len(group)} položek v TOC "
              f"{'(SEDÍ)' if len(headings) == len(group) else '(NESEDÍ)'}\n")

        # --- tohle je NOVÉ: skutečný výsledek find_heading_positions(), ne
        # jen syrové počítání nadpisů - ukazuje efekt fuzzy shody (texts_match)
        positions = find_heading_positions(raw, group)
        print("  Skutečné přiřazení (find_heading_positions):")
        for entry, pos in zip(group, positions):
            if pos is None:
                print(f"    ✗ {entry['title']!r} -> NENALEZENO (prázdný obsah)")
            else:
                b = raw[pos]
                print(f"    ✓ {entry['title']!r} -> str.{b['page']} "
                      f"id={b['block_id']} {b['text'][:50]!r}")
        print()

        for b in raw:
            marker = ">>> " if b["type"] in ("title", "heading") else "    "
            print(f"  {marker}str.{b['page']:>3} id={b['block_id']:>3} "
                  f"{b['type']:9} size={b['font_size']:>5} {b['text'][:65]!r}")
        print()


if __name__ == "__main__":
    main()
