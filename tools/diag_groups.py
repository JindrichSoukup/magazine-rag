"""Diagnostics for combined TOC groups - several contents entries sharing
one printed page (see create_toc.py and assign_articles.py).

For each such group it prints the ENTIRE sorted content of its page
range, exactly what find_heading_positions works on, with the
title/heading blocks marked. That shows how many there really are, how
their text differs from the titles in the contents, and where the gap in
expectations comes from - instead of guessing blind.

Usage (needs the files run_all.py produced for one specific issue):
    python -m tools.diag_groups output/2023-1/blocks.json \\
        output/2023-1/toc.json output/2023-1/page_map.json [profile]
"""
import json
import sys

from magazine_rag import profiles
from magazine_rag.console import setup_console
from magazine_rag.assign_articles import (
    cluster_columns,
    find_heading_positions,
    find_split_y,
    is_junk,
)


def ordered_page_blocks(blocks_by_page, pg, profile, page_height):
    page_blocks = [b for b in blocks_by_page.get(pg, [])
                   if not is_junk(b, profile, page_height)]
    return [b for col, y0, b in
            sorted(cluster_columns(page_blocks), key=lambda t: (t[0], t[1]))]


def main():
    setup_console()
    if len(sys.argv) not in (4, 5):
        print("usage: python -m tools.diag_groups blocks.json toc.json "
              "page_map.json [profile]")
        sys.exit(1)

    blocks_path, toc_path, page_map_path = sys.argv[1:4]
    profile = profiles.get(sys.argv[4] if len(sys.argv) == 5 else "ziva")
    blocks = json.loads(open(blocks_path, encoding="utf-8").read())
    page_height = max((b["bbox"][3] for b in blocks), default=0.0)
    toc = json.loads(open(toc_path, encoding="utf-8").read())
    page_map = json.loads(open(page_map_path, encoding="utf-8").read())
    label_to_page = page_map["label_to_page"]

    # --- the same steps 1-3 as in assemble_articles() --------------------
    resolved = []
    for entry in toc:
        pdf_page = label_to_page.get(entry["printed_page"])
        if pdf_page is None:
            print(f"  [warn] unresolved page {entry['printed_page']!r} "
                  f"for {entry['title']!r} - skipping")
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

    # --- print diagnostics for every multi-entry group -------------------
    multi_groups = [(gi, g) for gi, g in enumerate(groups) if len(g) > 1]
    print(f"Found {len(multi_groups)} multi-entry groups out of "
          f"{len(groups)} in total.\n")

    for gi, group in multi_groups:
        first_entry = group[0]
        start_pg, end_pg = first_entry["pdf_page_start"], first_entry["pdf_page_end"]

        print("=" * 78)
        print(f"GROUP #{gi}  |  pp. {start_pg}-{end_pg}  |  "
              f"{len(group)} contents entries:")
        for e in group:
            print(f"    - {e['title']!r}  (author: {e.get('author')!r})")
        print()

        raw = []
        for pg in range(start_pg, end_pg + 1):
            page_blocks = ordered_page_blocks(blocks_by_page, pg, profile,
                                              page_height)
            if pg == start_pg and gi > 0:
                split_y = find_split_y(page_blocks, first_entry)
                if split_y is not None:
                    before = [b for b in page_blocks if b["bbox"][3] <= split_y]
                    page_blocks = [b for b in page_blocks if b["bbox"][3] > split_y]
                    print(f"  (on p. {pg}, {len(before)} blocks separated off "
                          f"as belonging to the PREVIOUS article, per "
                          f"find_split_y)")
                else:
                    print(f"  (on p. {pg}: find_split_y found no cut - the "
                          f"whole page goes to this group)")
            raw.extend(page_blocks)

        headings = [b for b in raw if b["type"] in ("title", "heading")]
        print(f"\n  --> {len(headings)} title/heading blocks found vs. "
              f"{len(group)} contents entries "
              f"{'(MATCH)' if len(headings) == len(group) else '(MISMATCH)'}\n")

        # The actual result of find_heading_positions(), not just a raw
        # count of headings - this shows the effect of the fuzzy match.
        positions = find_heading_positions(raw, group)
        print("  Actual assignment (find_heading_positions):")
        for entry, pos in zip(group, positions):
            if pos is None:
                print(f"    MISS {entry['title']!r} -> NOT FOUND (empty content)")
            else:
                b = raw[pos]
                print(f"    OK   {entry['title']!r} -> p.{b['page']} "
                      f"id={b['block_id']} {b['text'][:50]!r}")
        print()

        for b in raw:
            marker = ">>> " if b["type"] in ("title", "heading") else "    "
            print(f"  {marker}p.{b['page']:>3} id={b['block_id']:>3} "
                  f"{b['type']:9} size={b['font_size']:>5} {b['text'][:65]!r}")
        print()


if __name__ == "__main__":
    main()
