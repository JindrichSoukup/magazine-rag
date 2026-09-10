"""Diagnostics for a contents entry that keeps failing to match across
issues.

Written for Živa's recurring back-matter item "Kontaktní adresy" (contact
addresses of the authors), which the contents page names one way and the
body of the issue another. The tool finds the group containing it and
prints what heading, if any, is actually found in the text - so that the
fuzzy match has something concrete to be judged against.

`--needle` makes it usable for any other stubborn entry; the default is
the Živa case it was written for.

Usage:
    python -m tools.diag_contact_addresses output/2019-2/blocks.json \\
        output/2019-2/toc.json output/2019-2/page_map.json [profile]
"""
import argparse
import json

from magrag import profiles
from magrag.console import setup_console
from magrag.assign_articles import (
    cluster_columns,
    find_split_y,
    is_junk,
    texts_match,
)

DEFAULT_NEEDLE = "kontaktní adresy"


def ordered_page_blocks(blocks_by_page, pg, profile, page_height):
    page_blocks = [b for b in blocks_by_page.get(pg, [])
                   if not is_junk(b, profile, page_height)]
    return [b for col, y0, b in
            sorted(cluster_columns(page_blocks), key=lambda t: (t[0], t[1]))]


def main():
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("blocks", help="blocks.json")
    ap.add_argument("toc", help="toc.json")
    ap.add_argument("page_map", help="page_map.json")
    ap.add_argument("--needle", default=DEFAULT_NEEDLE,
                    help="lower-case substring identifying the contents "
                         f"entry to investigate (default {DEFAULT_NEEDLE!r})")
    profiles.add_profile_argument(ap)
    args = ap.parse_args()

    profile = profiles.get(args.profile)
    blocks = json.loads(open(args.blocks, encoding="utf-8").read())
    page_height = max((b["bbox"][3] for b in blocks), default=0.0)
    toc = json.loads(open(args.toc, encoding="utf-8").read())
    page_map = json.loads(open(args.page_map, encoding="utf-8").read())
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

    needle = args.needle.lower()
    target_group = None
    target_gi = None
    for gi, group in enumerate(groups):
        if any(needle in e["title"].strip().lower() for e in group):
            target_group = group
            target_gi = gi
            break

    if target_group is None:
        print(f"No group found with {args.needle!r} in its title.")
        return

    first_entry = target_group[0]
    start_pg, end_pg = first_entry["pdf_page_start"], first_entry["pdf_page_end"]

    print(f"Group containing {args.needle!r}: pp. {start_pg}-{end_pg}, "
          f"{len(target_group)} entries:")
    for e in target_group:
        print(f"    - {e['title']!r}")
    print()

    raw = []
    for pg in range(start_pg, end_pg + 1):
        page_blocks = ordered_page_blocks(blocks_by_page, pg, profile, page_height)
        if pg == start_pg and target_gi > 0:
            split_y = find_split_y(page_blocks, first_entry)
            if split_y is not None:
                page_blocks = [b for b in page_blocks if b["bbox"][3] > split_y]
        raw.extend(page_blocks)

    headings = [b for b in raw if b["type"] in ("title", "heading")]
    print(f"Every title/heading block in this group ({len(headings)} in total):")
    for b in headings:
        print(f"    p.{b['page']} id={b['block_id']} {b['text'][:70]!r}")
    print()

    # Try the fuzzy match against EVERY heading found, to see how close
    # or far each one is.
    target_entry = next(e for e in target_group
                        if needle in e["title"].strip().lower())
    print(f"Fuzzy match of {target_entry['title']!r} against each heading:")
    for b in headings:
        print(f"    match={texts_match(target_entry['title'], b['text'])!s:5} "
              f"<- {b['text'][:60]!r}")


if __name__ == "__main__":
    main()
