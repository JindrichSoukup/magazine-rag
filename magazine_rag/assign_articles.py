"""Stage 2b: attach every text block to the article it belongs to.

Inputs:
  blocks.json    - per-page text blocks (from extract_blocks.py)
  toc.json       - article list with PRINTED page numbers (create_toc.py)
  page_map.json  - printed-label <-> pdf-page translation (build_page_map.py)

What it does:
  1. Resolves each TOC entry's printed page ("262", "CXXXIII", ...) to a
     real PDF page number, then sorts the TOC into true reading order
     (the JSON order of toc.json is NOT reading order - articles and back
     matter get interleaved during extraction).
  2. Builds page ranges: article N owns every PDF page from its own start
     page up to, but not including, the start page of the next DIFFERENT
     page. Consecutive TOC entries sharing the exact same start page -
     which happens when create_toc.py split one semicolon-joined combined
     entry into several - are treated as one GROUP that collectively owns
     that page range, then subdivided among themselves by locating each
     entry's own heading within the group's shared content. See
     find_heading_positions().
  3. On the FIRST page of a group's range, checks whether the previous
     article's tail actually spills onto this same page too - common when
     a short remainder of the previous article shares a page with this
     group's title, author and abstract. If so, the page's blocks are
     split at a vertical cut point instead of the whole page going to
     this group. See find_split_y(): this was a real bug, found by
     inspecting a mismatched article, where a piece on mycorrhiza leaked
     into an article on epigenetics purely because they shared a page.
  4. Tags every block with the article it falls in.
  5. Filters out obvious junk (running headers and footers, the copyright
     line, lone figure-number labels floating on top of images).
  6. Produces a *best-effort* column-aware reading order per article,
     purely so a human can eyeball the result. It does not need to be
     perfect for the later RAG step - see the note at the bottom.

Output: articles.json - one entry per article:
  {
    "title": ..., "author": ...,
    "printed_page_start": "262", "pdf_page_start": 4, "pdf_page_end": 7,
    "total_pdf_pages": 84,  # pages in the whole ISSUE, not the article -
                            # build_chunks.py needs it to recognise the
                            # cover pages at the end of the PDF
    "chunks": [ {block_id, page, type, text}, ... ],  # best-effort order
    "quality_flags": [],  # [] = confidently assigned. Otherwise some of:
                          #   "boundary_page_unverified" - the boundary
                          #     page was shared with the previous article
                          #     but find_split_y found no cut, so the
                          #     whole page was taken without certainty
                          #   "merged_fallback" - several contents entries
                          #     could not be told apart in the text at
                          #     all and were merged into one record
                          #     (see "original_titles")
                          #   "heading_not_found_empty" - the entry's own
                          #     heading was not found in the text; its
                          #     content ended up with a neighbouring entry
                          #     (see that entry's "absorbed_titles")
                          #   "absorbed_unmatched_siblings" - this record
                          #     also holds the content of a neighbouring
                          #     entry whose own heading was not found
                          #     (see "absorbed_titles")
    "full_text": "...",              # the article body as one string
    "full_text_paragraphs": [ {"page": 4, "text": "..."}, ... ],  # the
                            # same, but per paragraph with its page -
                            # build_chunks.py chunks from this, and it is
                            # there, not here, that covers get dropped
    "captions_text": "...", "captions_paragraphs": [ ... ],  # ditto for
                            # figure captions
  }
"""
import argparse
import json
import re
from pathlib import Path

from magazine_rag import profiles
from magazine_rag.console import setup_console
from magazine_rag.build_page_map import normalize_label
from magazine_rag.extract_blocks import looks_like_word_break, HYPHEN_BREAK_RE

COLUMN_GAP = 40  # pt; an x0 gap bigger than this starts a new column cluster
BODY_TYPES = {"title", "heading", "other", "body"}


def join_paragraphs(texts):
    """Join texts into one string, separating paragraphs with a blank line.

    But when the previous piece ends in a hyphenated word break (see
    extract_blocks.looks_like_word_break) and the next piece continues it,
    they are glued with neither space nor blank line, exactly as
    extract_blocks.smart_join() does within a page. That is needed again
    here because adjacent chunks can come from different columns or even
    different PDF pages, where extract_blocks.py never merged - so
    otherwise the full text would end up carrying things like
    "...dlouhodo-\n\nbé roční úhrny".
    """
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
    """The same logic as join_paragraphs (gluing across a hyphenated word
    break), but over (page, text) pairs, returning (page, text) pairs, so
    the page stays traceable even for a merged paragraph. When two
    paragraphs merge because of a broken word, the first one's page is
    kept: glued words almost always stay on one page, at most on adjacent
    ones.

    This matters for build_chunks.py: it is there, at the level of
    chunking for RAG, that cover and back pages get dropped - and for
    that it needs each paragraph's page, not one finished string.
    """
    out = []  # list of [page, text], mutable because of merging
    for page, text in items:
        text = text.strip()
        if not text:
            continue
        if out and looks_like_word_break(out[-1][1], text):
            out[-1][1] = HYPHEN_BREAK_RE.sub(r"\1", out[-1][1]) + text
        else:
            out.append([page, text])
    return [(p, t) for p, t in out]


def build_full_text_and_paragraphs(chunks):
    """Glue ONLY the article body (title/heading/other/body) into one
    readable text, paragraphs separated by a blank line - and return it
    both as a single string, for quick reading, and as a list of
    {"page", "text"} for build_chunks.py, which filters covers by page.

    Figure captions and panel annotations are deliberately NOT included.
    A figure or chart often sits physically in the middle of a column, so
    the text before it and the text after it really are two separate
    blocks in the PDF - they cannot be merged, because there is a figure
    between them visually. Our (column, y) ordering then places them
    "next to each other" with the caption in between, which in running
    text reads as a sentence cut in half. Leaving captions out of the
    body means the text before and after a figure lands in adjacent
    paragraphs - not a perfect join, but at least no sentence is severed -
    and the captions stay together in their own captions_text field, and
    always in chunks.
    """
    items = [(c["page"], c["text"].strip()) for c in chunks
             if c["type"] in BODY_TYPES and c["text"].strip()]
    merged = join_paragraphs_with_pages(items)
    full_text = "\n\n".join(t for _, t in merged)
    paragraphs = [{"page": p, "text": t} for p, t in merged]
    return full_text, paragraphs


def build_captions_text_and_paragraphs(chunks):
    """Figure captions - not panel annotations, which carry no content -
    kept together in the order they appeared in the article. Like
    build_full_text_and_paragraphs, returns both a string and a list with
    pages."""
    items = [(c["page"], c["text"].strip()) for c in chunks
             if c["type"] == "caption" and c["text"].strip()]
    merged = join_paragraphs_with_pages(items)
    captions_text = "\n\n".join(t for _, t in merged)
    paragraphs = [{"page": p, "text": t} for p, t in merged]
    return captions_text, paragraphs


def find_split_y(page_blocks, entry):
    """Find the Y coordinate of the cut on a page shared with the PREVIOUS
    article: above it (smaller y) is still the previous article, from it
    downwards THIS article begins.

    The y0 of the author block is used, being more precise because it
    sits closer to the real transition; when that is not found (no
    author, or the text does not match) we fall back to the y0 of the
    title. The cut is then compared against the BOTTOM edge (y1) of every
    block on the page, not against position in the (column, y) list -
    because two articles' content can interleave between columns on one
    page, typically via a last column running independently, so "every
    block before the title in reading order" would not be enough. See
    the comment at the call site.
    """
    author_text = (entry.get("author") or "").strip()
    if author_text:
        for b in page_blocks:
            if b["type"] == "other" and texts_match(author_text, b["text"]):
                return b["bbox"][1]
    title_text = entry["title"].strip()
    for b in page_blocks:
        if b["type"] in ("title", "heading") and texts_match(title_text, b["text"]):
            return b["bbox"][1]
    return None  # not found - do not split (safe default: change nothing)


# Prefixes the contents page uses but the heading in the body does not.
# Czech literals: these are the actual words printed in Živa's contents.
COMMON_TOC_PREFIXES = (
    "recenze:", "k výuce:", "upoutávka na knihu:", "zaujalo nás:",
    "biozvěst:",
)


def normalize_for_match(text: str) -> str:
    """Strip the common prefixes that appear in the contents but are
    missing from the heading in the body itself: the contents say
    "Recenze: Book title", the body just "Author: Book title with
    subtitle"."""
    t = text.strip().lower()
    for prefix in COMMON_TOC_PREFIXES:
        if t.startswith(prefix):
            return t[len(prefix):].strip()
    return t


def longest_common_substring_len(a: str, b: str) -> int:
    """Length of the longest contiguous common substring (textbook DP,
    O(len(a)*len(b)) - fine for short titles and headings)."""
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
    """An exact match, a substring either way round, OR a shared
    contiguous stretch of at least min_common characters.

    The last of those is needed for cases like "Fenomén Velká kotlina 1 -
    pohled geobotanický a horského ekologa", where the contents hold ONE
    entry covering TWO different headings: two authors, two "views" of
    the same subject. Neither heading contains the whole contents title,
    but both share a long enough distinctive stretch of it.
    """
    a = normalize_for_match(toc_title)
    b = normalize_for_match(block_text)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    return longest_common_substring_len(a, b) >= min_common


def find_heading_positions(ordered_blocks, entries):
    """Split a group's shared content between its individual TOC entries.

    When create_toc.py split one combined TOC record - several items
    under one page number, separated by a semicolon - into several
    `entries` sharing a pdf_page_start, their common block of content
    (`ordered_blocks`, already sorted by page and by column within a
    page) has to be divided between them.

    The more general and more robust rule is tried first: count all the
    title/heading blocks in `ordered_blocks`, and if there are exactly as
    many as there are `entries`, take them in order as the cut points. A
    quick shortcut that works reliably whenever the counts line up.

    When the counts disagree, a fuzzy match (see texts_match) is sought
    for each entry separately. That copes with reviews - the contents
    give a shortened title while the body says "Author: Full book title",
    but the full title is CONTAINED in the heading as a substring - and
    with entries covering two headings at once, where the unmatched
    "extra" heading is simply added to the PREVIOUS successfully located
    entry by the calling code rather than being lost.
    """
    heading_idxs = [i for i, b in enumerate(ordered_blocks)
                    if b["type"] in ("title", "heading")]
    if len(heading_idxs) == len(entries):
        return heading_idxs

    positions = []
    scores = []  # how well each entry matched the heading it landed on
    search_from = 0
    for entry in entries:
        title_text = entry["title"].strip()
        found = None
        for idx in range(search_from, len(ordered_blocks)):
            b = ordered_blocks[idx]
            if b["type"] in ("title", "heading") and texts_match(title_text, b["text"]):
                found = idx
                break
        score = (longest_common_substring_len(
                     normalize_for_match(title_text),
                     normalize_for_match(ordered_blocks[found]["text"]))
                 if found is not None else 0)
        prev = next((k for k in range(len(positions) - 1, -1, -1)
                     if positions[k] is not None), None)
        if (found is not None and prev is not None
                and positions[prev] == found and scores[prev] >= score):
            # Two entries on the SAME heading, and the earlier one matches
            # it at least as well: one heading covering both, e.g. a single
            # review of two books (2017/6, "Marine Mammals ... a ...
            # Cetacean Paleobiology"). The heading and its text stay with
            # the earlier entry, and this one is absorbed into it by the
            # caller (absorbed_unmatched_siblings) rather than stealing
            # the text and leaving the earlier one empty.
            #
            # When THIS entry matches better, both keep the heading's
            # position: the earlier entry then gets the text before the
            # heading and this one the heading onwards. That is right when
            # the earlier entry only matched through shared words and its
            # real text has no heading of its own (2020/6: "Knihy
            # Nakladatelství Academia" vs the heading "Ceny Nakladatelství
            # Academia za rok 2019 ...", with the book list above it).
            print(f"  [info] {title_text!r} shares a heading with "
                  f"{entries[prev]['title']!r}, which matches it better - "
                  f"treating them as one article")
            positions.append(None)
            scores.append(0)
            continue
        if found is None:
            print(f"  [warn] no heading of its own found for {title_text!r} "
                  f"in a combined contents record ({len(heading_idxs)} "
                  f"headings on the page vs. {len(entries)} contents "
                  f"entries, counts disagree) - it will get empty content, "
                  f"its text probably stayed with a neighbouring entry - "
                  f"check by hand")
        else:
            # Deliberately `found`, not `found + 1`: the next entry may
            # land on the same heading, and the check above decides who
            # gets it.
            search_from = found
        positions.append(found)  # None, or an index
        scores.append(score)
    return positions


def is_junk(b: dict, profile, page_height: float = 0.0) -> bool:
    """Repeating boilerplate that is not the content of any article.

    The first two tests are properties of a specific magazine - what its
    footer and imprint look like - and come from the profile. The third
    is generic and holds everywhere.
    """
    text = b["text"].strip()
    font = b["font"]
    size = b["font_size"]

    # running header/footer ("ziva.avcr.cz 262 živa 6/2014" and friends)
    if profile.is_footer_block(font, size, text, b["bbox"], page_height):
        return True

    # the imprint/copyright strip repeated on every page
    if profile.is_junk_text(text):
        return True

    # lone figure-number markers overlaid on photos and diagrams: a block
    # that is *just* "3" or "12" sitting in a tiny bounding box
    x0, y0, x1, y1 = b["bbox"]
    if text.isdigit() and (x1 - x0) < 15 and (y1 - y0) < 15:
        return True

    return False


def cluster_columns(blocks_on_page):
    """Tag each block with a column index (0, 1, 2, ...) based on its x0,
    so a page can be sorted left to right, top to bottom."""
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
    # See build_page_map: page height is derived from the blocks and is
    # needed by position-based footer detection.
    page_height = max((b["bbox"][3] for b in blocks), default=0.0)

    # 1) resolve + sort the TOC into real reading order --------------------
    resolved = []
    for entry in toc:
        # via normalize_label, because the contents page and the footer
        # need not write the same label identically ("032" vs "32")
        pdf_page = label_to_page.get(normalize_label(entry["printed_page"]))
        if pdf_page is None:
            print(f"  [warn] could not resolve printed page "
                  f"{entry['printed_page']!r} for {entry['title']!r} - skipping")
            continue
        resolved.append({**entry, "pdf_page_start": pdf_page})
    resolved.sort(key=lambda e: e["pdf_page_start"])

    # 2) group consecutive entries sharing the SAME pdf_page_start. Most
    #    groups have size 1 (an ordinary article with its own page);
    #    larger ones occur where create_toc.py split one combined TOC
    #    record on a semicolon --------------------------------------------
    groups = []
    for entry in resolved:
        if groups and groups[-1][0]["pdf_page_start"] == entry["pdf_page_start"]:
            groups[-1].append(entry)
        else:
            groups.append([entry])

    # 3) the end of the page range is computed PER GROUP, not per entry:
    #    the whole group ends where the NEXT group (a different page)
    #    begins ------------------------------------------------------------
    # Known limitation: this is the last page with any TEXT, not the page
    # count of the PDF. When the issue ends with pages that are only
    # pictures (a back cover photo), it comes out lower than the real
    # count, and build_chunks.filter_cover_pages (skip_last) then drops
    # the last pages of real content instead of the cover. Left as is
    # for now; the fix would be to carry the PDF's page count from
    # extract_blocks.
    last_pdf_page = max(b["page"] for b in blocks)
    for gi, group in enumerate(groups):
        end = (groups[gi + 1][0]["pdf_page_start"] - 1
               if gi + 1 < len(groups) else last_pdf_page)
        for entry in group:
            entry["pdf_page_end"] = end

    # 4) group blocks by page for fast lookup ------------------------------
    blocks_by_page = {}
    for b in blocks:
        blocks_by_page.setdefault(b["page"], []).append(b)

    def ordered_page_blocks(pg):
        page_blocks = [b for b in blocks_by_page.get(pg, [])
                       if not is_junk(b, profile, page_height)]
        return [b for col, y0, b in
                sorted(cluster_columns(page_blocks), key=lambda t: (t[0], t[1]))]

    # 5) split blocks between GROUPS - and on a boundary page, where the
    #    TOC says a NEW group starts, check whether the PREVIOUS group
    #    also ends there (see find_split_y above) --------------------------
    group_raw_blocks = [[] for _ in groups]
    boundary_unverified = [False] * len(groups)  # see quality_flags below

    for gi, group in enumerate(groups):
        first_entry = group[0]
        start_pg, end_pg = first_entry["pdf_page_start"], first_entry["pdf_page_end"]
        for pg in range(start_pg, end_pg + 1):
            page_blocks = ordered_page_blocks(pg)
            if pg == start_pg and gi > 0:
                split_y = find_split_y(page_blocks, first_entry)
                if split_y is not None:
                    for b in page_blocks:
                        # y1, the block's bottom edge, at or above the cut
                        # means the block finished BEFORE this group began,
                        # so it belongs to the previous group whatever
                        # column it sits in (content can interleave between
                        # columns on a page, see the comment above)
                        if b["bbox"][3] <= split_y:
                            group_raw_blocks[gi - 1].append(b)
                        else:
                            group_raw_blocks[gi].append(b)
                    continue
                else:
                    # We did not find this group's own title or author on
                    # this page, so there was nothing to decide by whether
                    # the previous article also ends here. The whole page
                    # went to THIS group, but unverified.
                    boundary_unverified[gi] = True
            group_raw_blocks[gi].extend(page_blocks)

    # 6) inside a group of more than one entry, split the blocks between
    #    the individual TOC entries by the position of each one's own
    #    heading.
    #
    #    If NOT A SINGLE heading is found for the whole group - the counts
    #    disagreed AND the text match failed for every entry, in practice
    #    because the semicolon here separated subtitles of ONE article
    #    rather than several separate ones - we DO NOT SPLIT AT ALL and
    #    merge the group back into one record with a joined title.
    #    Otherwise the group's entire content would silently disappear.
    #    That was a real bug, fixed after an audit of issue 2023/1.
    #
    #    At the same time every resulting record notes in "quality_flags"
    #    whether and where we had to fall back on a guess, so that the
    #    corpus and RAG answers can tell it apart from confidently
    #    assigned content. ------------------------------------------------
    final_entries = []  # (entry, raw_blocks); may hold FEWER items than
                        # `resolved` if some group got merged
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
            print(f"  [warn] no entry of the group on page "
                  f"{group[0]['pdf_page_start']} had a heading of its own - "
                  f"not splitting, merging back into one record: "
                  f"{combined_title!r}")
            merged = dict(group[0])
            merged["title"] = combined_title
            merged["author"] = next((e["author"] for e in group if e.get("author")), None)
            merged["quality_flags"] = ["merged_fallback"] + (
                ["boundary_page_unverified"] if boundary_unverified[gi] else [])
            merged["original_titles"] = [e["title"] for e in group]
            final_entries.append((merged, raw))
            continue

        # Content BEFORE the first heading found - normally none, but if
        # find_split_y above did not place the boundary with the genuinely
        # previous article exactly, something may be left there - must not
        # vanish without trace, so it goes to the first entry found rather
        # than being discarded.
        first_found_i = next(i for i, p in enumerate(positions) if p is not None)
        for i, (entry, pos) in enumerate(zip(group, positions)):
            entry = dict(entry)
            if pos is None:
                entry["quality_flags"] = ["heading_not_found_empty"]
                final_entries.append((entry, []))  # warning already printed
                continue
            lo = 0 if i == first_found_i else pos
            # Find the NEXT located (non-None) position among the group's
            # remaining entries, skipping any holes left by unmatched
            # neighbours, so we do not steal content that is not ours.
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

    # 7) build chunks from the assigned raw blocks - sorted by page, and
    #    by column within a page --------------------------------------------
    articles = []
    for art_idx, (entry, entry_blocks) in enumerate(final_entries):
        by_page = {}
        for b in entry_blocks:
            by_page.setdefault(b["page"], []).append(b)

        chunks = []
        for pg in sorted(by_page):
            for col, y0, b in sorted(cluster_columns(by_page[pg]),
                                     key=lambda t: (t[0], t[1])):
                chunks.append({
                    "block_id": b["block_id"],
                    "page": b["page"],
                    "type": b["type"],
                    "text": b["text"],
                })

        if not chunks:
            print(f"  [warn] article {entry['title']!r} has no blocks left "
                  f"after splitting the boundary pages - check by hand")

        # The id must be unique across the whole archive, not just the
        # issue, so it carries whatever identifies the issue: year and
        # issue for Živa ("2014-6-3"), just the issue for The MagPi, which
        # numbers its issues continuously and has no year ("150-3").
        article_id = "-".join(str(p) for p in (year, issue, art_idx)
                              if p not in (None, ""))
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
            "total_pdf_pages": last_pdf_page,  # pages in the whole issue,
            # not the article - build_chunks.py needs it to tell which
            # pages are the cover at the end of the PDF
            "quality_flags": entry.get("quality_flags", []),  # [] = confidently
            # assigned; otherwise see the list of values in the docstring
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
    ap = argparse.ArgumentParser(description="assign text blocks to articles")
    ap.add_argument("blocks", help="blocks.json (from extract_blocks)")
    ap.add_argument("toc", help="toc.json (from create_toc)")
    ap.add_argument("page_map", help="page_map.json (from build_page_map)")
    ap.add_argument("out", help="output articles.json")
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
    print(f"{len(articles)} articles "
          f"({sum(len(a['chunks']) for a in articles)} blocks) -> {args.out}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# NOTE on reading order and RAG:
#
# The "best-effort" ordering above is good enough to *read* an article and
# sanity-check the pipeline, but it will occasionally misplace a block -
# a long figure caption that happens to use a body-sized font gets sorted
# by its raw y position and may show up a paragraph too early or too late.
# That is a genuine, hard-to-fully-solve problem in multi-column magazine
# layout parsing.
#
# For the RAG step it barely matters: each block or chunk is already a
# reasonably self-contained paragraph (title plus intro, or subheading
# plus paragraph), so we embed and retrieve *chunks*, not a perfectly
# ordered document. A caption that ends up slightly out of place is simply
# retrieved as its own small chunk ("Fig. 3: karyotype scheme...") - which
# is often exactly what you would want anyway.
# ---------------------------------------------------------------------------
