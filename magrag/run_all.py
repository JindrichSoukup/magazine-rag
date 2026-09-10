"""Batch-process a whole archive: PDFs -> a corpus of articles -> chunks.

Walks every PDF in the input directory, runs the whole extraction part of
the pipeline on each (extract_blocks -> create_toc -> build_page_map ->
assign_articles), and finally assembles two files for embedding:

    <output>/corpus.json    - every article from every issue together
    <output>/chunks.jsonl   - chunks ready to be embedded

Each issue's intermediate results stay in `<output>/<year>-<issue>/`.
They are useful when debugging, because they show which stage something
went wrong in.

How the PDFs are named comes from `filename_pattern` in the source
profile (`ziva-YYYY-N.pdf` for Živa, `MagPi<N>.pdf` for The MagPi).

Usage:
    python -m magrag.run_all --profile ziva --input ./pdf --output ./output
    python -m magrag.run_all --profile magpi --input ./pdf --output ./output
"""
import argparse
import json
from pathlib import Path

from magrag import profiles
from magrag.assign_articles import assemble_articles
from magrag.build_chunks import build_chunks_for_article
from magrag.build_page_map import build_page_map
from magrag.console import setup_console
from magrag.create_toc import build_toc
from magrag.extract_blocks import extract_pdf


def find_issues(input_dir: Path, profile):
    """Find every PDF matching the profile and return them in issue order."""
    pattern = profile.filename_re
    found, skipped = [], []
    for path in sorted(input_dir.glob("*.pdf")):
        m = pattern.search(path.name)
        if not m:
            skipped.append(path.name)
            continue
        # A profile may have one group (continuous numbering) or two
        # (year plus issue). Normalise to a pair so the rest of the
        # pipeline need not care which magazine this is.
        groups = m.groups()
        year, issue = (groups[0], groups[1]) if len(groups) >= 2 else ("", groups[0])
        found.append((year, issue, path))
    if skipped:
        print(f"  [warn] {len(skipped)} PDFs do not match the pattern of "
              f"profile {profile.key!r} ({profile.filename_pattern}), "
              f"skipping: " + ", ".join(skipped[:5])
              + (" ..." if len(skipped) > 5 else ""))
    # numeric sort, so that "MagPi9" does not come before "MagPi10"
    found.sort(key=lambda t: (_sort_key(t[0]), _sort_key(t[1])))
    return found


def _sort_key(value: str):
    return (0, int(value)) if value.isdigit() else (1, value)


def issue_label(year: str, issue: str) -> str:
    return f"{year}-{issue}" if year else str(issue)


def process_one(pdf_path: Path, year: str, issue: str, out_dir: Path, profile):
    out_dir.mkdir(parents=True, exist_ok=True)
    label = issue_label(year, issue)

    def dump(name, data):
        (out_dir / name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[{label}] extracting blocks from {pdf_path.name} ...")
    blocks = extract_pdf(str(pdf_path), profile)
    dump("blocks.json", blocks)

    print(f"[{label}] reading the contents page ...")
    toc = build_toc(str(pdf_path), profile)
    dump("toc.json", toc)

    page_map = build_page_map(blocks, profile)
    dump("page_map.json", page_map)

    print(f"[{label}] assembling articles from blocks ...")
    articles = assemble_articles(blocks, toc, page_map["label_to_page"], profile,
                                 year=year, issue=issue)
    dump("articles.json", articles)

    n_chunks = sum(len(a["chunks"]) for a in articles)
    print(f"[{label}] done: {len(articles)} articles, {n_chunks} blocks")
    return articles


def main():
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True,
                    help="directory holding the issues' PDFs")
    ap.add_argument("--output", required=True, help="output directory")
    ap.add_argument("--skip-first", type=int, default=None,
                    help="how many pages at the start of an issue to ignore "
                         "when chunking (the cover); the default comes from "
                         "the profile")
    ap.add_argument("--skip-last", type=int, default=None,
                    help="how many pages at the end of an issue to ignore "
                         "when chunking (the back cover); the default comes "
                         "from the profile")
    profiles.add_profile_argument(ap)
    args = ap.parse_args()

    profile = profiles.get(args.profile)
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    issues = find_issues(input_dir, profile)
    print(f"Profile {profile.key!r}: found {len(issues)} issues to process.")

    all_articles = []
    failed = []
    for year, issue, pdf_path in issues:
        issue_out_dir = output_dir / issue_label(year, issue)
        try:
            all_articles.extend(
                process_one(pdf_path, year, issue, issue_out_dir, profile))
        except Exception as e:  # one broken issue must not stop the archive
            print(f"  [ERROR] {pdf_path.name}: {e!r}")
            failed.append(str(pdf_path))

    corpus_path = output_dir / "corpus.json"
    corpus_path.write_text(json.dumps(all_articles, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    skip_first = (args.skip_first if args.skip_first is not None
                  else profile.skip_first_pages)
    skip_last = (args.skip_last if args.skip_last is not None
                 else profile.skip_last_pages)
    print(f"\nBuilding chunks for embedding (skip_first={skip_first}, "
          f"skip_last={skip_last}) ...")
    all_chunks = []
    for article in all_articles:
        if not article.get("full_text", "").strip():
            continue
        all_chunks.extend(
            build_chunks_for_article(article, profile, skip_first, skip_last))

    chunks_path = output_dir / "chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print("\n=== Summary ===")
    print(f"Processed successfully: {len(issues) - len(failed)} / {len(issues)} issues")
    print(f"Articles in total: {len(all_articles)}")
    print(f"Blocks in articles.json: "
          f"{sum(len(a['chunks']) for a in all_articles)}")
    print(f"Chunks for embedding: {len(all_chunks)}")
    print(f"Combined corpus: {corpus_path}")
    print(f"Embedding chunks: {chunks_path}")
    if failed:
        print("These issues failed (see the error above and fix by hand):")
        for f in failed:
            print("  -", f)


if __name__ == "__main__":
    main()
