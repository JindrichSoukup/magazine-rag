"""Dávkové zpracování celého archivu: PDF -> korpus článků -> chunky.

Projde všechna PDF ve vstupní složce, na každé pustí celou extrakční část
pipeline (extract_blocks -> create_toc -> build_page_map -> assign_articles)
a nakonec z toho složí dva soubory pro embedding:

    <output>/corpus.json    - všechny články ze všech čísel pohromadě
    <output>/chunks.jsonl   - chunky připravené k zaembeddování

Mezivýsledky každého čísla zůstávají v `<output>/<rok>-<číslo>/` - hodí se
při ladění, protože je z nich vidět, ve které fázi se co pokazilo.

Jak se PDF pojmenovávají, určuje `filename_pattern` v profilu zdroje
(u Živy `ziva-RRRR-C.pdf`, u MagPi `MagPi<N>.pdf`).

Použití:
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
    """Najdi všechna PDF odpovídající profilu a vrať je v pořadí vydání."""
    pattern = profile.filename_re
    found, skipped = [], []
    for path in sorted(input_dir.glob("*.pdf")):
        m = pattern.search(path.name)
        if not m:
            skipped.append(path.name)
            continue
        # Profil může mít jednu skupinu (průběžné číslování) nebo dvě
        # (ročník + číslo). Sjednotíme to na dvojici, ať zbytek pipeline
        # nemusí řešit, který časopis to je.
        groups = m.groups()
        year, issue = (groups[0], groups[1]) if len(groups) >= 2 else ("", groups[0])
        found.append((year, issue, path))
    if skipped:
        print(f"  [warn] {len(skipped)} PDF neodpovídá vzoru profilu "
              f"{profile.key!r} ({profile.filename_pattern}), přeskakuji: "
              + ", ".join(skipped[:5]) + (" ..." if len(skipped) > 5 else ""))
    # číselné řazení, ať "MagPi9" nepředběhne "MagPi10"
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

    print(f"[{label}] extrahuji bloky z {pdf_path.name} ...")
    blocks = extract_pdf(str(pdf_path), profile)
    dump("blocks.json", blocks)

    print(f"[{label}] tahám obsah čísla (TOC) ...")
    toc = build_toc(str(pdf_path), profile)
    dump("toc.json", toc)

    page_map = build_page_map(blocks, profile)
    dump("page_map.json", page_map)

    print(f"[{label}] skládám články z bloků ...")
    articles = assemble_articles(blocks, toc, page_map["label_to_page"], profile,
                                 year=year, issue=issue)
    dump("articles.json", articles)

    n_chunks = sum(len(a["chunks"]) for a in articles)
    print(f"[{label}] hotovo: {len(articles)} článků, {n_chunks} bloků")
    return articles


def main():
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", required=True, help="složka s PDF čísel časopisu")
    ap.add_argument("--output", required=True, help="výstupní složka")
    ap.add_argument("--skip-first", type=int, default=None,
                    help="kolik stránek na začátku čísla ignorovat při "
                         "chunkování (obálka); výchozí hodnota je v profilu")
    ap.add_argument("--skip-last", type=int, default=None,
                    help="kolik stránek na konci čísla ignorovat při "
                         "chunkování (zadní obálka); výchozí hodnota je v profilu")
    profiles.add_profile_argument(ap)
    args = ap.parse_args()

    profile = profiles.get(args.profile)
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    issues = find_issues(input_dir, profile)
    print(f"Profil {profile.key!r}: nalezeno {len(issues)} čísel ke zpracování.")

    all_articles = []
    failed = []
    for year, issue, pdf_path in issues:
        issue_out_dir = output_dir / issue_label(year, issue)
        try:
            all_articles.extend(
                process_one(pdf_path, year, issue, issue_out_dir, profile))
        except Exception as e:  # ať jedno rozbité číslo nezastaví celý archiv
            print(f"  [CHYBA] {pdf_path.name}: {e!r}")
            failed.append(str(pdf_path))

    corpus_path = output_dir / "corpus.json"
    corpus_path.write_text(json.dumps(all_articles, ensure_ascii=False, indent=2),
                           encoding="utf-8")

    skip_first = args.skip_first if args.skip_first is not None else profile.skip_first_pages
    skip_last = args.skip_last if args.skip_last is not None else profile.skip_last_pages
    print(f"\nSkládám chunky pro embedding (skip_first={skip_first}, "
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

    print("\n=== Souhrn ===")
    print(f"Zpracováno úspěšně: {len(issues) - len(failed)} / {len(issues)} čísel")
    print(f"Celkem článků: {len(all_articles)}")
    print(f"Celkem bloků v articles.json: "
          f"{sum(len(a['chunks']) for a in all_articles)}")
    print(f"Celkem chunků pro embedding: {len(all_chunks)}")
    print(f"Souhrnný korpus: {corpus_path}")
    print(f"Embedding chunky: {chunks_path}")
    if failed:
        print("Selhala tato čísla (podívej se na chybu výše a oprav ručně):")
        for f in failed:
            print("  -", f)


if __name__ == "__main__":
    main()
