"""
Dávkové spuštění celého pipeline (extract_blocks -> create_toc ->
build_page_map -> assign_articles -> build_chunks) přes celý archiv čísel
Živy.

Očekává soubory pojmenované ziva-RRRR-C.pdf (např. ziva-2024-1.pdf) v jedné
vstupní složce. Pro každé číslo vyrobí podsložku ve výstupu s mezivýsledky
(blocks.json, toc.json, page_map.json, articles.json) - ty se hodí pro
ladění/kontrolu. Nakonec sesbírá VŠECHNY články ze VŠECH čísel do jednoho
souboru ziva_corpus.json a rovnou z něj postaví ziva_embedding_chunks.jsonl
(viz build_chunks.py) - to je vstup pro embedding/RAG.

Použití:
    python run_all.py --input ./pdf --output ./output
    python run_all.py --input ./pdf --output ./output --skip-first 2 --skip-last 3

Struktura vstupní složky:
    pdf/ziva-2014-6.pdf
    pdf/ziva-2015-1.pdf
    pdf/ziva-2015-2.pdf
    ...
"""
import argparse
import json
import re
from pathlib import Path

from extract_blocks import extract_pdf
from create_toc import build_toc
from build_page_map import build_page_map
from assign_articles import assemble_articles
from build_chunks import build_chunks_for_article, SKIP_FIRST_PAGES, SKIP_LAST_PAGES

FILENAME_RE = re.compile(r"ziva-(\d{4})-(\d)\.pdf$", re.IGNORECASE)


def find_issues(input_dir: Path):
    """Najdi všechny ziva-RRRR-C.pdf a vrať je seřazené podle roku a čísla."""
    found = []
    for path in input_dir.glob("ziva-*.pdf"):
        m = FILENAME_RE.search(path.name)
        if not m:
            print(f"  [warn] přeskakuji, neodpovídá vzoru ziva-RRRR-C.pdf: {path.name}")
            continue
        year, issue = m.group(1), m.group(2)
        found.append((year, issue, path))
    found.sort(key=lambda t: (t[0], t[1]))
    return found


def process_one(pdf_path: Path, year: str, issue: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{year}-{issue}] extrahuji bloky z {pdf_path.name} ...")
    blocks = extract_pdf(str(pdf_path))
    (out_dir / "blocks.json").write_text(
        json.dumps(blocks, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[{year}-{issue}] tahám obsah čísla (TOC) ...")
    toc = build_toc(str(pdf_path))
    (out_dir / "toc.json").write_text(
        json.dumps(toc, ensure_ascii=False, indent=2), encoding="utf-8")

    page_map = build_page_map(blocks)
    (out_dir / "page_map.json").write_text(
        json.dumps(page_map, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[{year}-{issue}] skládám články z bloků ...")
    articles = assemble_articles(blocks, toc, page_map["label_to_page"],
                                  year=year, issue=issue)
    (out_dir / "articles.json").write_text(
        json.dumps(articles, ensure_ascii=False, indent=2), encoding="utf-8")

    n_chunks = sum(len(a["chunks"]) for a in articles)
    print(f"[{year}-{issue}] hotovo: {len(articles)} článků, {n_chunks} chunků")
    return articles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="složka s PDF soubory ziva-RRRR-C.pdf")
    ap.add_argument("--output", required=True, help="výstupní složka")
    ap.add_argument("--skip-first", type=int, default=SKIP_FIRST_PAGES,
                     help="kolik stránek na začátku PDF čísla ignorovat při chunkování (obálka)")
    ap.add_argument("--skip-last", type=int, default=SKIP_LAST_PAGES,
                     help="kolik stránek na konci PDF čísla ignorovat při chunkování (zadní obálka)")
    args = ap.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    issues = find_issues(input_dir)
    print(f"Nalezeno {len(issues)} čísel ke zpracování.")

    all_articles = []
    failed = []
    for year, issue, pdf_path in issues:
        issue_out_dir = output_dir / f"{year}-{issue}"
        try:
            articles = process_one(pdf_path, year, issue, issue_out_dir)
            all_articles.extend(articles)
        except Exception as e:  # ať jedno rozbité číslo nezastaví celý archiv
            print(f"  [CHYBA] {pdf_path.name}: {e!r}")
            failed.append(str(pdf_path))

    corpus_path = output_dir / "ziva_corpus.json"
    corpus_path.write_text(json.dumps(all_articles, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    print(f"\nSkládám chunky pro embedding (skip_first={args.skip_first}, "
          f"skip_last={args.skip_last}) ...")
    all_chunks = []
    for article in all_articles:
        if not article.get("full_text", "").strip():
            continue
        all_chunks.extend(
            build_chunks_for_article(article, args.skip_first, args.skip_last))

    chunks_path = output_dir / "ziva_embedding_chunks.jsonl"
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print("\n=== Souhrn ===")
    print(f"Zpracováno úspěšně: {len(issues) - len(failed)} / {len(issues)} čísel")
    print(f"Celkem článků: {len(all_articles)}")
    print(f"Celkem chunků (bloky, articles.json): {sum(len(a['chunks']) for a in all_articles)}")
    print(f"Celkem chunků pro embedding (chunks_path níže): {len(all_chunks)}")
    print(f"Souhrnný korpus: {corpus_path}")
    print(f"Embedding chunky: {chunks_path}")
    if failed:
        print("Selhala tato čísla (podívej se na chybu výše a oprav ručně):")
        for f in failed:
            print("  -", f)


if __name__ == "__main__":
    main()
