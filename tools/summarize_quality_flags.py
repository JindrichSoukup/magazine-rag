"""
Shrnutí quality_flags napříč celým korpusem (ziva_corpus.json) - kolik
článků bylo přiřazeno s nejistotou, jaké nejistoty se týkají, rozpad podle
čísla i podle typu vlajky.

Použití:
    python summarize_quality_flags.py output/ziva_corpus.json
    python summarize_quality_flags.py output/ziva_corpus.json --list
"""
import argparse
import json
from collections import Counter, defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", help="ziva_corpus.json")
    ap.add_argument("--list", action="store_true",
                     help="vypiš i každý jednotlivý flagged článek zvlášť")
    ap.add_argument("--unique-only", action="store_true",
                     help="u --list vypiš JEN články, jejichž titulek se mezi "
                          "flagged články neopakuje - ať jde snadno oddělit "
                          "od té známé opakující se admin kategorie (Aktuality, "
                          "Kontaktní adresy...) něco jedinečného, co stojí za "
                          "bližší pohled")
    ap.add_argument("--top", type=int, default=15,
                     help="kolik nejvíc postižených čísel vypsat (výchozí 15)")
    args = ap.parse_args()

    articles = json.loads(open(args.corpus, encoding="utf-8").read())
    total = len(articles)
    if total == 0:
        print("Prázdný korpus.")
        return

    flag_counts = Counter()
    per_issue_flag_counts = defaultdict(Counter)
    flagged_articles = []

    for a in articles:
        flags = a.get("quality_flags", [])
        issue_key = f"{a['year']}-{a['issue']}"
        if flags:
            flagged_articles.append(a)
            for f in flags:
                flag_counts[f] += 1
                per_issue_flag_counts[issue_key][f] += 1

    n_flagged = len(flagged_articles)
    print(f"Celkem článků v korpusu: {total}")
    print(f"Článků s aspoň jedním quality_flag: {n_flagged} ({100 * n_flagged / total:.1f} %)\n")

    print("=== Rozpad podle typu vlajky (článek může mít víc než jednu) ===")
    if not flag_counts:
        print("  (žádné - celý korpus je bez vlajek)")
    for flag, count in flag_counts.most_common():
        print(f"  {flag:30} {count:6}   ({100 * count / total:.1f} % všech článků)")
    print()

    print("=== Kolik obsahu (chunků) je v ohrožení ===")
    total_chunks = sum(len(a["chunks"]) for a in articles)
    flagged_chunks = sum(len(a["chunks"]) for a in flagged_articles)
    empty_articles = sum(1 for a in articles if not a["chunks"])
    print(f"  Chunků celkem v korpusu: {total_chunks}")
    if total_chunks:
        print(f"  Chunků ve flagged článcích: {flagged_chunks} "
              f"({100 * flagged_chunks / total_chunks:.1f} %)")
    print(f"  Úplně prázdných článků (0 chunků): {empty_articles}")
    print()

    all_issues = sorted(set(f"{a['year']}-{a['issue']}" for a in articles))
    issues_with_flags = set(per_issue_flag_counts.keys())
    print(f"=== Čísla ===")
    print(f"  Čísel celkem: {len(all_issues)}")
    print(f"  S aspoň 1 flagged článkem: {len(issues_with_flags)}")
    print(f"  Úplně čistých (bez jediné vlajky): {len(all_issues) - len(issues_with_flags)}")
    print()

    print(f"=== Čísla s nejvíc flagged články (top {args.top}) ===")
    issue_totals = Counter({k: sum(c.values()) for k, c in per_issue_flag_counts.items()})
    for issue_key, n in issue_totals.most_common(args.top):
        detail = ", ".join(f"{f}={c}" for f, c in per_issue_flag_counts[issue_key].most_common())
        print(f"  {issue_key:10} {n:3}  ({detail})")
    print()

    title_counts = Counter(a["title"] for a in flagged_articles)

    if args.list:
        to_show = [a for a in flagged_articles if not args.unique_only
                   or title_counts[a["title"]] == 1]
        label = "jedinečných" if args.unique_only else ""
        print(f"=== Detailní seznam {label} flagged článků ({len(to_show)}) ===")
        for a in sorted(to_show, key=lambda a: (a["year"], a["issue"])):
            extra = ""
            if a.get("absorbed_titles"):
                extra += f" | absorbed={a['absorbed_titles']}"
            if a.get("original_titles"):
                extra += f" | original={a['original_titles']}"
            print(f"  {a['year']}-{a['issue']:>2} | {a['title'][:55]!r:57} | "
                  f"{a['quality_flags']}{extra}")
        print()

    # Nejčastější titulky mezi flagged články - ať jde poznat, jestli jde
    # pořád o tu stejnou známou kategorii (Aktuality, Kontaktní adresy...),
    # co se opakuje každé číslo, nebo se tam schovává něco jedinečného
    print("=== Nejčastější titulky mezi flagged články (opakující se napříč čísly) ===")
    for title, count in title_counts.most_common(20):
        if count > 1:
            print(f"  {count:3}x  {title[:65]!r}")
    unique_titles = [t for t, c in title_counts.items() if c == 1]
    print(f"\n  Jedinečných (jen 1x) titulků: {len(unique_titles)} z {len(title_counts)} celkem")


if __name__ == "__main__":
    main()
