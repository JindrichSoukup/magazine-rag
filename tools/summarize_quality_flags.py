"""Summarise quality_flags across the whole corpus.

How many articles were assigned with some uncertainty, what kinds of
uncertainty, broken down by issue and by flag type.

Usage:
    python -m tools.summarize_quality_flags output/corpus.json
    python -m tools.summarize_quality_flags output/corpus.json --list
"""
import argparse
import json
from collections import Counter, defaultdict

from magazine_rag.console import setup_console


def main():
    setup_console()
    ap = argparse.ArgumentParser(
        description="summarise quality_flags across the corpus")
    ap.add_argument("corpus", help="corpus.json")
    ap.add_argument("--list", action="store_true",
                    help="also print every flagged article individually")
    ap.add_argument("--unique-only", action="store_true",
                    help="with --list, print ONLY articles whose title does "
                         "not repeat among the flagged ones - so that "
                         "something genuinely unusual is easy to separate "
                         "from the familiar recurring administrative "
                         "category (news, contact addresses...)")
    ap.add_argument("--top", type=int, default=15,
                    help="how many worst-affected issues to list (default 15)")
    args = ap.parse_args()

    articles = json.loads(open(args.corpus, encoding="utf-8").read())
    total = len(articles)
    if total == 0:
        print("Empty corpus.")
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
    print(f"Articles in the corpus: {total}")
    print(f"Articles with at least one quality_flag: {n_flagged} "
          f"({100 * n_flagged / total:.1f}%)\n")

    print("=== By flag type (an article may carry more than one) ===")
    if not flag_counts:
        print("  (none - the whole corpus is unflagged)")
    for flag, count in flag_counts.most_common():
        print(f"  {flag:30} {count:6}   ({100 * count / total:.1f}% of all articles)")
    print()

    print("=== How much content is affected ===")
    total_chunks = sum(len(a["chunks"]) for a in articles)
    flagged_chunks = sum(len(a["chunks"]) for a in flagged_articles)
    empty_articles = sum(1 for a in articles if not a["chunks"])
    print(f"  Chunks in the corpus: {total_chunks}")
    if total_chunks:
        print(f"  Chunks in flagged articles: {flagged_chunks} "
              f"({100 * flagged_chunks / total_chunks:.1f}%)")
    print(f"  Entirely empty articles (0 chunks): {empty_articles}")
    print()

    all_issues = sorted(set(f"{a['year']}-{a['issue']}" for a in articles))
    issues_with_flags = set(per_issue_flag_counts.keys())
    print("=== Issues ===")
    print(f"  Issues in total: {len(all_issues)}")
    print(f"  With at least one flagged article: {len(issues_with_flags)}")
    print(f"  Entirely clean (not one flag): "
          f"{len(all_issues) - len(issues_with_flags)}")
    print()

    print(f"=== Issues with the most flagged articles (top {args.top}) ===")
    issue_totals = Counter({k: sum(c.values())
                            for k, c in per_issue_flag_counts.items()})
    for issue_key, n in issue_totals.most_common(args.top):
        detail = ", ".join(f"{f}={c}"
                           for f, c in per_issue_flag_counts[issue_key].most_common())
        print(f"  {issue_key:10} {n:3}  ({detail})")
    print()

    title_counts = Counter(a["title"] for a in flagged_articles)

    if args.list:
        to_show = [a for a in flagged_articles if not args.unique_only
                   or title_counts[a["title"]] == 1]
        label = "unique " if args.unique_only else ""
        print(f"=== Detailed list of {label}flagged articles ({len(to_show)}) ===")
        for a in sorted(to_show, key=lambda a: (a["year"], a["issue"])):
            extra = ""
            if a.get("absorbed_titles"):
                extra += f" | absorbed={a['absorbed_titles']}"
            if a.get("original_titles"):
                extra += f" | original={a['original_titles']}"
            print(f"  {a['year']}-{a['issue']:>2} | {a['title'][:55]!r:57} | "
                  f"{a['quality_flags']}{extra}")
        print()

    # The commonest titles among flagged articles, so it is visible
    # whether this is still the same familiar category that recurs every
    # issue (news, contact addresses...) or whether something unique is
    # hiding in there.
    print("=== Commonest titles among flagged articles (recurring across issues) ===")
    for title, count in title_counts.most_common(20):
        if count > 1:
            print(f"  {count:3}x  {title[:65]!r}")
    unique_titles = [t for t, c in title_counts.items() if c == 1]
    print(f"\n  Unique (appearing once): {len(unique_titles)} of "
          f"{len(title_counts)} titles")


if __name__ == "__main__":
    main()
