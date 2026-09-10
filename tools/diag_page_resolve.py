"""Diagnostics for "could not resolve printed page".

Prints the numbering runs build_page_map.py detected alongside the label
being looked for, so it is visible why it fell into no run at all -
whether it is missing entirely, or merely outside the range of an
otherwise correctly found run.

Usage:
    python -m tools.diag_page_resolve output/2020-6/page_map.json CXXV
"""
import json
import sys

from magrag.build_page_map import int_to_label, label_to_int
from magrag.console import setup_console


def main():
    setup_console()
    page_map_path, missing_label = sys.argv[1:3]
    page_map = json.loads(open(page_map_path, encoding="utf-8").read())

    print(f"Label being looked for: {missing_label!r}")
    scheme, value = label_to_int(missing_label)
    print(f"  -> scheme={scheme}, value={value}\n")

    runs = page_map.get("runs", [])
    print(f"{len(runs)} numbering runs detected:")
    for r in runs:
        lo_label = int_to_label(r["scheme"], r["value_min"])
        hi_label = int_to_label(r["scheme"], r["value_max"])
        in_range = (r["scheme"] == scheme
                    and r["value_min"] <= value <= r["value_max"])
        marker = " <-- the wanted value would belong here" if in_range else ""
        print(f"    scheme={r['scheme']:7} range {lo_label}-{hi_label} "
              f"(values {r['value_min']}-{r['value_max']}, "
              f"offset={r['offset']}){marker}")

    print()
    is_covered = any(r["scheme"] == scheme
                     and r["value_min"] <= value <= r["value_max"] for r in runs)
    if is_covered:
        print("The label DOES fall inside a run, so it should be in "
              "label_to_page. Check whether label_to_page really contains "
              "exactly this key - case, spaces?")
    else:
        print("The label falls into NO detected run. The surrounding pages "
              "probably had no footer or page number to detect directly, so "
              "the offset for this run is missing altogether.")

    label_to_page = page_map.get("label_to_page", {})
    print(f"\nIs {missing_label!r} directly in label_to_page? "
          f"{missing_label in label_to_page}")
    # Show a few neighbouring values in the same scheme that DID resolve.
    if scheme:
        neighbors = []
        for delta in range(-3, 4):
            lbl = int_to_label(scheme, value + delta)
            if lbl in label_to_page:
                neighbors.append((lbl, label_to_page[lbl]))
        print(f"Neighbouring labels (+-3) found in label_to_page: {neighbors}")


if __name__ == "__main__":
    main()
