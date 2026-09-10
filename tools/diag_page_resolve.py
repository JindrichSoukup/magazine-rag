"""
Diagnostika pro "could not resolve printed page" - vypíše detekované
úseky číslování (runs) z build_page_map.py a hledaný popisek, ať je vidět,
proč se do žádného úseku netrefil (chybí zcela, nebo jen mimo rozsah
nějakého jinak správně nalezeného úseku).

Použití:
    python diag_page_resolve.py output/2020-6/page_map.json CXXV
"""
import json
import sys

sys.path.insert(0, ".")
from magrag.build_page_map import label_to_int, int_to_label


def main():
    page_map_path, missing_label = sys.argv[1:3]
    page_map = json.loads(open(page_map_path, encoding="utf-8").read())

    print(f"Hledaný popisek: {missing_label!r}")
    scheme, value = label_to_int(missing_label)
    print(f"  -> schéma={scheme}, hodnota={value}\n")

    runs = page_map.get("runs", [])
    print(f"Detekováno {len(runs)} úseků číslování:")
    for r in runs:
        lo_label = int_to_label(r["scheme"], r["value_min"])
        hi_label = int_to_label(r["scheme"], r["value_max"])
        in_range = (r["scheme"] == scheme and r["value_min"] <= value <= r["value_max"])
        marker = " <-- hledaná hodnota by sem patřila" if in_range else ""
        print(f"    schéma={r['scheme']:7} rozsah {lo_label}-{hi_label} "
              f"(hodnoty {r['value_min']}-{r['value_max']}, offset={r['offset']}){marker}")

    print()
    is_covered = any(r["scheme"] == scheme and r["value_min"] <= value <= r["value_max"] for r in runs)
    if is_covered:
        print("Popisek SPADÁ do některého úseku - měl by být v label_to_page. "
              "Zkontrolujte, jestli label_to_page opravdu obsahuje přesně tenhle klíč "
              "(velká/malá písmena, mezery?).")
    else:
        print("Popisek NESPADÁ do žádného detekovaného úseku - na okolních "
              "stránkách se pravděpodobně vůbec nenašla patička/číslo stránky "
              "k přímé detekci, takže offset pro tenhle úsek chybí úplně.")

    label_to_page = page_map.get("label_to_page", {})
    print(f"\nJe {missing_label!r} přímo v label_to_page? "
          f"{missing_label in label_to_page}")
    # ukaž pár sousedních hodnot ve stejném schématu, co RESOLVED být můžou
    if scheme:
        neighbors = []
        for delta in range(-3, 4):
            lbl = int_to_label(scheme, value + delta)
            if lbl in label_to_page:
                neighbors.append((lbl, label_to_page[lbl]))
        print(f"Sousední popisky (+-3) nalezené v label_to_page: {neighbors}")


if __name__ == "__main__":
    main()
