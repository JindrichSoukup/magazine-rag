"""Zlatý test celé extrakční části: dá tatáž PDF pořád tentýž výstup?

Problém, který tenhle test řeší: pipeline je jedna dlouhá heuristika a
skoro každá úprava někde jinde v ní může tiše posunout výsledek o pár
bloků. Ruční kontrola ("vypadá to pořád stejně") tohle nezachytí.

Problém, který tenhle test má: vstupem je autorsky chráněné PDF a jeho
výstupem plný text článků. Ani jedno nesmí do repozitáře.

Řešení: fixture neobsahuje obsah, ale **otisk** - počty článků a bloků
plus SHA-256 serializovaného výstupu každé fáze. Na změnu to reaguje
stejně citlivě jako porovnání obsahu, ale nezveřejňuje z časopisu ani
písmeno. Kdo PDF má, test si pustí; kdo ne, tomu se přeskočí.

Vygenerování/aktualizace fixture (dělejte to vědomě, ne "ať to projde" -
změna otisku znamená, že se změnil výstup pipeline):

    python -m tests.test_pipeline_golden --update cesta/k/cislu.pdf ziva
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from magrag.assign_articles import assemble_articles
from magrag.build_page_map import build_page_map
from magrag.create_toc import build_toc
from magrag.extract_blocks import extract_pdf
from magrag.profiles import get

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "golden.json"

# Kde se hledá zdrojové PDF. Proměnná prostředí má přednost, ať jde test
# pustit i nad jiným číslem, než jaké má autor náhodou ve složce data/.
ENV_VAR = "MAGRAG_GOLDEN_PDF"
DEFAULT_PDF = Path("data") / "ziva-2014-6.pdf"


def digest(obj) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def run_stages(pdf_path: str, profile):
    """Extrakční část pipeline až po hotové články."""
    blocks = extract_pdf(pdf_path, profile)
    toc = build_toc(pdf_path, profile)
    page_map = build_page_map(blocks, profile)
    articles = assemble_articles(blocks, toc, page_map["label_to_page"], profile,
                                 year="2014", issue="6")
    return blocks, toc, page_map, articles


def fingerprint(pdf_path: str, profile) -> dict:
    blocks, toc, page_map, articles = run_stages(pdf_path, profile)
    return {
        "profile": profile.key,
        "pdf_name": Path(pdf_path).name,
        "counts": {
            "blocks": len(blocks),
            "toc_entries": len(toc),
            "resolved_labels": len(page_map["label_to_page"]),
            "articles": len(articles),
            "article_chunks": sum(len(a["chunks"]) for a in articles),
        },
        "sha256": {
            "blocks": digest(blocks),
            "toc": digest(toc),
            "page_map": digest(page_map),
            "articles": digest(articles),
        },
    }


def find_pdf():
    from_env = os.environ.get(ENV_VAR)
    if from_env and Path(from_env).is_file():
        return from_env
    if DEFAULT_PDF.is_file():
        return str(DEFAULT_PDF)
    return None


@pytest.fixture(scope="module")
def golden():
    if not FIXTURE_PATH.is_file():
        pytest.skip(f"chybí {FIXTURE_PATH} - vygenerujte ho, viz docstring")
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def actual(golden):
    pdf = find_pdf()
    if pdf is None:
        pytest.skip(
            f"zdrojové PDF není k dispozici (nastavte {ENV_VAR}, nebo dejte "
            f"soubor do {DEFAULT_PDF}); do repozitáře nepatří kvůli právům")
    if Path(pdf).name != golden["pdf_name"]:
        pytest.skip(f"fixture je pro {golden['pdf_name']}, ne pro {Path(pdf).name}")
    return fingerprint(pdf, get(golden["profile"]))


@pytest.mark.parametrize("stage", ["blocks", "toc", "page_map", "articles"])
def test_stage_output_is_unchanged(actual, golden, stage):
    assert actual["sha256"][stage] == golden["sha256"][stage], (
        f"výstup fáze {stage!r} se změnil oproti zlatému otisku. Pokud je to "
        f"záměr, aktualizujte fixture a v commitu popište, co se změnilo a proč."
    )


def test_counts_are_unchanged(actual, golden):
    assert actual["counts"] == golden["counts"]


def test_running_twice_gives_the_same_result(golden):
    """Determinismus není samozřejmost - stačí, aby se někde v pipeline
    iterovalo přes množinu, a pořadí bloků se začne mezi běhy měnit."""
    pdf = find_pdf()
    if pdf is None:
        pytest.skip("zdrojové PDF není k dispozici")
    profile = get(golden["profile"])
    assert fingerprint(pdf, profile) == fingerprint(pdf, profile)


def _update(pdf_path: str, profile_key: str):
    data = fingerprint(pdf_path, get(profile_key))
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"zapsáno {FIXTURE_PATH}")
    print(json.dumps(data["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from magrag.console import setup_console

    setup_console()
    if len(sys.argv) >= 3 and sys.argv[1] == "--update":
        _update(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "ziva")
    else:
        print(__doc__)
