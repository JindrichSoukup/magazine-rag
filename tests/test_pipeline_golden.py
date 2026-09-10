"""A golden test of the whole extraction stack: does the same PDF still
produce the same output?

The problem this test solves: the pipeline is one long heuristic, and
almost any change anywhere in it can quietly shift the result by a few
blocks. Checking by hand ("it still looks the same") does not catch that.

The problem this test has: the input is a copyrighted PDF and its output
is the full text of articles. Neither may enter the repository.

The solution: the fixture holds no content, only a **fingerprint** -
counts of articles and blocks plus a SHA-256 of each stage's serialised
output. It reacts to a change just as sensitively as comparing the
content would, while publishing not one letter of the magazine. Whoever
has the PDF can run it; for everyone else it skips.

Generating or updating the fixture - do it deliberately, not to "make it
pass": a changed fingerprint means the pipeline's output changed.

    python -m tests.test_pipeline_golden --update path/to/issue.pdf ziva
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import pytest

from magazine_rag.assign_articles import assemble_articles
from magazine_rag.build_page_map import build_page_map
from magazine_rag.create_toc import build_toc
from magazine_rag.extract_blocks import extract_pdf
from magazine_rag.profiles import get

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "golden.json"

# Where the source PDF is looked for. The environment variable wins, so
# the test can also be run over an issue other than whichever one the
# author happens to have in data/.
ENV_VAR = "MAGAZINE_RAG_GOLDEN_PDF"
DEFAULT_PDF = Path("data") / "ziva-2014-6.pdf"


def digest(obj) -> str:
    blob = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def run_stages(pdf_path: str, profile):
    """The extraction part of the pipeline, up to finished articles."""
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
        pytest.skip(f"{FIXTURE_PATH} is missing - generate it, see the docstring")
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def actual(golden):
    pdf = find_pdf()
    if pdf is None:
        pytest.skip(
            f"the source PDF is not available (set {ENV_VAR}, or put the file "
            f"in {DEFAULT_PDF}); it does not belong in the repository for "
            f"rights reasons")
    if Path(pdf).name != golden["pdf_name"]:
        pytest.skip(f"the fixture is for {golden['pdf_name']}, "
                    f"not {Path(pdf).name}")
    return fingerprint(pdf, get(golden["profile"]))


@pytest.mark.parametrize("stage", ["blocks", "toc", "page_map", "articles"])
def test_stage_output_is_unchanged(actual, golden, stage):
    assert actual["sha256"][stage] == golden["sha256"][stage], (
        f"the output of stage {stage!r} changed against the golden "
        f"fingerprint. If that is intended, update the fixture and describe "
        f"in the commit what changed and why."
    )


def test_counts_are_unchanged(actual, golden):
    assert actual["counts"] == golden["counts"]


def test_running_twice_gives_the_same_result(golden):
    """Determinism is not a given - it takes only one iteration over a
    set somewhere in the pipeline for block order to start varying
    between runs."""
    pdf = find_pdf()
    if pdf is None:
        pytest.skip("the source PDF is not available")
    profile = get(golden["profile"])
    assert fingerprint(pdf, profile) == fingerprint(pdf, profile)


def _update(pdf_path: str, profile_key: str):
    data = fingerprint(pdf_path, get(profile_key))
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"wrote {FIXTURE_PATH}")
    print(json.dumps(data["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    from magazine_rag.console import setup_console

    setup_console()
    if len(sys.argv) >= 3 and sys.argv[1] == "--update":
        _update(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "ziva")
    else:
        print(__doc__)
