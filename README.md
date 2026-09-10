# magrag — a magazine's PDF archive → corpus → RAG

A pipeline that turns a magazine's back-catalogue of PDFs into a
structured corpus of articles with metadata and citable page numbers,
embeds it, and answers questions over it with references to the source.

It grew up on the archive of **Živa**, a Czech natural-history monthly
(75 issues, 2,941 articles, ~35,000 embedding chunks), and is written so
that moving it to another magazine means writing one file — see [Source
profile](#source-profile).

> **Rights.** No magazine content lives in this repository — no PDFs, no
> text extracted from them. Živa is under copyright (Academia / Czech
> Academy of Sciences), so the archive cannot be published; the code can.
> For a public demo there is a profile for
> [The MagPi](https://magpi.raspberrypi.com/issues), which is published
> under CC BY-NC-SA.

---

## Two phases, and the second one is the point

The project happened in two separate stints, and each taught something
different.

**First by hand, one layer at a time**, to find out what a system like
this actually decides — including what it decides silently for you, by
default, when you are not looking. The output of that phase is not only
scripts but two notes that make sense without the rest of the repository:

- [**A layer-by-layer map of RAG decisions**](docs/rag-decision-checklist.md)
  — ten layers of a RAG system and, in each, the questions that get
  settled. It deliberately describes *what is decided*, not how to
  decide it, and it applies to any RAG project, not just this one.
- [**RAG: build vs. buy**](docs/rag-build-vs-buy.md) — what hands-on
  building tells you when somebody has to run a thing like this inside a
  larger organisation. The main conclusion: build vs. buy is not one
  question, it is a question **per layer**, and the most expensive work
  is the work vendor demos never show.

**Then generalisation**: turn something tuned to one magazine into a
system with swappable source profiles, and verify it by moving it onto a
second, unrelated magazine. Porting is the only honest test of
generalisation, and so it proved — five bugs surfaced that the first
magazine could never have shown, and **not one of them crashed**. The
pipeline finished every time and printed a contented summary.

How both phases went, including what was decided wrongly, is in the
[project log](docs/project-log.md).

---

## Why this is not `PyPDFLoader` + `RecursiveCharacterTextSplitter`

Because that does not work on magazine typesetting. The interesting part
of this project is neither the embedding nor the vector search — those
are an afternoon. The interesting part is the road from "a PDF" to "an
article with its author, volume and page number":

- Text does not run by page but by **article**, and an article spills
  across pages. Where it ends is knowable only from the contents page.
- The printed page number is **not** the PDF page number, and the
  difference is not constant: within one issue the numbering routinely
  alternates Arabic (main articles) → Roman (supplement) → Arabic again,
  each with a different offset.
- One physical page carries **the end of one article and the start of
  the next**. Cutting by block order is not enough — the last column
  often runs independently of the rest of the page.
- Justified text breaks words with a hyphen. Without gluing them back,
  the corpus receives `opaková-` and `ní` as two separate words.
- A figure caption mid-column **cuts a sentence in half** when the text
  is assembled naively from top to bottom.

Every one of those is a specific bug found only on real data. The course
of it is written up in the [project log](docs/project-log.md), including
what was decided wrongly and why.

---

## How it works

```
 an issue's PDF
    │
    ├─► extract_blocks    text blocks with font, size, position and type
    │                     (title/heading/other/body/caption/annotation)
    ├─► create_toc        the contents: article → author → printed page
    ├─► build_page_map    printed page → PDF page (numbering runs)
    ├─► assign_articles   blocks → articles, shared boundary pages included
    │                     + quality_flags wherever the pipeline is unsure
    ├─► build_chunks      articles → ~1200-character chunks with a header
    ├─► build_embeddings  chunks → vectors (local model, with checkpoints)
    ├─► build_chroma      vectors → Chroma
    ├─► assemble_context  query → sources with citations (window + promote)
    └─► answer            sources + question → an LLM answer with references
```

`extract_blocks` is the only step that knows what a specific magazine's
typesetting looks like. Everything after it works with the structure
`{page, type, font, bbox, text}` alone and ports unchanged.

---

## Quick start

```bash
git clone <url> && cd magrag
python -m venv .venv && . .venv/Scripts/activate   # Linux/macOS: . .venv/bin/activate
pip install -e ".[dev]"          # PDF extraction only
pip install -r requirements.txt  # the whole pipeline, versions pinned
```

Put the PDFs in one directory and run the whole archive at once:

```bash
python -m magrag.run_all --profile ziva --input ./pdf --output ./output
```

That produces `output/<year>-<issue>/{blocks,toc,page_map,articles}.json`
for each issue, useful when debugging, plus a combined
`output/corpus.json` and `output/chunks.jsonl`.

The rest of the way to answers:

```bash
python -m magrag.build_embeddings --input output/chunks.jsonl \
    --output-dir output --model intfloat/multilingual-e5-base

python -m tools.check_embeddings --chunks output/chunks.jsonl \
    --vectors output/ziva_embeddings__intfloat__multilingual-e5-base.npy \
    --ids     output/ziva_embeddings__intfloat__multilingual-e5-base_ids.json

python -m magrag.build_chroma --chunks output/chunks.jsonl \
    --vectors output/ziva_embeddings__intfloat__multilingual-e5-base.npy \
    --ids     output/ziva_embeddings__intfloat__multilingual-e5-base_ids.json \
    --db-dir ./chroma_db --collection ziva --overwrite

python -m magrag.answer --db-dir ./chroma_db --collection ziva \
    --model intfloat/multilingual-e5-base \
    --chunks output/chunks.jsonl --corpus output/corpus.json
```

Without `--query`, `answer` runs interactively. With `--dry-run` it
prints the finished prompt and sends nothing to the API — handy for
tuning retrieval for free.

---

## Source profile

Everything that makes one magazine differ from another lives in a single
`SourceProfile` instead of being scattered across five scripts: font
names and point sizes for block classification, the shape of the running
footer, which page carries the contents, the filename pattern, the
language of the metadata header and the system prompt.

| Profile | Typography | Footer | Note |
|---|---|---|---|
| `ziva` | hand-written, absolute sizes | by magazine name | the reference, tuned on 75 issues |
| `magpi` | adaptive | by position on the page | open licence, suitable for a demo |
| `adaptive` | adaptive | — | the starting point for an unknown magazine |

**Adaptive classification** is the answer to nobody knowing a new
magazine's font names. Instead of absolute values the pipeline counts how
many characters are set in each combination of font family and size. The
most voluminous combination is by definition the body text, and every
rule is then relative to it ("a title is 1.7x larger than the text").
Weighting by characters rather than by block count matters: a page
carries many titles but little of their text.

What portability costs can be measured. The same issue of Živa processed
by both profiles, where the adaptive one knows nothing about Živa — not
the font name, not where the contents are, not what stands in the footer.

| | hand-written `ziva` | adaptive |
|---|---|---|
| articles found | 37 | 37 |
| title contained in the adaptive one | — | 36 of 37 |
| byte-identical article text | — | 20 of 37 |
| articles with `quality_flags` | 3 | 4 |

The adaptive profile finds the same articles. Its titles are longer,
though: they include the author line, because a generic profile cannot
know that the contents list authors separately and set them apart by
colour. The differences in article text are paragraph boundaries, not
lost content.

### Verified on a second magazine

The `magpi` profile is tested on three real issues downloaded from
`magpi.raspberrypi.com/issues` (150, 152, 155; born-digital PDF, 132
pages). Without a single hand-entered value about the typesetting, the
pipeline pulls **85 articles and 656 chunks** out of them, with no
duplicates and no invented authors.

It did not come free. The road to that number exposed five bugs the first
magazine could never have shown, and **not one of them crashed** — the
pipeline finished every time and printed a contented summary:

| Finding | Why Živa never showed it |
|---|---|
| the contents say `032`, the footer `32` | Živa does not zero-pad page numbers |
| titles carry the control character `U+0007` | a decorative bullet set in a symbol font |
| the entry number is glued to a tab and a bullet | likewise |
| the contents are spread across three pages | Živa always has them on one |
| the contents page carries three kinds of number | Živa has no decorative callouts |

The last of those is the most interesting. Beside the real page numbers,
The MagPi's contents carry decorative callouts (a large white number with
a short caption) and the number of the contents page itself in the
footer. Hard-coding which is which does not work: The MagPi redesigned
between issues 150 and 152, changing fonts, sizes and the number format.
It is solved by the same reasoning as block classification, one storey
up — looking for the **most frequent "number style + style of the title
right after it" pair**, because the contents are a list and that pair
repeats for every entry.

### Adding a new magazine

```bash
python -m tools.inspect_fonts path/to/issue.pdf
```

It prints a typography histogram with sample text, a ready-made
suggestion of rules to paste into a profile, and the candidates for the
running footer. On the sample issue of Živa it produces exactly what was
originally derived by hand (`MeliorCE` at 9pt as body text, `živa 6/2014`
and `ziva.avcr.cz` as the footer).

Then copy `magrag/profiles/magpi.py`, edit it, and register it in
`magrag/profiles/__init__.py`. When hand calibration does not pay off,
because the magazine redesigned several times over the archive, leave
`adaptive=True`.

---

## Reproducibility

- **Pinned versions** in `requirements.txt`. PyMuPDF changes how it
  splits a page into blocks between versions, and that is the input to
  everything else.
- **A golden test** (`tests/test_pipeline_golden.py`) compares a SHA-256
  fingerprint of each stage's output against a pinned value. The fixture
  holds no magazine content, only fingerprints and counts — it reacts to
  a change as sensitively as comparing the text would, while publishing
  not one letter. Without the source PDF it skips itself:

  ```bash
  pytest -q                                       # 105 tests
  MAGRAG_GOLDEN_PDF=path/to/issue.pdf pytest -q   # including the golden test
  ```

- **`quality_flags`** on every article admit where the pipeline had to
  fall back on a guess (`boundary_page_unverified`, `merged_fallback`,
  ...). For a summary across the archive:
  `python -m tools.summarize_quality_flags output/corpus.json`. On Živa,
  8.8% of articles carry at least one flag and 70% of those belong to a
  single well-understood category, the administrative back matter of an
  issue.

Generated data (`output/`, `chroma_db/`, the vectors) and source PDFs are
in `.gitignore`. Živa's corpus is 127 MB, the chunks 95 MB and the
vectors 120 MB — over GitHub's per-file limit, and above all they do not
belong there for rights reasons.

---

## Layout

```
magrag/            the pipeline (one module per stage)
  profiles/        source profiles + system prompts
  typography.py    block classification, hand-written and adaptive
  console.py       UTF-8 on stdout (or it dies on the first message)
tools/             diagnostics and calibration, not part of a production run
tests/             105 tests; the golden test skips without a PDF
docs/              the project log and notes on RAG decisions
```

## Documentation

- [Project log](docs/project-log.md) — what was built, what turned up
  and why it was decided that way. The most interesting read in the
  repository.
- [RAG: build vs. buy](docs/rag-build-vs-buy.md) — what to take from
  this if you are running something similar in a larger organisation.
- [A layer-by-layer map of RAG decisions](docs/rag-decision-checklist.md)
  — what gets decided in each layer of a RAG system, explicitly or
  silently by default.

## Licence

Code: MIT (see [LICENSE](LICENSE)). The licence of this project does
**not** cover the content of the magazines it processes — that is
governed by the publishers' rights.
