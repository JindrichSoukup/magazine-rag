# Živa RAG pipeline — project log

Digitising the archive of the Czech natural-history magazine **Živa**
(Nakladatelství Academia) into a structured corpus suitable for RAG
(retrieval-augmented generation): from raw PDFs through article parsing,
chunking, embeddings and a vector store to a retrieval strategy for an
LLM.

Scope: 75 issues, 2,941 articles, ~60,500 block chunks, ~35,000
embedding chunks.

---

## Phase 1 — Extracting blocks from the PDF (`extract_blocks.py`)

**Goal:** get structured text blocks out of the PDF (fitz/PyMuPDF), each
with its page, type, font and position.

- Reconstructed the missing extraction script from the surviving outputs
  (`blocks.json`, `toc.json`) and a sample PDF.
- Heuristic classification of block type
  (`title`/`heading`/`other`/`body`/`caption`/`annotation`) by font and
  size, tuned specifically to Živa's typesetting (MeliorCE = the serif
  body text and titles, HelveticaCE/Arial = the sans captions and
  footers).
- **Fix — words broken across a hyphen:** justified InDesign text breaks
  words at the end of a line with a hyphen ("dlouhodo- bé"). Added
  `smart_join()` with detection (a letter, then a trailing hyphen, then
  a continuation in lowercase), applied both when assembling lines *and*
  when merging adjacent blocks on the same page.
- **Telling `annotation` from `caption`:** panel labels and scale bars
  stuck onto a figure (`a`, `b`, `1 cm`, a run of numbers along a chart
  axis) carry no content — a new `annotation` type, excluded from the
  final text. It also detects the legends of colour scales on maps and
  charts.
- **Catching captions set in the body typeface:** Živa's longer figure
  captions do not always use the small sans font; they are recognised by
  the pattern "figure number followed by a capital" at the start of the
  block.
- **Widened `heading` detection:** the smaller subheadings in the back
  matter use size 13, bold and non-bold alike, outside the original
  range of 14 to 18.
- **A real bug — stylistically unrelated text merged:** PyMuPDF itself
  can join visually close but unrelated text into one raw block,
  typically the end of one review plus the heading "Kontaktní adresy
  autorů" right below it. The consequence: the longer, unimportant
  fragment "won" the dominant style and the heading vanished under a
  `body` classification. **Fix:** `split_lines_by_style()`, splitting a
  raw block by line wherever the font size changes markedly (>3pt). It
  took two iterations: the first version also compared weight, which
  needlessly tore apart figure captions with a bold number inside;
  corrected to compare size only.

**Decision:** cover pages (the first and last two) are **not** ignored
during extraction (see Phase 3). Extraction stays complete and
independent of what the later phases do with it.

---

## Phase 2 — The contents page and page mapping (`create_toc.py`, `build_page_map.py`)

- Reconstructed and generalised the contents parser (a bold MeliorCE
  number = the printed page; the colour of the first span separates the
  title from the author).
- **Abbreviated page ranges:** `"XXXI–II"` (= XXXI to XXXII) is not a
  clean Roman numeral, so `RANGE_RE` was added and only the starting
  value is extracted.
- **Combined contents entries:** where several items share one page
  number, separated by a semicolon, `create_toc.py` splits them into
  separate records with the same page (resolution in Phase 4).
- **Mapping printed page to PDF page:** the numbering is not one
  continuous sequence — within an issue it routinely alternates Arabic
  (main articles) → Roman (supplement) → Arabic again, with a different
  offset each time. `build_page_map.py` detects these contiguous runs
  itself and fills the gaps (a missing footer on an edge page) within a
  run, with a safe cap of five pages so that it stays a genuine
  correction rather than guessing into a large gap.
- **Case-insensitive Roman numerals:** a typesetting slip (`"CXLVIiI"`
  with a lowercase `i`) — the regex was case-sensitive, fixed in both
  independent places (`build_page_map.py`, `create_toc.py`), with
  normalisation to upper case.

**Accepted known limitations (errors in the source, not the pipeline):**
a few printed page numbers are demonstrably wrong in the source PDF (an
editorial copy-paste from the previous issue, or bad alignment). Such
records are skipped safely with a warning, rather than guessing what the
editor "meant".

---

## Phase 3 — Assigning blocks to articles (`assign_articles.py`)

The most complex and most iterated part of the pipeline.

- The basic model: an article owns the pages from its own start up to,
  but not including, the start of the next article.
- **A real bug — a shared boundary page:** a page can physically contain
  the end of one article *and* the start of another. Found while
  investigating a mismatch: content about mycorrhiza ended up in a
  completely different article about epigenetics because they shared a
  page. The last of three columns often runs independently of the rest
  of the page, so "everything before the title in reading order" is not
  enough. **Fix:** `find_split_y()`, a cut by **Y coordinate** across
  all columns rather than by position in the list, using the position of
  the block holding the new article's author or title.
- **Groups of records sharing a page** (the result of the combined TOC
  records from Phase 2): generalised to "N items share one page", not
  just two. Splitting within a group is primarily done by **counting
  headings** (more robust than text: reviews tend to have a shortened
  title in the contents but a completely different heading in the body,
  with the book author's full name), falling back to a **fuzzy text
  match** (exact match → substring after stripping common prefixes →
  longest common substring of at least 15 characters, which also handles
  a contents entry legitimately covering two headings in the text, e.g.
  one feature presented from two authors' viewpoints).
- **A safety net against silent data loss:** when not one match is found
  in a whole group, no split happens at all — the group merges back into
  one record with a joined title, rather than the content disappearing
  without a trace. A real bug, fixed after an audit of issue 2023/1.
- **A real bug — `find_split_y` never succeeded on most back-matter
  articles:** it looked only for `type=="title"`, which is reserved for
  main articles, not for `"heading"`, which is what reviews, obituaries
  and shorter formats use, and it used an exact text match rather than a
  fuzzy one. After the fix, the number of "unverified" boundary pages in
  the test sample fell from 18 to 3 articles out of 37 — a real
  improvement in accuracy across the archive, not cosmetics.
- **`quality_flags`** — a new field on every article, admitting
  explicitly where the pipeline had to fall back on a guess:
  `boundary_page_unverified`, `heading_not_found_empty`,
  `absorbed_unmatched_siblings` (plus `absorbed_titles`),
  `merged_fallback` (plus `original_titles`).

**Result of the quality_flags audit across the archive** (2,941
articles): 8.8% of articles carry at least one flag, but 70% of those
belong to a single well-understood category, the administrative back
matter of an issue (news, contact addresses, the biologist's calendar,
the editorial). Of the **unique** cases, almost all turned out to be the
same category with a specific name; the one genuinely different finding,
two merged reviews, turned out to be a real editorial error in the
source (two items printed in swapped order), not a pipeline error — and
the content stayed safely preserved, merely under a different title.

---

## Phase 4 — Chunking for RAG (`build_chunks.py`)

- **Decision:** chunk from `full_text`, the continuous article text,
  rather than from individual blocks. Blocks are far too uneven in
  length, from one character to hundreds, for embedding.
- `full_text` deliberately **excludes** captions and annotations: a
  figure mid-column otherwise cuts sentences in half (see the shared
  columns in Phase 3). Captions have their own `captions_text` field.
- A target chunk size of ~1200 characters with an overlap (later
  narrowed to a `window=1` chunk around a hit at retrieval time, see
  Phase 6, rather than enlarging the chunk overlap itself).
- **A deliberate decision, not an implicit one:** cover pages (the front
  and back two) are excluded **here**, at the level of "what goes into
  RAG", and not earlier in extraction or article assignment. It required
  carrying page numbers from `assign_articles.py` all the way here
  (`full_text_paragraphs`, with a `page` on each paragraph instead of
  one string) — nothing is lost, only data added.
- **A real bug:** `split_oversized_paragraph()` could not split a
  paragraph with not one piece of punctuation (a table or a data
  listing) — it returned it whole, up to four times the limit. Fixed
  with a fallback to splitting on word boundaries.
- **A safety limit on tokens:** rather than shrinking all chunks
  pre-emptively because of a rare exception (the 512-token limit of
  standard embedding models), only the specific text that really
  overflows is truncated (`enforce_max_length()` in `embed.py`) - an
  explicit decision, not an accident.
- The metadata header (`"Časopis: Živa\nRočník:...\nČlánek:...\nAutoři:...\nText:..."`)
  — contextual chunking. An experiment confirmed that "Časopis: Živa",
  constant across the whole corpus, dilutes the embedding needlessly,
  but it was left as it is on request (simplicity over
  micro-optimisation).

---

## Phase 5 — Embedding

**Decisions, with reasons rather than defaults:**

- A local model via `sentence-transformers` rather than an API: the
  content is purely Czech, the run is one-off and repeatable, and there
  was an educational interest in seeing the mechanism underneath. An API
  is a future extension (the embedding architecture is an
  interchangeable function).
- `normalize_embeddings=True`, float32 (float16 brings no benefit on a
  CPU), batch size not treated as a risk (embedding models are an order
  of magnitude smaller than an LLM).
- Vector dimension deliberately not treated as a selection criterion on
  a small dataset: the model carries its dimension, it is not an
  independent knob.
- Candidates were compared (`compare_models.py`) on our own test queries
  with known answers, not on a general benchmark.
- `embed.py`: `_load_model()` tries `local_files_only=True` first (no
  network) and falls back to an online download only when the model is
  not cached yet — this removes a pointless network delay on every
  repeated run.
- **Checkpointing** (`build_embeddings.py`): progressive saving via
  `np.memmap` in batches, so that a long run (hours on a CPU) can be
  interrupted and resumed safely. Verified by a real interruption
  mid-run.
- **A statistical sanity check** (`check_embeddings.py`): instead of one
  anecdotal pair, hundreds of randomly drawn pairs (same article vs.
  different article), measuring the **relative** separation (the share
  of same-article pairs above the median of different-article pairs)
  rather than absolute similarity numbers, which mislead because
  embedding spaces are anisotropic. Result over the full archive: 98.8%
  separation.

---

## Phase 6 — Vector store and retrieval (Chroma)

- **Decision:** Chroma rather than FAISS — simpler code (metadata and
  filtering built in) at the price of approximate rather than exact
  search, which at this dataset's size is practically indistinguishable.
- Vectors are handed to Chroma **already computed**, not through its own
  `embedding_function`, which keeps embedding an independently
  replaceable component.
- **A real bug:** `collection.add()` on an existing ID silently
  overwrites nothing — no error, no effect — so after regenerating the
  corpus the old results kept coming back. Fixed with `upsert()` plus an
  `--overwrite` flag for a complete rebuild when the data structure
  changes.
- **Retrieval strategy** (`assemble_context.py`): top_N=10 candidates,
  grouped by article; an article with three or more hits in the top 10
  is promoted to **the whole article** from the corpus (a strong signal
  that the query is aimed at it in full); the others get window
  expansion (one chunk either side of a hit, overlapping windows
  merged) — a consciously chosen compromise between "a short chunk only"
  and "the whole article for everything".
- A citation on every block of context (magazine, year, issue, article,
  authors, pages) — settled as a must at the very start of the retrieval
  discussion.

---

## Diagnostic tools (built along the way, not at the end)

- `diag_groups.py` — prints the sorted content of a multi-entry group
  with the headings marked and the actual assignment result.
- `diag_contact_addresses.py`, `diag_page_resolve.py`,
  `diag_page_spans.py` — targeted diagnostics for specific recurring
  failures.
- `summarize_quality_flags.py` — a summary of `quality_flags` across the
  corpus, broken down by type and by issue, with a filter for unique
  (non-recurring) titles to separate familiar noise from a new problem.

---

## Key decisions at a glance (why, not just what)

| Decision | Alternative considered | Why this way |
|---|---|---|
| Covers trimmed at chunking time | Trim during extraction | Keep the raw digitisation complete and reusable |
| Chunk from `full_text`, not from blocks | Chunk the blocks directly | Consistent size for embedding |
| A local embedding model | An API straight away | Purely Czech text, educational interest, zero cost per experiment |
| Chroma | FAISS plus our own metadata store | Simplicity of code over full control |
| `window=1`, `promote_threshold=3` | Always a chunk / always a whole article | A compromise: short chunks are not enough, whole articles are waste when the query is not aimed at one source |
| `quality_flags` as metadata, not a silent fix | Guess and merge blindly | Transparency over false certainty |

---

## Loose ends at the time (all since closed)

- Assembling the LLM prompt from `assemble_context.py` output (system
  instruction plus context plus question).
- Possible reranking (a cross-encoder) as a second pass if the
  window/promote heuristic proved insufficient in practice.
- The isolated "swapped review order" error (2019/6) remains knowingly
  unfixed (see known limitations).

---

## Phase 7 — Answer generation (`answer.py`)

The last loose end from the previous version. The system instruction sat
in a file nobody read, and the pipeline stopped at printed context.

- The prompt has three parts and each lives somewhere different on
  purpose: the **system instruction** in the source profile (the
  citation rules and the language of the answer belong to the corpus,
  not the pipeline), the **context** as numbered sources with a full
  citation and a FULL ARTICLE / excerpt marker, and the **question** at
  the very end, so that the stable part of the prompt can be cached.
- The FULL ARTICLE / excerpt marker is not decoration: the system
  instruction refers to it. On an excerpt the model may admit that the
  fragment starts mid-thought; on a whole article such a caveat would be
  false caution.
- `--dry-run` prints the finished prompt and sends nothing to the API.
  Retrieval can be tuned for free and without a key.

---

## Phase 8 — Refactoring onto source profiles

**The problem:** the pipeline was usable on one specific magazine. Font
names, point sizes, the shape of the footer, the contents page and the
magazine's name were hard-wired in six different places across five
scripts.

**The solution:** all of that is now **data** in a single
`SourceProfile`, not an `if` in the middle of a parser. Adding a
magazine means writing one profile.

- **Adaptive classification** (`profiles/adaptive.py`) — the answer to
  nobody knowing a new magazine's font names. The reference font size is
  derived from the document (the most voluminous family-and-size
  combination by CHARACTER count, not by block count: a page carries
  many titles but little of their text) and every rule is relative to
  it. Portable without calibration, at the cost of not seeing
  distinctions that are not typographic — the author line is nearly the
  same size as the text.
- **Two footer-detection strategies**, because one is not enough:
  `keyword` by content (Živa's footer carries its own name and domain)
  and `position` by place on the page (The MagPi's footer carries only a
  number).
- **`tools/inspect_fonts.py`** — calibration for a new profile. It
  prints a typography histogram with samples, a suggested set of rules
  and the footer candidates. As a check: on the sample issue of Živa it
  produces exactly what had originally been derived by hand.

**A real bug found while writing the tests:** `label_to_int()` compared
Roman numerals case-insensitively, but `roman_to_int()` understood upper
case only — so `"CXLVIiI"` (a typesetting slip) quietly returned 146
instead of 148. It never showed in the pipeline, because
`extract_label()` normalises the token first, but the diagnostic tools
call `label_to_int()` directly. Normalisation belongs inside. A second
finding: `ROMAN_RE` without an upper length bound declares any long
enough run of the letters I/V/X/L/C/D/M a Roman numeral — and under
position-based footer detection that really happens.

**Reproducibility:**

- Pinned versions in `requirements.txt`. PyMuPDF changes how it splits a
  page into blocks between versions, and that is the input to everything
  else.
- **A golden test** compares a SHA-256 fingerprint of each stage's
  output. The fixture holds no magazine content, only fingerprints and
  counts — as sensitive as comparing the text, while publishing not one
  letter. It skips without the PDF.
- The whole refactor is verified by regression: `run_all` over the
  sample issue produces `blocks`, `toc`, `page_map`, `articles`,
  `corpus` and `chunks` **byte-identical** to the output from before the
  refactor.

**The most visible fix:** the pipeline died with a `UnicodeEncodeError`
on its first print on any console using cp1252, the Windows default. It
worked only in Spyder, whose stdout is UTF-8 — exactly the kind of bug
the author never sees and everyone who clones the project hits within
two seconds.

**What the adaptive profile costs in accuracy** (measured on the sample
issue of Živa, where the adaptive profile knows nothing at all about
Živa — not the font name, not where the contents are, not what stands in
the footer): 36 articles against 37, all 36 titles identical to the hand
profile's, 19 of them byte-identical in full text too. The missing
article was lost on an unmapped printed page number; the remaining
differences are paragraph boundaries, not lost content. The number of
articles carrying `quality_flags` is the same for both profiles (3).

**A bug found by that measurement:** the first version of the adaptive
profile inherited the default `footer_detection="keyword"` with an empty
list of keywords. Footer detection then found not one printed page
number, the contents had nothing to map onto, and the pipeline
**quietly produced zero articles** — exactly the kind of failure that
goes unnoticed without a comparison against a known result, because
nothing crashes. Guarded twice over: the adaptive profile uses
`"position"`, and `is_footer_block()` under the `"keyword"` strategy
with no keywords returns `False` instead of declaring every small piece
of text on the page a footer.

---

## Phase 9 — Verification on a second magazine (The MagPi)

A test of whether source profiles really work or merely look tidy. Three
real MagPi issues (150, 152, 155), born-digital PDF, 132 pages, not one
hand-entered value about the typesetting.

**The progression was 10 → 39 → 92 articles**, each jump exposing a bug
Živa could not have shown and — this is the point — **not one of them
crashed**. The pipeline finished every time, printed a summary and
looked content.

1. **The page labels did not meet.** The MagPi's contents give `032`,
   the footer on the page `32`. The page mapping worked perfectly: one
   contiguous run, offset 0, 97 of 132 pages detected directly. The two
   sides simply did not connect. Fixed with `normalize_label()`, which
   normalises the form at both ends. Živa does not zero-pad, so such a
   mismatch never arose there.

2. **Control characters in titles.** The decorative bullet before a
   contents entry comes out of the PDF as `U+0007` and stayed in the
   title. It broke the text comparison between a contents title and the
   heading found in the body (`texts_match`). Fixed in `clean()`, by
   comparing characters rather than by a regex with a range: a range is
   hard to read and easy to get wrong.

3. **The contents are spread across three pages.** `find_toc_pages()`
   took only the best one, so two thirds of the issue were never
   processed. Fixed with two thresholds: absolute (to tell the contents
   from a page with a few stray numbers) and relative to the best page
   (to pick up its counterparts). On issue 150 that also fixed the case
   where the "best" page was the weaker half of a spread.

**A bug in the calibration tool itself.** `inspect_fonts` recommended
`"keyword"` for The MagPi when the right answer is `"position"`. The
"print what repeats near the edge" filter reliably hides page numbers:
the magazine's name is the SAME string on every page, whereas the page
number is DIFFERENT on every page. They have to be counted differently.
Plus two errors in the denominator: the shares were computed against the
whole issue even when only the first N pages had been examined, and
numbers were summed by occurrence rather than by page (the number is
often set at both top and bottom, so the sum exceeded the page count —
on Živa it printed "140 of 84"). After the fix the tool recommends
`"position"` for both magazines, which is what actually works for both.

**What remained unresolved at this point:** splitting title from author
by the colour of the first span is a purely Živa heuristic. On The MagPi
the section name ("Tutorials") came out as the author, and on the
contents page a few numbers in a different format were mistaken for the
start of a new entry, producing duplicate records (about a quarter in
practice). It did not affect the corpus's usability — the article text
and page ranges were right — but the author metadata needed a rule of
its own for The MagPi.

The Živa regression passed after each of these changes: `blocks`, `toc`,
`page_map`, `articles` and `chunks` all stay byte-identical.

---

## Phase 10 — Duplicate records and invented authors on The MagPi

The two remaining defects from the previous phase. Both traced to the
same cause: the contents page was being read by rules cut for Živa, when
The MagPi's are different.

**What is actually on The MagPi's contents page.** Not one kind of
number but **three**, and only one of them is a contents entry:

1. the real page numbers (`RobotoSerif` at 8.5pt in black on issue 155,
   `Rajdhani` at 14pt on issue 150),
2. decorative callouts — a large white number with a short caption, set
   into a coloured panel,
3. the number of the contents page itself in the running footer.

Kinds 2 and 3 created false entries, which then looked like duplicates
in the corpus ("Contents", "Top Projects", the same article twice).

**Why it could not be written into the profile.** The MagPi redesigned
between issues 150 and 152: fonts changed (`Rajdhani`/`RobotoSlab` to
`Roboto*`), as did sizes and the number format (`22` to `032`). One set
of hard-coded values would hold for part of the archive only, and would
quietly produce nonsense on the rest.

**The solution: the same reasoning as block classification, one storey
up.** The contents are a list, so the **"number style + style of the
title right after it" pair** repeats for every entry. The most frequent
such pair is by definition the real one; a decorative callout leads into
a different kind of text and there are an order of magnitude fewer of
them. Measured on real data the gap is comfortable: on Živa the second
real pair is at 79% of the first, on The MagPi the first decorative
callout at 14%.

Two iterations, both instructive:

- **The first version took only the single most frequent number style.**
  It worked on The MagPi; on Živa the article count fell from 37 to 23,
  because Živa uses two equally valid sizes of number (9 and 10pt) and
  half the entries were lost. Without a comparison against a second
  magazine it would have gone unnoticed.
- **The second version took only the single most frequent text style.**
  The same mistake one step further on: Živa mixes title sizes too. Only
  a threshold on the frequency of the whole pair, rather than of an
  individual style, fits both.

**Authors.** Splitting title from author by the colour of the first span
is a purely Živa thing. The MagPi's contents do not list authors at all,
so the section name ("Tutorials", "Project Showcase") was being produced
as the author. A new `toc_has_authors` field turns it off. It is off for
the adaptive profile too, deliberately: a missing author is an empty
field, whereas a badly split title is a damaged title **and** an
invented author at once.

**A side fix:** `font_family()` did not understand an optical size in a
font name, so `RobotoSerif-20ptRegular` and `RobotoSerif-Italic` came
out as two different families. An italic run inside a title was then
discarded as a foreign style.

**Result.** The MagPi: 85 articles from three issues, no duplicate page,
no invented author. Živa through the adaptive profile: 37 articles,
exactly as many as the hand profile (previously 36), and 36 of the hand
profile's 37 titles are contained verbatim in the adaptive ones. The
hand profile's regression stays byte-identical.

---

## Phase 11 — English throughout

The repository was written in Czech: comments, docstrings, CLI help and
printed messages. For a public portfolio piece that is a barrier, so all
of it is now English.

What stays Czech is the part that is **data about Czech text** rather
than prose, and translating it would break the parser:

- Živa's footer keywords and imprint markers, and its filename pattern.
- The Czech connectives in colour-scale legends
  (`LEGEND_WORDS = {"do", "nad", "pod", "až"}`) and the Czech capitals
  in the caption-lead and sentence-splitting regexes.
- The Czech system prompt used to answer questions about a Czech
  magazine, and the Czech chunk header and citation that Živa's profile
  declares — the language of these follows the corpus, not the code.
- The retrieval test queries in `tools/compare_models.py`: a retrieval
  test has to be in the language of the documents.

Two things came out of the translation rather than being cosmetic:

- The chunk metadata header moved out of the `SourceProfile` default and
  into `ziva.py`. The shared default is now English and Živa states its
  Czech one explicitly, which makes the "the language follows the
  corpus" rule visible instead of implied.
- Source citations followed the same route: `citation_template`,
  `page_single_label` and `page_range_label` moved into the profile.
  Živa keeps its Czech citation ("str. 4-7, autoři: ...") and The MagPi
  gets an English one with no year, because its issues are numbered
  continuously.

`tools/diag_kontaktni_adresy.py` became `tools/diag_contact_addresses.py`
and gained a `--needle` argument, so it is no longer hard-wired to the
one Živa entry it was written for.

The Czech originals of the documentation are kept outside the repository
for the author's own reference; the published versions are these.
