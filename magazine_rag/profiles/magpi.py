"""Profile: The MagPi (Raspberry Pi Press), licensed CC BY-NC-SA 3.0.

Why The MagPi: the pipeline grew up on Živa, whose contents are under
copyright and cannot be part of a public demo. The MagPi is structurally
almost the same case - a monthly in born-digital PDF, multi-column
setting, covers, a contents page at a predictable place, a running footer
with the page number - but it is published under an open licence, so it
can be shown. Issues are downloaded from magpi.raspberrypi.com/issues.

**The typography here is not calibrated by hand but derived from the
document** (`adaptive=True`, see `profiles/adaptive.py`). That is a
deliberate trade: a hand profile like `ziva.py` is more accurate, but it
assumes somebody has verified font names and point sizes for every
volume. The MagPi has redesigned several times over a decade, so one set
of absolute values would not hold across the archive anyway; relative
rules survive it.

If you want hand-calibrated accuracy for one particular volume, run

    python -m tools.inspect_fonts path/to/MagPi155.pdf

and copy the values from its output into a profile of your own, following
`ziva.py` (fill in `families` and set `adaptive=False`). Everything else
below still applies.

**What had to be set differently from Živa:**

* The footer is found **by position**, not by keyword. Živa puts its own
  name and domain in the footer, so it can be caught by content; The
  MagPi puts only the page number there, so what decides is "small text
  near the bottom edge that is essentially just a number".
* Issues are numbered continuously (MagPi 1, 2, ... 155) rather than by
  year and issue. The filename pattern therefore has a single group and
  `run_all.py` turns it into a pair with an empty year.
* The styles of contents entries are derived from the page itself. The
  MagPi redesigned between issues 150 and 152: fonts changed
  (Rajdhani/RobotoSlab to Roboto*), as did sizes and the page-number
  format ("22" to "032").
* The chunk metadata header and the system prompt are in English - the
  language follows the corpus, not the pipeline.
"""
from .adaptive import RELATIVE_RULES
from . import FontFamily, SourceProfile

PROFILE = SourceProfile(
    key="magpi",
    journal_name="The MagPi",
    language="en",

    # "MagPi155.pdf", "The-MagPi-155.pdf", "MagPi-155.pdf", "magpi_155.pdf"
    filename_pattern=r"(?:the[-_ ]?)?magpi[-_ ]?(\d{1,3})\.pdf$",

    # The contents are not always on the same page (the amount of front
    # advertising varies) and are spread across two or three pages - see
    # create_toc.find_toc_pages.
    toc_page_indices=(),
    # Entry styles are derived from the page itself. Hard-coded values
    # would hold for part of the archive only and would quietly produce
    # nonsense on the rest. See create_toc.detect_entry_styles.
    toc_adaptive_styles=True,
    # The MagPi's contents do not list authors. If the title were split
    # anyway, the section name ("Tutorials", "Project Showcase") would
    # come out as the author.
    toc_has_authors=False,

    # The MagPi's footer holds nothing but the page number, so there is
    # no content to latch onto.
    footer_detection="position",
    footer_max_size=12.0,
    footer_zone=0.90,
    header_zone=0.08,
    footer_max_tokens=5,

    adaptive=True,
    relative_rules=RELATIVE_RULES,
    families=(
        FontFamily(name="body-family", prefixes=(), fallback="body"),
        FontFamily(name="other-family", prefixes=(), fallback="caption",
                   detect_annotations=True),
    ),
    default_block_type="body",

    skip_first_pages=2,
    skip_last_pages=2,

    chunk_header_template=(
        "Magazine: {journal}\n"
        "Issue: {issue}\n"
        "Article: {title}\n"
        "Authors: {author}\n"
        "Text: {text}"
    ),
    unknown_author_label="unknown",
    # No year: The MagPi numbers its issues continuously.
    citation_template="{title} ({journal} {issue}, {pages}), authors: {author}",

    system_prompt_file="generic_en.txt",
)
