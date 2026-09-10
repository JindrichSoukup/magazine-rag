"""Profile: adaptive - classification without knowing any font names.

A hand-written profile (see `ziva.py`) is more accurate, but it assumes
somebody has already worked out that the main typeface is called
`MeliorCE` and that a title is 18 points. On a new magazine nobody knows
those values, and finding them by hand is an hour of work.

The adaptive profile does not know them either - it **derives them from
the document itself**:

1. Walk the whole PDF and count how many CHARACTERS are set in each
   (font family, size) combination. Weighting by characters rather than
   by block count matters: a page carries many titles but little of their
   text.
2. The most voluminous combination is by definition the article's body
   text, which fixes `body_size` and `body_family`.
3. Everything else is classified **relative** to that reference (a title
   is 1.7x larger than body text) and by whether it is the same font
   family as the text or a different one (captions, footers, the cover).

That carries the whole heuristic to another magazine with no code change
and no calibration. Portability costs accuracy: the adaptive profile
cannot see distinctions that are not typographic but semantic - in Živa,
for instance, the author line (`other`), which is nearly the same size as
the text. When such a distinction matters, write a hand profile; the
adaptive one is a good start and a safety net.
"""
from . import FontFamily, RelativeRule, SourceProfile

# Ratios are against the size of body text (1.0). The bounds leave a gap
# between categories on purpose: magazine typography jumps in visible
# steps rather than continuously, so a sensible band is safer than a
# sharp threshold.
RELATIVE_RULES = (
    # Markedly larger type in the text family = an article title.
    RelativeRule("title", ratio_min=1.7, same_family=True),
    # The intermediate step = a chapter heading.
    RelativeRule("heading", ratio_min=1.18, ratio_max=1.7, same_family=True),
    # Bold at body-text size = a subheading. Without requiring bold this
    # would swallow the body text, so `bold` is mandatory here.
    RelativeRule("heading", ratio_min=0.95, ratio_max=1.18, bold=True,
                 same_family=True),
    # Conspicuously large type outside the text family = cover furniture.
    RelativeRule("title", ratio_min=2.2, same_family=False),
)

PROFILE = SourceProfile(
    key="adaptive",
    journal_name="",          # filled in from --journal-name on the CLI
    language="en",

    filename_pattern=r"(\d{4})-(\d+)\.pdf$",

    # Nobody knows in advance what an unknown magazine puts in its footer,
    # so position on the page has to decide rather than content. Without
    # this, not a single printed page number would be found, the contents
    # would have nothing to map onto, and the pipeline would quietly
    # produce zero articles.
    footer_detection="position",
    footer_max_size=12.0,
    footer_zone=0.90,
    header_zone=0.08,
    footer_max_tokens=5,

    adaptive=True,
    relative_rules=RELATIVE_RULES,
    # In the adaptive profile families are not identified by name but by
    # whether they are the body-text family or not. The fallbacks are the
    # same as in a hand profile, and for the same reason: an unknown size
    # in the text family is almost certainly text, outside it almost
    # certainly a caption.
    families=(
        FontFamily(name="body-family", prefixes=(), fallback="body"),
        FontFamily(name="other-family", prefixes=(), fallback="caption",
                   detect_annotations=True),
    ),
    default_block_type="body",

    # Without knowing the magazine, the contents are not looked for on a
    # fixed page but detected (see create_toc.find_toc_pages), and the
    # styles of their entries are derived from the page itself
    # (detect_entry_styles).
    toc_page_indices=(),
    toc_adaptive_styles=True,
    # Whether the contents list authors cannot be known from an armchair.
    # Not splitting is the safer default: a missing author is an empty
    # field, whereas a badly split title is a damaged title *and* an
    # invented author at once.
    toc_has_authors=False,

    skip_first_pages=2,
    skip_last_pages=2,

    system_prompt_file="generic_en.txt",
)
