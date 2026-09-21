"""Profile: Živa (Nakladatelství Academia / Czech Academy of Sciences),
volumes 2014-2024.

The reference profile - the rules below are tuned on 75 real issues and
pinned by `tests/test_pipeline_golden.py`.

RIGHTS: the contents of Živa are under copyright. This profile describes
only the typesetting (font names, point sizes, the shape of the footer);
no magazine content lives in this repository and none may, extracted full
texts included. The openly licensed alternative for a public demo is the
`magpi` profile.

The Czech string literals below are *data about Czech text* - the words
that actually appear in this magazine's footer and imprint. They are not
prose and must not be translated.
"""
from . import FontFamily, SizeRule, SourceProfile

# Živa sets all of its running text in one serif face (MeliorCE) and
# everything *around* the text - figure captions, footers, the cover - in
# a sans face (HelveticaCE, occasionally Arial). Splitting into those two
# families is the load-bearing idea of the whole classification: in the
# serif family an unknown size is almost certainly body text, in the sans
# family almost certainly a caption.
SERIF = FontFamily(
    name="serif",
    prefixes=("MeliorCE",),
    rules=(
        SizeRule("title", size_min=18, bold=True),
        # 13-18 bold: a chapter heading inside an article. The upper bound
        # is harmlessly inclusive here - anything from 18 up was already
        # taken by the rule above.
        SizeRule("heading", size_min=13, size_max=18, bold=True),
        # The smaller subheadings in the back matter ("Kontaktní údaje pro
        # předplatitele", "Vědci z Akademie věd ČR oceněni Českou hlavou")
        # are exactly 13pt but not bold.
        SizeRule("heading", size_min=13, size_max=13, bold=False),
        # The author line under an article title.
        SizeRule("other", size_min=11.5, size_max=12.5, bold=False),
    ),
    fallback="body",
)

SANS = FontFamily(
    name="sans",
    prefixes=("HelveticaCE", "Arial"),
    rules=(
        SizeRule("title", size_min=30),      # the big cover number, "6 /2014"
        SizeRule("other", size_min=11, size_max=13, bold=True),  # cover blurbs
    ),
    fallback="caption",
    # Panel labels (a/b/c), scale bars ("1 cm") and runs of numbers along
    # a chart axis are set exactly like captions but carry no content.
    # They can only be told apart by the shape of the text, not by the
    # font. See is_diagram_annotation().
    detect_annotations=True,
)

PROFILE = SourceProfile(
    key="ziva",
    journal_name="Živa",
    language="cs",

    filename_pattern=r"ziva-(\d{4})-(\d)\.pdf$",

    # In Živa the contents are always on the third physical PDF page.
    toc_page_indices=(2,),
    toc_page_number_prefixes=("MeliorCE",),
    toc_page_number_bold=True,
    toc_has_authors=True,   # the contents list authors, separated by span colour
    toc_drop_markers=(
        "© Nakladatelství Academia",
        "SSČ AV ČR",
        "Přetisk článků",
        "http://ziva.avcr.cz",
        "www.ziva.avcr.cz",
    ),
    toc_skip_span_markers=(
        "ziva.avcr.cz",
        "© Nakladatelství Academia",
        "Přetisk článků",
    ),

    # "ziva.avcr.cz 262 živa 6/2014" or "živa 6/2014 263 ziva.avcr.cz"
    footer_pattern=r"živa\s+\d/\d{4}|ziva\.avcr\.cz",
    footer_font_prefixes=("HelveticaCE-Bold",),
    footer_max_size=9.0,
    footer_keywords=("živa", "ziva"),
    footer_skip_tokens=("ziva.avcr.cz", "http://ziva.avcr.cz"),

    junk_marker_sets=(("Nakladatelství Academia", "Přetisk"),),

    families=(SERIF, SANS),
    default_block_type="body",

    skip_first_pages=2,   # front cover + inside front cover
    skip_last_pages=2,    # back cover + advertisement for the next issue

    # The corpus is Czech, so the header baked into the embedded text is
    # Czech too - the header's language follows the corpus, not the code.
    chunk_header_template=(
        "Časopis: {journal}\n"
        "Ročník: {year}\n"
        "Číslo: {issue}\n"
        "Článek: {title}\n"
        "Autoři: {author}\n"
        "Text: {text}"
    ),
    unknown_author_label="neuvedeno",
    citation_template=(
        "{title} ({journal} {year}/{issue}, {pages}), autoři: {author}"),
    # PDF pages, not the printed ones (see the base profile)
    page_single_label="str. {page} v PDF",
    page_range_label="str. {start}-{end} v PDF",
    captions_label="Popisky obrázků:",

    system_prompt_file="ziva_cs.txt",
)
