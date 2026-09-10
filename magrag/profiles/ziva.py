"""Profil: Živa (Nakladatelství Academia / AV ČR), ročníky 2014-2024.

Referenční profil - pravidla níž jsou vyladěná na 75 reálných čísel
a jsou zafixovaná testem `tests/test_profile_ziva.py`.

POZOR NA PRÁVA: obsah Živy je autorsky chráněný. Tenhle profil popisuje
jen sazbu (jména fontů, velikosti písma, tvar patičky) - žádný obsah
časopisu v repozitáři není a být nesmí, včetně vytažených plných textů.
Volně licencovaná alternativa pro veřejné demo je profil `magpi`.
"""
from . import FontFamily, SizeRule, SourceProfile

# Živa sází celý text jedním patkovým písmem (MeliorCE) a všechno "okolo"
# textu - popisky obrázků, patičky, obálku - bezpatkovým (HelveticaCE,
# místy Arial). Rozdělení na tyhle dvě rodiny je nosná myšlenka celé
# klasifikace: v patkové rodině je neznámá velikost skoro jistě běžný text,
# v bezpatkové skoro jistě popisek obrázku.
SERIF = FontFamily(
    name="serif",
    prefixes=("MeliorCE",),
    rules=(
        SizeRule("title", size_min=18, bold=True),
        # 13-18 tučně: nadpis kapitoly uvnitř článku. Horní mez je tu
        # nezávadně "včetně" - cokoli od 18 výš už sebralo pravidlo nad tím.
        SizeRule("heading", size_min=13, size_max=18, bold=True),
        # Menší podnadpisy v zadní části čísla ("Kontaktní údaje pro
        # předplatitele", "Vědci z Akademie věd ČR oceněni Českou hlavou")
        # mají přesně velikost 13, ale nejsou tučné.
        SizeRule("heading", size_min=13, size_max=13, bold=False),
        # Řádek s autorem/autory pod titulkem článku.
        SizeRule("other", size_min=11.5, size_max=12.5, bold=False),
    ),
    fallback="body",
)

SANS = FontFamily(
    name="sans",
    prefixes=("HelveticaCE", "Arial"),
    rules=(
        SizeRule("title", size_min=30),      # velké číslo na obálce, "6 /2014"
        SizeRule("other", size_min=11, size_max=13, bold=True),  # titulky na obálce
    ),
    fallback="caption",
    # Panelové značky (a/b/c), měřítka ("1 cm") a řady čísel na ose grafu
    # jsou sázené stejně jako popisky, ale nenesou žádný obsah - odliší se
    # až podle tvaru textu, ne podle fontu. Viz is_diagram_annotation().
    detect_annotations=True,
)

PROFILE = SourceProfile(
    key="ziva",
    journal_name="Živa",
    language="cs",

    filename_pattern=r"ziva-(\d{4})-(\d)\.pdf$",

    # Obsah čísla je v Živě vždy na 3. fyzické stránce PDF.
    toc_page_indices=(2,),
    toc_page_number_prefixes=("MeliorCE",),
    toc_page_number_bold=True,
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

    # "ziva.avcr.cz 262 živa 6/2014" nebo "živa 6/2014 263 ziva.avcr.cz"
    footer_pattern=r"živa\s+\d/\d{4}|ziva\.avcr\.cz",
    footer_font_prefixes=("HelveticaCE-Bold",),
    footer_max_size=9.0,
    footer_keywords=("živa", "ziva"),
    footer_skip_tokens=("ziva.avcr.cz", "http://ziva.avcr.cz"),

    junk_marker_sets=(("Nakladatelství Academia", "Přetisk"),),

    families=(SERIF, SANS),
    default_block_type="body",

    skip_first_pages=2,   # přední obálka + vnitřní strana obálky
    skip_last_pages=2,    # zadní obálka + inzerce příštího čísla

    system_prompt_file="ziva_cs.txt",
)
