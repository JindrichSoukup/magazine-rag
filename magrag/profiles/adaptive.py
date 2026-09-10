"""Profil: adaptivní - klasifikace bez znalosti jmen fontů.

Ruční profil (viz `ziva.py`) je přesnější, ale předpokládá, že někdo předem
zjistil, že se hlavní písmo jmenuje `MeliorCE` a že titulek má 18 bodů.
U nového časopisu tyhle hodnoty nikdo nezná a zjišťovat je ručně je práce
na hodinu.

Adaptivní profil je nezná taky - **odvodí si je z dokumentu samotného**:

1. Projde celé PDF a spočítá, kolik ZNAKŮ je vysázeno kterou kombinací
   (rodina písma, velikost). Vážení podle znaků, ne podle počtu bloků, je
   podstatné: titulků je na stránce hodně kusů, ale málo textu.
2. Nejobjemnější kombinace je z definice běžný text článku - tím je daná
   `body_size` a `body_family`.
3. Všechno ostatní se klasifikuje **relativně** k téhle referenci
   (titulek = 1,7× větší písmo než text) a podle toho, jestli jde o stejnou
   rodinu písma jako text, nebo o jinou (= popisky, patičky, obálka).

Tím se celá heuristika přenese na jiný časopis beze změny kódu i bez
kalibrace. Za přenositelnost se platí přesností: adaptivní profil nepozná
věci, které se od běžného textu neliší typograficky, ale jen významem -
u Živy třeba řádek s autorem článku (`other`), který má skoro stejnou
velikost jako text. Když na takovém rozlišení záleží, vyplatí se napsat
ruční profil; adaptivní je dobrý start a záchranná síť.
"""
from . import FontFamily, RelativeRule, SourceProfile

# Poměry jsou vůči velikosti běžného textu (1,0). Hranice jsou volené tak,
# aby mezi kategoriemi byla mezera - typografie časopisů skáče po výrazných
# krocích, ne plynule, takže rozumné pásmo je bezpečnější než ostrý práh.
RELATIVE_RULES = (
    # Výrazně větší písmo v rodině textu = titulek článku.
    RelativeRule("title", ratio_min=1.7, same_family=True),
    # Mezistupeň = nadpis kapitoly.
    RelativeRule("heading", ratio_min=1.18, ratio_max=1.7, same_family=True),
    # Tučné v velikosti běžného textu = podnadpis. Bez tučnosti by to
    # sebralo běžný text, proto je `bold` tady povinné.
    RelativeRule("heading", ratio_min=0.95, ratio_max=1.18, bold=True,
                 same_family=True),
    # Nápadně velké písmo mimo rodinu textu = obálka / číslo čísla.
    RelativeRule("title", ratio_min=2.2, same_family=False),
)

PROFILE = SourceProfile(
    key="adaptive",
    journal_name="",          # doplní se z --journal-name na příkazové řádce
    language="en",

    filename_pattern=r"(\d{4})-(\d+)\.pdf$",

    # Co stojí v patičce neznámého časopisu, nikdo předem neví - rozhodovat
    # tedy musí poloha na stránce, ne obsah. Bez toho by se nenašlo jediné
    # tištěné číslo stránky, obsah čísla by se neměl na co namapovat
    # a pipeline by tiše vyrobila nula článků.
    footer_detection="position",
    footer_max_size=12.0,
    footer_zone=0.90,
    header_zone=0.08,
    footer_max_tokens=5,

    adaptive=True,
    relative_rules=RELATIVE_RULES,
    # Rodiny se u adaptivního profilu neurčují jménem, ale tím, jestli jde
    # o rodinu běžného textu, nebo ne. Fallbacky jsou ale stejné jako
    # u ručního profilu a platí stejná úvaha: neznámá velikost v rodině
    # textu je skoro jistě text, mimo ni skoro jistě popisek.
    families=(
        FontFamily(name="body-family", prefixes=(), fallback="body"),
        FontFamily(name="other-family", prefixes=(), fallback="caption",
                   detect_annotations=True),
    ),
    default_block_type="body",

    # Bez znalosti konkrétního časopisu se obsah čísla nehledá na pevné
    # stránce, ale detekcí (viz create_toc.find_toc_pages), a styly jeho
    # položek se odvodí ze stránky samotné (detect_entry_styles).
    toc_page_indices=(),
    toc_adaptive_styles=True,
    # Jestli obsah uvádí autory, se od stolu nepozná. Nedělit je bezpečnější:
    # chybějící autor je prázdné pole, kdežto špatně rozdělený titulek je
    # poškozený titulek i vymyšlený autor zároveň.
    toc_has_authors=False,

    skip_first_pages=2,
    skip_last_pages=2,

    system_prompt_file="generic_en.txt",
)
