"""Profil: The MagPi (Raspberry Pi Press), licence CC BY-NC-SA 3.0.

Proč zrovna MagPi: pipeline vznikla nad Živou, jejíž obsah je autorsky
chráněný a nemůže být součástí veřejného dema. MagPi je strukturně skoro
totožný případ - měsíčník v born-digital PDF, vícesloupcová sazba, obálka,
obsah čísla na pevném místě, běžící patička s číslem stránky - ale vychází
pod otevřenou licencí, takže se z něj dá udělat ukázka, kterou jde
zveřejnit. Čísla se stahují z magpi.raspberrypi.com/issues.

**Typografie se tu nekalibruje ručně, ale odvozuje z dokumentu**
(`adaptive=True`, viz `profiles/adaptive.py`). Je to vědomý kompromis:
ruční profil jako `ziva.py` je přesnější, ale předpokládá, že někdo pro
každý ročník ověřil jména fontů a velikosti písma. MagPi za deset let
několikrát změnil grafiku, takže jedna sada absolutních hodnot by stejně
neplatila napříč archivem - relativní pravidla to ustojí.

Chcete-li přesnost ruční kalibrace pro konkrétní ročník, pusťte

    python -m tools.inspect_fonts cesta/k/MagPi155.pdf

a hodnoty z výpisu přepište do vlastního profilu podle vzoru `ziva.py`
(stačí doplnit `families` a `adaptive=False`) - zbytek nastavení níž
zůstává v platnosti.

**Co bylo potřeba nastavit jinak než u Živy:**

* Patička se hledá **podle polohy**, ne podle klíčového slova. Živa má
  v patičce vlastní název a doménu, takže se dá chytit textem; MagPi má
  v patičce jen číslo stránky, takže rozhoduje "malý text u dolního okraje
  stránky, který je v podstatě jen číslo".
* Čísla jsou průběžná (MagPi 1, 2, ... 155), ne ročník + číslo. Vzor
  jména souboru má proto jen jednu skupinu a `run_all.py` z ní udělá
  dvojici s prázdným ročníkem.
* Metadatová hlavička chunku i systémový prompt jsou anglicky - jazyk
  se řídí korpusem, ne jazykem pipeline.
"""
from .adaptive import RELATIVE_RULES
from . import FontFamily, SourceProfile

PROFILE = SourceProfile(
    key="magpi",
    journal_name="The MagPi",
    language="en",

    # "MagPi155.pdf", "The-MagPi-155.pdf", "MagPi-155.pdf", "magpi_155.pdf"
    filename_pattern=r"(?:the[-_ ]?)?magpi[-_ ]?(\d{1,3})\.pdf$",

    # Obsah čísla nebývá vždy na téže straně (mění se rozsah úvodní inzerce)
    # a bývá rozložený přes dvě až tři stránky - viz create_toc.find_toc_pages.
    toc_page_indices=(),
    # Styly položek obsahu se odvozují ze stránky samotné. MagPi mezi čísly
    # 150 a 152 předělal grafiku: změnily se fonty (Rajdhani/RobotoSlab ->
    # Roboto*), velikosti i formát čísel stránek ("22" -> "032"). Napevno
    # zadané hodnoty by tedy platily jen pro část archivu a na zbytku by
    # tiše vyrobily nesmysly. Viz create_toc.detect_entry_styles.
    toc_adaptive_styles=True,
    # Obsah MagPi u položek autory neuvádí. Kdyby se titulek přesto dělil,
    # jako autor by vyšel název rubriky ("Tutorials", "Project Showcase").
    toc_has_authors=False,

    # Patička MagPi obsahuje jen číslo stránky, na obsah se chytit nedá.
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

    system_prompt_file="generic_en.txt",
)
