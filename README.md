# Ziva PDF → RAG pipeline (stage 1–4)

Pořadí kroků (dá se spustit i ručně po jednom, kvůli ladění):

```
python extract_blocks.py ziva-2014-6.pdf ziva_blocks.json
python create_toc.py     ziva-2014-6.pdf ziva_toc.json
python build_page_map.py                       # čte ziva_blocks.json, píše page_map.json
python assign_articles.py                      # čte ziva_blocks.json + ziva_toc.json + page_map.json, píše ziva_articles.json
```

Nebo hromadně přes celý archiv:

```
pip install pymupdf

pdf/
  ziva-2014-6.pdf
  ziva-2015-1.pdf
  ...

python run_all.py --input ./pdf --output ./output
```

Výstup: `output/<rok>-<číslo>/{blocks,toc,page_map,articles}.json` pro každé
číslo + souhrnný `output/ziva_corpus.json` se všemi články ze všech čísel
(má `article_id` typu `"2014-6-0"`, takže je bezpečné je sloučit dohromady).

## Co je potřeba pohlídat u jiných čísel

* `extract_blocks.py` – heuristika typu bloku (`title`/`heading`/`other`/
  `body`/`caption`) je založená na fontu `MeliorCE`/`HelveticaCE` a
  konkrétních velikostech písma (viz `classify()`). Pokud se v jiných
  ročnících liší barva/velikost nadpisů (jak jste zmiňoval), stejná
  velikost/tučnost by měla platit i tak – ale stojí za to zkontrolovat pár
  čísel napříč roky (`extract_blocks.py` + rychlý pohled na `type` v JSONu).
* `create_toc.py` – počítá s tím, že obsah čísla je vždy na 3. fyzické
  stránce PDF (`toc_page_indices=(2,)`, 0-indexováno). Pokud by se to u
  starších/novějších čísel lišilo, stačí tenhle parametr změnit (nebo
  `build_toc()` zavolat pro víc indexů najednou).
* `build_page_map.py`/`assign_articles.py` na fontu/heuristikách
  `extract_blocks.py` nezávisí o nic víc, než že očekávají pole
  `page`, `block_id`, `type`, `font`, `font_size`, `bbox`, `text` u
  každého bloku – takže i kdyby `extract_blocks.py` bylo nutné pro jiný
  layout upravit, zbytek pipeline by měl fungovat beze změny.

## Update: zalomení pomlčkou i mezi chunky (ne jen uvnitř stránky)

`extract_blocks.py`'s `smart_join()` řešil jen zalomení uvnitř jedné
stránky. Chunky z různých sloupců (nemerguje se, protože mezi nimi je
vizuálně jiný blok/obrázek) nebo z různých stránek napříč jedním článkem
se ale skládaly obyčejným `"\n\n".join(...)` bez kontroly pomlčky. Na
vzorovém čísle to způsobovalo 27 rozbitých slov typu `"opaková-"` +
`"ní..."` místo `"opakování"`. `assign_articles.py` teď při skládání
`full_text`/`captions_text` používá stejnou detekci (naimportovanou z
`extract_blocks.py`), takže se to slepí bez ohledu na to, kde přesně k
zalomení došlo.

## Update: číslování stránek přes celý archiv (build_page_map.py v3)

Ukázalo se na reálných číslech, že tištěné číslování stránek není jedna
souvislá řada – v jednom PDF se běžně střídá: arabské číslo (hlavní články)
→ římské číslo (příloha) → arabské číslo znovu (další články), a offset
(rozdíl mezi PDF stránkou a tištěným číslem) je pro každý takový úsek jiný.
`build_page_map.py` teď detekuje tyhle souvislé úseky sám ("runs" v
`page_map.json`) a dopočítává chybějící popisky (třeba celostránková fotka
bez patičky) v rámci téhož úseku. Opravil jsem taky bug, kdy se jednopísmenné
římské číslice ("I", "V", "X", "C"...) mylně zahazovaly jako "nejednoznačné".

## Update: build_chunks.py napojený do run_all.py

`run_all.py` teď po sesbírání `ziva_corpus.json` rovnou zavolá i chunkování
a vyrobí `output/ziva_embedding_chunks.jsonl` - nemusíte spouštět
`build_chunks.py` zvlášť (pořád ale jde, kdyby se hodilo přegenerovat
chunky s jinými parametry bez opakování celé extrakce z PDF). Parametry
`--skip-first`/`--skip-last` (výchozí 2/2) jdou nastavit i na `run_all.py`.

## Update: build_chunks.py (stage 5) + kde se řeší obálkové stránky

Nový skript `build_chunks.py` sekundárně nasekává `full_text_paragraphs`/
`captions_paragraphs` (viz níž) na chunky vhodné pro embedding - cílová
velikost ~1200 znaků, s malým překryvem mezi sousedními chunky, plus
metadata (ročník/číslo/článek/autoři) jako samostatná pole i jako hlavička
zapečená přímo do `embedding_text` (tzv. "contextual chunking" - pomáhá to
relevanci vyhledávání, protože osamocený chunk bez kontextu embedding
modelu i LLM říká míň).

**Konceptuální rozhodnutí, které stálo za probrání:** poslední článek
v každém čísle dostával `pdf_page_end` až do úplně poslední PDF stránky
(protože po něm už nic dalšího není v TOC) - a tahle poslední stránka bývá
samostatná obálková fotka, co s článkem obsahově nesouvisí (viz "IV.
obálka" test). Řešilo by se to na 3 místech s různými kompromisy:

1. Zahodit stránky obálky už v `extract_blocks.py` (nejjednodušší, ale
   nevratně - `blocks.json` by přestal být kompletní syrová digitalizace).
2. Oříznout rozsah stránek v `assign_articles.py` (opravuje `articles.json`
   jako artefakt, ale je to natvrdo zadrátované pravidlo uprostřed
   pipeline, které s "jak digitalizuju PDF" nemá nic společného).
3. **(zvoleno)** Nechat `blocks.json`/`articles.json` kompletní se vším
   (žádná ztráta dat), a teprve `build_chunks.py` - tam, kde se skutečně
   rozhoduje "co jde do RAG" - stránky obálky vyfiltruje. Aby to šlo udělat
   přesně (ne jen hledáním textového markeru "obálka"), `assign_articles.py`
   teď u každého článku navíc ukládá `full_text_paragraphs`/
   `captions_paragraphs` (list `{"page": N, "text": "..."}` místo jednoho
   stringu) a `total_pdf_pages` (celkový počet stran PDF čísla). Nic se
   tím neztrácí - `full_text`/`captions_text` jako hotové stringy tam
   zůstávají dál pro rychlý náhled/čtení.

`build_chunks.py` pak přes `--skip-first`/`--skip-last` (výchozí 2/2)
vynechá odstavce z první/poslední N stránek PDF při stavbě chunků.

Mimochodem oprava bugu: `split_oversized_paragraph` neuměl rozsekat
odstavec, který nemá ŽÁDNOU tečku/otazník/vykřičník (typicky výpis dat
z tabulky/grafu) - vrátil ho tak, jak byl, i kdyby byl 4x delší než limit.
Teď má fallback na sekání po slovech.

## Update: full_text a captions_text jsou teď oddělené

Ukázalo se, že mít popisky obrázků namíchané (byť označené) přímo v
`full_text` je problém: obrázek/graf často sedí uprostřed sloupce, takže
text před ním a za ním jsou v PDF opravdu dva samostatné bloky - a naše
řazení podle (sloupec, y) je pak vmáčklo vedle sebe s popiskem obrázku
mezi nimi, čímž v souvislém textu vypadalo, že je věta uprostřed přeťatá
popiskem ("...jeho" → [popisek grafu] → "dopad významně ovlivňují...").

Řešení: `full_text` teď skládáme jen z `title`/`heading`/`other`/`body`
chunků (žádné captions/annotations). Popisky obrázků mají svoje vlastní
pole `captions_text` (spojené za sebou, ve svém pořadí). Věta se tak sice
pořád může rozdělit do dvou odstavců místo plynulého navázání, ale aspoň
ji nepřetne cizí obsah uprostřed.

Zároveň jsem rozšířil detekci `annotation` o legendy barevných škál na
mapách/grafech ("pod -150 -100 až -50 0 až 50 100 až 150 nad 150", "do 3
3–4 4–5 ... nad 12") a krátké jednotky v hranaté závorce ("[°C]", "[%]").

## Update: chytání popisků sazených v běžném písmu (bez malého fontu)

Delší popisky obrázků v Živě jsou často sazené STEJNÝM písmem jako tělo
článku (MeliorCE, ne malá bezpatková caption-sazba), takže je `classify()`
podle fontu vůbec nerozezná od běžného textu. `extract_blocks.py` teď navíc
kontroluje, jestli `body`-blok začíná typickým odkazem na číslo obrázku
("1 a 2 Nejnápadnějším příznakem...", "3 Schéma normálního...", "9 a 10
..."). Zkoušel jsem to kombinovat i s kontrolou, že blok sedí v PDF hned
vedle obrázku, ale u přechodů mezi články bývá popisek v surovém pořadí
bloků docela daleko od "svého" obrázku (mezi nimi je titulek/autor/abstrakt
dalšího článku) - spoléhá se tedy jen na tvar textu + minimální délku 20
znaků (ať to nechytne krátké číslované nadpisy typu "1. Úvod"). Na
vzorovém čísle to opravilo všech 12 dosud nezachycených popisků beze
zjištěného falešného poplachu.

## Update: rozlišení "annotation" vs "caption"

`extract_blocks.py` teď umí odlišit dvě různé věci, které předtím obojí
padaly do `type: "caption"`:

* **`"caption"`** – skutečný popisek obrázku (i útržkovitý, ale pořád jde
  o smysluplný text popisující, co je vidět).
* **`"annotation"`** – editorská "hantýrka" nalepená přímo na obrázek/schéma:
  panelové značky (`a`, `b`, `1`, `2`...), měřítko (`1 cm`, `0,2 mm`),
  řady čísel na ose grafu (`35 30 25 20 15 10 5 0`). Nenese žádný obsah,
  proto ho `assign_articles.py` do `full_text` vůbec nedává (v `chunks`
  zůstává, pro případ potřeby).

**Známé omezení:** kratší anatomické/technické útržky rozeseté kolem
composite obrázků (např. "substantia", "nigra", "gen IT15", nebo dvoupísmenná
zkratka do legendy jako "Mh", "Cu") se od skutečného popisku nedají čistě
podle tvaru textu odlišit – zůstávají jako `"caption"`. Vyžadovalo by to znát
polohu textu vůči konkrétnímu obrázku, ne jen jeho obsah.

## Známá nepřesnost (a proč nevadí)

Sloupcové řazení bloků v `assign_articles.py` je "best effort" – u
víceloupcových stránek s obrázky se občas popisek obrázku (nebo pořadí
chunků obecně) zařadí o kousek jinde, než by čtenář čekal. Od poslední
úpravy to už ale nemůže rozbít plynulost `full_text` (ten popisky vůbec
neobsahuje) - týká se to jen pořadí uvnitř `chunks`/`captions_text`. Pro
RAG to nevadí: chunk je pořád smysluplný samostatný kus textu se správnými
metadaty (článek/autor/rok/číslo/stránka), jen v aproximativním pořadí.
