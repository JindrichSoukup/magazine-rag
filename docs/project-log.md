# Živa RAG Pipeline — deník projektu

Digitalizace archivu časopisu **Živa** (Nakladatelství Academia) do strukturovaného korpusu vhodného pro RAG (retrieval-augmented generation): od syrových PDF, přes parsování článků, chunking, embeddingy, vektorové úložiště, až po strategii vyhledávání pro LLM.

Rozsah: 75 čísel, 2 941 článků, ~60 500 chunků (bloky), ~35 000 embedding chunků.

---

## Fáze 1 — Extrakce bloků z PDF (`extract_blocks.py`)

**Cíl:** dostat z PDF (fitz/PyMuPDF) strukturované textové bloky se stránkou, typem, fontem a pozicí.

- Rekonstrukce chybějícího extrakčního skriptu na základě dochovaných výstupů (`ziva_blocks.json`, `ziva_toc.json`) a vzorového PDF.
- Heuristická klasifikace typu bloku (`title`/`heading`/`other`/`body`/`caption`/`annotation`) podle fontu a velikosti — vyladěno konkrétně na sazbu Živy (MeliorCE = patkové tělo textu/titulky, HelveticaCE/Arial = bezpatkové popisky/patičky).
- **Oprava — zalomení slov přes pomlčku:** zarovnaný text v InDesignu láme slova na konci řádku pomlčkou ("dlouhodo- bé"). Přidána `smart_join()` s detekcí (písmeno + pomlčka na konci + pokračování malým písmenem), aplikovaná při skládání řádků *i* při slévání sousedních bloků na téže stránce.
- **Rozlišení `annotation` vs. `caption`:** panelové značky/měřítka nalepené na obrázek (`a`, `b`, `1 cm`, řady čísel na ose grafu) nenesou obsah — nový typ `annotation`, vyřazený z finálního textu. Zahrnuje i detekci legend barevných škál na mapách/grafech (`"pod -150 -100 až -50..."`).
- **Chytání popisků sazených v běžném písmu textu:** delší popisky obrázků v Živě nemají vždy malý bezpatkový font — poznají se podle vzorce "číslo obrázku + velké písmeno" na začátku bloku (`"1 a 2 Nejnápadnějším příznakem..."`).
- **Rozšíření detekce `heading`:** menší podnadpisy v zadní části čísla používají velikost 13 (bold i non-bold), mimo původní rozsah 14–18.
- **Skutečný bug — sléváno stylisticky nesourodého textu:** PyMuPDF sám dokáže spojit vizuálně blízký, ale obsahově nesouvisející text do jednoho syrového bloku (typicky konec jedné recenze + nadpis "Kontaktní adresy autorů" hned pod ním). Důsledek: dominantní styl "vyhrál" delší/nedůležitý fragment a nadpis zmizel pod klasifikací `body`. **Oprava:** `split_lines_by_style()` — rozdělení syrového bloku podle řádků, když se velikost písma mezi nimi výrazně změní (>3pt). Vyžadovalo dvě iterace (první verze srovnávala i tučnost, což zbytečně roztrhalo popisky obrázků s tučným číslem uvnitř — opraveno na porovnání jen podle velikosti).

**Rozhodnutí:** obálkové stránky (první/poslední 2) se **neignorují** už při extrakci (viz Fáze 3) — extrakce zůstává kompletní, nezávislá na tom, co s ní pozdější fáze udělají.

---

## Fáze 2 — Obsah čísla a mapování stránek (`create_toc.py`, `build_page_map.py`)

- Rekonstrukce/zobecnění parseru obsahu (tučné MeliorCE číslo = tištěná stránka, barva prvního spanu odlišuje titulek od autora).
- **Zkrácené rozsahy stránek:** `"XXXI–II"` (= XXXI až XXXII) není čistá římská číslice → přidán `RANGE_RE`, extrahuje se jen počáteční hodnota.
- **Sloučené položky obsahu:** když je pod jedním číslem stránky víc položek oddělených středníkem, `create_toc.py` je rozdělí na samostatné záznamy se stejnou stránkou (řešení viz Fáze 4).
- **Mapování tištěná stránka → PDF stránka:** číslování není jedna souvislá řada — v jednom čísle se běžně střídá arabské (hlavní články) → římské (příloha) → arabské znovu, s jiným offsetem pokaždé. `build_page_map.py` detekuje tyhle souvislé úseky ("runs") samo a dopočítává mezery (chybějící patička na okrajové stránce) v rámci jednoho úseku — s bezpečným stropem (max. 5 stránek), aby to zůstalo opravdová korekce, ne hádání do velké mezery.
- **Case-insensitivita římských číslic:** sazečský šotek (`"CXLVIiI"` s malým `i`) — regex byl case-sensitive, opraveno na obou nezávislých místech (`build_page_map.py`, `create_toc.py`), s normalizací na velká písmena.

**Přijaté known limitations (chyby zdroje, ne pipeline):** pár tištěných čísel stránek je ve zdrojovém PDF prokazatelně špatně (editorský copy-paste z minulého čísla, nebo špatné zarovnání) — tyhle záznamy se bezpečně přeskočí s varováním, místo aby se hádalo, co editor "měl na mysli".

---

## Fáze 3 — Přiřazení bloků článkům (`assign_articles.py`)

Nejsložitější a nejvíc iterovaná část pipeline.

- Základní model: článek vlastní stránky od svého začátku po (nezahrnutě) začátek dalšího článku.
- **Skutečný bug — sdílená hraniční stránka:** stránka může fyzicky obsahovat konec jednoho článku *a* začátek druhého (nalezeno při zkoumání mismatche: obsah o mykorhize omylem skončil u úplně jiného článku o epigenetice, protože sdíleli stránku). Poslední (3.) sloupec často běží nezávisle na zbytku stránky, takže "vše před titulkem v pořadí čtení" nestačí — **oprava:** `find_split_y()`, řez podle **Y-souřadnice** napříč všemi sloupci (ne podle pořadí v seznamu), použije pozici bloku s autorem/titulkem nového článku.
- **Skupiny záznamů se stejnou stránkou** (výsledek sloučených TOC záznamů z Fáze 2): zobecnění na "N položek sdílí jednu stránku", ne jen 2. Rozdělení uvnitř skupiny primárně **počítáním nadpisů** (robustnější než text — recenze mívají v obsahu zkrácený titulek, ale v textu úplně jiný nadpis s celým jménem autora knihy), záložně **fuzzy textovou shodou** (přesná shoda → podřetězec po odstranění běžných prefixů → nejdelší společný podřetězec ≥15 znaků — řeší i případ, kdy jeden záznam v obsahu legitimně pokrývá dva nadpisy v textu, např. "Fenomén Velká kotlina" se dvěma pohledy dvou autorů).
- **Bezpečnostní síť proti tiché ztrátě dat:** když se nenajde ani jedna shoda v celé skupině, nerozděluje se vůbec — sloučí se zpátky do jednoho záznamu se spojeným titulkem, místo aby obsah beze stopy zmizel (reálně nalezená chyba, opravena po auditu čísla 2023/1).
- **Skutečný bug — `find_split_y` nikdy neuspěl u většiny zadních článků:** hledal jen `type=="title"` (vyhrazeno pro hlavní články), ne `"heading"` (běžné pro recenze/nekrology/kratší útvary), a používal přesnou textovou shodu místo fuzzy. Po opravě kleslo množství "neověřených" hraničních stránek na testovacím vzorku z 18 na 3 článků (z 37) — reálné zlepšení přesnosti napříč celým archivem, ne jen kosmetika.
- **`quality_flags`** — nové pole u každého článku, explicitně přiznávající, kde se pipeline musela spolehnout na fallback/hádání: `boundary_page_unverified`, `heading_not_found_empty`, `absorbed_unmatched_siblings` (+ `absorbed_titles`), `merged_fallback` (+ `original_titles`).

**Výsledek auditu quality_flags přes celý archiv** (2 941 článků): 8,8 % článků má aspoň jednu vlajku, ale 70 % z nich patří do jediné dobře známé kategorie (administrativní zadní strana čísla — Aktuality, Kontaktní adresy, Kalendář biologa, Editorial...). Z **jedinečných** případů se ověřilo, že jde skoro vždy o stejnou kategorii jen s konkrétním jménem; jediný skutečně odlišný nález (dvě sloučené recenze) se ukázal jako reálná editorská chyba ve zdroji (prohozené pořadí položek), ne chyba pipeline — a obsah zůstal bezpečně zachovaný, jen pod jiným titulkem.

---

## Fáze 4 — Chunking pro RAG (`build_chunks.py`)

- **Rozhodnutí:** chunkovat z `full_text` (plynulý text článku), ne z jednotlivých bloků — bloky mají příliš nesourodou délku (od jednoho znaku po stovky) pro embedding.
- `full_text` cíleně **neobsahuje** popisky/anotace — obrázek uprostřed sloupce jinak trhá věty napůl (viz sdílené sloupce v Fázi 3). Popisky mají vlastní pole `captions_text`.
- Cílová velikost chunku ~1200 znaků, s překryvem (posléze zmenšeno na `window=1` chunk okolo zásahu při retrievalu, viz Fáze 6 — spíš než zvětšovat překryv chunků samotných).
- **Rozhodnutí (promyšlené, ne implicitní):** obálkové stránky (přední/zadní 2) se vyřazují **až tady**, na úrovni „co jde do RAG“ — ne dřív v extrakci ani v přiřazení článků. Vyžadovalo protáhnout čísla stránek z `assign_articles.py` až sem (`full_text_paragraphs` s polem `page` u každého odstavce místo jednoho stringu) — nic se tím neztrácí, jen přibylo dat.
- **Skutečný bug:** `split_oversized_paragraph()` neuměl rozdělit odstavec bez jediné interpunkce (tabulka/výčet dat) — vracel ho vcelku, i 4× delší než limit. Fallback na sekání po slovech.
- **Bezpečnostní limit tokenů:** místo preventivního zmenšení všech chunků kvůli vzácným výjimkám (limit 512 tokenů u standardních embedding modelů) se ořezává jen konkrétní text, co limit skutečně přesáhne (`enforce_max_length()` v `embed.py`) — zjištěno explicitním rozhodnutím, ne náhodou.
- Metadata hlavička (`"Časopis: Živa\nRočník:...\nČlánek:...\nAutoři:...\nText:..."`) — tzv. contextual chunking, ověřeno experimentem, že "Časopis: Živa" (konstantní přes celý korpus) zbytečně ředí embedding, ale ponecháno beze změny na žádost (jednoduchost > mikrooptimalizace).

---

## Fáze 5 — Embedding

**Rozhodnutí (s odůvodněním, ne default):**
- Lokální model přes `sentence-transformers` (ne API) — jazykově čistě český obsah, jednorázový/opakovatelný běh, edukační zájem vidět mechaniku pod pokličkou. API jako budoucí rozšíření (architektura embeddingu jako vyměnitelná funkce).
- `normalize_embeddings=True`, float32 (float16 na CPU bez výhody), batch size neřešeno jako riziko (embedding modely jsou o řád menší než LLM).
- Dimenze vektoru záměrně neřešena jako kritérium výběru u malého datasetu — model si nese dimenzi, není to nezávislý knoflík.
- Porovnání kandidátů (`compare_models.py`) na vlastních testovacích dotazech se známou odpovědí, ne obecný benchmark.
- `embed.py`: `_load_model()` zkouší nejdřív `local_files_only=True` (žádná síť), spadá na online stažení jen když model ještě není v cache — řeší zbytečné síťové zpomalení při opakovaném spouštění.
- **Checkpointing** (`build_embeddings.py`): průběžné ukládání přes `np.memmap` po dávkách, ať jde velký běh (hodiny na CPU) bezpečně přerušit/navázat. Ověřeno reálným přerušením uprostřed běhu.
- **Statistický sanity check** (`check_embeddings.py`): místo jednoho anekdotického páru — stovky náhodně losovaných párů (stejný/různý článek), měří se **relativní** oddělení (% párů ze stejného článku nad mediánem různých), ne absolutní čísla podobnosti (embedding prostory bývají anizotropní, absolutní čísla zavádí). Výsledek na plném archivu: 98,8 % separace.

---

## Fáze 6 — Vektorové úložiště a retrieval (Chroma)

- **Rozhodnutí:** Chroma místo FAISS — jednodušší kód (vestavěná metadata a filtrování) za cenu aproximativního (ne exaktního) vyhledávání — na velikosti tohoto datasetu prakticky nerozlišitelné.
- Vektory se do Chromy dodávají **hotové** (ne přes její vlastní `embedding_function`) — udržuje embedding jako nezávisle vyměnitelnou komponentu.
- **Skutečný bug:** `collection.add()` na existující ID mlčky nic nepřepíše (žádná chyba, žádný efekt) — po přegenerování korpusu se tak vracely staré výsledky. Oprava: `upsert()` + `--overwrite` flag pro kompletní přestavbu při změně struktury dat.
- **Retrieval strategie** (`assemble_context.py`): top_N=10 kandidátů, seskupení podle článku; článek s ≥3 zásahy v top_10 se povýší na **celý článek** z korpusu (silný signál, že dotaz cílí na něj celý); ostatní dostanou window expansion (±1 chunk okolo zásahu, sloučení překrývajících se oken) — vědomě zvolený kompromis mezi "jen krátký chunk" a "celý článek pro všechno".
- Citace u každého bloku kontextu (časopis/ročník/číslo/článek/autoři/strany) — rozhodnuto jako "must" hned na začátku diskuze o retrievalu.

---

## Diagnostické nástroje (vytvořené průběžně, ne až na konci)

- `diag_groups.py` — vypíše seřazený obsah vícepoložkové skupiny s vyznačenými nadpisy a skutečným výsledkem přiřazení.
- `diag_kontaktni_adresy.py`, `diag_page_resolve.py`, `diag_page_spans.py` — cílené diagnostiky pro konkrétní opakující se selhání.
- `summarize_quality_flags.py` — souhrn `quality_flags` napříč celým korpusem, rozpad podle typu i čísla, filtr na jedinečné (neopakující se) tituly pro odlišení známého šumu od nového problému.

---

## Přehled klíčových rozhodnutí (proč, ne jen co)

| Rozhodnutí | Alternativa zvážená | Proč tahle volba |
|---|---|---|
| Obálka se ořezává až při chunkování | Ořezat hned při extrakci | Zachovat syrovou digitalizaci kompletní a znovupoužitelnou |
| Chunking z `full_text`, ne z bloků | Chunkovat přímo bloky | Konzistentní velikost pro embedding |
| Lokální embedding model | Rovnou API | Čistě český text, edukační zájem, nulové náklady na experimenty |
| Chroma | FAISS + vlastní metadata store | Jednoduchost kódu > plná kontrola |
| `window=1`, `promote_threshold=3` | Vždy jen chunk / vždy celý článek | Kompromis: krátké chunky nestačí, celé články jsou plýtvání, když dotaz necílí na 1 zdroj |
| `quality_flags` jako metadata, ne tichá oprava | Slepě hádat/spojovat | Transparentnost nad falešnou jistotou |

---

## Otevřené konce / co zbývá

- Sestavení promptu pro LLM z `assemble_context.py` výstupu (systémová instrukce + kontext + otázka).
- Případný reranking (cross-encoder) jako druhá fáze, pokud window/promote heuristika v praxi nestačí.
- Případ ojedinělé chyby "prohozené pořadí recenzí" (2019/6) zůstává vědomě neopravený (viz known limitations).

---

## Fáze 7 — Generování odpovědi (`answer.py`)

Poslední otevřený konec z předchozí verze. Systémová instrukce ležela
v souboru, který nikdo nečetl, a pipeline končila u vypsaného kontextu.

- Prompt má tři části a každá je jinde schválně: **systémová instrukce**
  v profilu zdroje (pravidla citování i jazyk odpovědi patří ke korpusu,
  ne k pipeline), **kontext** jako očíslované Zdroje s plnou citací
  a značkou CELÝ ČLÁNEK / výřez, a **dotaz** až úplně na konci, aby
  stabilní část promptu šla cachovat.
- Značka CELÝ ČLÁNEK / výřez není kosmetika — systémová instrukce se na ni
  odvolává. U výřezu smí model přiznat, že úryvek začíná uprostřed
  myšlenky; u celého článku by taková výhrada byla falešná opatrnost.
- `--dry-run` vypíše hotový prompt a nic neposílá do API. Ladit retrieval
  se dá zadarmo a bez klíče.

---

## Fáze 8 — Refaktor na profily zdroje

**Problém:** pipeline byla použitelná na jeden konkrétní časopis. Jména
fontů, velikosti písma, tvar patičky, stránka s obsahem a název časopisu
byly zadrátované na šesti různých místech napříč pěti skripty.

**Řešení:** všechno tohle je teď **data** v jednom `SourceProfile`, ne `if`
uprostřed parseru. Přidat časopis znamená napsat jeden profil.

- **Adaptivní klasifikace** (`profiles/adaptive.py`) — odpověď na to, že
  u nového časopisu nikdo nezná jména fontů. Referenční velikost písma se
  odvodí z dokumentu (nejobjemnější kombinace rodina+velikost podle počtu
  ZNAKŮ, ne podle počtu bloků — titulků je na stránce hodně kusů, ale málo
  textu) a všechna pravidla jsou relativní k ní. Přenositelné bez kalibrace,
  za cenu toho, že nepozná rozdíly, které nejsou typografické (řádek
  s autorem má skoro stejnou velikost jako text).
- **Dvě strategie detekce patičky**, protože jedna nestačí: `keyword`
  podle obsahu (Živa má v patičce vlastní název a doménu) a `position`
  podle polohy na stránce (MagPi má v patičce jen číslo).
- **`tools/inspect_fonts.py`** — kalibrace nového profilu. Vypíše histogram
  sazby s ukázkami, návrh pravidel a kandidáty na patičku. Kontrola: na
  vzorovém čísle Živy z něj vypadne přesně to, co bylo původně odvozené
  ručně.

**Skutečný bug nalezený při psaní testů:** `label_to_int()` porovnávala
římské číslice case-insensitive, ale `roman_to_int()` uměla jen velká
písmena — `"CXLVIiI"` (sazečský šotek) tedy tiše vracelo 146 místo 148.
V pipeline se to neprojevilo, protože `extract_label()` token normalizuje
dřív, ale diagnostické nástroje volají `label_to_int()` napřímo. Normalizace
patří dovnitř. Druhý nález: `ROMAN_RE` bez horní meze délky prohlásí za
římskou číslici jakýkoli dost dlouhý shluk písmen I/V/X/L/C/D/M — u detekce
patičky podle polohy se to reálně stane.

**Reprodukovatelnost:**

- Zamčené verze v `requirements.txt`. PyMuPDF mezi verzemi mění, jak dělí
  stránku na bloky, a to je vstup úplně všeho ostatního.
- **Zlatý test** porovnává SHA-256 otisk výstupu každé fáze. Fixture
  neobsahuje obsah časopisu, jen otisky a počty — reaguje stejně citlivě
  jako porovnání textu, ale nezveřejňuje ani písmeno. Bez PDF se přeskočí.
- Celý refaktor je ověřený regresí: `run_all` nad vzorovým číslem dává
  `blocks`, `toc`, `page_map`, `articles`, `corpus` i `chunks` **bajtově
  shodné** s výstupem před refaktorem.

**Oprava, která byla nejvíc vidět:** pipeline padala na `UnicodeEncodeError`
na prvním printu na každé konzoli s kódováním cp1252 (výchozí stav na
Windows). Fungovala jen ve Spyderu, který má stdout v UTF-8 — tedy přesně
ten druh chyby, kterou autor nikdy nevidí a každý, kdo si projekt naklonuje,
do ní narazí do dvou sekund.

**Kolik adaptivní profil stojí přesnosti** (měřeno na vzorovém čísle Živy,
kde adaptivní profil neví o Živě vůbec nic — ani jméno fontu, ani kde je
obsah čísla, ani co stojí v patičce): 36 článků proti 37, všech 36 titulků
shodných s ručním profilem, 19 z nich má bajtově shodný i celý text.
Chybějící článek se ztratil na nenamapovaném tištěném čísle stránky, zbylé
rozdíly jsou hranice odstavců, ne ztracený obsah. Počet článků označených
`quality_flags` je u obou profilů stejný (3).

**Bug nalezený tímhle měřením:** první verze adaptivního profilu dědila
výchozí `footer_detection="keyword"` s prázdným seznamem klíčových slov.
Detekce patičky pak nenašla jediné tištěné číslo stránky, obsah čísla se
neměl na co namapovat a pipeline **tiše vyrobila nula článků** — přesně ten
typ selhání, který se bez porovnání se známým výsledkem nepozná, protože
nic nespadne. Ošetřeno dvakrát: adaptivní profil používá `"position"`,
a `is_footer_block()` u strategie `"keyword"` bez klíčových slov vrací
`False` místo toho, aby za patičku prohlásila každý malý text na stránce.
