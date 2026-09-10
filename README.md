# magrag — archiv časopisu v PDF → korpus → RAG

Pipeline, která z ročníků časopisu v PDF udělá strukturovaný korpus článků
s metadaty a citovatelnými čísly stránek, zaembedduje ho a odpovídá nad ním
na otázky s odkazy na zdroj.

Vzniklo to nad archivem přírodovědného měsíčníku **Živa** (75 čísel, 2 941
článků, ~35 000 embedding chunků) a je to napsané tak, aby se to dalo
přenést na jiný časopis výměnou jednoho souboru — viz [Profil
zdroje](#profil-zdroje).

> **Práva.** V repozitáři není žádný obsah časopisu — ani PDF, ani z nich
> vytažený text. Živa je autorsky chráněná (Academia / AV ČR), takže archiv
> nejde zveřejnit; kód ano. Pro veřejné demo je připravený profil pro
> [The MagPi](https://magpi.raspberrypi.com/issues), který vychází pod
> licencí CC BY-NC-SA.

---

## Proč to není `PyPDFLoader` + `RecursiveCharacterTextSplitter`

Protože to na časopisecké sazbě nefunguje. Zajímavá část tohoto projektu
není embedding ani vektorové vyhledávání — ta je hotová za odpoledne.
Zajímavá je cesta od „PDF" k „článek s autorem, ročníkem a číslem stránky":

- Text nejde po stránkách, ale po **článcích**, a jeden článek se přes
  stránky přelévá. Kde končí, se nedozvíte odjinud než z obsahu čísla.
- Tištěné číslo stránky **není** číslo stránky v PDF, a rozdíl není
  konstantní: v jednom čísle se běžně střídá arabské číslování (hlavní
  články) → římské (příloha) → arabské znovu, pokaždé s jiným posunem.
- Jedna fyzická stránka nese **konec jednoho článku a začátek dalšího**.
  Řezat podle pořadí bloků nestačí — poslední sloupec často běží nezávisle
  na zbytku stránky.
- Zarovnaný text láme slova pomlčkou. Bez slepení zpátky se do korpusu
  dostane `opaková-` a `ní` jako dvě různá slova.
- Popisek obrázku uprostřed sloupce **přetne větu v půlce**, když se text
  skládá naivně shora dolů.

Každá z těch věcí je jeden konkrétní bug, který se našel až na reálných
datech. Průběh je zapsaný v [deníku projektu](docs/project-log.md) —
včetně toho, co se rozhodlo špatně a proč.

---

## Jak to funguje

```
 PDF čísla
    │
    ├─► extract_blocks    bloky textu s fontem, velikostí, polohou a typem
    │                     (title/heading/other/body/caption/annotation)
    ├─► create_toc        obsah čísla: článek → autor → tištěná stránka
    ├─► build_page_map    tištěná stránka → stránka PDF (úseky číslování)
    ├─► assign_articles   bloky → články, včetně sdílených hraničních stran
    │                     + quality_flags tam, kde si pipeline není jistá
    ├─► build_chunks      články → chunky ~1200 znaků s metadatovou hlavičkou
    ├─► build_embeddings  chunky → vektory (lokální model, s checkpointy)
    ├─► build_chroma      vektory → Chroma
    ├─► assemble_context  dotaz → zdroje s citacemi (window + promote)
    └─► answer            zdroje + otázka → odpověď LLM s odkazy na zdroj
```

Krok `extract_blocks` je jediný, který ví, jak vypadá sazba konkrétního
časopisu. Všechno za ním pracuje už jen se strukturou
`{page, type, font, bbox, text}` a je přenositelné beze změny.

---

## Rychlý start

```bash
git clone <url> && cd magrag
python -m venv .venv && . .venv/Scripts/activate   # Linux/macOS: . .venv/bin/activate
pip install -e ".[dev]"          # jen extrakce z PDF
pip install -r requirements.txt  # celá pipeline se zamčenými verzemi
```

Dejte PDF do jedné složky a pusťte celý archiv najednou:

```bash
python -m magrag.run_all --profile ziva --input ./pdf --output ./output
```

Vznikne `output/<rok>-<číslo>/{blocks,toc,page_map,articles}.json` pro každé
číslo (užitečné při ladění), plus souhrnný `output/corpus.json` a
`output/chunks.jsonl`.

Zbytek cesty k odpovědím:

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

`answer` bez `--query` běží interaktivně. S `--dry-run` vypíše hotový prompt
a nic neposílá do API — hodí se na ladění retrievalu zadarmo.

---

## Profil zdroje

Všechno, čím se jeden časopis liší od jiného, je v jednom `SourceProfile`
místo roztroušené po pěti skriptech: jména fontů a velikosti písma pro
klasifikaci bloků, tvar běžící patičky, stránka s obsahem čísla, vzor
pojmenování souborů, jazyk metadatové hlavičky a systémová instrukce.

| Profil | Sazba | Patička | Poznámka |
|---|---|---|---|
| `ziva` | ruční, absolutní velikosti | podle názvu časopisu | referenční, vyladěný na 75 čísel |
| `magpi` | adaptivní | podle polohy na stránce | otevřená licence, vhodné pro demo |
| `adaptive` | adaptivní | — | základ pro neznámý časopis |

**Adaptivní klasifikace** je odpověď na to, že u nového časopisu nikdo
nezná jména fontů. Místo absolutních hodnot si pipeline spočítá, kolik
znaků je vysázeno kterou kombinací rodiny a velikosti písma. Nejobjemnější
kombinace je z definice běžný text — a všechna pravidla jsou pak relativní
k ní („titulek je 1,7× větší než text"). Vážení podle znaků, ne podle počtu
bloků, je podstatné: titulků je na stránce hodně kusů, ale málo textu.

Kolik se za přenositelnost platí, jde změřit: stejné číslo Živy zpracované
oběma profily, kde adaptivní neví o Živě vůbec nic — ani jméno fontu, ani
kde je obsah čísla, ani co stojí v patičce.

| | ruční `ziva` | adaptivní |
|---|---|---|
| nalezené články | 37 | 37 |
| titulek obsažen v adaptivním | — | 36 z 37 |
| bajtově shodný text článku | — | 20 z 37 |
| články s `quality_flags` | 3 | 4 |

Adaptivní profil najde tytéž články. Jeho titulky jsou ale delší: obsahují
i řádek s autorem, protože obecný profil nemůže vědět, že obsah čísla
autory uvádí zvlášť a odděluje je barvou. Rozdíly v textu článků jsou
hranice odstavců, ne ztracený obsah.

### Ověřeno na druhém časopise

Profil `magpi` je otestovaný na třech reálných číslech stažených z
`magpi.raspberrypi.com/issues` (150, 152, 155; born-digital PDF, 132 stran).
Bez jediné ručně zadané hodnoty o sazbě z nich pipeline vytáhne **85 článků
a 656 chunků**, bez duplicit a bez vymyšlených autorů.

Nebylo to zadarmo. Cesta k tomu číslu ukázala pět chyb, které byly na Živě
neviditelné, a **žádná z nich nespadla** — pipeline pokaždé doběhla a vypsala
spokojený souhrn:

| Nález | Proč to Živa nikdy neukázala |
|---|---|
| obsah čísla uvádí `032`, patička `32` | Živa čísla stránek nedoplňuje nulami |
| titulky nesou řídicí znak `U+0007` | ozdobná odrážka sázená symbolovým fontem |
| číslo položky je slepené s tabulátorem a odrážkou | totéž |
| obsah je rozložený přes tři stránky | Živa má obsah vždy na jedné |
| na stránce s obsahem jsou troje různá čísla | Živa ozdobné upoutávky nemá |

Poslední z nich je nejzajímavější. Vedle skutečných čísel stránek stojí
v obsahu MagPi ozdobné upoutávky (velké bílé číslo s krátkým popiskem)
a číslo samotné stránky s obsahem v patičce. Napevno zadat, které je které,
nejde: MagPi mezi čísly 150 a 152 předělal grafiku včetně fontů, velikostí
i formátu čísel. Řeší se to stejnou úvahou jako klasifikace bloků, jen
o patro výš — hledá se **nejčastější dvojice „styl čísla + styl titulku
hned za ním"**, protože obsah je seznam a ta dvojice se v něm opakuje
u každé položky.

### Přidání nového časopisu

```bash
python -m tools.inspect_fonts cesta/k/cislu.pdf
```

Vypíše histogram sazby s ukázkami textu, hotový návrh pravidel k vložení do
profilu a kandidáty na běžící patičku. Na vzorovém čísle Živy z něj vypadne
přesně to, co bylo původně odvozené ručně (`MeliorCE` @ 9 b jako běžný text,
`živa 6/2014` a `ziva.avcr.cz` jako patička).

Pak zkopírujte `magrag/profiles/magpi.py`, upravte a zaregistrujte
v `magrag/profiles/__init__.py`. Když se ruční kalibrace nevyplatí (časopis
během archivu několikrát změnil grafiku), nechte `adaptive=True`.

---

## Reprodukovatelnost

- **Zamčené verze** v `requirements.txt`. PyMuPDF mezi verzemi mění, jak
  dělí stránku na bloky, a to je vstup úplně všeho ostatního.
- **Zlatý test** (`tests/test_pipeline_golden.py`) porovnává SHA-256 otisk
  výstupu každé fáze proti zafixované hodnotě. Fixture neobsahuje obsah
  časopisu, jen otisky a počty — na změnu reaguje stejně citlivě jako
  porovnání textu, ale nezveřejňuje ani písmeno. Bez zdrojového PDF se sám
  přeskočí:

  ```bash
  pytest -q                                      # 85 testů
  MAGRAG_GOLDEN_PDF=cesta/k/cislu.pdf pytest -q  # včetně zlatého testu
  ```

- **`quality_flags`** u každého článku přiznávají, kde se pipeline musela
  spolehnout na fallback (`boundary_page_unverified`, `merged_fallback`, …).
  Souhrn přes celý archiv: `python -m tools.summarize_quality_flags`.
  Na Živě má aspoň jednu vlajku 8,8 % článků a 70 % z nich patří do jediné
  dobře známé kategorie (administrativní zadní strana čísla).

Generovaná data (`output/`, `chroma_db/`, vektory) a zdrojová PDF jsou
v `.gitignore`. Korpus Živy má 127 MB, chunky 95 MB a vektory 120 MB — přes
limit GitHubu na jeden soubor, a hlavně to tam nepatří kvůli právům.

---

## Struktura

```
magrag/            pipeline (jeden modul na fázi)
  profiles/        profily zdroje + systémové instrukce
  typography.py    klasifikace bloků, ruční i adaptivní
  console.py       UTF-8 na stdout (jinak spadne na první české hlášce)
tools/             diagnostika a kalibrace, nepatří do produkčního běhu
tests/             85 testů; zlatý test se bez PDF přeskočí
docs/              deník projektu a poznámky k rozhodování o RAG
```

## Dokumentace

- [Deník projektu](docs/project-log.md) — co se stavělo, na co se přišlo
  a proč se to rozhodlo takhle. Nejzajímavější čtení z celého repozitáře.
- [RAG: build vs. buy](docs/rag-build-vs-buy.md) — co si z toho odnést,
  když podobnou věc řídíte ve větší organizaci.
- [Přehled rozhodnutí podle vrstvy](docs/rag-decision-checklist.md) — co se
  v každé vrstvě RAG systému rozhoduje, explicitně nebo tiše defaultem.

## Licence

Kód: MIT (viz [LICENSE](LICENSE)). Obsah zpracovávaných časopisů licence
tohoto projektu **nepokrývá** — řídí se právy vydavatele.
