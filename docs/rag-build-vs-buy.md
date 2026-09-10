# RAG: build vs. buy v rámci větší organizace — tři otázky a odpovědi

Tři navazující otázky a odpovědi z konverzace o tom, co si z tříddenní práce na home-made RAG komponentách odnést pro někoho, kdo podobné věci řeší/manageuje/orchestruje ve větší organizaci/korporátu. Přesné citace, beze změny.

---

## Otázka 1

**Q:** Tři dny jsem strávil vytvářením home-made kompenent RAG. Co si z toho můžu vzít pro někoho, kdo by takové věci měl řešit/manageovat/orchestrovat v rámci větší organizace/korporátu?

**A:**

Dobrá otázka na destilaci — pojďme to rozebrat na věci, co jsou specifické pro *tenhle* projekt vs. co je obecně přenositelné pro řízení podobné práce ve větší organizaci.

### 1. Nejdražší část práce je neviditelná v demo prezentacích

Skoro celý čas šel do "nudné" části — parsování PDF, hranice článků, oprava zalomených slov. Embedding a vektorové vyhledávání (to, co vypadá "zajímavě" a co prodávají vendor demos) zabraly zlomek času. **Manažerský důsledek:** když někdo odhaduje timeline RAG projektu podle toho, jak rychle se dá "zapojit LangChain a Pinecone", odhaduje špatně — realistický odhad musí počítat s tím, že datově specifická práce se nedá dobře odhadnout předem, protože se objevuje až při kontaktu s reálnými daty.

### 2. Tiché chyby jsou nebezpečnější než pády — a to je manažerský problém, ne jen technický

Skoro žádný z bugů, co jsme našli (sdílené stránky, sloučený nesourodý text, `find_split_y` co nikdy nefungoval), nespadl s chybou. Kód běžel, tvářil se v pořádku, výsledek byl tiše špatně. **Přenositelná lekce:** "proběhlo to bez chyby" není přijatelné kritérium hotovosti u datové pipeline. Potřebujete explicitní, měřitelnou vrstvu důvěry (u nás `quality_flags` + souhrnný audit) jako **první-třídní deliverable**, ne dodatečnou kontrolu, kterou uděláte, "když zbude čas" — protože nezbyde.

### 3. Cena iterace je sama o sobě designové rozhodnutí

Každá oprava v extrakci znamenala přegenerovat celý archiv — hodiny výpočtu. Museli jsme si proto řešit checkpointing, `upsert` místo úplného přestavění indexu, testování na jednom čísle před spuštěním na 75. **Pro řízení týmu:** ptejte se dopředu "kolik stojí jedna iterace opravy", ne až když vám někdo řekne, že přegenerování zabere noc. Tohle přímo určuje, jak rychle se dá tým učit z chyb.

### 4. Nástroje na diagnostiku nejsou overhead — jsou součástí práce

Bokem vzniklo přes půl tuctu malých diagnostických skriptů (dump konkrétní stránky, souhrn nejistoty napříč archivem...). Nebyly plánované dopředu, vznikaly, jak se objevovaly konkrétní záhady. **Lekce pro plánování:** pokud v odhadu na "postavit datovou pipeline" není budget na "postavit nástroje na to, abych zjistil/a, proč pipeline dělá, co dělá", odhad je neúplný.

### 5. Cena systému určuje hlavně vaše vlastní architektura, ne ceník vendora

Když jsme počítali náklady, ukázalo se, že rozdíl mezi "levným" a "drahým" hostingem byl u nás zanedbatelný — to, co skutečně hýbe cenou, jsou **vaše vlastní rozhodnutí** (kolik kontextu posíláte do LLM, `promote_threshold`, velikost okna). **Pro někoho, kdo řídí rozpočet:** než vyjednávat slevu u vendora, podívejte se, jestli vlastní architektura neposílá 5× víc kontextu, než je potřeba.

### 6. "Build vs. buy" není binární — je to otázka, kde přesně leží vaše diferenciace

Zkusili jsme si nasimulovat, jak by se stejná data zpracovala obecným nástrojem (NotebookLM-styl) — a ukázalo se, že pro **generický** dotaz to funguje obstojně, ale pro cokoliv vyžadujícího přesnou citaci/hranice článků selhává úplně. **Přenositelně:** obecné nástroje jsou optimalizované na generický případ; pokud je vaše skutečná hodnota v přesnosti nad specifickými/nestandardními daty, "jen to nahrát do hotového nástroje" vás nedovede tam, kam potřebujete — ale zjistit tohle *předem*, ne po nasazení, vyžaduje malý prototyp/pilot, ne jen čtení marketingových materiálů.

### 7. Rozhodnutí bez zdůvodnění se špatně dědí

Celou dobu jsem vás nutil k rozhodnutím s explicitním "proč", ne jen "co" (embedding model, Chroma vs. FAISS, velikost chunků). Až budete psát dokumentaci/handoff pro tým, tohle je přesně to, co přežije — "proč jsme to udělali takhle" se ztrácí jako první, a je to přesně to, co potřebuje příští člověk, když se rozhodne to změnit.

---

## Otázka 2

**Q:** Takže opravdu existují situace, kde build může být jednoznačně lepší než buy?

**A:**

Ano, ale trochu jinak, než se to obvykle rámuje — a je to vlastně vidět přímo na tom, co jsme dnes udělali.

### Klíčový postřeh: "build vs. buy" není jedna otázka, je to otázka **po vrstvách**

Podívejte se, co jsme v tomhle projektu skutečně udělali:

| vrstva | co jsme udělali | proč |
|---|---|---|
| Embedding model | **koupili/použili** hotový (`multilingual-e5-base`) | commodity — nikdo nezíská výhodu tím, že si postaví vlastní transformer |
| Vektorové vyhledávání | **koupili/použili** hotové (Chroma) | commodity — HNSW index si nikdo nepíše sám |
| PDF parsování Živy, hranice článků, oprava sazečských chyb | **postavili** od nuly | tady žádný "buy" ani neexistuje |

Takže to, co jsme dnes nazvali "home-made RAG", je ve skutečnosti **z 80 % buy + z 20 % build** — jen ten build je ta viditelná, časově náročná část. Klíčová chyba v uvažování je předstírat, že "build vs. buy" je jedno rozhodnutí pro celý systém.

### Kdy je build na té zbývající vrstvě jednoznačně správně

1. **Když generický nástroj daný problém neřeší vůbec, ne jen hůř.** Naše sdílené hraniční stránky, editorské typo (`CXLVIiI`), zalomení slov — žádný vendor nepostaví obecné řešení pro "specifika InDesign exportu 70 let starého českého časopisu". Tady není alternativa "buy" na stole — je to buď postavit si to, nebo se problém neřeší.

2. **Když je přesnost/dohledatelnost sama o sobě produkt, ne detail.** Pokud je hodnota vašeho systému "vím přesně, který autor a která strana", ne "nějaká odpověď z nějakého zdroje", pak "koupit obecný nástroj, co tohle nedělá" není o nic levnější varianta — je to jiný produkt, který neřeší vaše zadání.

3. **Když potřebujete auditovatelnost, kterou black-box nástroj nedá.** `quality_flags` fungují, protože víme přesně, co se stalo uvnitř. To se nedá dostat z hostovaného "nahraj PDF, zeptej se" nástroje — nemáte přístup dovnitř.

### Ale — a tohle je důležité pro poctivost odpovědi

"Build je správně" **neznamená "je to zadarmo"**. Pořád to nese náklad — na každou opravu jsme museli přegenerovat celý archiv, udržovat diagnostiku, testovat regresi. Rozhodnutí stavět tuhle vrstvu je rozhodnutí **přijmout tenhle údržbový náklad natrvalo**, ne jednorázový výdaj. Manažersky správná otázka není "umíme si to postavit", ale "chceme tenhle typ údržby vlastnit dlouhodobě, nebo radši zaplatit někomu, kdo to (hůř, ale) udělá za nás".

**Shrnutí:** build je jednoznačně lepší přesně tam, kde se vaše diferenciace potkává s idiosynkrazií vašich dat — a to je užší množina případů, než "build je obecně lepší". Většina RAG stacku (embedding, vektorové úložiště, LLM) zůstává komoditou, kde buy dává smysl skoro vždy.

---

## Otázka 3

**Q:** Takže po této zkušenosti vím, že RAG ve smyslu buy nemusí být jen celý blackbox (typu notebookLM) ale dá se to kupovat po částech ...

**A:**

Přesně tak — a je to vlastně přirozené pokračování té vrstvové úvahy z minula. Škála **není** "black box vs. postav si všechno", je to spektrum, kde si vybíráte **po vrstvách**, kterou koupíte hotovou a kterou postavíte.

### Celý RAG stack rozepsaný na vrstvy — a co se dá koupit na každé

| vrstva | co se dá koupit samostatně | co jsme udělali my |
|---|---|---|
| **Parsování/ingest dokumentů** | Unstructured.io, LlamaParse, Google Document AI Layout Parser — služby specificky na "PDF → čisté chunky" | postaveno na míru (`extract_blocks.py`) |
| **Chunking strategie** | často součást parsovací služby výše, nebo frameworky (LangChain text splitters) | postaveno na míru (article-aware) |
| **Embedding** | OpenAI/Cohere/Voyage API | koupeno-ekvivalent, jen lokálně hostěný model |
| **Vektorové úložiště** | Pinecone/Weaviate/Qdrant Cloud | koupeno-ekvivalent (Chroma, lokálně) |
| **Reranking** | Cohere Rerank API | neřešeno zatím |
| **Orchestrace/"lepidlo"** | LangChain, LlamaIndex, Haystack | postaveno na míru (`assemble_context.py`) |
| **Generování odpovědi** | Claude/GPT/Gemini API | koupeno |
| **Celý balík najednou** | NotebookLM, Glean, ChatGPT s nahráním souboru | (to jsme si simulovali jako srovnání) |

Všimněte si — **dokonce i tu vrstvu, co jsme dělali nejvíc ručně** (parsování PDF), lze dnes koupit jako samostatnou službu (Unstructured.io, LlamaParse). Otázka není "existuje to koupené", ale "řeší to *moje* konkrétní idiosynkrazie" — a to nezjistíte z marketingové stránky, zjistíte to pilotem na reálných datech (přesně to, co jsme si nakousli tím naivním srovnáním).

### Proč tohle je důležitější postřeh, než zní

Manažersky to znamená, že rozhodnutí "build vs. buy" se **neděje jednou na začátku projektu** — děje se **na každém švu** mezi vrstvami, a to průběžně, jak se dozvídáte víc o datech. Důsledky:

1. **Švy mezi koupenými komponentami jsou pořád vaše práce.** I kdybyste koupil/a embedding, vektor DB i LLM API zvlášť, to "lepidlo" mezi nimi (u nás `assemble_context.py` — kdy povýšit na celý článek, jak sestavit prompt) je pořád něco, co musíte navrhnout a vlastnit vy. Composabilita neznamená nulovou integrační práci.

2. **Nakupovat po vrstvách vyžaduje, abyste znal/a svůj vlastní stack vrstvu po vrstvě** — přesně tu tabulku výš. Bez ní se snadno stane, že se "buduje", protože se nikdo nezeptal, jestli konkrétní vrstva už není komodita.

3. **Cena flexibility je závislost na víc švech, ne na jednom vendorovi.** Fully black-box řešení (NotebookLM) má jeden bod selhání/vyjednávání. Composable stack jich má pět-šest — každý s vlastním rizikem výpadku/změny cen/API breaking change.

**Shrnutí:** ano — a tohle je asi nejpřenositelnější věc z celého projektu pro řízení podobné práce dál: než se ptát "build, nebo buy", rozepište si stack na vrstvy jako tu tabulku, a ptejte se to zvlášť pro každou.
