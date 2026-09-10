# RAG — přehled rozhodnutí a parametrů podle vrstvy

Referenční seznam otázek, které se řeší (explicitně, nebo tiše defaultní hodnotou) v každé vrstvě RAG systému. Pro každý bod: co se rozhoduje, ne jak přesně rozhodnout — to je vždy na konkrétním případu.

---

## 1. Ingest / parsování dokumentů

- Jaké formáty musí systém zvládnout (PDF, DOCX, HTML, obrázky/OCR, tabulky)?
- Je potřeba OCR? Jak dobrá musí být jeho přesnost?
- Jak se zachází s layoutem — víceloupcový text, tabulky, hlavičky/patičky?
- Jak se zachovává struktura (nadpisy, hierarchie sekcí)?
- Jak se extrahují metadata (autor, datum, název, číslo stránky)?
- Jak se odstraňuje/označuje boilerplate (opakující se hlavičky, patičky, čísla stránek)?
- Jak se zachází s obrázky, popisky, anotacemi — jsou součástí textu, nebo oddělené?
- Podpora více jazyků v jednom dokumentu?
- Build vs. buy: vlastní parser, nebo služba (Unstructured.io, LlamaParse, Document AI Layout Parser)?

## 2. Chunking

- Cílová velikost chunku (tokeny/znaky) — kompromis granularita vs. konzistence pro embedding.
- Metoda: pevná velikost, rekurzivní/hierarchická, sémantická (podle odstavců/vět), podle struktury dokumentu (podle nadpisů/sekcí)?
- Překryv mezi chunky — kolik, a jak (posledních N znaků, celé odstavce)?
- Co s tabulkami, kódem, obrázky/popisky — vlastní chunky, nebo součást okolního textu?
- Kontextová hlavička u chunku (metadata vepsaná do embedovaného textu) — ano/ne, co všechno?
- Jaká metadata se ukládají ke chunku (zdroj, stránka, sekce, datum, autor, přístupová práva)?
- Hierarchické/"parent-child" chunky (malý chunk pro vyhledání, větší pro kontext)?
- Co dělat s chunkem, co přesáhne tokenový limit embedding modelu (zahodit/ořezat/rozdělit)?
- Jak nakládat s obálkovými/nevztaženými částmi dokumentu (titulní strana, rejstřík, reklama)?

## 3. Embedding

- Model: lokální (open-weight) vs. hostovaný přes API?
- Jazyková způsobilost — monolingvální vs. multilingvální model, jak dobře pokrývá cílový jazyk?
- Dimenze vektoru — kompromis přesnost/náklady na úložiště a rychlost vyhledávání.
- Normalizace vektorů (jednotková délka) — ovlivňuje, jaká metrika podobnosti dává smysl.
- Asymetrické embeddingy (rozdílný prefix/instrukce pro dotaz vs. pro dokument)?
- Přesnost čísel (float32/float16/kvantizace) — úložiště vs. kvalita.
- Batch size / propustnost při zpracování velkého korpusu.
- Co se stane při změně modelu — nutnost přeindexovat celý korpus?
- Cena za token (pokud API), latence, rate limity.
- Checkpointing/odolnost dlouhého indexačního běhu proti přerušení.

## 4. Vektorové úložiště

- Hostovaná služba vs. self-hosted (knihovna/kontejner)?
- Typ indexu: exaktní (flat) vs. aproximativní (HNSW, IVF, PQ) — kompromis přesnost/rychlost/škálovatelnost.
- Podpora metadata filtrů (a jestli se filtruje před, nebo až po vektorovém vyhledávání)?
- Hybridní vyhledávání (vektor + klíčová slova/BM25) — ano/ne, jak se váží?
- Multi-tenancy / izolace dat mezi uživateli nebo odděleními?
- Strategie aktualizace: `upsert` po jednotlivých záznamech, vs. kompletní znovu-sestavení indexu?
- Škálovatelnost (sharding, replikace) při růstu objemu dat.
- Zálohování/perzistence dat.
- Cenový model (úložiště podle GB, poplatky za čtení/zápis, paušální minima).

## 5. Strategie vyhledávání (retrieval)

- `top_k` — kolik kandidátů se vytáhne před dalším zpracováním?
- Metrika podobnosti (kosinová, skalární součin, eukleidovská vzdálenost)?
- Metadata filtry v dotazu (rok, autor, typ dokumentu, přístupová práva)?
- Transformace dotazu před vyhledáním (rozšíření dotazu, HyDE, více variant dotazu)?
- Reranking — druhá, přesnější fáze (cross-encoder) nad prvotními kandidáty; ano/ne, jaký model?
- Diverzita výsledků (např. MMR) — vyhýbat se vracení skoro identických chunků?
- Agregace na úrovni dokumentu — kdy vrátit jen chunk, kdy celý dokument/sekci?
- Kolik okolního kontextu přibalit ke každému zásahu (okno sousedních chunků)?
- Deduplikace překrývajících se/opakujících se výsledků?
- Váha mezi vektorovým a klíčovým vyhledáváním u hybridního přístupu?

## 6. Sestavení promptu / kontextu pro LLM

- Systémový prompt — pravidla groundingu, formát citací, tón, jazyk odpovědi.
- Kolik tokenů kontextu poslat (rozpočet vstupu vs. místo na odpověď)?
- Formát citací — inline odkaz na zdroj, seznam na konci, obojí?
- Jak nakládat s kontextem, co otázku nepokrývá vůbec/jen částečně?
- Pořadí chunků v promptu (nejrelevantnější první/poslední — modely mívají slabší pozornost uprostřed dlouhého kontextu)?
- Co cachovat (opakující se systémový prompt/instrukce) kvůli nákladům?

## 7. Generování odpovědi (LLM)

- Který model — kompromis kvalita/cena/latence (nemusí to být nejsilnější dostupný model)?
- Teplota a další parametry vzorkování?
- Maximální délka odpovědi?
- Streamování odpovědi, nebo čekat na celek?
- Potřeba strukturovaného výstupu (JSON, nástroje/function calling)?
- Záložní model/strategie při výpadku nebo zahlcení primárního?

## 8. Evaluace a kvalita

- Jaké metriky sledovat (přesnost/úplnost retrievalu, věrnost odpovědi zdroji, relevance)?
- Jak vzniká testovací sada (ručně sestavené otázky se známou odpovědí, nebo automaticky)?
- Automatizované vs. lidské hodnocení odpovědí?
- Průběžné monitorování v provozu (drift kvality, zpětná vazba uživatelů)?
- Jak se měří a hlásí důvěryhodnost/nejistota u zpracovaných dat (viz `quality_flags` v tomhle projektu)?

## 9. Provoz a infrastruktura

- Požadavky na latenci (real-time chat vs. dávkové zpracování)?
- Sledování a rozpočet nákladů (kde přesně utrácíte — embedding, úložiště, generování)?
- Frekvence a proces reindexace při aktualizaci zdrojových dat?
- Verzování — co se stane se starým indexem při změně embedding modelu?
- Bezpečnost/soukromí (citlivá data, řízení přístupu na úrovni dokumentu/chunku)?
- Požadavky na rezidenci dat / compliance?

## 10. Guardrails a dohledatelnost

- Detekce/omezení halucinací (odpovědi mimo dodaný kontext)?
- Ověřování, že citace v odpovědi skutečně odpovídají použitým zdrojům?
- Filtrování nevhodného obsahu?
- Jak systém reaguje, když poctivá odpověď je "nevím" — je to podporované, nebo se model tlačí k odpovědi za každou cenu?

---

## Průřezová otázka nad celým seznamem

Pro každou vrstvu: **build, nebo buy** — a pokud buy, od koho a za jakou cenu/závislost? (Viz předchozí diskuze — tohle se řeší zvlášť pro každou vrstvu, ne jednou za celý systém.)
