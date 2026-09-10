"""Poslední fáze: z nalezeného kontextu nechej LLM složit odpověď s citacemi.

Do téhle chvíle pipeline končila u vypsaného kontextu (`assemble_context.py`)
a systémová instrukce ležela v souboru, který nikdo nečetl. Tenhle modul ten
konec dopojí: vezme dotaz, pustí retrieval, poskládá prompt a odpověď
odstreamuje.

Prompt má tři části a každá je jinde schválně:

* **systémová instrukce** - v profilu zdroje (`profiles/prompts/*.txt`),
  protože pravidla citování i jazyk odpovědi patří ke korpusu, ne k pipeline;
* **kontext** - očíslované Zdroje z retrievalu, každý s plnou citací a
  značkou, jestli jde o celý článek, nebo jen výřez;
* **dotaz** - až úplně na konci, aby stabilní část promptu šla cachovat.

Model odpovídá **jen** z dodaných Zdrojů. To není zdvořilostní fráze
v instrukci: bez uzemnění na citovatelný text je RAG jen drahý způsob, jak
si nechat od jazykového modelu potvrdit vlastní domněnku.

Použití:
    python -m magrag.answer --db-dir ./chroma_db --collection ziva \\
        --model intfloat/multilingual-e5-base \\
        --chunks output/chunks.jsonl --corpus output/corpus.json \\
        --query "Jak se u nás šíří bolševník?"

Bez `--query` se spustí interaktivní smyčka (index i model se načtou jen
jednou). S `--dry-run` se vypíše hotový prompt a nic se nikam neposílá -
hodí se na ladění retrievalu bez utrácení za tokeny.
"""
import argparse
import os
import sys

import chromadb

from magrag import profiles
from magrag.assemble_context import (
    DEFAULT_PROMOTE_THRESHOLD,
    DEFAULT_TOP_N,
    DEFAULT_WINDOW,
    assemble_context,
    build_body_sequences,
    build_corpus_index,
    count_tokens,
    load_jsonl,
    run_search,
)
from magrag.console import setup_console
from magrag.embed import embed

# Claude Opus 5 - poslední generace, 1M kontextu. Odpovědi nad archivem
# časopisu bývají delší souvislý text, ne jednořádková odpověď, proto
# se streamuje a max_tokens je štědré.
DEFAULT_LLM_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 8000


def format_sources(blocks) -> str:
    """Kontextové bloky -> očíslované "Zdroje" pro prompt.

    Značka CELÝ ČLÁNEK / výřez tu není kosmetika - systémová instrukce se
    na ni odvolává. U výřezu smí model přiznat, že úryvek začíná uprostřed
    myšlenky; u celého článku by taková výhrada byla falešná opatrnost.
    """
    parts = []
    for i, b in enumerate(blocks, 1):
        tag = "CELÝ ČLÁNEK" if b["mode"] == "full_article" else "výřez"
        parts.append(f"--- Zdroj {i} [{tag}] ---\n{b['citation']}\n\n{b['text']}")
    return "\n\n".join(parts)


def build_prompt(question: str, blocks) -> str:
    if not blocks:
        return (f"Otázka: {question}\n\n"
                "Kontext: (retrieval nenašel žádné relevantní úryvky)")
    return f"{format_sources(blocks)}\n\n---\n\nOtázka: {question}"


def stream_answer(client, prompt: str, system_prompt: str, model: str,
                  max_tokens: int) -> str:
    """Odstreamuj odpověď na stdout a vrať ji celou jako string."""
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
        # Bezpečnostní klasifikátor může požadavek odmítnout (HTTP 200,
        # stop_reason "refusal"). Serverový fallback přesměruje takový
        # případ na jiný model podle kategorie, místo aby uživatel dostal
        # prázdnou odpověď.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        print()
        final = stream.get_final_message()

    if final.stop_reason == "refusal":
        detail = getattr(final, "stop_details", None)
        print(f"\n[model požadavek odmítl: "
              f"{getattr(detail, 'category', 'neuvedeno')}]", file=sys.stderr)
    elif final.stop_reason == "max_tokens":
        print(f"\n[odpověď byla uříznuta na {max_tokens} tokenech - "
              f"zvyšte --max-tokens]", file=sys.stderr)
    return "".join(b.text for b in final.content if b.type == "text")


def make_client():
    """Vytvoř klienta Anthropic API, nebo srozumitelně vysvětli, co chybí."""
    try:
        import anthropic
    except ImportError:
        raise SystemExit(
            "Chybí balíček 'anthropic' (pip install anthropic). Retrieval "
            "jde zkoušet i bez něj - použijte --dry-run.")
    if not (os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("[pozor] není nastavený ANTHROPIC_API_KEY; klient zkusí "
              "přihlášení uložené příkazem 'ant auth login'.", file=sys.stderr)
    return anthropic.Anthropic()


def main():
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db-dir", required=True, help="složka s Chroma databází")
    ap.add_argument("--collection", default="ziva")
    ap.add_argument("--model", required=True, help="embedding model (retrieval)")
    ap.add_argument("--chunks", required=True, help="chunks.jsonl")
    ap.add_argument("--corpus", required=True, help="corpus.json")
    ap.add_argument("--query", default=None,
                    help="jednorázový dotaz; bez něj interaktivní smyčka")
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    ap.add_argument("--promote-threshold", type=int,
                    default=DEFAULT_PROMOTE_THRESHOLD)
    ap.add_argument("--llm-model", default=DEFAULT_LLM_MODEL,
                    help=f"model pro generování odpovědi (výchozí {DEFAULT_LLM_MODEL})")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--dry-run", action="store_true",
                    help="vypiš hotový prompt a skonči; nic se neposílá do API")
    profiles.add_profile_argument(ap)
    args = ap.parse_args()

    profile = profiles.get(args.profile)
    system_prompt = profile.system_prompt()

    print("Nahrávám chunky, korpus a embedding model ...")
    import json
    from pathlib import Path

    chunks = load_jsonl(args.chunks)
    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    body_sequences = build_body_sequences(chunks)
    corpus_index = build_corpus_index(corpus)

    coll = chromadb.PersistentClient(path=args.db_dir).get_collection(args.collection)
    embed(["zahřívací dotaz"], model_name=args.model, is_query=True,
          show_progress=False)
    print(f"Připraveno ({coll.count()} chunků, {len(corpus)} článků, "
          f"profil {profile.key!r}).\n")

    client = None if args.dry_run else make_client()

    def handle(question: str):
        hits = run_search(coll, args.model, question, args.top_n)
        blocks = assemble_context(hits, corpus_index, body_sequences,
                                  args.window, args.promote_threshold)
        prompt = build_prompt(question, blocks)
        n_tokens = count_tokens(system_prompt) + count_tokens(prompt)
        print(f"[{len(blocks)} zdrojů, ~{n_tokens} tokenů promptu]\n")
        if args.dry_run:
            print(f"=== SYSTÉMOVÁ INSTRUKCE ===\n{system_prompt}\n")
            print(f"=== PROMPT ===\n{prompt}")
            return
        stream_answer(client, prompt, system_prompt, args.llm_model,
                      args.max_tokens)

    if args.query:
        handle(args.query)
        return

    print("Interaktivní režim; prázdný řádek nebo Ctrl+C ukončí.\n")
    while True:
        try:
            question = input("Dotaz> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        handle(question)
        print()


if __name__ == "__main__":
    main()
