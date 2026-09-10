"""The last stage: let an LLM compose an answer, with citations, from the
retrieved context.

Until this module the pipeline stopped at printed context and the system
prompt sat in a file nobody read. This connects that end: it takes a
question, runs retrieval, assembles the prompt and streams the answer.

The prompt has three parts and each lives somewhere different on purpose:

* the **system prompt** lives in the source profile
  (`profiles/prompts/*.txt`), because the citation rules and the language
  of the answer belong to the corpus, not to the pipeline;
* the **context** is the numbered sources from retrieval, each with a full
  citation and a marker saying whether it is a whole article or an
  excerpt;
* the **question** comes last, so that the stable part of the prompt can
  be cached.

The model answers **only** from the supplied sources. That is not a
politeness in the instruction: without grounding in citable text, RAG is
just an expensive way to have a language model confirm your own guess.

Usage:
    python -m magrag.answer --db-dir ./chroma_db --collection ziva \\
        --model intfloat/multilingual-e5-base \\
        --chunks output/chunks.jsonl --corpus output/corpus.json \\
        --query "How is giant hogweed spreading here?"

Without `--query` it runs an interactive loop, loading the index and the
model once. With `--dry-run` it prints the finished prompt and sends
nothing anywhere - useful for tuning retrieval without spending tokens.
"""
import argparse
import json
import os
import sys
from pathlib import Path

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

# Claude Opus 5 - latest generation, 1M context. Answers over a magazine
# archive tend to be a longer piece of continuous prose rather than a
# one-liner, hence streaming and a generous max_tokens.
DEFAULT_LLM_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 8000


def format_sources(blocks) -> str:
    """Context blocks -> the numbered "Sources" for the prompt.

    The FULL ARTICLE / excerpt marker is not decoration: the system
    prompt refers to it. On an excerpt the model may admit that the
    fragment starts mid-thought; on a whole article such a caveat would
    be false caution.
    """
    parts = []
    for i, b in enumerate(blocks, 1):
        tag = "FULL ARTICLE" if b["mode"] == "full_article" else "excerpt"
        parts.append(f"--- Source {i} [{tag}] ---\n{b['citation']}\n\n{b['text']}")
    return "\n\n".join(parts)


def build_prompt(question: str, blocks) -> str:
    if not blocks:
        return (f"Question: {question}\n\n"
                "Context: (retrieval found no relevant excerpts)")
    return f"{format_sources(blocks)}\n\n---\n\nQuestion: {question}"


def stream_answer(client, prompt: str, system_prompt: str, model: str,
                  max_tokens: int) -> str:
    """Stream the answer to stdout and return it whole as a string."""
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": prompt}],
        # A safety classifier may decline the request (HTTP 200,
        # stop_reason "refusal"). The server-side fallback routes such a
        # case to another model by category instead of leaving the user
        # with an empty answer.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
        print()
        final = stream.get_final_message()

    if final.stop_reason == "refusal":
        detail = getattr(final, "stop_details", None)
        print(f"\n[the model declined the request: "
              f"{getattr(detail, 'category', 'unspecified')}]", file=sys.stderr)
    elif final.stop_reason == "max_tokens":
        print(f"\n[the answer was cut off at {max_tokens} tokens - raise "
              f"--max-tokens]", file=sys.stderr)
    return "".join(b.text for b in final.content if b.type == "text")


def make_client():
    """Build an Anthropic API client, or explain clearly what is missing."""
    try:
        import anthropic
    except ImportError:
        raise SystemExit(
            "The 'anthropic' package is missing (pip install anthropic). "
            "Retrieval can still be tried without it - use --dry-run.")
    if not (os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("[note] ANTHROPIC_API_KEY is not set; the client will try the "
              "credentials stored by 'ant auth login'.", file=sys.stderr)
    return anthropic.Anthropic()


def main():
    setup_console()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db-dir", required=True, help="directory of the Chroma database")
    ap.add_argument("--collection", default="ziva")
    ap.add_argument("--model", required=True, help="embedding model (retrieval)")
    ap.add_argument("--chunks", required=True, help="chunks.jsonl")
    ap.add_argument("--corpus", required=True, help="corpus.json")
    ap.add_argument("--query", default=None,
                    help="a one-off question; without it an interactive loop runs")
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    ap.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    ap.add_argument("--promote-threshold", type=int,
                    default=DEFAULT_PROMOTE_THRESHOLD)
    ap.add_argument("--llm-model", default=DEFAULT_LLM_MODEL,
                    help=f"model used to generate the answer "
                         f"(default {DEFAULT_LLM_MODEL})")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the finished prompt and stop; nothing is sent "
                         "to the API")
    profiles.add_profile_argument(ap)
    args = ap.parse_args()

    profile = profiles.get(args.profile)
    system_prompt = profile.system_prompt()

    print("Loading chunks, corpus and the embedding model ...")
    chunks = load_jsonl(args.chunks)
    corpus = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    body_sequences = build_body_sequences(chunks)
    corpus_index = build_corpus_index(corpus)

    coll = chromadb.PersistentClient(path=args.db_dir).get_collection(args.collection)
    embed(["warm-up query"], model_name=args.model, is_query=True,
          show_progress=False)
    print(f"Ready ({coll.count()} chunks, {len(corpus)} articles, "
          f"profile {profile.key!r}).\n")

    client = None if args.dry_run else make_client()

    def handle(question: str):
        hits = run_search(coll, args.model, question, args.top_n)
        blocks = assemble_context(hits, corpus_index, body_sequences, profile,
                                  args.window, args.promote_threshold)
        prompt = build_prompt(question, blocks)
        n_tokens = count_tokens(system_prompt) + count_tokens(prompt)
        print(f"[{len(blocks)} sources, ~{n_tokens} prompt tokens]\n")
        if args.dry_run:
            print(f"=== SYSTEM PROMPT ===\n{system_prompt}\n")
            print(f"=== PROMPT ===\n{prompt}")
            return
        stream_answer(client, prompt, system_prompt, args.llm_model,
                      args.max_tokens)

    if args.query:
        handle(args.query)
        return

    print("Interactive mode; an empty line or Ctrl+C ends the session.\n")
    while True:
        try:
            question = input("Question> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            break
        handle(question)
        print()


if __name__ == "__main__":
    main()
