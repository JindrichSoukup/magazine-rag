"""The embedding step, behind a single function.

`embed()` hides which model or provider actually does the work. The rest
of the pipeline (storage, Chroma, retrieval) calls only this and neither
knows nor cares whether a local sentence-transformers model is inside, or
later an API.

Usage:
    from magrag.embed import embed
    vectors = embed(["some text", "more text"],
                    model_name="intfloat/multilingual-e5-base")
"""
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_BATCH_SIZE = 32

# Some models (the whole E5 family) require a "query: " / "passage: "
# prefix in front of the text; without it retrieval quality drops
# noticeably. Handling it centrally means it cannot be forgotten at one
# of the call sites.
QUERY_PREFIX_MODELS = {
    "intfloat/multilingual-e5-base": "query: ",
    "intfloat/multilingual-e5-small": "query: ",
    "intfloat/multilingual-e5-large": "query: ",
}
PASSAGE_PREFIX_MODELS = {
    "intfloat/multilingual-e5-base": "passage: ",
    "intfloat/multilingual-e5-small": "passage: ",
    "intfloat/multilingual-e5-large": "passage: ",
}


@lru_cache(maxsize=None)
def _load_model(model_name: str) -> SentenceTransformer:
    """Load the model once per process, however often embed() is called
    (typically once for the chunks and once per query).

    By default `SentenceTransformer(model_name)` always reaches out to the
    HuggingFace Hub to check for a newer revision, even when the model is
    already in the local cache - which costs seconds on every run. So try
    `local_files_only=True` first (cache only, no network) and fall back to
    a normal download only when the model isn't there at all.
    """
    print(f"  [embed] loading model {model_name} ...")
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        print(f"  [embed] {model_name!r} not in the local cache - downloading ...")
        return SentenceTransformer(model_name)


def enforce_max_length(texts, model: SentenceTransformer, model_name: str,
                       sample_size: int = 5):
    """Truncate only the texts that actually exceed the model's input limit.

    BERT/XLM-R style models (the whole E5 family) have a hard cap on input
    length (`model.max_seq_length`, typically 512 tokens). Anything longer
    is silently truncated *inside* `model.encode()` - no error, no warning -
    and what gets lost is the tail of the text, which here is the "Text:"
    part, i.e. exactly the content that matters most.

    Rather than shrinking every chunk pre-emptively because of this rare
    case (see TARGET_CHARS in build_chunks.py), we keep the normal chunk
    size and truncate - loudly, not silently - only the texts that really
    do overflow. Tokenisation uses that specific model's tokenizer rather
    than a characters-per-token estimate, because the ratio varies a lot
    between models and languages.
    """
    max_len = getattr(model, "max_seq_length", None)
    if not max_len:
        return list(texts)  # model with no known limit (e.g. a future API model)

    tokenizer = model.tokenizer
    out, examples = [], []
    for t in texts:
        token_ids = tokenizer.encode(t, truncation=False)
        if len(token_ids) > max_len:
            examples.append((t, len(token_ids)))
            # Let the tokenizer do the truncating so it can reserve room
            # for special tokens (CLS/SEP); a plain token_ids[:max_len]
            # would not.
            truncated_ids = tokenizer.encode(t, truncation=True, max_length=max_len)
            t = tokenizer.decode(truncated_ids, skip_special_tokens=True)
        out.append(t)

    if examples:
        print(f"  [embed] WARNING: {len(examples)}/{len(texts)} texts exceeded "
              f"max_seq_length={max_len} tokens of model {model_name!r} - "
              f"they were truncated at the end.")
        for t, n in examples[:sample_size]:
            print(f"    - {n} tokens: {t[:80]!r}...")
    return out


def embed(texts, model_name: str = DEFAULT_MODEL, is_query: bool = False,
          batch_size: int = DEFAULT_BATCH_SIZE, show_progress: bool = True) -> np.ndarray:
    """Return an array of vectors for the given texts.

    Vectors are float32 and normalised to unit length, so a dot product is
    the cosine similarity.

    Pass `is_query=True` when embedding a QUERY rather than chunks to be
    indexed: models that need it (the E5 family) get the "query: " prefix
    instead of "passage: ". Models without that quirk get no prefix.
    """
    model = _load_model(model_name)

    prefix_map = QUERY_PREFIX_MODELS if is_query else PASSAGE_PREFIX_MODELS
    prefix = prefix_map.get(model_name, "")
    prepared = [f"{prefix}{t}" for t in texts] if prefix else list(texts)
    prepared = enforce_max_length(prepared, model, model_name)

    vectors = model.encode(
        prepared,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
    return vectors.astype(np.float32)


def embedding_dim(model_name: str = DEFAULT_MODEL) -> int:
    model = _load_model(model_name)
    if hasattr(model, "get_embedding_dimension"):
        return model.get_embedding_dimension()
    return model.get_sentence_embedding_dimension()
