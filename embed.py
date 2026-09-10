"""
Jádro embedding kroku - jedna funkce embed(), za kterou se schová
konkrétní model/provider. Zbytek pipeline (ukládání, Chroma, vyhledávání)
bude volat jen tohle a nebude vědět/zajímat se, jestli je uvnitř lokální
sentence-transformers, nebo později třeba OpenAI/Cohere API.

Použití:
    from embed import embed
    vektory = embed(["nějaký text", "další text"], model_name="intfloat/multilingual-e5-base")
"""
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "intfloat/multilingual-e5-base"
DEFAULT_BATCH_SIZE = 32

# Některé modely (celá rodina E5) vyžadují prefix "query: "/"passage: " před
# textem, jinak jsou výsledky výrazně horší - viz diskuze o E5/BGE.
# Tady se to řeší centrálně, ať to nejde zapomenout na jednom z volacích míst.
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
    """Model se stahuje/nahrává jen jednou za proces (cache), i když embed()
    zavoláme vícekrát s tím samým jménem (např. jednou pro chunky, jednou
    pro dotaz).

    SentenceTransformer(model_name) defaultně VŽDYCKY zkontroluje
    HuggingFace Hub přes síť (jestli náhodou nevyšla novější verze), i když
    je model už stažený v lokální cache - to stojí čas navíc při každém
    spuštění. Zkusíme proto nejdřív local_files_only=True (čistě z cache,
    žádná síť) a jen když model ještě není stažený vůbec, spadneme zpátky
    na normální (síťové) stažení."""
    print(f"  [embed] nahrávám model {model_name} ...")
    try:
        return SentenceTransformer(model_name, local_files_only=True)
    except Exception:
        print(f"  [embed] '{model_name}' není v lokální cache - stahuji ze sítě ...")
        return SentenceTransformer(model_name)


def enforce_max_length(texts, model: SentenceTransformer, model_name: str, sample_size: int = 5):
    """Modely typu BERT/XLM-R (celá E5 rodina) mají tvrdý strop na délku
    vstupu (model.max_seq_length, typicky 512 tokenů) - text nad limit by
    se jinak tiše ořízl UVNITŘ model.encode(), bez chyby a bez varování,
    a přišli bychom tak o konec textu (u nás zrovna o konec "Text:" části,
    tedy o obsah, na kterém nejvíc záleží).

    Místo abychom kvůli téhle - v praxi vzácné - výjimce preventivně
    zmenšovali cílovou velikost úplně všech chunků (viz TARGET_CHARS
    v build_chunks.py), děláme to opačně: necháváme běžnou délku chunků
    a explicitně (a nahlas, ne potichu) zkrátíme jen ty texty, které limit
    doopravdy přesáhnou. Tokenizace se dělá tokenizerem KONKRÉTNÍHO modelu,
    ne odhadem podle počtu znaků - u různých modelů/jazyků totiž vychází
    jinak, kolik tokenů připadá na znak."""
    max_len = getattr(model, "max_seq_length", None)
    if not max_len:
        return list(texts)  # model bez známého limitu (např. budoucí API model)

    tokenizer = model.tokenizer
    out, examples = [], []
    for t in texts:
        token_ids = tokenizer.encode(t, truncation=False)
        if len(token_ids) > max_len:
            examples.append((t, len(token_ids)))
            # truncation=True necháme tokenizeru, ať si sám pohlídá místo
            # na speciální tokeny (CLS/SEP apod.) - proto to neděláme jen
            # prostým token_ids[:max_len]
            truncated_ids = tokenizer.encode(t, truncation=True, max_length=max_len)
            t = tokenizer.decode(truncated_ids, skip_special_tokens=True)
        out.append(t)

    if examples:
        print(f"  [embed] POZOR: {len(examples)}/{len(texts)} textů přesahovalo "
              f"max_seq_length={max_len} tokenů modelu {model_name!r} - "
              f"byly zkráceny na konci.")
        for t, n in examples[:sample_size]:
            print(f"    - {n} tokenů: {t[:80]!r}...")
    return out


def embed(texts, model_name: str = DEFAULT_MODEL, is_query: bool = False,
          batch_size: int = DEFAULT_BATCH_SIZE, show_progress: bool = True) -> np.ndarray:
    """Vrať pole vektorů (float32, normalizovaných na jednotkovou délku -
    takže skalární součin == kosinová podobnost) pro seznam textů.

    is_query=True použijte při embedování DOTAZU (ne chunků k indexaci) -
    u modelů, které to vyžadují (E5 rodina), se text zabalí prefixem
    "query: " místo "passage: ". Pro modely bez téhle zvláštnosti se prefix
    prostě nepřidá."""
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
