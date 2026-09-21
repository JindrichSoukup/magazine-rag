"""Compute embeddings for every chunk in chunks.jsonl and write them to
disk as two files side by side, linked by ORDER (line N in the .jsonl
corresponds to row N in the .npy):

    <profile>_embeddings__<model>.npy       - array (n_chunks, dim), float32
    <profile>_embeddings__<model>_ids.json  - chunk_ids in the same order

The "embedding_text" field is embedded (with its metadata header), not
the bare "text" - see the note on contextual chunking in build_chunks.py.

CHECKPOINTING
-------------
On a large archive of tens of thousands of chunks, the computation can
take hours on a CPU. So that a run can be interrupted safely (Ctrl+C, a
crash, the machine going down) and then resumed where it stopped, instead
of "compute everything, save at the end" it works like this:

  1. Vectors are written as they go into a temporary <model>.raw file via
     np.memmap - an on-disk array that can be written piecewise without
     holding the whole result in memory, and crucially independent of
     whether the process runs to completion.
  2. Every --checkpoint-every chunks (500 by default), how much is done
     is recorded in <model>_progress.json.
  3. Run the script again with the same --output-dir and --model and it
     first checks for a run in progress; if there is one, it continues
     from the last checkpoint instead of starting over.
  4. Once everything is done, the .raw file is converted into a normal
     .npy - exactly the format the rest of the pipeline and Chroma
     expect - and the temporary files (.raw, _progress.json) are removed.

NOTE: a checkpoint is only valid for the SAME input .jsonl; only the line
count is checked, not the content. Change the input file between the
interruption and the resume - a different order or different chunks - and
the indices would drift apart. Left this way for simplicity: one does not
normally change the input between two sittings of "compute the
embeddings".

Usage:
    python -m magazine_rag.build_embeddings --input output/chunks.jsonl \\
        --output-dir output --model intfloat/multilingual-e5-base

    # after an interruption, just run exactly the same command again
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from magazine_rag import profiles
from magazine_rag.console import setup_console
from magazine_rag.embed import embed, embedding_dim, DEFAULT_MODEL, DEFAULT_BATCH_SIZE

DEFAULT_CHECKPOINT_EVERY = 500


def load_chunks(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def paths_for(out_dir: Path, model_name: str, prefix: str = "embeddings"):
    """One place where every filename is assembled, so that the final
    .npy/_ids.json, the temporary .raw and _progress.json always agree.

    `prefix` keeps different magazines' output apart in the same
    directory; the model name is in the filename so that several
    candidates can be compared without deleting anything (see
    tools/compare_models.py).

    Suffixes are appended to the string, never set with with_suffix():
    a model name may contain a dot ("nomic-embed-text-v1.5"), and
    with_suffix() would replace the ".5" instead of adding to it.
    """
    slug = model_name.replace("/", "__")
    base = out_dir / f"{prefix}__{slug}"
    return {
        "npy": Path(f"{base}.npy"),
        "ids": Path(f"{base}_ids.json"),
        "raw": Path(f"{base}.raw"),
        "progress": Path(f"{base}_progress.json"),
    }


def format_eta(seconds: float) -> str:
    if seconds == float("inf") or seconds != seconds:  # inf or NaN
        return "?"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def main():
    setup_console()
    ap = argparse.ArgumentParser(description="compute embeddings for the chunks")
    ap.add_argument("--input", required=True, help="chunks.jsonl")
    ap.add_argument("--output-dir", required=True,
                    help="where to write the .npy and _ids.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                    help="batch for model.encode() itself (the compute unit)")
    ap.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY,
                    help="how many chunks between saving progress to disk")
    profiles.add_profile_argument(ap)
    args = ap.parse_args()

    profile = profiles.get(args.profile)

    chunks = load_chunks(Path(args.input))
    n = len(chunks)
    print(f"Loaded {n} chunks from {args.input}")

    dim = embedding_dim(args.model)
    print(f"Model: {args.model} (dim={dim})")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = paths_for(out_dir, args.model, prefix=f"{profile.key}_embeddings")

    if p["npy"].exists():
        print(f"{p['npy']} already exists (finished earlier) - doing nothing. "
              f"Delete it to recompute.")
        return

    texts = [c["embedding_text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks]

    # --- find or create a checkpoint -------------------------------------
    if p["progress"].exists() and p["raw"].exists():
        progress = json.loads(p["progress"].read_text(encoding="utf-8"))
        if progress.get("model") != args.model or progress.get("total") != n:
            raise SystemExit(
                f"The run in progress in {p['raw'].name} does not match the "
                f"current input or model (it was for model="
                f"{progress.get('model')!r}, total={progress.get('total')}) - "
                f"delete {p['raw'].name} and {p['progress'].name}, or use a "
                f"different --output-dir."
            )
        done = progress["done"]
        print(f"Resuming the previous run: {done}/{n} chunks already done.")
    else:
        done = 0
        mm_init = np.memmap(p["raw"], dtype="float32", mode="w+", shape=(n, dim))
        mm_init.flush()
        del mm_init
        p["ids"].write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
        p["progress"].write_text(
            json.dumps({"model": args.model, "total": n, "done": 0}),
            encoding="utf-8")

    mm = np.memmap(p["raw"], dtype="float32", mode="r+", shape=(n, dim))

    # --- main loop: embed in checkpoint_every sized pieces ---------------
    idx = done
    t0 = time.time()
    try:
        while idx < n:
            end = min(idx + args.checkpoint_every, n)
            vectors = embed(texts[idx:end], model_name=args.model, is_query=False,
                            batch_size=args.batch_size, show_progress=False)
            mm[idx:end] = vectors
            mm.flush()
            idx = end
            p["progress"].write_text(
                json.dumps({"model": args.model, "total": n, "done": idx}),
                encoding="utf-8")

            elapsed = time.time() - t0
            rate = (idx - done) / elapsed if elapsed > 0 else 0
            eta = (n - idx) / rate if rate > 0 else float("inf")
            print(f"  {idx}/{n} done ({rate:.2f} chunks/s, "
                  f"about {format_eta(eta)} left)")
    except KeyboardInterrupt:
        print(f"\nInterrupted at {idx}/{n}. Checkpoint saved - run the same "
              f"command again and it will carry on from here.")
        return

    # --- finished: convert the temporary .raw into a normal .npy ---------
    final = np.array(mm)  # read into memory - fine at our sizes
    del mm
    np.save(p["npy"], final)
    p["raw"].unlink()
    p["progress"].unlink()

    print(f"Done. Vectors: {p['npy']}  (shape={final.shape}, dtype={final.dtype})")
    print(f"IDs (same order): {p['ids']}")


if __name__ == "__main__":
    main()
