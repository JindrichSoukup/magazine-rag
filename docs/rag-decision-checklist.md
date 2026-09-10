# RAG — a layer-by-layer map of the decisions

A reference list of the questions that get settled — explicitly, or
silently by a default value — in every layer of a RAG system. Each entry
says *what* is being decided, not how to decide it: that always depends
on the case at hand.

---

## 1. Ingest and document parsing

- Which formats must the system handle (PDF, DOCX, HTML, images/OCR, tables)?
- Is OCR needed? How accurate does it have to be?
- How is layout handled — multi-column text, tables, headers and footers?
- How is structure preserved (headings, section hierarchy)?
- How is metadata extracted (author, date, title, page number)?
- How is boilerplate removed or marked (repeating headers, footers, page numbers)?
- How are images, captions and annotations treated — part of the text, or separate?
- Several languages inside one document?
- Build vs. buy: your own parser, or a service (Unstructured.io, LlamaParse, Document AI Layout Parser)?

## 2. Chunking

- Target chunk size (tokens or characters) — granularity against consistency for embedding.
- Method: fixed size, recursive/hierarchical, semantic (by paragraph or sentence), or by document structure (by heading or section)?
- Overlap between chunks — how much, and how (the last N characters, whole paragraphs)?
- What about tables, code, images and captions — chunks of their own, or part of the surrounding text?
- A contextual header on the chunk (metadata written into the embedded text) — yes or no, and how much?
- What metadata is stored with a chunk (source, page, section, date, author, access rights)?
- Hierarchical "parent-child" chunks (a small chunk to find it, a larger one for context)?
- What happens to a chunk that exceeds the embedding model's token limit (drop, truncate, split)?
- How are the cover and unrelated parts of a document handled (title page, index, advertising)?

## 3. Embedding

- Model: local (open-weight) or hosted behind an API?
- Language competence — monolingual or multilingual, and how well does it cover the target language?
- Vector dimension — accuracy against storage cost and search speed.
- Vector normalisation (unit length) — it determines which similarity metric makes sense.
- Asymmetric embeddings (a different prefix or instruction for the query and for the document)?
- Numeric precision (float32/float16/quantisation) — storage against quality.
- Batch size and throughput when processing a large corpus.
- What happens when the model changes — must the whole corpus be reindexed?
- Cost per token (if an API), latency, rate limits.
- Checkpointing: how a long indexing run survives interruption.

## 4. Vector store

- A hosted service or self-hosted (a library or a container)?
- Index type: exact (flat) or approximate (HNSW, IVF, PQ) — accuracy against speed and scale.
- Support for metadata filters (and whether filtering happens before or after the vector search)?
- Hybrid search (vector plus keyword/BM25) — yes or no, and how are the two weighted?
- Multi-tenancy and isolation of data between users or departments?
- Update strategy: `upsert` per record, or a complete rebuild of the index?
- Scalability (sharding, replication) as the data grows.
- Backup and persistence.
- Pricing model (storage per GB, read/write charges, flat minimums).

## 5. Retrieval strategy

- `top_k` — how many candidates are pulled before further processing?
- Similarity metric (cosine, dot product, Euclidean distance)?
- Metadata filters in the query (year, author, document type, access rights)?
- Query transformation before search (query expansion, HyDE, several query variants)?
- Reranking — a second, more accurate pass (a cross-encoder) over the initial candidates; yes or no, and which model?
- Result diversity (e.g. MMR) — should near-identical chunks be avoided?
- Document-level aggregation — when to return just a chunk, when a whole document or section?
- How much surrounding context to attach to each hit (a window of neighbouring chunks)?
- Deduplication of overlapping or repeating results?
- The weighting between vector and keyword search in a hybrid approach?

## 6. Assembling the prompt and context for the LLM

- The system prompt — grounding rules, citation format, tone, language of the answer.
- How many tokens of context to send (input budget against room for the answer)?
- Citation format — an inline reference, a list at the end, or both?
- What to do with context that does not cover the question at all, or only partly?
- The order of chunks in the prompt (most relevant first or last — models tend to attend less to the middle of a long context)?
- What to cache (a repeating system prompt or instruction) for cost?

## 7. Answer generation (the LLM)

- Which model — quality against cost and latency (it need not be the strongest one available)?
- Temperature and the other sampling parameters?
- Maximum answer length?
- Stream the answer, or wait for the whole thing?
- Is structured output needed (JSON, tools/function calling)?
- A fallback model or strategy when the primary one is down or saturated?

## 8. Evaluation and quality

- Which metrics to track (retrieval precision and recall, faithfulness of the answer to its sources, relevance)?
- How is the test set built (hand-written questions with known answers, or generated)?
- Automated or human evaluation of the answers?
- Ongoing monitoring in production (quality drift, user feedback)?
- How is the trustworthiness or uncertainty of the processed data measured and reported (see `quality_flags` in this project)?

## 9. Operations and infrastructure

- Latency requirements (real-time chat against batch processing)?
- Cost tracking and budget (where exactly the money goes — embedding, storage, generation)?
- How often and by what process is the index rebuilt when the source data changes?
- Versioning — what happens to the old index when the embedding model changes?
- Security and privacy (sensitive data, access control at document or chunk level)?
- Data residency and compliance requirements?

## 10. Guardrails and traceability

- Detecting and limiting hallucination (answers outside the supplied context)?
- Verifying that the citations in an answer really correspond to the sources used?
- Filtering inappropriate content?
- How does the system behave when the honest answer is "I don't know" — is that supported, or is the model pushed to answer at any cost?

---

## The question that cuts across the whole list

For every layer: **build or buy** — and if buy, from whom, at what price
and at what dependency? This gets settled separately for each layer, not
once for the system as a whole. See
[RAG: build vs. buy](rag-build-vs-buy.md).
