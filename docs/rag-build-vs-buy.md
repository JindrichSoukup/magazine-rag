# RAG build vs. buy in a larger organisation — three questions

Three follow-on questions and answers from a conversation about what
three days of building home-made RAG components is worth to someone who
has to run, manage or orchestrate work like this inside a larger
organisation.

Translated from the Czech original; the substance is unchanged.

---

## Question 1

**Q:** I spent three days building home-made RAG components. What can I
take from that for somebody who has to run, manage or orchestrate work
like this inside a larger organisation?

**A:**

A good question to distil. Let us separate what is specific to *this*
project from what carries over to managing similar work at scale.

### 1. The most expensive part of the work is invisible in demos

Almost all the time went into the "boring" part: parsing the PDF,
article boundaries, repairing broken words. Embedding and vector search —
the part that looks interesting and that vendor demos sell — took a
fraction of it. **The management consequence:** anyone estimating a RAG
project's timeline by how quickly one can "plug in LangChain and
Pinecone" is estimating wrongly. A realistic estimate has to allow for
data-specific work that cannot be scoped in advance, because it only
appears on contact with real data.

### 2. Silent errors are more dangerous than crashes — and that is a management problem, not just a technical one

Almost none of the bugs we found — shared pages, stylistically unrelated
text merged into one block, a `find_split_y` that never worked — failed
with an error. The code ran, looked fine, and the result was quietly
wrong. **The transferable lesson:** "it ran without an error" is not an
acceptable definition of done for a data pipeline. You need an explicit,
measurable layer of confidence — here `quality_flags` plus a summary
audit — as a **first-class deliverable**, not a check you do "if there
is time left", because there will not be.

### 3. The cost of an iteration is itself a design decision

Every fix in extraction meant regenerating the whole archive: hours of
computation. That is why we had to deal with checkpointing, with
`upsert` rather than a full index rebuild, with testing on one issue
before running over 75. **For running a team:** ask up front "what does
one fix-and-check cycle cost", not once somebody tells you a regeneration
takes overnight. That number directly determines how fast the team can
learn from its mistakes.

### 4. Diagnostic tools are not overhead — they are part of the work

More than half a dozen small diagnostic scripts appeared along the way:
dump a specific page, summarise uncertainty across the archive, and so
on. None was planned; each appeared as a specific mystery did. **The
planning lesson:** if an estimate for "build a data pipeline" contains no
budget for "build the tools that tell me why the pipeline does what it
does", the estimate is incomplete.

### 5. What a system costs is mostly decided by your own architecture, not the vendor's price list

When we worked the costs out, the difference between "cheap" and
"expensive" hosting turned out to be negligible for us. What actually
moves the bill is **your own decisions**: how much context you send to
the LLM, the promote threshold, the window size. **For anyone holding a
budget:** before negotiating a discount with a vendor, check whether your
own architecture is sending five times more context than it needs to.

### 6. "Build vs. buy" is not binary — it is a question of where exactly your differentiation sits

We simulated how the same data would be handled by a general-purpose
tool of the NotebookLM kind. For a **generic** question it works
respectably; for anything requiring precise citation or article
boundaries it fails outright. **Transferably:** general tools are
optimised for the generic case. If your real value lies in accuracy over
specific or non-standard data, "just upload it into a finished tool" will
not get you where you need to be — but finding that out *beforehand*,
rather than after deployment, takes a small prototype or pilot, not
reading marketing material.

### 7. Decisions without a reason are inherited badly

Throughout, I pushed for decisions with an explicit "why", not just a
"what": the embedding model, Chroma versus FAISS, the chunk size. When
you come to write documentation or a handover for a team, that is
exactly what survives. "Why we did it this way" is the first thing to be
lost, and it is exactly what the next person needs when they decide to
change it.

---

## Question 2

**Q:** So are there really situations where build is unambiguously
better than buy?

**A:**

Yes, but somewhat differently from how it is usually framed — and it is
visible in what we did today.

### The key observation: "build vs. buy" is not one question, it is a question **per layer**

Look at what this project actually did:

| Layer | What we did | Why |
|---|---|---|
| Embedding model | **bought** (used a finished one, `multilingual-e5-base`) | a commodity — nobody gains an advantage by building their own transformer |
| Vector search | **bought** (used a finished one, Chroma) | a commodity — nobody writes their own HNSW index |
| Parsing Živa's PDFs, article boundaries, repairing typesetting slips | **built** from scratch | there is no "buy" here to begin with |

So what we called a "home-made RAG" is in fact **80% buy and 20%
build** — it is just that the build is the visible, time-consuming part.
The key mistake in reasoning is to pretend "build vs. buy" is one
decision for the whole system.

### When build is unambiguously right on that remaining layer

1. **When a general tool does not solve the problem at all, not merely
   less well.** Our shared boundary pages, editorial typos (`CXLVIiI`),
   hyphenated word breaks — no vendor will build a general solution for
   "the quirks of an InDesign export of a 70-year-old Czech magazine".
   There is no "buy" option on the table here: either you build it, or
   the problem goes unsolved.

2. **When accuracy and traceability are the product, not a detail.** If
   the value of your system is "I know exactly which author and which
   page", not "some answer from some source", then buying a general tool
   that does not do that is not the cheaper option — it is a different
   product that does not address your brief.

3. **When you need auditability a black box cannot give you.**
   `quality_flags` work because we know exactly what happened inside.
   You cannot get that from a hosted "upload a PDF, ask a question"
   tool: you have no access to the inside.

### But — and this matters for an honest answer

"Build is right" **does not mean "it is free"**. It still carries a cost:
every fix meant regenerating the whole archive, maintaining the
diagnostics, running the regression. Deciding to build this layer is
deciding to **take on that maintenance cost permanently**, not to make a
one-off outlay. The right management question is not "can we build it",
but "do we want to own this kind of maintenance long term, or would we
rather pay someone to do it for us, worse".

**In short:** build is unambiguously better exactly where your
differentiation meets the idiosyncrasy of your data — and that is a
narrower set of cases than "build is generally better". Most of a RAG
stack (embedding, the vector store, the LLM) stays a commodity, where
buy makes sense almost always.

---

## Question 3

**Q:** So after this experience I know that RAG in the "buy" sense need
not be one black box, of the NotebookLM kind — it can be bought in
pieces.

**A:**

Exactly — and it is a natural continuation of the per-layer reasoning.
The scale is **not** "black box versus build everything"; it is a
spectrum where you choose, **layer by layer**, which one to buy finished
and which to build.

### The whole RAG stack written out by layer, and what can be bought at each

| Layer | What can be bought separately | What we did |
|---|---|---|
| **Document parsing / ingest** | Unstructured.io, LlamaParse, Google Document AI Layout Parser — services specifically for "PDF to clean chunks" | built to measure (`extract_blocks.py`) |
| **Chunking strategy** | often part of the parsing service above, or a framework (LangChain text splitters) | built to measure (article-aware) |
| **Embedding** | OpenAI/Cohere/Voyage APIs | bought-equivalent, only hosted locally |
| **Vector store** | Pinecone/Weaviate/Qdrant Cloud | bought-equivalent (Chroma, local) |
| **Reranking** | Cohere Rerank API | not addressed yet |
| **Orchestration, the glue** | LangChain, LlamaIndex, Haystack | built to measure (`assemble_context.py`) |
| **Answer generation** | Claude/GPT/Gemini APIs | bought |
| **The whole thing at once** | NotebookLM, Glean, ChatGPT with a file upload | (what we simulated for comparison) |

Note that **even the layer we did most by hand**, PDF parsing, can now be
bought as a standalone service. The question is not "does a bought
version exist" but "does it solve *my* particular idiosyncrasies" — and
you will not learn that from a marketing page, you learn it from a pilot
on real data, which is precisely what our naive comparison started.

### Why this observation matters more than it sounds

For management it means the build-vs-buy decision **does not happen once
at the start of a project**. It happens **at every seam** between layers,
continuously, as you learn more about the data. Consequences:

1. **The seams between bought components are still your work.** Even
   having bought the embedding, the vector database and the LLM API
   separately, the glue between them — here `assemble_context.py`: when
   to promote to a whole article, how to assemble the prompt — is still
   something you have to design and own. Composability does not mean
   zero integration work.

2. **Buying layer by layer requires knowing your own stack layer by
   layer** — precisely the table above. Without it, it is easy to end up
   building because nobody asked whether that particular layer had
   already become a commodity.

3. **The price of flexibility is dependence on more seams, not on one
   vendor.** A fully black-box solution has one point of failure and one
   negotiation. A composable stack has five or six, each with its own
   risk of an outage, a price change or a breaking API change.

**In short:** yes — and this is probably the most transferable thing in
the whole project. Rather than asking "build or buy", write the stack out
by layer as in the table above, and ask the question separately for each.
