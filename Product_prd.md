# Product PRD — Local Wikipedia RAG Assistant

> A PRD written for an AI coding agent so that the system can be regenerated end-to-end from this single document.

## 1. Goal
Build a fully local ChatGPT-style assistant that answers questions about famous people and famous places. All components — the language model, embedding model, vector store and database — must run on a developer laptop. No external LLM API may be called.

## 2. Users & primary use cases
- **Learner / curious user** who wants to ask grounded questions about historical figures and landmarks without an internet round-trip to a hosted LLM.
- **Demo audience / instructor** evaluating retrieval quality, hallucination control, and architectural choices.

## 3. Functional requirements

### 3.1 Ingest
- Pull plain-text Wikipedia content for at least **20 famous people** and **20 famous places**.
- Must include the mandatory minimum set defined in the assignment brief.
- Use the MediaWiki REST/Action API directly via standard-library HTTP (no `wikipedia-api` package).
- Persist each page as `data/<type>__<slug>.txt` with a small header (`TITLE:`, `TYPE:`).
- Skip already-downloaded pages on re-run; log failures but never abort the whole batch.

### 3.2 Chunking
- Hand-written, no third-party splitter.
- Strategy: paragraph-aware sliding window. Pack consecutive paragraphs greedily up to ~1000 chars; for paragraphs longer than the limit, slide a 1000-char window with 200-char overlap.
- Rationale for 1000/200: large enough that each chunk usually contains a complete fact (Wikipedia paragraphs are dense), small enough to keep retrieval precision high. Halves the chunk count vs 500/100, which materially shortens CPU embedding time.
- Must be deterministic so the index is reproducible.

### 3.3 Embedding & storage
- Embed every chunk with a **local** model. Default: `nomic-embed-text` via Ollama.
- Embedding requests are issued from a small thread pool (default `EMBED_CONCURRENCY=4`); order is preserved. This overlaps tokenization, HTTP and JSON encoding for a ~2-3x speedup on multi-core CPUs.
- Vector store: **Chroma**, persistent on disk under `storage/chroma/`.
- **Single collection** with a `type` metadata field (`person` | `place`) — see "Design choices" below.
- Mirror chunks in a small **SQLite** table (`storage/chunks.sqlite`) for inspection / rebuild.
- Build script must support an idempotent `--reset` flag.

### 3.4 Query routing
- Decide whether the user query targets a person, a place, or both:
  1. Look for any known entity title as a substring of the query (case-insensitive).
  2. Otherwise check keyword cues (`who`, `where`, …).
  3. Otherwise default to **both** (never starve retrieval).
- The route must be transparent: include `types`, `matched_titles`, and a human-readable `reason`.

### 3.5 Retrieval
- Embed the query with the same model as the index.
- For single-type routes: cosine-similarity search with `where={"type": <t>}`.
- For mixed routes: split the top-k budget across types and merge by distance, so comparison questions always have evidence from both sides.

### 3.6 Generation
- Use a **local** LLM via Ollama. Default: `llama3.2:3b`.
- Prompt the model with a system instruction that:
  - restricts answers to the supplied context,
  - mandates the literal phrase **"I don't know based on the available data."** when the context is insufficient,
  - asks for concise (2–5 sentence) answers and structured comparisons.
- Temperature ≤ 0.3 to reduce hallucination.

### 3.7 Chat interfaces
- **CLI** (`app_cli.py`) supporting commands: `/sources on|off`, `/k <n>`, `/reset`, `/stats`, `/help`, `/quit`.
- **Streamlit UI** (`app_streamlit.py`) with chat history, sidebar controls (top-k, show-sources, reset), Ollama health indicator, latency display, and an expandable "Sources" panel per answer.

### 3.8 Observability
- Each answer reports per-stage latency (route / embed / retrieve / generate / total).
- Sources panel shows `type`, `title`, distance, and chunk text.

## 4. Non-functional requirements
- **Localhost only.** No outbound calls except (a) MediaWiki during ingest and (b) `localhost:11434` for Ollama.
- **Reproducibility.** Same input pages + same models → same chunks → same answers (modulo LLM sampling).
- **Resettability.** A single command (`/reset` or `--reset`) wipes the vector store.
- **Resilience.** Each ingest failure is logged; build_index batches embeddings (size 32) so a single failure can be retried without rebuilding from scratch.

## 5. Design choices (explained)

### 5.1 Single vector store with `type` metadata (Option B)
- Mixed queries (person + place) are answered with one similarity search.
- One embedding space, one persistence directory, one reset operation.
- Filtering by `type` in Chroma is O(1) extra cost.

### 5.2 Native chunker
- The assignment explicitly asks for language-native functionality. A 60-line paragraph-aware splitter is enough; we avoid LangChain's `RecursiveCharacterTextSplitter`. Chunk size (1000) and overlap (200) were chosen empirically on Wikipedia biographies: large enough to contain a complete fact, small enough to preserve retrieval precision, and roughly halves embedding time vs. 500/100.

### 5.3 Rule-based router
- Deterministic and explainable. Wikipedia entity titles are stable, so substring matching covers the vast majority of real queries; cue words handle the rest.

### 5.4 Stdlib HTTP for Ollama
- Keeps the dependency set tiny (Chroma + Streamlit) and demonstrates the protocol explicitly.

## 6. Out of scope (explicitly)
- Streaming token output.
- Multi-turn memory / coreference (each query is independent).
- Re-ranking with a cross-encoder.
- Authentication, rate limiting, multi-user state.
- Production deployment — see `recommendation.md`.

## 7. Acceptance tests
The assistant must pass these qualitatively (run from a fresh laptop after the README steps):

| Question                                                   | Expected behaviour                                  |
|------------------------------------------------------------|-----------------------------------------------------|
| Who was Albert Einstein and what is he known for?          | Grounded answer citing relativity / E=mc².          |
| What did Marie Curie discover?                             | Mentions polonium / radium.                         |
| Compare Lionel Messi and Cristiano Ronaldo                 | Two-sided comparison drawing on both pages.         |
| Which famous place is located in Turkey?                   | Returns Hagia Sophia and/or Cappadocia.             |
| Which person is associated with electricity?               | Routes to people; returns Tesla.                    |
| Who is the president of Mars?                              | Returns "I don't know based on the available data." |
| Tell me about a random unknown person John Doe.            | Returns "I don't know based on the available data." |

## 8. Deliverables
- Working source code (this repo)
- `README.md`
- `Product_prd.md` (this file)
- `recommendation.md`
- 5-minute demo video link in the README
