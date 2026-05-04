# Production deployment recommendation

This prototype is a **single-user laptop app**. Moving it to production for a small/medium audience (e.g. an internal company assistant or a public demo) needs changes in five areas: serving model inference, scaling the vector store, managing data freshness, hardening the API, and observability.

## 1. Inference: replace the laptop Ollama with a managed model server

| Layer       | Prototype          | Production                                                                 |
|-------------|--------------------|----------------------------------------------------------------------------|
| LLM serving | Ollama on laptop   | **vLLM** or **Text Generation Inference (TGI)** on a GPU node, behind an OpenAI-compatible HTTP API. |
| Embeddings  | Ollama nomic-embed | **Infinity** or **TEI** (Text Embeddings Inference); same model, batched. |
| Model size  | `llama3.2:3b`      | `llama3.1:8b` or `Qwen2.5-7B-Instruct` (quality bump) at fp16 / AWQ.       |
| Hardware    | CPU                | 1× L4 / A10G GPU is enough for 7-8B at low concurrency.                    |

Wrap inference behind a single internal hostname so the app code doesn't change — only the `OLLAMA_HOST` env var.

## 2. Vector store: graduate from local Chroma

Local Chroma is fine up to ~1M chunks single-process. For real workloads:

- **pgvector on managed Postgres** — easiest, gives you SQL + vectors in one place, plus transactions for atomic re-index.
- **Qdrant** or **Weaviate** — better filter performance and HNSW tuning, supports multi-tenant collections.
- Index parameters: HNSW `M=32, ef_construction=200`, query `ef=128`. Re-evaluate after measuring recall@k on a labeled set.

## 3. Data freshness & ingestion

- Move the ingest job into a scheduled worker (Airflow / Prefect / a plain cron + container).
- Track each Wikipedia page's `revid`; only re-embed chunks whose source changed (content hash per chunk).
- Store raw extracts in object storage (S3 / Azure Blob), not local disk, so re-indexing is reproducible from a snapshot.
- Add an ETag/`If-Modified-Since` check against the MediaWiki API to be a polite citizen and reduce cost.

## 4. Retrieval quality

- Add **BM25 hybrid** retrieval (Postgres `tsvector` or OpenSearch) and fuse with dense scores via Reciprocal Rank Fusion.
- Add a **cross-encoder re-ranker** (e.g. `bge-reranker-base`) on the top-50 to produce top-5. Worth ~10–20 % NDCG in our experience.
- Add **query rewriting**: for short queries, expand with the LLM into 2-3 paraphrases, retrieve for each, and union.

## 5. API & UI

- Replace Streamlit with a **FastAPI** backend exposing `/chat`, `/ingest`, `/admin/reset`, plus a thin Next.js (or React) front-end. Streamlit is great for demos, painful for multi-user state.
- Stream tokens with **server-sent events**; users perceive ~3× lower latency.
- Cache `(query → answer)` for popular questions in Redis with a 24h TTL; cache `(query → embedding)` indefinitely.

## 6. Safety, security, governance

- Input validation: cap question length, reject prompt-injection patterns (`ignore previous instructions`, system-prompt overrides) at the API edge.
- Output filtering: a small classifier or regex pass to flag PII / off-topic answers.
- Authn/Authz: API keys per tenant, per-key rate limits (e.g. 30 req/min) at the gateway (Nginx, Traefik, Cloudflare).
- Secrets via env vars / a vault — never bake them into images.
- Containerize with a non-root user, read-only root FS, and a minimal base image.

## 7. Observability

- **Logs**: structured JSON, one record per request with `route.types`, `top_k`, retrieval distances, latency per stage, and the LLM's `prompt_tokens` / `eval_tokens`.
- **Metrics** (Prometheus/Grafana): p50/p95/p99 of each pipeline stage, retrieval recall against a small offline eval set, "I don't know" rate.
- **Traces** (OpenTelemetry): one span per pipeline stage; correlates LLM latency with retrieval bottlenecks.
- **Eval harness**: a CI job that runs ~50 reference Q&A pairs against the deployed system on every model or prompt change, with regression alerts.

## 8. Cost & scaling notes

| Concurrency target | Suggested setup                                                        |
|--------------------|------------------------------------------------------------------------|
| 1 user, demo       | Laptop + Ollama (this repo)                                            |
| <50 RPS            | 1× L4 GPU for LLM, CPU embedding service, 1× Postgres+pgvector         |
| 50–500 RPS         | Autoscaled vLLM replicas behind a load balancer; Qdrant cluster (3 nodes); Redis cache |

## 9. Migration path from this repo

1. Containerize: `Dockerfile` for the API, separate compose service for Ollama.
2. Swap `OLLAMA_HOST` to a TGI/vLLM endpoint via env var.
3. Replace `chromadb.PersistentClient` calls with a thin adapter behind the same `vectorstore` interface.
4. Move `src/ingest.py` into a scheduled worker.
5. Add the FastAPI/Next.js front-end and retire the Streamlit app.

The current code already separates concerns (`embedder`, `vectorstore`, `router`, `rag`), so each step above is a focused change rather than a rewrite.
