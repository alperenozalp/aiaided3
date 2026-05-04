"""
Thin Ollama HTTP client for embeddings and chat completions.

Uses only the standard library (urllib + json). No external SDK.
Ollama exposes:
  POST /api/embeddings   {model, prompt}        -> {"embedding": [...]}
  POST /api/generate     {model, prompt, ...}   -> {"response": "...", ...}
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import List

from config import EMBED_BATCH_SIZE, EMBED_CONCURRENCY, EMBED_MODEL, LLM_MODEL, OLLAMA_HOST


class OllamaError(RuntimeError):
    pass


def _post(path: str, payload: dict, timeout: int = 120) -> dict:
    url = f"{OLLAMA_HOST}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise OllamaError(
            f"Cannot reach Ollama at {OLLAMA_HOST}. "
            f"Is `ollama serve` running and the model pulled? Detail: {e}"
        ) from e


def embed(text: str, model: str = EMBED_MODEL) -> List[float]:
    res = _post("/api/embeddings", {"model": model, "prompt": text})
    emb = res.get("embedding")
    if not emb:
        raise OllamaError(f"No embedding returned (model={model}). Response: {res}")
    return emb


def _embed_batch(texts: List[str], model: str) -> List[List[float]]:
    """Call Ollama's batch embed endpoint (`/api/embed`) once for a list of texts."""
    if not texts:
        return []
    res = _post("/api/embed", {"model": model, "input": texts}, timeout=300)
    embs = res.get("embeddings")
    if not embs or len(embs) != len(texts):
        # Fallback to legacy single-text endpoint if the server is older.
        return [embed(t, model=model) for t in texts]
    return embs


def embed_many(
    texts: List[str],
    model: str = EMBED_MODEL,
    concurrency: int = EMBED_CONCURRENCY,
    batch_size: int = EMBED_BATCH_SIZE,
) -> List[List[float]]:
    """Embed many texts using Ollama's batch endpoint, with concurrent batches.

    Order is preserved.
    """
    if not texts:
        return []
    if len(texts) <= batch_size or concurrency <= 1:
        return _embed_batch(texts, model)

    # Split into batches and run them concurrently.
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(lambda b: _embed_batch(b, model), batches))
    out: List[List[float]] = []
    for r in results:
        out.extend(r)
    return out


def generate(prompt: str, model: str = LLM_MODEL, temperature: float = 0.2) -> str:
    res = _post(
        "/api/generate",
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=300,
    )
    return res.get("response", "").strip()


def ping() -> bool:
    try:
        with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False
