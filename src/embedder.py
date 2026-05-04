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

from config import EMBED_CONCURRENCY, EMBED_MODEL, LLM_MODEL, OLLAMA_HOST


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


def embed_many(
    texts: List[str],
    model: str = EMBED_MODEL,
    concurrency: int = EMBED_CONCURRENCY,
) -> List[List[float]]:
    """Embed a batch of texts in parallel HTTP calls.

    Order is preserved. concurrency=1 falls back to sequential.
    """
    if concurrency <= 1 or len(texts) <= 1:
        return [embed(t, model=model) for t in texts]
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        return list(ex.map(lambda t: embed(t, model=model), texts))


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
