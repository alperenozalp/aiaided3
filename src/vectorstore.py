"""
Vector store wrapper around Chroma (persistent client).

Design choice: Option B — a SINGLE collection with a `type` metadata field
("person" or "place"). Rationale:
  * Comparison queries that mix people AND places are answered with one
    similarity search over the union of vectors instead of two parallel
    searches that must then be merged/re-ranked.
  * Metadata filtering ($eq / $in / $or) is cheap in Chroma and keeps the
    code path uniform regardless of whether the query is person-only,
    place-only, or mixed.
  * One collection means one embedding space, one persistence layer, one
    place to clear/reset.

A small SQLite table mirrors chunk text + metadata so we can rebuild or
inspect the corpus without touching Chroma internals.

Important Windows note: chroma>=0.5 will, on first `get_or_create_collection`,
silently download a ~80 MB ONNX MiniLM model unless an `embedding_function`
is supplied. We never need it (we always pass `embeddings=...` to `add()`),
so we install a no-op embedding function to suppress the download — without
this the build can hang for minutes on slow connections / antivirus.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional

import chromadb
from chromadb.config import Settings

from config import CHROMA_DIR, COLLECTION_NAME, SQLITE_PATH, TOP_K


# ---------- No-op embedding function (prevents Chroma's auto-download) ----------

class _NoopEmbeddingFunction:
    """Stops Chroma from downloading its default ONNX embedder.

    Embeddings are always supplied explicitly via `add(embeddings=...)`,
    so Chroma never has cause to call this. We only register it so
    `get_or_create_collection` doesn't fall back to the default model.
    """

    def __call__(self, input):  # pragma: no cover - never invoked
        raise RuntimeError(
            "NoopEmbeddingFunction was called; embeddings must be supplied externally."
        )

    def name(self) -> str:
        return "noop"


# ---------- SQLite (chunk metadata mirror) ----------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    doc_id       TEXT NOT NULL,
    title        TEXT NOT NULL,
    type         TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL,
    text         TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(type);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(SQLITE_PATH)
    conn.executescript(_SCHEMA)
    return conn


# ---------- Chroma ----------

_CLIENT: Optional["chromadb.api.ClientAPI"] = None
_COLLECTION = None
# Chroma's HNSW writer is happiest with smallish add() calls on Windows;
# 32 keeps each insert well under a second and avoids long stalls.
_ADD_BATCH = 32


def _client() -> "chromadb.api.ClientAPI":
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
    return _CLIENT


def get_collection():
    global _COLLECTION
    if _COLLECTION is None:
        _COLLECTION = _client().get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=_NoopEmbeddingFunction(),
        )
    return _COLLECTION


def reset() -> None:
    """Wipe Chroma collection AND SQLite mirror."""
    global _COLLECTION
    c = _client()
    try:
        c.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _COLLECTION = c.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=_NoopEmbeddingFunction(),
    )
    if Path(SQLITE_PATH).exists():
        Path(SQLITE_PATH).unlink()


def add_chunks(records: List[dict], embeddings: List[List[float]]) -> None:
    """records: dicts with id, doc_id, title, type, chunk_index, text."""
    if not records:
        return
    coll = get_collection()

    # Insert in small sub-batches; large single add() calls have been
    # observed to stall for minutes on Windows.
    for i in range(0, len(records), _ADD_BATCH):
        sub_recs = records[i : i + _ADD_BATCH]
        sub_embs = embeddings[i : i + _ADD_BATCH]
        coll.add(
            ids=[r["id"] for r in sub_recs],
            embeddings=sub_embs,
            documents=[r["text"] for r in sub_recs],
            metadatas=[
                {
                    "doc_id": r["doc_id"],
                    "title": r["title"],
                    "type": r["type"],
                    "chunk_index": r["chunk_index"],
                }
                for r in sub_recs
            ],
        )

    with _connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO chunks(id,doc_id,title,type,chunk_index,text) "
            "VALUES (?,?,?,?,?,?)",
            [
                (r["id"], r["doc_id"], r["title"], r["type"], r["chunk_index"], r["text"])
                for r in records
            ],
        )


def query(
    embedding: List[float],
    types: Optional[Iterable[str]] = None,
    top_k: int = TOP_K,
) -> List[dict]:
    """types: e.g. ["person"], ["place"], or None for both.

    Returns a list of {id, text, title, type, chunk_index, distance}, best-first.
    """
    coll = get_collection()
    where = None
    if types is not None:
        types = list(types)
        if len(types) == 1:
            where = {"type": types[0]}
        elif len(types) > 1:
            where = {"type": {"$in": types}}

    res = coll.query(
        query_embeddings=[embedding],
        n_results=top_k,
        where=where,
    )

    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res.get("distances", [[None] * len(ids)])[0]
    return [
        {
            "id": i,
            "text": d,
            "title": m.get("title"),
            "type": m.get("type"),
            "chunk_index": m.get("chunk_index"),
            "distance": dist,
        }
        for i, d, m, dist in zip(ids, docs, metas, dists)
    ]


def stats() -> dict:
    coll = get_collection()
    total = coll.count()
    with _connect() as conn:
        rows = conn.execute("SELECT type, COUNT(*) FROM chunks GROUP BY type").fetchall()
    return {"total_chunks": total, "by_type": dict(rows)}
