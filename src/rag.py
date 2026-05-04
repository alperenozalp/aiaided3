"""
End-to-end RAG pipeline: route -> embed query -> retrieve -> generate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from config import TOP_K
from src import embedder, vectorstore
from src.router import Route, route


SYSTEM_PROMPT = """You are a careful assistant answering questions about famous people and places using ONLY the context below.

Rules:
- Answer ONLY using facts present in the context.
- If the context does not contain the answer, reply exactly: "I don't know based on the available data."
- Be concise. Prefer 2-5 sentences unless a comparison is asked.
- Do NOT invent dates, numbers, or names that are not in the context.
- When comparing two entities, structure the answer clearly (e.g. short bullet list).
"""


def _build_prompt(question: str, contexts: List[dict]) -> str:
    if not contexts:
        ctx_block = "(no context retrieved)"
    else:
        parts = []
        for i, c in enumerate(contexts, 1):
            parts.append(f"[{i}] ({c['type']} - {c['title']}) {c['text']}")
        ctx_block = "\n\n".join(parts)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{ctx_block}\n\n"
        f"QUESTION: {question}\n\n"
        f"ANSWER:"
    )


@dataclass
class RAGResult:
    question: str
    answer: str
    route: Route
    contexts: List[dict] = field(default_factory=list)
    latency_ms: dict = field(default_factory=dict)


def answer(question: str, top_k: int = TOP_K, show_sources: bool = True) -> RAGResult:
    import time

    t0 = time.time()
    r = route(question)
    t1 = time.time()

    q_emb = embedder.embed(question)
    t2 = time.time()

    # If we route to BOTH and the user clearly mentioned one entity of each,
    # split the budget so each type gets representation.
    if set(r.types) == {"person", "place"}:
        half = max(2, top_k // 2)
        person_hits = vectorstore.query(q_emb, types=["person"], top_k=half)
        place_hits = vectorstore.query(q_emb, types=["place"], top_k=half)
        merged = sorted(person_hits + place_hits, key=lambda x: x["distance"] or 1e9)
        contexts = merged[:top_k]
    else:
        contexts = vectorstore.query(q_emb, types=r.types, top_k=top_k)
    t3 = time.time()

    prompt = _build_prompt(question, contexts)
    text = embedder.generate(prompt)
    t4 = time.time()

    return RAGResult(
        question=question,
        answer=text or "I don't know based on the available data.",
        route=r,
        contexts=contexts if show_sources else [],
        latency_ms={
            "route": int((t1 - t0) * 1000),
            "embed": int((t2 - t1) * 1000),
            "retrieve": int((t3 - t2) * 1000),
            "generate": int((t4 - t3) * 1000),
            "total": int((t4 - t0) * 1000),
        },
    )
