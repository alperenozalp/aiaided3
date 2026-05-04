"""
Native chunker: paragraph-aware sliding window over characters.

Strategy:
1. Split document on blank lines into paragraphs.
2. Greedily pack paragraphs into chunks <= CHUNK_SIZE characters.
3. If a single paragraph is larger than CHUNK_SIZE, slide a window over it
   with CHUNK_OVERLAP characters of overlap.

This avoids breaking sentences when possible while still bounding chunk size
for large documents.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

from config import CHUNK_OVERLAP, CHUNK_SIZE


@dataclass
class Chunk:
    text: str
    index: int  # position within source document


_PARA_SPLIT = re.compile(r"\n\s*\n+")


def _window(text: str, size: int, overlap: int) -> List[str]:
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    out: List[str] = []
    for start in range(0, len(text), step):
        piece = text[start : start + size]
        if not piece.strip():
            continue
        out.append(piece)
        if start + size >= len(text):
            break
    return out


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    paragraphs = [p.strip() for p in _PARA_SPLIT.split(text) if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paragraphs:
        if len(p) > size:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(_window(p, size, overlap))
            continue
        if not buf:
            buf = p
        elif len(buf) + 1 + len(p) <= size:
            buf = f"{buf}\n\n{p}"
        else:
            chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)

    return [Chunk(text=c, index=i) for i, c in enumerate(chunks)]


def chunk_iter(docs: Iterable[tuple[str, str]]) -> Iterable[tuple[str, Chunk]]:
    """Yield (doc_id, chunk) pairs."""
    for doc_id, text in docs:
        for ch in chunk_text(text):
            yield doc_id, ch
