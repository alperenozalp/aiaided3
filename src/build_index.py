"""
Build the index: read .txt files in data/, chunk, embed, store in Chroma + SQLite.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR  # noqa: E402
from src import embedder, vectorstore  # noqa: E402
from src.chunker import chunk_text  # noqa: E402


def _parse_doc(path: Path) -> tuple[str, str, str]:
    """Returns (title, type, body). Header format set by ingest.py."""
    text = path.read_text(encoding="utf-8")
    title = path.stem
    kind = "person" if path.stem.startswith("person__") else "place"
    body = text
    lines = text.splitlines()
    if lines and lines[0].startswith("TITLE: "):
        title = lines[0][len("TITLE: "):].strip()
        # skip header lines until blank line
        for i, ln in enumerate(lines):
            if ln.strip() == "":
                body = "\n".join(lines[i + 1 :])
                break
    return title, kind, body


def build(reset: bool = False) -> None:
    if reset:
        print("Resetting vector store...")
        vectorstore.reset()

    files = sorted(DATA_DIR.glob("*.txt"))
    if not files:
        print(f"No documents in {DATA_DIR}. Run `python -m src.ingest` first.")
        return

    if not embedder.ping():
        print("ERROR: Ollama is not reachable. Start it with `ollama serve` "
              f"and pull the embedding model: `ollama pull {embedder.EMBED_MODEL}`.")
        return

    total_chunks = 0
    batch_records: list[dict] = []
    batch_texts: list[str] = []
    # Bigger flush window: embed_many now uses Ollama's /api/embed batch
    # endpoint with concurrent batches, so larger flushes => fewer round trips.
    BATCH = 256

    def flush():
        nonlocal batch_records, batch_texts
        if not batch_records:
            return
        embs = embedder.embed_many(batch_texts)
        vectorstore.add_chunks(batch_records, embs)
        batch_records, batch_texts = [], []

    for path in files:
        title, kind, body = _parse_doc(path)
        doc_id = path.stem
        chunks = chunk_text(body)
        print(f"  {kind:6s} {title:35s} -> {len(chunks):3d} chunks")
        for ch in chunks:
            cid = f"{doc_id}::{ch.index}"
            batch_records.append(
                {
                    "id": cid,
                    "doc_id": doc_id,
                    "title": title,
                    "type": kind,
                    "chunk_index": ch.index,
                    "text": ch.text,
                }
            )
            batch_texts.append(ch.text)
            total_chunks += 1
            if len(batch_records) >= BATCH:
                flush()
    flush()

    print()
    print(f"Indexed {total_chunks} chunks across {len(files)} documents.")
    print("Stats:", vectorstore.stats())


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--reset", action="store_true", help="Wipe existing index first")
    args = p.parse_args()
    build(reset=args.reset)
