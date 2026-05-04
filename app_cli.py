"""
Command-line chat interface for the Local Wikipedia RAG Assistant.

Commands during chat:
  /sources on|off   toggle showing retrieved chunks
  /k <n>            set top_k
  /reset            wipe the vector store
  /stats            show index stats
  /help             show this help
  /quit             exit
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import embedder, vectorstore
from src.rag import answer_stream, retrieve


HELP = """Commands:
  /sources on|off   toggle showing retrieved chunks
  /k <n>            set top_k (current: {k})
  /reset            wipe the vector store
  /stats            show index stats
  /help             show this help
  /quit             exit
"""


def main() -> None:
    show_sources = True
    top_k = 5

    print("Local Wikipedia RAG Assistant (CLI)")
    print("Type your question, or /help for commands. Ctrl+C to exit.")
    if not embedder.ping():
        print("\n[!] Ollama not reachable at", embedder.OLLAMA_HOST)
        print("    Start it with `ollama serve` and pull the models:")
        print(f"      ollama pull {embedder.EMBED_MODEL}")
        print(f"      ollama pull {embedder.LLM_MODEL}")
        print()

    try:
        stats = vectorstore.stats()
        print(f"Index: {stats}\n")
    except Exception as e:
        print(f"[!] Vector store not ready: {e}")
        print("    Run: python -m src.ingest && python -m src.build_index\n")

    while True:
        try:
            q = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.startswith("/"):
            cmd, *rest = q[1:].split(maxsplit=1)
            cmd = cmd.lower()
            if cmd in ("quit", "exit"):
                break
            elif cmd == "help":
                print(HELP.format(k=top_k))
            elif cmd == "sources" and rest:
                show_sources = rest[0].strip().lower() == "on"
                print(f"  sources = {show_sources}")
            elif cmd == "k" and rest:
                try:
                    top_k = max(1, int(rest[0]))
                    print(f"  top_k = {top_k}")
                except ValueError:
                    print("  usage: /k <n>")
            elif cmd == "reset":
                vectorstore.reset()
                print("  vector store reset.")
            elif cmd == "stats":
                print(" ", vectorstore.stats())
            else:
                print("  unknown command. /help for list.")
            continue

        try:
            import sys as _sys
            import time as _time

            t0 = _time.time()
            r, contexts = retrieve(q, top_k=top_k)
            t1 = _time.time()
            print("\nbot > ", end="", flush=True)
            chunks: list[str] = []
            for tok in answer_stream(q, contexts):
                chunks.append(tok)
                _sys.stdout.write(tok)
                _sys.stdout.flush()
            text = "".join(chunks).strip() or "I don't know based on the available data."
            t2 = _time.time()
            print("\n")
        except Exception as e:
            print(f"  [error] {e}")
            continue

        print(f"  route: {r.types}  ({r.reason})")
        print(
            f"  latency_ms: retrieve={int((t1 - t0) * 1000)} "
            f"generate={int((t2 - t1) * 1000)} total={int((t2 - t0) * 1000)}"
        )
        if show_sources and contexts:
            print("  sources:")
            for i, c in enumerate(contexts, 1):
                snippet = c["text"][:140].replace("\n", " ")
                print(f"    [{i}] {c['type']} - {c['title']}  d={c['distance']:.3f}")
                print(f"        {snippet}...")
        print()


if __name__ == "__main__":
    main()
