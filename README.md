# Local Wikipedia RAG Assistant

A ChatGPT-style assistant that answers questions about **20 famous people and 20 famous places** using **only local resources**: a local LLM (Ollama), a local embedding model, and a local vector database. No external LLM API is used.

- **Project 3 — combine both into a Retrieval-Augmented Generation (RAG) system**

---

## Architecture at a glance

```
┌──────────┐   ┌─────────┐   ┌──────────┐   ┌────────────┐   ┌────────────┐   ┌────────┐
│ Wikipedia│──▶│ Ingest  │──▶│ Chunker  │──▶│ Embedder   │──▶│ Chroma DB  │──▶│ Router │
│   API    │   │ (.txt)  │   │ overlap  │   │ Ollama     │   │ + SQLite   │   │ + RAG  │
└──────────┘   └─────────┘   └──────────┘   │ nomic-embed│   │ mirror     │   └───┬────┘
                                            └────────────┘   └────────────┘       │
                                                                            ┌─────▼─────┐
                                                                            │ Ollama LLM│
                                                                            │ llama3.2  │
                                                                            └───────────┘
```

| Layer       | Choice                              | Why                                                                 |
|-------------|-------------------------------------|---------------------------------------------------------------------|
| Ingest      | `urllib` + MediaWiki API            | Standard library only; matches the "language native" guideline.     |
| Chunking    | Paragraph-aware sliding window      | Hand-written; 1000-char chunks with 200-char overlap; respects paragraph boundaries; bounded for big docs. |
| Embeddings  | Ollama `nomic-embed-text` (parallel)| Fast, local, 768-dim, designed for retrieval; HTTP calls issued from a thread pool to overlap I/O.        |
| Vector DB   | Chroma (persistent) — single store  | Option B: one collection + `type` metadata; simplest mixed queries. |
| Mirror      | SQLite                              | Keeps raw chunks queryable / inspectable independent of Chroma.     |
| Router      | Rule-based (entity-name + cues)     | Deterministic, transparent, zero training.                          |
| LLM         | Ollama `llama3.2:3b`                | 3B fits on a laptop; fast first-token latency.                      |
| UI          | Streamlit + CLI                     | Streamlit for demo; CLI for scripting / debugging.                  |

### Vector-store design choice (Option A vs B)
We use **Option B: a single Chroma collection with a `type` metadata field** (`"person"` or `"place"`). Reasons:

- **Mixed queries are first-class** ("Compare the Eiffel Tower and the Statue of Liberty", "Compare Albert Einstein and Nikola Tesla"). One similarity search over the union plus filtering avoids the merge/re-rank logic two collections would require.
- **Simpler operations**: one persistence path, one embedding space, one place to reset.
- **Cheap filtering**: Chroma's `where={"type": {"$in": [...]}}` is essentially free.

If retrieval ever needed type-specific tuning (e.g. different chunk sizes per type), splitting into two collections later is straightforward — the SQLite mirror lets us rebuild without re-fetching.

---

## Prerequisites

- **Python 3.10+**
- **[Ollama](https://ollama.com/download)** installed and running locally

### Install Ollama

- **Windows / macOS**: download the installer from <https://ollama.com/download>.
- **Linux**: `curl -fsSL https://ollama.com/install.sh | sh`

After install, open a terminal and verify:
```bash
ollama --version
```

Start the Ollama background server (a system tray app on Windows/macOS, or `ollama serve` on Linux):
```bash
ollama serve
```

Pull the two models we use:
```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b
```

---

## Setup

```powershell
# 1. clone / open the project
cd aiaided3

# 2. (recommended) create a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows PowerShell
# source .venv/bin/activate      # macOS/Linux

# 3. install Python dependencies
pip install -r requirements.txt
```

---

## Running the system

### 1) Ingest Wikipedia data
Fetches plain-text extracts for the 20 people and 20 places listed in `config.py` and saves them under `data/`.

```powershell
python -m src.ingest
```

Re-running is safe — files that already exist are skipped.

### 2) Build the vector index
Chunks every document, embeds each chunk via Ollama, and stores everything in Chroma + SQLite.

```powershell
python -m src.build_index            # incremental
python -m src.build_index --reset    # wipe and rebuild
```

### 3) Chat

**Streamlit UI (recommended for demo):**
```powershell
streamlit run app_streamlit.py
```
Open the URL it prints (usually <http://localhost:8501>).

**CLI:**
```powershell
python app_cli.py
```

CLI commands during chat:
| Command          | Effect                                  |
|------------------|-----------------------------------------|
| `/sources on`    | show retrieved chunks under each answer |
| `/sources off`   | hide chunks                             |
| `/k 8`           | retrieve top-8 chunks                   |
| `/stats`         | print index stats                       |
| `/reset`         | wipe the vector store                   |
| `/help`          | show help                               |
| `/quit`          | exit                                    |

---

## Example queries

People:
- *Who was Albert Einstein and what is he known for?*
- *What did Marie Curie discover?*
- *Why is Nikola Tesla famous?*
- *Compare Lionel Messi and Cristiano Ronaldo.*
- *What is Frida Kahlo known for?*

Places:
- *Where is the Eiffel Tower located?*
- *Why is the Great Wall of China important?*
- *What is Machu Picchu?*
- *What was the Colosseum used for?*
- *Where is Mount Everest?*

Mixed:
- *Which famous place is located in Turkey?*
- *Which person is associated with electricity?*
- *Compare Albert Einstein and Nikola Tesla.*
- *Compare the Eiffel Tower and the Statue of Liberty.*

Failure cases (the assistant should say *"I don't know based on the available data."*):
- *Who is the president of Mars?*
- *Tell me about a random unknown person John Doe.*

---

## Project layout

```
aiaided3/
├── app_cli.py              # CLI chat
├── app_streamlit.py        # Streamlit UI
├── config.py               # Paths, models, entity lists
├── requirements.txt
├── README.md
├── Product_prd.md
├── recommendation.md
├── data/                   # raw Wikipedia .txt files (created by ingest)
├── storage/
│   ├── chroma/             # Chroma persistent client
│   └── chunks.sqlite       # chunk metadata mirror
└── src/
    ├── ingest.py           # Wikipedia API → data/
    ├── chunker.py          # paragraph-aware sliding window
    ├── embedder.py         # Ollama HTTP client (stdlib)
    ├── vectorstore.py      # Chroma + SQLite wrapper
    ├── router.py           # rule-based query router
    ├── build_index.py      # chunk + embed + store
    └── rag.py              # end-to-end pipeline
```

---

## Demo video


> **Demo:** _<paste link>_

---


