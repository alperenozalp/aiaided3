"""Central configuration for the Local Wikipedia RAG Assistant."""
from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
STORAGE_DIR = ROOT_DIR / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma"
SQLITE_PATH = STORAGE_DIR / "chunks.sqlite"

DATA_DIR.mkdir(exist_ok=True)
STORAGE_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# --- Ollama ---
OLLAMA_HOST = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:3b"

# --- Chunking ---
# Larger chunks (1000 chars, ~200 tokens) cut indexing time on CPU and give
# the LLM more coherent context per retrieved snippet. The 200-char overlap
# keeps facts that span chunk boundaries findable.
CHUNK_SIZE = 1000         # characters
CHUNK_OVERLAP = 200       # characters

# --- Embedding (parallelism) ---
# nomic-embed-text on Ollama serializes per-process internally, but issuing
# requests concurrently still overlaps tokenization, JSON encoding and HTTP,
# giving ~2-3x speedup on multi-core CPUs.
EMBED_CONCURRENCY = 4

# --- Retrieval ---
TOP_K = 5
COLLECTION_NAME = "wiki_rag"

# --- Entities to ingest (Wikipedia page titles) ---
PEOPLE = [
    "Albert Einstein",
    "Marie Curie",
    "Leonardo da Vinci",
    "William Shakespeare",
    "Ada Lovelace",
    "Nikola Tesla",
    "Lionel Messi",
    "Cristiano Ronaldo",
    "Taylor Swift",
    "Frida Kahlo",
    "Isaac Newton",
    "Charles Darwin",
    "Mahatma Gandhi",
    "Wolfgang Amadeus Mozart",
    "Vincent van Gogh",
    "Stephen Hawking",
    "Mustafa Kemal Atatürk",
    "Pablo Picasso",
    "Cleopatra",
    "Alan Turing",
]

PLACES = [
    "Eiffel Tower",
    "Great Wall of China",
    "Taj Mahal",
    "Grand Canyon",
    "Machu Picchu",
    "Colosseum",
    "Hagia Sophia",
    "Statue of Liberty",
    "Giza pyramid complex",
    "Mount Everest",
    "Stonehenge",
    "Petra",
    "Acropolis of Athens",
    "Sydney Opera House",
    "Niagara Falls",
    "Mount Fuji",
    "Angkor Wat",
    "Cappadocia",
    "Burj Khalifa",
    "Christ the Redeemer (statue)",
]
