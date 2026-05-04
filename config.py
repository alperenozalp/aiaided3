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
# Larger chunks (1400 chars, ~280 tokens) cut indexing time substantially on
# CPU (fewer embeddings to compute) and give the LLM more coherent context
# per retrieved snippet. The 150-char overlap keeps facts that span chunk
# boundaries findable.
CHUNK_SIZE = 1400         # characters
CHUNK_OVERLAP = 150       # characters

# --- Embedding (parallelism + batching) ---
# We use Ollama's batch endpoint /api/embed (input=list) which embeds many
# texts in one HTTP call, eliminating most per-request overhead. On top of
# that we still issue a few batches concurrently to overlap I/O.
EMBED_BATCH_SIZE = 32
EMBED_CONCURRENCY = 8

# --- Retrieval ---
TOP_K = 5
COLLECTION_NAME = "wiki_rag"

# --- LLM generation ---
# keep_alive tells Ollama to keep the model resident in RAM between calls.
# "30m" avoids the multi-second cold-start on every question.
OLLAMA_KEEP_ALIVE = "30m"
# Cap the answer length. Most factual answers fit in 256 tokens; this stops
# the model from rambling and shaves seconds off generation time.
LLM_NUM_PREDICT = 256
# Context window. With TOP_K=5 chunks ~ 1400 chars each + prompt overhead,
# 4096 is plenty. Larger windows slow down generation on CPU.
LLM_NUM_CTX = 4096

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
