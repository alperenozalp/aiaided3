"""
Wikipedia ingestion using only the standard library (urllib + json).

Fetches plain-text extracts of each page in config.PEOPLE / config.PLACES
via the MediaWiki API and stores them as UTF-8 .txt files under data/.

Each file is named "<type>__<slug>.txt" so its category is recoverable
from the filename alone (used by the chunker / vector-store builder).
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

# Allow running as a script: `python src/ingest.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, PEOPLE, PLACES  # noqa: E402

WIKI_API = "https://en.wikipedia.org/w/api.php"
# Wikipedia asks for a contact-y UA so they can reach you if you misbehave.
USER_AGENT = "LocalWikiRAG/1.0 (student project; contact: localhost) python-urllib"
MAX_RETRIES = 5
BASE_BACKOFF = 2.0  # seconds; doubled per retry


def slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return s or "untitled"


def _http_get_json(url: str) -> dict:
    """GET with retries that respect 429/5xx and the Retry-After header."""
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
                "Accept-Encoding": "identity",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504):
                retry_after = e.headers.get("Retry-After") if e.headers else None
                try:
                    wait = float(retry_after) if retry_after else BASE_BACKOFF * (2 ** attempt)
                except ValueError:
                    wait = BASE_BACKOFF * (2 ** attempt)
                wait = min(wait, 60.0)
                print(f"    [retry {attempt + 1}/{MAX_RETRIES}] HTTP {e.code}; sleeping {wait:.1f}s")
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError as e:
            last_err = e
            wait = BASE_BACKOFF * (2 ** attempt)
            print(f"    [retry {attempt + 1}/{MAX_RETRIES}] URLError; sleeping {wait:.1f}s")
            time.sleep(wait)
    assert last_err is not None
    raise last_err


def fetch_extract(title: str) -> str:
    """Fetch plain-text extract of a Wikipedia page (full article)."""
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": "1",
        "redirects": "1",
        "titles": title,
    }
    url = f"{WIKI_API}?{urllib.parse.urlencode(params)}"
    data = _http_get_json(url)

    pages = data.get("query", {}).get("pages", {})
    if not pages:
        raise RuntimeError(f"No pages returned for '{title}'")
    page = next(iter(pages.values()))
    if "missing" in page:
        raise RuntimeError(f"Wikipedia page missing for '{title}'")
    extract = page.get("extract", "").strip()
    if not extract:
        raise RuntimeError(f"Empty extract for '{title}'")
    return extract


def ingest_one(title: str, kind: str) -> Path:
    out_path = DATA_DIR / f"{kind}__{slugify(title)}.txt"
    if out_path.exists() and out_path.stat().st_size > 500:
        print(f"  [skip] {kind}: {title} (already exists)")
        return out_path
    text = fetch_extract(title)
    header = f"TITLE: {title}\nTYPE: {kind}\n\n"
    out_path.write_text(header + text, encoding="utf-8")
    print(f"  [ok]   {kind}: {title} ({len(text):,} chars)")
    return out_path


def main() -> None:
    print(f"Ingesting into {DATA_DIR}")
    failures: list[str] = []
    for title in PEOPLE:
        try:
            ingest_one(title, "person")
        except Exception as e:
            failures.append(f"person/{title}: {e}")
            print(f"  [FAIL] person: {title}: {e}")
        time.sleep(1.0)  # be polite

    for title in PLACES:
        try:
            ingest_one(title, "place")
        except Exception as e:
            failures.append(f"place/{title}: {e}")
            print(f"  [FAIL] place: {title}: {e}")
        time.sleep(1.0)

    print()
    print(f"Done. People: {len(PEOPLE)}, Places: {len(PLACES)}, Failures: {len(failures)}")
    if failures:
        print("Failures:")
        for f in failures:
            print(" -", f)


if __name__ == "__main__":
    main()
