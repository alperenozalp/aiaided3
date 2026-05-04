"""
Rule-based query router.

Decides whether a query targets people, places, or both, by:
  1. Looking for any known entity title (case-insensitive substring) in
     the query — most reliable signal.
  2. Falling back to keyword cues ("who", "where", "compare", ...).
  3. Defaulting to BOTH if nothing matches, so we never starve retrieval.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from config import PEOPLE, PLACES

PERSON_CUES = {"who", "whom", "whose", "person", "scientist", "artist", "player",
               "singer", "writer", "inventor", "physicist", "leader"}
PLACE_CUES = {"where", "located", "place", "city", "country", "monument",
              "tower", "wall", "mountain", "building", "site"}
COMPARE_CUES = {"compare", "vs", "versus", "and", "between"}


@dataclass
class Route:
    types: List[str]          # subset of ["person", "place"]
    matched_titles: List[str] # entity titles found verbatim in query
    reason: str               # human-readable explanation


def _find_titles(query: str, titles: List[str]) -> List[str]:
    q = query.lower()
    hits = []
    for t in titles:
        if t.lower() in q:
            hits.append(t)
    return hits


def route(query: str) -> Route:
    person_hits = _find_titles(query, PEOPLE)
    place_hits = _find_titles(query, PLACES)

    if person_hits and place_hits:
        return Route(
            types=["person", "place"],
            matched_titles=person_hits + place_hits,
            reason=f"Query mentions person ({person_hits}) and place ({place_hits}).",
        )
    if person_hits:
        return Route(
            types=["person"],
            matched_titles=person_hits,
            reason=f"Query mentions person(s): {person_hits}.",
        )
    if place_hits:
        return Route(
            types=["place"],
            matched_titles=place_hits,
            reason=f"Query mentions place(s): {place_hits}.",
        )

    tokens = {w.strip(".,?!").lower() for w in query.split()}
    has_person_cue = bool(tokens & PERSON_CUES)
    has_place_cue = bool(tokens & PLACE_CUES)

    if has_person_cue and not has_place_cue:
        return Route(types=["person"], matched_titles=[], reason="Person keyword cue.")
    if has_place_cue and not has_person_cue:
        return Route(types=["place"], matched_titles=[], reason="Place keyword cue.")

    return Route(
        types=["person", "place"],
        matched_titles=[],
        reason="No clear signal — searching both stores.",
    )
