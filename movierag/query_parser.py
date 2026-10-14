"""Step 3a: self-querying. Natural language -> structured QueryPlan.

"a documentary under 90 minutes about the ocean"
    -> QueryPlan(search_text="the ocean", kinds=["documentary"], max_runtime_min=90)

Why: embeddings are bad at hard constraints. "under 90 minutes" is a number
to filter on, not a vibe to match. So an LLM splits the request into
(filters) + (text to search semantically).

Belt and braces: small LLMs sometimes return broken JSON or miss a number.
A rule-based parser runs too, and fills in or replaces what the LLM missed.
"""

from __future__ import annotations

import logging
import re

from llama_index.core.prompts import PromptTemplate

from .schemas import GENRES, QueryPlan

log = logging.getLogger(__name__)

PARSE_PROMPT = PromptTemplate(
    """You turn a movie-night request into search parameters.

Rules:
- search_text: themes, mood, plot or style only (no length, year or format words).
- genres: only from this list: {genres}. Use [] if unsure.
- kinds: only film, series, documentary, short, youtube - and only if the user asked for that format.
- max_runtime_min: minutes, e.g. "under 2 hours" -> 120, "1h30" -> 90.
- min_year / max_year: e.g. "from the 90s" -> 1990 and 1999, "after 2015" -> 2016.
- similar_to: a title the user gives as a reference ("like Devs" -> "Devs").
- wants_new: true only if the user wants things they have not saved yet.

Example
Request: something like Interstellar but a series, nothing older than 2010
Answer: {{"search_text": "space exploration, time, family sacrifice", "genres": ["Science Fiction"], "kinds": ["series"], "max_runtime_min": null, "min_year": 2010, "max_year": null, "similar_to": "Interstellar", "wants_new": false}}

Request: {query}
Answer:"""
)

# ----------------------------------------------------------------- rules
_GENRE_WORDS = {
    "Action": ["action", "fight", "fighting", "martial art", "kung fu", "kung-fu"],
    "Adventure": ["adventure", "quest", "treasure"],
    "Animation": ["animation", "animated", "anime", "cartoon"],
    "Comedy": ["comedy", "comedies", "funny", "laugh", "lighthearted", "light-hearted"],
    "Crime": ["crime", "heist", "gangster", "mafia", "detective"],
    "Documentary": ["documentary", "documentaries"],
    "Drama": ["drama", "dramatic", "emotional"],
    "Family": ["family", "kids", "children"],
    "Fantasy": ["fantasy", "magic", "dragon", "mythical", "myth"],
    "History": ["history", "historical", "biopic", "true story"],
    "Horror": ["horror", "scary", "creepy", "terrifying"],
    "Music": ["music", "musical", "musician", "violin"],
    "Mystery": ["mystery", "whodunit", "puzzle"],
    "Romance": ["romance", "romantic", "love story"],
    "Science Fiction": ["sci-fi", "scifi", "science fiction", "space", "alien", "robot", "dystopia", "time travel"],
    "Thriller": ["thriller", "suspense", "tense"],
    "War": ["war", "soldier", "battle", "wwii", "ww2", "wwi"],
    "Western": ["western", "cowboy"],
}


def _minutes(num: str, unit: str) -> int:
    value = float(num.replace(",", "."))
    return int(round(value * 60)) if unit.lower().startswith("h") else int(round(value))


def heuristic_plan(query: str) -> QueryPlan:
    q = query.lower()
    plan = QueryPlan(search_text=query)

    # --- length: "under 90 min", "less than 2 hours", "under an hour", "2-hour max", "1h30"
    m = re.search(
        r"(?:under|less than|shorter than|max(?:imum)?|at most|no longer than|up to|<|i have|i've got|only have)\s*"
        r"(?:an?\s+)?(\d+(?:[.,]\d+)?)?\s*(hours?|hrs?|h|minutes?|mins?)\b",
        q,
    )
    if m:
        plan.max_runtime_min = _minutes(m.group(1) or "1", m.group(2))
    m = re.search(r"(\d+(?:[.,]\d+)?)[- ]?(hours?|hrs?|h|minutes?|mins?)\s+max", q)
    if m:
        plan.max_runtime_min = _minutes(m.group(1), m.group(2))
    m = re.search(r"\b(\d)\s*h\s*(\d{2})\b", q)  # "under 1h30"
    if m and re.search(r"under|less than|shorter than|max|at most|no longer than|up to|<|have", q):
        plan.max_runtime_min = int(m.group(1)) * 60 + int(m.group(2))

    # --- years: "90s", "1980s", "80s or older", "after 2015", "before 2000", "since 2018"
    m = re.search(r"\b(?:(19|20)(\d)0|(\d)0)'?s\b", q)
    if m:
        if m.group(1):
            start = int(m.group(1) + m.group(2) + "0")
        else:
            d = int(m.group(3))
            start = (1900 if d >= 3 else 2000) + d * 10
        plan.min_year, plan.max_year = start, start + 9
        if re.search(r"or older|or earlier|and older|or before", q):
            plan.min_year = None
        if re.search(r"or newer|or later|and newer|or after", q):
            plan.max_year = None
    m = re.search(r"\b(after|since|newer than|from)\s+((?:19|20)\d\d)\b", q)
    if m:
        plan.min_year = int(m.group(2)) + (1 if m.group(1) in ("after", "newer than") else 0)
    m = re.search(r"\b(before|older than|until|pre-?)\s*((?:19|20)\d\d)\b", q)
    if m:
        plan.max_year = int(m.group(2)) - (0 if m.group(1) == "until" else 1)

    # --- formats
    kinds: list[str] = []
    if re.search(r"\bdocumentar|\bdocu\b|\bdocs?\b", q):
        kinds.append("documentary")
    if re.search(r"\byoutube\b|\bvideo essays?\b", q):
        kinds.append("youtube")
    if re.search(r"\b(?:tv )?series\b|\btv shows?\b|\ba show\b|\bseasons?\b|\bepisodes?\b|\bbinge", q):
        kinds.append("series")
    if re.search(r"\bshort films?\b|\bshorts\b", q):
        kinds.append("short")
    if re.search(r"\bbut an? (?:film|movie)\b", q):
        kinds = ["film"]
    elif re.search(r"\b(?:film|movie)s?\b", q) and not kinds:
        kinds.append("film")
    plan.kinds = kinds  # type: ignore[assignment]

    # --- genres from keywords
    plan.genres = [
        g for g, words in _GENRE_WORDS.items()
        if any(re.search(rf"\b{re.escape(w)}(?:s|es)?\b", q) for w in words)
    ]
    if "documentary" in kinds or "youtube" in kinds:
        # "a documentary about space" is not Science Fiction: for non-fiction the
        # topic words belong to the semantic search, not to a genre filter.
        plan.genres = []

    # --- reference title: "like Devs", "similar to The Abyss"
    m = re.search(
        r"\b(?:like|similar to|in the vein of)\s+['\"]?([A-Z0-9][\w:'\- ]*?)['\"]?"
        r"(?=[,.!?]|\s+but\b|\s+with\b|\s+and\b|\s+from\b|$)",
        query,
    )
    if m:
        plan.similar_to = m.group(1).strip()

    plan.wants_new = bool(
        re.search(r"\bnew\b|haven'?t (?:saved|seen)|not (?:in|on) my list|\bdiscover|outside my list|beyond my list", q)
    )
    plan.search_text = _strip_constraints(query)
    return plan


def _strip_constraints(query: str) -> str:
    """Remove the parts already turned into filters, so they don't pollute search.

    Otherwise BM25 happily matches "under 2 hours" with *Under* the Skin.
    """
    t = query
    patterns = [
        r"(?:under|less than|shorter than|max(?:imum)?|at most|no longer than|up to|i have|i've got|only have)\s*"
        r"(?:an?\s+hour|\d+(?:[.,]\d+)?\s*(?:h\s*\d{2}|hours?|hrs?|h|minutes?|mins?)\b)",
        r"\d+(?:[.,]\d+)?[- ]?(?:hours?|hrs?|h|minutes?|mins?)\s+max",
        r"\b(?:after|since|newer than|from|before|older than|until)\s+(?:19|20)\d\d\b",
        r"\b(?:from the\s+)?(?:19|20)?\d0'?s\b(?:\s+or\s+(?:older|earlier|newer|later))?",
        r"\b(?:i )?haven'?t (?:saved|seen)\b|\bnot (?:in|on) my list\b",
        r"\b(?:show me|i want|i'd like|give me|recommend|suggest|please|tonight|something)\b",
        r"\b(?:an?|some|new)\b(?=\s+(?:movie|film|series|documentar|youtube|video))",
        r"\b(?:movies?|films?|tv series|series|tv shows?|documentar(?:y|ies)|youtube|video essays?|videos?|short films?)\b",
    ]
    for p in patterns:
        t = re.sub(p, " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*,[\s,]*", ", ", t)  # "war,, with" -> "war, with"
    t = re.sub(r"\s{2,}", " ", t).strip(" ,.-")
    t = re.sub(r"^(?:an?|some)\s+", "", t, flags=re.IGNORECASE)  # orphaned article
    t = re.sub(r"\s+(?:but|and|with|or)$", "", t, flags=re.IGNORECASE)  # dangling conjunction
    return t or query


def _normalise(plan: QueryPlan) -> QueryPlan:
    lookup = {g.lower(): g for g in GENRES}
    lookup.update({"sci-fi": "Science Fiction", "scifi": "Science Fiction", "sf": "Science Fiction",
                   "historical": "History", "martial arts": "Action"})
    plan.genres = list(dict.fromkeys(lookup[g.lower()] for g in plan.genres if g.lower() in lookup))
    if plan.max_runtime_min is not None and not 1 <= plan.max_runtime_min <= 1000:
        plan.max_runtime_min = None
    if not plan.search_text.strip():
        plan.search_text = " ".join(plan.genres) or "a good movie"
    return plan


def parse_query(query: str, llm=None) -> tuple[QueryPlan, str]:
    """Returns (plan, how) where how is 'llm', 'llm+rules' or 'rules'."""
    rules = heuristic_plan(query)
    if llm is None or type(llm).__name__ == "MockLLM":  # offline mode: rules only
        return _normalise(rules), "rules"
    try:
        plan = llm.structured_predict(QueryPlan, PARSE_PROMPT, genres=", ".join(GENRES), query=query)
    except Exception as exc:  # broken JSON, validation error, timeout...
        log.warning("LLM query parsing failed (%s); using rules only", exc)
        return _normalise(rules), "rules"

    how = "llm"
    # Hard constraints: trust explicit numbers found by the regexes.
    for field in ("max_runtime_min", "min_year", "max_year"):
        if getattr(rules, field) is not None and getattr(plan, field) != getattr(rules, field):
            setattr(plan, field, getattr(rules, field))
            how = "llm+rules"
    for kind in rules.kinds:
        if kind in ("documentary", "youtube") and kind not in plan.kinds:
            plan.kinds.append(kind)
            how = "llm+rules"
    if rules.wants_new and not plan.wants_new:
        plan.wants_new, how = True, "llm+rules"
    return _normalise(plan), how
