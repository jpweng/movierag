"""Step 3: the full RAG loop.

    question
      -> parse (LLM + rules)            : what do you mean?          (query_parser.py)
      -> filter my catalogue            : hard constraints in Python
      -> hybrid search + rerank         : best matches from MY list  (index.py)
      -> discover (TMDb / YouTube)      : top up with NEW items if needed
      -> generate                       : LLM writes the answer, citing [L1] / [N2]
      -> check citations                : cheap hallucination guard
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

from llama_index.core.prompts import PromptTemplate

from .index import MovieIndex, rank_in_memory, rerank
from .query_parser import parse_query
from .schemas import QueryPlan, Title
from .tmdb import MOVIE_NAME_TO_ID, SHARED_TO_TV_ID, TMDbClient
from .youtube import YouTubeClient

ANSWER_PROMPT = PromptTemplate(
    """You are a friendly movie-night assistant. The user asked:
"{question}"

Their constraints: {constraints}
{relaxed}
CANDIDATES FROM THE USER'S SAVED LIST:
{my_list}

NEW SUGGESTIONS (not saved yet):
{new}

Instructions:
- Recommend up to {k_list} candidates from the saved list first, then up to {k_new} new suggestions if any are given.
- Only recommend candidates listed above. Never invent titles or facts.
- For each pick write: the title and year, the label in brackets like [L1] or [N2], then one or two sentences on why it fits the request, using only the candidate's description.
- Respect the constraints. If nothing fits well, say so honestly.
- Be concise. Use a short bullet list.

Answer:"""
)

_STOP = {"about", "with", "something", "movie", "movies", "film", "films", "that", "this", "want", "watch",
         "tonight", "good", "great", "like", "some", "from", "under", "minutes", "hours", "series", "show",
         "documentary", "documentaries", "youtube", "video", "videos", "please", "really", "very", "into"}


@dataclass
class Recommendation:
    question: str
    plan: QueryPlan
    parsed_by: str
    relaxations: list[str]
    my_list: list[tuple[Title, float]]
    new: list[tuple[Title, float]]
    answer: str = ""
    cited: list[str] = field(default_factory=list)
    bad_citations: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def labels(self) -> dict[str, Title]:
        out = {f"L{i}": t for i, (t, _) in enumerate(self.my_list, 1)}
        out.update({f"N{i}": t for i, (t, _) in enumerate(self.new, 1)})
        return out

    def show(self) -> None:
        print(f"Q: {self.question}")
        print(f"plan ({self.parsed_by}): search='{self.plan.search_text}' | {self.plan.constraints_text()}")
        for r in self.relaxations:
            print(f"  relaxed: {r}")
        print("\nFrom your list:")
        for i, (t, s) in enumerate(self.my_list, 1):
            print(f"  [L{i}] {t.title} ({t.year}) {t.kind}, {t.runtime_min or '?'} min  score={s:.3f}")
        if self.new:
            print("New:")
            for i, (t, s) in enumerate(self.new, 1):
                print(f"  [N{i}] {t.title} ({t.year}) {t.kind}, {t.runtime_min or '?'} min  score={s:.3f}  {t.url or ''}")
        if self.answer:
            print("\n" + self.answer)
        if self.bad_citations:
            print(f"\n!! The answer cites labels that don't exist: {self.bad_citations}")
        print("\n" + " | ".join(f"{k} {v:.1f}s" for k, v in self.timings.items()))


# ---------------------------------------------------------------- filtering
def _passes(t: Title, plan: QueryPlan, genres: bool = True, years: bool = True) -> bool:
    kinds = set(plan.kinds) - {"youtube"}
    if plan.kinds and not kinds:
        return False  # only YouTube requested: nothing in the saved list qualifies
    if kinds and t.kind not in kinds:
        return False
    if plan.max_runtime_min and t.runtime_min and t.runtime_min > plan.max_runtime_min:
        return False  # unknown runtime passes: better to show it than to hide it
    if years and t.year:
        if plan.min_year and t.year < plan.min_year:
            return False
        if plan.max_year and t.year > plan.max_year:
            return False
    if genres and plan.genres and not set(plan.genres) & set(t.genres):
        return False
    return True


def filter_titles(titles: list[Title], plan: QueryPlan, min_results: int = 1) -> tuple[set[str], list[str]]:
    """Hard filters first; if nothing survives, relax the softest ones."""
    relaxed: list[str] = []
    ids = {t.item_id for t in titles if _passes(t, plan)}
    if len(ids) < min_results and plan.genres:
        ids = {t.item_id for t in titles if _passes(t, plan, genres=False)}
        relaxed.append(f"no saved title matched genres {plan.genres}; ignored the genre filter")
    if len(ids) < min_results and (plan.min_year or plan.max_year):
        ids = {t.item_id for t in titles if _passes(t, plan, genres=False, years=False)}
        relaxed.append("no saved title matched the years; ignored the year filter")
    return ids, relaxed


def _find_in_list(titles: list[Title], name: Optional[str]) -> Optional[Title]:
    if not name:
        return None
    n = name.lower().strip()
    for t in titles:
        if t.title.lower() == n or (t.original_title or "").lower() == n:
            return t
    for t in titles:
        if n in t.title.lower():
            return t
    return None


# ---------------------------------------------------------------- discovery
def discover(
    plan: QueryPlan,
    tmdb: TMDbClient,
    youtube: YouTubeClient,
    exclude: set[str],
    seeds: list[Title],
    reference: Optional[Title] = None,
    max_details: int = 30,
) -> list[Title]:
    """Find candidates the user has NOT saved, honouring the hard constraints."""
    found: list[Title] = []
    kinds = set(plan.kinds) or {"film"}

    if tmdb.available and kinds - {"youtube"}:
        pool: list[tuple[str, int]] = []  # (media_type, id), in priority order

        # 1. "like X": TMDb's own item-to-item recommendations
        if reference and reference.tmdb_id:
            media = "tv" if reference.item_id.startswith("tv:") else "movie"
            pool += [(media, r["id"]) for r in tmdb.recommendations(media, reference.tmdb_id)[:10]]
        elif plan.similar_to:
            hits = tmdb.search("movie", plan.similar_to) or tmdb.search("tv", plan.similar_to)
            if hits:
                media = "movie" if "title" in hits[0] else "tv"
                pool += [(media, r["id"]) for r in tmdb.recommendations(media, hits[0]["id"])[:10]]

        # 2. "because you saved X": recommendations seeded by the best matches in your list
        for seed in seeds[:2]:
            if seed.tmdb_id:
                media = "tv" if seed.item_id.startswith("tv:") else "movie"
                pool += [(media, r["id"]) for r in tmdb.recommendations(media, seed.tmdb_id)[:6]]

        # 3. Structured discovery: genres, runtime, years (+ topic keywords)
        words = [w for w in re.findall(r"[a-zA-Z]{4,}", plan.search_text.lower()) if w not in _STOP][:3]
        keyword_ids = [k for k in (tmdb.search_keyword(w) for w in words) if k]
        medias = []
        if kinds & {"film", "documentary", "short"}:
            medias.append("movie")
        if "series" in kinds:
            medias.append("tv")
        for media in medias:
            date_key = "primary_release_date" if media == "movie" else "first_air_date"
            params: dict = {"include_adult": "false", "vote_count.gte": 50 if "documentary" in kinds else 150}
            genre_map = MOVIE_NAME_TO_ID if media == "movie" else SHARED_TO_TV_ID
            if "documentary" in kinds:
                params["with_genres"] = "99"
            elif plan.genres:
                params["with_genres"] = "|".join(str(genre_map[g]) for g in plan.genres if g in genre_map)
            if media == "movie":
                cap = plan.max_runtime_min
                if "short" in kinds and kinds == {"short"}:
                    cap = min(cap or 40, 40)
                if cap:
                    params["with_runtime.lte"] = cap
            if plan.min_year:
                params[f"{date_key}.gte"] = f"{plan.min_year}-01-01"
            if plan.max_year:
                params[f"{date_key}.lte"] = f"{plan.max_year}-12-31"
            if keyword_ids:
                pool += [(media, r["id"]) for r in tmdb.discover(
                    media, sort_by="popularity.desc", with_keywords="|".join(map(str, keyword_ids)), **params)[:10]]
            pool += [(media, r["id"]) for r in tmdb.discover(media, sort_by="popularity.desc", **params)[:10]]
            pool += [(media, r["id"]) for r in tmdb.discover(
                media, sort_by="vote_average.desc", **{**params, "vote_count.gte": params["vote_count.gte"] * 4})[:10]]

        seen: set[str] = set()
        for media, tid in pool:
            item_id = f"{media}:{tid}"
            if item_id in seen or item_id in exclude:
                continue
            seen.add(item_id)
            if len(seen) > max_details:
                break
            d = tmdb.details(media, tid)
            if not d:
                continue
            t = tmdb.to_title(media, d, source="tmdb_discovery")
            if _passes(t, plan, genres=False) and (t.overview or "").strip():
                found.append(t)

    if "youtube" in kinds and youtube.available:
        found += youtube.search(plan.search_text, max_results=12, max_minutes=plan.max_runtime_min)
    return found


# ---------------------------------------------------------------- the loop
class MovieRecommender:
    def __init__(self, index: MovieIndex, llm=None, reranker=None, tmdb: TMDbClient | None = None,
                 youtube: YouTubeClient | None = None, embed_model=None):
        self.index = index
        self.titles = index.titles
        self.llm = llm
        self.reranker = reranker
        self.tmdb = tmdb or TMDbClient(offline=True)
        self.youtube = youtube or YouTubeClient(None, offline=True)
        self.embed_model = embed_model or index.embed_model

    def retrieve_my_list(self, plan: QueryPlan, k: int = 5, method: str = "hybrid", use_reranker: bool = True):
        allowed, relaxed = filter_titles(self.titles, plan)
        reference = _find_in_list(self.titles, plan.similar_to)
        query = plan.search_text
        if reference:  # "like Devs": search with Devs' own description, and don't recommend Devs itself
            query = f"{plan.search_text}\n{reference.overview[:400]}"
            allowed.discard(reference.item_id)
        hits = self.index.search(query, method=method, top_k=k, allowed_ids=allowed,
                                 reranker=self.reranker if use_reranker else None)
        return self.index.titles_for(hits), relaxed, reference, query

    def recommend(self, question: str, k_list: int = 5, k_new: int = 3, parse_with_llm: bool = True,
                  generate: bool = True, method: str = "hybrid", use_reranker: bool = True) -> Recommendation:
        t0 = time.perf_counter()
        timings: dict[str, float] = {}

        plan, how = parse_query(question, self.llm if parse_with_llm else None)
        timings["parse"] = time.perf_counter() - t0

        t = time.perf_counter()
        mine, relaxed, reference, query = self.retrieve_my_list(plan, k_list, method, use_reranker)
        timings["retrieve"] = time.perf_counter() - t

        new: list[tuple[Title, float]] = []
        need_new = plan.wants_new or len(mine) < k_list or bool({"documentary", "youtube"} & set(plan.kinds))
        if need_new and k_new > 0:
            t = time.perf_counter()
            candidates = discover(plan, self.tmdb, self.youtube, exclude={x.item_id for x in self.titles},
                                  seeds=[x for x, _ in mine], reference=reference)
            hits = rank_in_memory(candidates, query, self.embed_model, top_k=max(k_new * 4, 10))
            if self.reranker is not None and use_reranker and hits:
                hits = rerank(self.reranker, query, hits, k_new)
            by_id = {c.item_id: c for c in candidates}
            new = [(by_id[h.node.node_id], float(h.score or 0)) for h in hits[:k_new]]
            timings["discover"] = time.perf_counter() - t

        rec = Recommendation(question, plan, how, relaxed, mine, new, timings=timings)
        if generate and self.llm is not None:
            t = time.perf_counter()
            rec.answer = self._generate(rec, k_list, k_new)
            timings["generate"] = time.perf_counter() - t
            rec.cited = sorted(set(re.findall(r"\[([LN]\d+)\]", rec.answer)))
            rec.bad_citations = [c for c in rec.cited if c not in rec.labels()]
        timings["total"] = time.perf_counter() - t0
        return rec

    def _generate(self, rec: Recommendation, k_list: int, k_new: int) -> str:
        return self.llm.complete(build_prompt(rec, k_list, k_new)).text.strip()


def context_text(rec: Recommendation) -> str:
    """The candidates exactly as the LLM sees them (also used by the judge)."""

    def block(items, prefix):
        if not items:
            return "(none)"
        return "\n\n".join(f"[{prefix}{i}] {t.card()}" for i, (t, _) in enumerate(items, 1))

    return block(rec.my_list, "L"), block(rec.new, "N")


def build_prompt(rec: Recommendation, k_list: int = 5, k_new: int = 3) -> str:
    my_list, new = context_text(rec)
    return ANSWER_PROMPT.format(
        question=rec.question,
        constraints=rec.plan.constraints_text(),
        relaxed=("Note: " + "; ".join(rec.relaxations) + "\n") if rec.relaxations else "",
        my_list=my_list,
        new=new,
        k_list=min(k_list, 3),
        k_new=k_new,
    )
