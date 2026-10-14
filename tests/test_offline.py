"""End-to-end test of the plumbing, with no network, no GPU and no model downloads.

TMDb is replaced by a fake that invents plausible payloads, the embedding model
by a hashing bag-of-words, and the LLM by LlamaIndex's MockLLM.

Run:  uv run pytest -q
"""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pytest
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import MockLLM

from movierag.catalogue import build_catalogue, load_bookmarks, load_catalogue, save_catalogue
from movierag.evaluate import evaluate_retrieval, plan_field_matches, retrieval_metrics
from movierag.index import MovieIndex, rrf_fuse
from movierag.query_parser import parse_query
from movierag.recommender import MovieRecommender, filter_titles
from movierag.schemas import QueryPlan
from movierag.tmdb import TMDbClient
from movierag.youtube import YouTubeClient, iso8601_minutes

ROOT = Path(__file__).resolve().parent.parent


class HashEmbedding(BaseEmbedding):
    """Deterministic toy embedding: hashed bag of words, L2-normalised."""

    dim: int = 256

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for w in re.findall(r"[a-z]{3,}", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._vec(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._vec(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._vec(query)


class FakeTMDb(TMDbClient):
    """Answers like TMDb would, using the reels' own titles."""

    def __init__(self, bookmarks):
        super().__init__(api_key="fake", offline=False)
        self.by_imdb = {}
        for b in bookmarks:
            m = re.search(r"tt\d+", b.imdb_url or "")
            if m:
                self.by_imdb[m.group(0)] = b
        self.names = {}
        self.imdb_of = {}

    def _get(self, path, **params):
        self.calls += 1
        if path.startswith("/find/"):
            imdb = path.split("/")[-1]
            b = self.by_imdb.get(imdb)
            if not b:
                return {"movie_results": [], "tv_results": [], "tv_episode_results": []}
            tid = int(imdb[2:]) % 100000
            self.names[tid] = (b.title, b.year, b.evidence or b.caption or "")
            self.imdb_of[tid] = imdb
            kind = (b.kind or "").lower()
            if kind == "tv episode":
                return {"movie_results": [], "tv_results": [], "tv_episode_results": [{"show_id": 777, "name": b.title}]}
            if "series" in kind:
                return {"movie_results": [], "tv_results": [{"id": tid}], "tv_episode_results": []}
            return {"movie_results": [{"id": tid}], "tv_results": [], "tv_episode_results": []}
        m = re.fullmatch(r"/(movie|tv)/(\d+)", path)
        if m:
            media, tid = m.group(1), int(m.group(2))
            name, year, plot = self.names.get(tid, (f"Discovered {tid}", 2020, f"A documentary about space number {tid}."))
            base = {
                "id": tid, "overview": plot, "genres": [{"id": 18, "name": "Drama"}],
                "keywords": {"keywords": [{"name": "space"}], "results": [{"name": "space"}]},
                "external_ids": {"imdb_id": self.imdb_of.get(tid, f"tt{tid:07d}")},
                "watch/providers": {"results": {"CH": {"flatrate": [{"provider_name": "Netflix"}]}}},
                "vote_average": 7.1, "vote_count": 900,
            }
            if tid >= 90000:
                base["genres"] = [{"id": 99, "name": "Documentary"}]
            if media == "movie":
                return {**base, "title": name, "original_title": name, "release_date": f"{year or 2000}-05-01",
                        "runtime": 85 if tid >= 90000 else 110,
                        "credits": {"crew": [{"job": "Director", "name": "Some Director"}], "cast": [{"name": "Actor A"}]}}
            return {**base, "name": name, "original_name": name, "first_air_date": f"{year or 2000}-01-01",
                    "episode_run_time": [50], "number_of_seasons": 2, "created_by": [{"name": "Creator"}],
                    "aggregate_credits": {"cast": [{"name": "Actor B"}]}, "genres": [{"id": 10765, "name": "Sci-Fi & Fantasy"}]}
        if path.startswith("/discover/") or path.endswith("/recommendations"):
            return {"results": [{"id": 90000 + i} for i in range(8)]}
        if path == "/search/keyword":
            return {"results": [{"id": 1, "name": params.get("query", "")}]}
        if path.startswith("/search/"):
            return {"results": [{"id": 90001, "title": "x"}]}
        return None


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    bookmarks = load_bookmarks(ROOT / "movies-reels.json")
    tmdb = FakeTMDb(bookmarks)
    titles = build_catalogue(bookmarks, tmdb, ratings_cache_dir=None, progress=lambda *_: None)
    path = tmp_path_factory.mktemp("cat") / "catalogue.json"
    save_catalogue(titles, path)
    titles = load_catalogue(path)
    embed = HashEmbedding()
    index = MovieIndex(titles, embed, persist_dir=tmp_path_factory.mktemp("chroma"))
    rec = MovieRecommender(index, llm=MockLLM(max_tokens=40), tmdb=tmdb, youtube=YouTubeClient(None, offline=True))
    return {"bookmarks": bookmarks, "titles": titles, "index": index, "rec": rec}


def test_catalogue_groups_and_merges(world):
    titles = world["titles"]
    assert 60 < len(titles) < len(world["bookmarks"])
    assert len({t.item_id for t in titles}) == len(titles)
    kalki = [t for t in titles if "Kalki" in t.title]
    assert len(kalki) == 1 and len(kalki[0].bookmarks) == 2  # saved twice -> one title
    assert sum(t.enriched for t in titles) > 60


def test_vector_bm25_hybrid(world):
    index = world["index"]
    for method in ("vector", "bm25", "hybrid"):
        hits = index.search("Paganini violin virtuoso", method=method, top_k=5)
        assert hits, method
    top = index.titles_for(index.search("Paganini violin", method="bm25", top_k=3))
    assert any("Violinist" in t.title for t, _ in top)


def test_filters_respected(world):
    index = world["index"]
    plan = QueryPlan(search_text="anything", kinds=["series"])
    allowed, _ = filter_titles(world["titles"], plan)
    hits = index.titles_for(index.search("war", top_k=10, allowed_ids=allowed))
    assert hits and all(t.kind == "series" for t, _ in hits)


def test_relaxation():
    from movierag.schemas import Title

    titles = [Title(item_id="a", title="A", genres=["Drama"], year=2001)]
    ids, relaxed = filter_titles(titles, QueryPlan(search_text="x", genres=["Western"]))
    assert ids == {"a"} and relaxed


def test_rrf_prefers_agreement():
    from llama_index.core.schema import NodeWithScore, TextNode

    a, b, c = (TextNode(id_=x, text=x) for x in "abc")
    fused = rrf_fuse([[NodeWithScore(node=a, score=1), NodeWithScore(node=b, score=0.9)],
                      [NodeWithScore(node=b, score=30), NodeWithScore(node=c, score=20)]])
    assert fused[0].node.node_id == "b"


def test_recommend_end_to_end(world):
    r = world["rec"].recommend("a documentary under 90 minutes about space", k_list=3, k_new=3)
    assert r.plan.kinds == ["documentary"] and r.plan.max_runtime_min == 90
    assert r.new and all(t.kind == "documentary" and (t.runtime_min or 0) <= 90 for t, _ in r.new)
    assert all(t.item_id not in world["index"].by_id for t, _ in r.new)
    assert isinstance(r.answer, str)


def test_like_reference_excluded(world):
    r = world["rec"].recommend("something like Devs", k_list=5, k_new=0, generate=False)
    assert r.my_list and all(t.title != "Devs" for t, _ in r.my_list)


def test_parser_rules():
    plan, how = parse_query("an 80s or older kung fu movie")
    assert how == "rules" and plan.max_year == 1989 and plan.min_year is None and "Action" in plan.genres
    plan, _ = parse_query("a comedy under 1h30")
    assert plan.max_runtime_min == 90
    assert plan_field_matches(plan, {"genres": ["Comedy"], "max_runtime_min": 90}) == {"genres": True, "max_runtime_min": True}


def test_metrics_and_eval_set(world):
    assert retrieval_metrics(["x", "y"], {"y"}, 5) == {"hit@5": 1.0, "recall@5": 1.0, "mrr": 0.5}
    import json

    data = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))
    imdb_in_list = {t.imdb_id for t in world["titles"]}
    for q in data["retrieval"]:
        missing = set(q["relevant_imdb"]) - imdb_in_list
        assert not missing, (q["question"], missing)
    index = world["index"]
    by_id = index.by_id
    avg, rows = evaluate_retrieval(
        data["retrieval"],
        lambda q: [by_id[h.node.node_id].imdb_id for h in index.search(q, "hybrid", top_k=5)],
    )
    assert 0 <= avg["mrr"] <= 1 and len(rows) == len(data["retrieval"])


def test_youtube_duration():
    assert iso8601_minutes("PT1H02M30S") == 63
    assert iso8601_minutes("PT45S") == 1
    assert iso8601_minutes(None) is None
