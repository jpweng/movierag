"""TMDb (The Movie Database) client: the enrichment + discovery data source.

IMDb has no free general-purpose API, so TMDb is the standard workaround:
plots, runtimes, genres, keywords, cast, streaming providers per country,
and an IMDb-ID lookup (/find) that connects your bookmarks to TMDb records.

Docs: https://developer.themoviedb.org/reference/intro/getting-started
"""

from __future__ import annotations

import logging
import re
import statistics
import time
from typing import Any, Optional

import requests

from .http import JsonCache, cache_key
from .schemas import Title

log = logging.getLogger(__name__)

BASE_URL = "https://api.themoviedb.org/3"

# Stable TMDb genre IDs (see /genre/movie/list and /genre/tv/list).
MOVIE_GENRES = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance",
    878: "Science Fiction", 10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western",
}
TV_GENRES = {
    10759: "Action & Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 10762: "Kids", 9648: "Mystery",
    10763: "News", 10764: "Reality", 10765: "Sci-Fi & Fantasy", 10766: "Soap",
    10767: "Talk", 10768: "War & Politics", 37: "Western",
}
# TV genre names -> our shared vocabulary (schemas.GENRES)
TV_TO_SHARED = {
    "Action & Adventure": ["Action", "Adventure"],
    "Sci-Fi & Fantasy": ["Science Fiction", "Fantasy"],
    "War & Politics": ["War"],
    "Kids": ["Family"],
    "News": [], "Reality": [], "Soap": ["Drama"], "Talk": [],
}
MOVIE_NAME_TO_ID = {v: k for k, v in MOVIE_GENRES.items()}
SHARED_TO_TV_ID = {
    "Action": 10759, "Adventure": 10759, "Animation": 16, "Comedy": 35, "Crime": 80,
    "Documentary": 99, "Drama": 18, "Family": 10751, "Fantasy": 10765, "Mystery": 9648,
    "Science Fiction": 10765, "War": 10768, "Western": 37,
}


def normalise_genres(names: list[str]) -> list[str]:
    out: list[str] = []
    for name in names:
        for g in TV_TO_SHARED.get(name, [name]):
            if g != "TV Movie" and g not in out:
                out.append(g)
    return out


def parse_tmdb_url(url: str | None) -> Optional[tuple[str, int]]:
    """'https://www.themoviedb.org/movie/12345-foo' -> ('movie', 12345)."""
    if not url:
        return None
    m = re.search(r"themoviedb\.org/(movie|tv)/(\d+)", url)
    return (m.group(1), int(m.group(2))) if m else None


class TMDbError(RuntimeError):
    pass


class TMDbClient:
    def __init__(
        self,
        api_key: str | None = None,
        read_token: str | None = None,
        language: str = "en-US",
        region: str = "CH",
        cache: JsonCache | None = None,
        offline: bool = False,
        min_interval_s: float = 0.03,
    ):
        self.api_key = api_key
        self.read_token = read_token
        self.language = language
        self.region = region
        self.cache = cache
        self.offline = offline or not (api_key or read_token)
        self.min_interval_s = min_interval_s
        self._last_call = 0.0
        self.session = requests.Session()
        if read_token:
            self.session.headers["Authorization"] = f"Bearer {read_token}"
        self.calls = 0  # network calls made (cache hits don't count)

    @classmethod
    def from_config(cls, cfg) -> "TMDbClient":
        return cls(
            api_key=cfg.tmdb_api_key,
            read_token=cfg.tmdb_read_token,
            language=cfg.language,
            region=cfg.watch_region,
            cache=JsonCache(cfg.cache_dir / "tmdb", max_age_days=30),
            offline=cfg.offline,
        )

    @property
    def available(self) -> bool:
        return not self.offline

    # ------------------------------------------------------------ transport
    def _get(self, path: str, **params: Any) -> Optional[dict]:
        params = {"language": self.language, **params}
        url = BASE_URL + path
        key = cache_key(url, params)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return None if hit.get("__not_found__") else hit
        if self.offline:
            return None

        if self.api_key and not self.read_token:
            params["api_key"] = self.api_key
        for attempt in range(4):
            wait = self.min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            resp = self.session.get(url, params=params, timeout=20)
            self.calls += 1
            if resp.status_code == 429:  # rate limited: back off and retry
                time.sleep(float(resp.headers.get("Retry-After", 2 ** attempt)))
                continue
            if resp.status_code == 404:
                if self.cache is not None:
                    self.cache.set(key, {"__not_found__": True})
                return None
            if resp.status_code == 401:
                raise TMDbError("TMDb rejected the credentials (401). Check TMDB_API_KEY / TMDB_READ_TOKEN.")
            resp.raise_for_status()
            data = resp.json()
            if self.cache is not None:
                self.cache.set(key, data)
            return data
        raise TMDbError(f"TMDb kept rate-limiting {path}")

    # ------------------------------------------------------------ endpoints
    def find_by_imdb(self, imdb_id: str) -> Optional[tuple[str, int, Optional[str]]]:
        """IMDb ID -> (media_type, tmdb_id, episode_name or None)."""
        data = self._get(f"/find/{imdb_id}", external_source="imdb_id")
        if not data:
            return None
        if data.get("movie_results"):
            return "movie", data["movie_results"][0]["id"], None
        if data.get("tv_results"):
            return "tv", data["tv_results"][0]["id"], None
        if data.get("tv_episode_results"):  # a saved episode -> index the whole show
            ep = data["tv_episode_results"][0]
            return "tv", ep["show_id"], ep.get("name")
        return None

    def details(self, media_type: str, tmdb_id: int) -> Optional[dict]:
        extra = "keywords,external_ids,watch/providers," + (
            "credits" if media_type == "movie" else "aggregate_credits"
        )
        return self._get(f"/{media_type}/{tmdb_id}", append_to_response=extra)

    def discover(self, media_type: str, **params: Any) -> list[dict]:
        data = self._get(f"/discover/{media_type}", **params)
        return (data or {}).get("results", [])

    def recommendations(self, media_type: str, tmdb_id: int) -> list[dict]:
        data = self._get(f"/{media_type}/{tmdb_id}/recommendations")
        return (data or {}).get("results", [])

    def search_keyword(self, word: str) -> Optional[int]:
        """'submarine' -> TMDb keyword id, used by /discover?with_keywords=..."""
        data = self._get("/search/keyword", query=word)
        results = (data or {}).get("results", [])
        exact = [r for r in results if r.get("name", "").lower() == word.lower()]
        return (exact or results or [{}])[0].get("id")

    def search(self, media_type: str, query: str, year: int | None = None) -> list[dict]:
        params: dict[str, Any] = {"query": query, "include_adult": "false"}
        if year:
            params["year" if media_type == "movie" else "first_air_date_year"] = year
        data = self._get(f"/search/{media_type}", **params)
        return (data or {}).get("results", [])

    # ------------------------------------------------------------ parsing
    def to_title(self, media_type: str, d: dict, source: str = "my_list") -> Title:
        """Turn a /movie or /tv details payload into our Title shape."""
        is_movie = media_type == "movie"
        genres = normalise_genres([g["name"] for g in d.get("genres", [])])

        if is_movie:
            runtime = d.get("runtime") or None
            date = d.get("release_date") or ""
            kw = [k["name"] for k in (d.get("keywords") or {}).get("keywords", [])]
            crew = (d.get("credits") or {}).get("crew", [])
            directors = [c["name"] for c in crew if c.get("job") == "Director"]
            cast = [c["name"] for c in (d.get("credits") or {}).get("cast", [])[:6]]
            countries = [c.get("iso_3166_1") for c in d.get("production_countries", [])]
            title, original = d.get("title"), d.get("original_title")
        else:
            runtimes = [r for r in d.get("episode_run_time") or [] if r]
            if not runtimes and d.get("last_episode_to_air"):
                runtimes = [d["last_episode_to_air"].get("runtime") or 0]
            runtime = int(statistics.median(runtimes)) if any(runtimes) else None
            date = d.get("first_air_date") or ""
            kw = [k["name"] for k in (d.get("keywords") or {}).get("results", [])]
            directors = [c["name"] for c in d.get("created_by", [])]
            cast = [c["name"] for c in (d.get("aggregate_credits") or {}).get("cast", [])[:6]]
            countries = list(d.get("origin_country") or [])
            title, original = d.get("name"), d.get("original_name")

        if "Documentary" in genres:
            kind = "documentary"
        elif not is_movie:
            kind = "series"
        elif runtime and runtime <= 40:
            kind = "short"
        else:
            kind = "film"

        providers_block = (d.get("watch/providers") or {}).get("results", {}).get(self.region, {})
        providers = [p["provider_name"] for p in providers_block.get("flatrate", [])]

        imdb_id = (d.get("external_ids") or {}).get("imdb_id") or d.get("imdb_id")
        return Title(
            item_id=f"{media_type}:{d['id']}",
            source=source,  # type: ignore[arg-type]
            kind=kind,  # type: ignore[arg-type]
            title=title or original or "?",
            original_title=original,
            year=int(date[:4]) if date[:4].isdigit() else None,
            runtime_min=runtime,
            genres=genres,
            overview=d.get("overview") or "",
            tagline=d.get("tagline") or None,
            keywords=kw,
            directors=directors,
            cast=cast,
            countries=[c for c in countries if c],
            original_language=d.get("original_language"),
            seasons=d.get("number_of_seasons"),
            tmdb_id=d["id"],
            tmdb_rating=d.get("vote_average"),
            tmdb_votes=d.get("vote_count"),
            imdb_id=imdb_id,
            providers=providers,
            url=f"https://www.themoviedb.org/{media_type}/{d['id']}",
            enriched=True,
        )
