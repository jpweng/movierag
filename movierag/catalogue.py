"""Step 1 of the pipeline: saved reels -> a clean catalogue of Titles.

bookmarks (91 reels)  --group by IMDb id-->  ~80 titles  --TMDb-->  enriched titles
                                                           --IMDb dump--> ratings

The catalogue (catalogue.json) is the single source of truth. The vector
index, the BM25 index and the filters are all derived from it, so you can
delete the index at any time and rebuild it in seconds.
"""

from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Iterable

from pydantic import TypeAdapter

from .imdb import download_ratings, load_ratings
from .schemas import Bookmark, Title
from .tmdb import TMDbClient, parse_tmdb_url

log = logging.getLogger(__name__)

_KIND_FROM_BOOKMARK = {
    "film": "film", "tv movie": "film", "video": "film", "animated anthology film": "film",
    "tv series": "series", "tv mini series": "series", "anime tv series": "series", "tv episode": "series",
    "short": "short", "short film": "short",
}


def load_bookmarks(path: Path) -> list[Bookmark]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    reels = raw["reels"] if isinstance(raw, dict) else raw
    fields = set(Bookmark.model_fields)
    return [Bookmark(**{k: v for k, v in r.items() if k in fields}) for r in reels]


def imdb_id_of(b: Bookmark) -> str | None:
    m = re.search(r"tt\d{6,9}", b.imdb_url or "")
    return m.group(0) if m else None


def _confidence(statuses: Iterable[str | None]) -> str:
    s = " ".join((x or "").lower() for x in statuses)
    if "confirmed" in s:
        return "confirmed"
    if "candidate" in s or "tentative" in s:
        return "candidate"
    return "unknown"


def group_bookmarks(bookmarks: list[Bookmark]) -> tuple["OrderedDict[str, list[Bookmark]]", list[Bookmark]]:
    """Several reels can point to the same film (Kalki 2898 AD was saved twice)."""
    groups: OrderedDict[str, list[Bookmark]] = OrderedDict()
    skipped: list[Bookmark] = []
    for b in bookmarks:
        if (b.match_status or "").lower().startswith("unresolved"):
            skipped.append(b)
            continue
        key = imdb_id_of(b)
        if key:
            key = f"imdb:{key}"
        elif parse_tmdb_url(b.tmdb_url):
            media, tid = parse_tmdb_url(b.tmdb_url)  # type: ignore[misc]
            key = f"{media}:{tid}"
        elif b.title:
            key = f"title:{b.title.lower()}:{b.year}"
        else:
            skipped.append(b)
            continue
        groups.setdefault(key, []).append(b)
    return groups, skipped


def _fallback_title(key: str, group: list[Bookmark]) -> Title:
    """When TMDb can't help (offline, no key, not found): keep what the reel told us."""
    first = group[0]
    kind = _KIND_FROM_BOOKMARK.get((first.kind or "").lower(), "film")
    imdb = imdb_id_of(first)
    overview = " ".join(dict.fromkeys(b.evidence for b in group if b.evidence))
    return Title(
        item_id=key if not key.startswith("title:") else f"bm:{first.id}",
        kind=kind,  # type: ignore[arg-type]
        title=first.title or "?",
        year=first.year,
        overview=overview,
        imdb_id=imdb,
        url=first.imdb_url,
    )


def build_catalogue(
    bookmarks: list[Bookmark],
    tmdb: TMDbClient,
    ratings_cache_dir: Path | None = None,
    progress: Callable[[str], None] = print,
) -> list[Title]:
    groups, skipped = group_bookmarks(bookmarks)
    progress(f"{len(bookmarks)} reels -> {len(groups)} distinct titles ({len(skipped)} unresolved, skipped)")

    by_id: OrderedDict[str, Title] = OrderedDict()
    for i, (key, group) in enumerate(groups.items(), 1):
        title: Title | None = None
        episode_note = None
        resolved: tuple[str, int] | None = None

        if key.startswith("imdb:") and tmdb.available:
            hit = tmdb.find_by_imdb(key.split(":", 1)[1])
            if hit:
                resolved, episode_note = (hit[0], hit[1]), hit[2]
        elif key.startswith(("movie:", "tv:")):
            media, tid = key.split(":")
            resolved = (media, int(tid))

        if resolved and tmdb.available:
            details = tmdb.details(*resolved)
            if details:
                title = tmdb.to_title(resolved[0], details)
        if title is None:
            title = _fallback_title(key, group)

        # Merge: two saved episodes of Doctor Who -> one "Doctor Who" title.
        existing = by_id.get(title.item_id)
        if existing:
            existing.bookmarks.extend(group)
            title = existing
        else:
            title.bookmarks = list(group)
            by_id[title.item_id] = title
        if episode_note:
            title.keywords = list(dict.fromkeys(title.keywords + [f"episode: {episode_note}"]))
        title.confidence = _confidence(b.match_status for b in title.bookmarks)  # type: ignore[assignment]

        if i % 10 == 0:
            progress(f"  enriched {i}/{len(groups)}")

    titles = list(by_id.values())

    if ratings_cache_dir is not None and not tmdb.offline:
        path = download_ratings(ratings_cache_dir)
        if path:
            wanted = {t.imdb_id for t in titles if t.imdb_id}
            ratings = load_ratings(path, wanted)
            for t in titles:
                if t.imdb_id in ratings:
                    t.imdb_rating, t.imdb_votes = ratings[t.imdb_id]
            progress(f"IMDb ratings attached to {sum(t.imdb_rating is not None for t in titles)} titles")

    enriched = sum(t.enriched for t in titles)
    progress(f"catalogue: {len(titles)} titles, {enriched} enriched from TMDb, TMDb network calls: {tmdb.calls}")
    return titles


_ADAPTER = TypeAdapter(list[Title])


def save_catalogue(titles: list[Title], path: Path) -> None:
    Path(path).write_bytes(_ADAPTER.dump_json(titles, indent=2))


def load_catalogue(path: Path) -> list[Title]:
    return _ADAPTER.validate_json(Path(path).read_bytes())
