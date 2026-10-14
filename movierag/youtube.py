"""YouTube Data API v3: discovery of videos (video essays, docs, shorts...).

Quota: a search costs 100 units, a videos.list call 1 unit; the free daily
quota is 10,000 units, so ~90 searches/day. The JSON cache keeps re-runs free.
Docs: https://developers.google.com/youtube/v3/docs/search/list
"""

from __future__ import annotations

import re
from typing import Optional

import requests

from .http import JsonCache, cache_key
from .schemas import Title

API = "https://www.googleapis.com/youtube/v3"


def iso8601_minutes(duration: str | None) -> Optional[int]:
    """'PT1H02M30S' -> 63 (rounded up)."""
    if not duration:
        return None
    m = re.fullmatch(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration)
    if not m:
        return None
    d, h, mi, s = (int(x) if x else 0 for x in m.groups())
    seconds = d * 86400 + h * 3600 + mi * 60 + s
    return max(1, -(-seconds // 60))


class YouTubeClient:
    def __init__(self, api_key: str | None, cache: JsonCache | None = None, offline: bool = False):
        self.api_key = api_key
        self.cache = cache
        self.offline = offline or not api_key
        self.session = requests.Session()

    @classmethod
    def from_config(cls, cfg) -> "YouTubeClient":
        return cls(cfg.youtube_api_key, JsonCache(cfg.cache_dir / "youtube", max_age_days=7), cfg.offline)

    @property
    def available(self) -> bool:
        return not self.offline

    def _get(self, endpoint: str, **params) -> Optional[dict]:
        url = f"{API}/{endpoint}"
        key = cache_key(url, params)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        if self.offline:
            return None
        resp = self.session.get(url, params={**params, "key": self.api_key}, timeout=20)
        if resp.status_code == 403:
            raise RuntimeError(f"YouTube API refused the request (quota or key): {resp.text[:300]}")
        resp.raise_for_status()
        data = resp.json()
        if self.cache is not None:
            self.cache.set(key, data)
        return data

    def search(
        self,
        query: str,
        max_results: int = 10,
        max_minutes: int | None = None,
        language: str = "en",
    ) -> list[Title]:
        # YouTube only offers coarse duration buckets: short <4min, medium 4-20, long >20.
        duration = "any"
        if max_minutes is not None:
            duration = "short" if max_minutes < 4 else "medium" if max_minutes <= 20 else "any"
        found = self._get(
            "search",
            part="snippet",
            q=query,
            type="video",
            maxResults=max_results,
            videoDuration=duration,
            relevanceLanguage=language,
            safeSearch="moderate",
        )
        ids = [it["id"]["videoId"] for it in (found or {}).get("items", []) if it.get("id", {}).get("videoId")]
        if not ids:
            return []
        details = self._get("videos", part="snippet,contentDetails,statistics", id=",".join(ids))
        titles = []
        for v in (details or {}).get("items", []):
            sn, cd, st = v.get("snippet", {}), v.get("contentDetails", {}), v.get("statistics", {})
            minutes = iso8601_minutes(cd.get("duration"))
            if max_minutes is not None and minutes is not None and minutes > max_minutes:
                continue  # exact filter on top of the coarse bucket
            published = sn.get("publishedAt", "")
            titles.append(
                Title(
                    item_id=f"yt:{v['id']}",
                    source="youtube",
                    kind="youtube",
                    title=sn.get("title", "?"),
                    year=int(published[:4]) if published[:4].isdigit() else None,
                    runtime_min=minutes,
                    overview=(sn.get("description") or "")[:800],
                    keywords=(sn.get("tags") or [])[:15],
                    channel=sn.get("channelTitle"),
                    views=int(st["viewCount"]) if st.get("viewCount", "").isdigit() else None,
                    url=f"https://www.youtube.com/watch?v={v['id']}",
                    enriched=True,
                )
            )
        return titles
