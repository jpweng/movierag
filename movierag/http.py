"""Tiny JSON-on-disk cache for API calls.

Why bother: TMDb/YouTube calls are slow-ish and rate-limited, and a notebook
gets re-run dozens of times. Caching makes re-runs instant and free, and makes
the pipeline reproducible (same inputs -> same index).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional


class JsonCache:
    def __init__(self, directory: Path, max_age_days: Optional[float] = None):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.max_age_s = None if max_age_days is None else max_age_days * 86400

    def _path(self, key: str) -> Path:
        digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
        return self.dir / digest[:2] / f"{digest}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        if self.max_age_s is not None and time.time() - path.stat().st_mtime > self.max_age_s:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def set(self, key: str, value: Any) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def get_or_fetch(self, key: str, fetch: Callable[[], Any]) -> Any:
        hit = self.get(key)
        if hit is not None:
            return hit
        value = fetch()
        if value is not None:
            self.set(key, value)
        return value


def cache_key(url: str, params: dict[str, Any]) -> str:
    # Never put secrets in the cache key (the key is hashed anyway, but be tidy).
    clean = {k: v for k, v in sorted(params.items()) if k not in {"api_key", "key"}}
    return url + "?" + json.dumps(clean, sort_keys=True, default=str)
