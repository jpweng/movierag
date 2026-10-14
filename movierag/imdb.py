"""IMDb ratings from IMDb's official (non-commercial) dataset dump.

IMDb has no free API, but it publishes daily TSV files for personal and
non-commercial use: https://developer.imdb.com/non-commercial-datasets/
title.ratings.tsv.gz (~8 MB) maps tconst (tt1234567) -> rating + votes.
"""

from __future__ import annotations

import csv
import gzip
import io
import time
from pathlib import Path

import requests

RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"


def download_ratings(cache_dir: Path, max_age_days: float = 7.0) -> Path | None:
    path = Path(cache_dir) / "title.ratings.tsv.gz"
    fresh = path.exists() and time.time() - path.stat().st_mtime < max_age_days * 86400
    if fresh:
        return path
    try:
        resp = requests.get(RATINGS_URL, timeout=60)
        resp.raise_for_status()
    except requests.RequestException:
        return path if path.exists() else None  # stale file beats nothing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    return path


def load_ratings(path: Path, wanted: set[str]) -> dict[str, tuple[float, int]]:
    """Stream the 1.5M-row file and keep only the IDs we care about."""
    out: dict[str, tuple[float, int]] = {}
    with gzip.open(path, "rb") as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8"), delimiter="\t")
        next(reader, None)  # header: tconst averageRating numVotes
        for tconst, rating, votes in reader:
            if tconst in wanted:
                out[tconst] = (float(rating), int(votes))
    return out
