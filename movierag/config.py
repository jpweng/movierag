"""Central configuration, read once from environment variables / a .env file.

Everything that is a *choice* (which model, which backend, which region)
lives here, so notebooks stay short and experiments are one-line changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    value = _env(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    # --- data -----------------------------------------------------------
    bookmarks_path: Path
    # Generated state (index, caches) lives OUTSIDE Google Drive by default:
    # Chroma uses SQLite, and SQLite + a syncing cloud folder = corrupted locks.
    home: Path

    # --- external APIs --------------------------------------------------
    tmdb_api_key: str | None
    tmdb_read_token: str | None
    youtube_api_key: str | None
    watch_region: str
    language: str

    # --- models ---------------------------------------------------------
    llm_backend: str  # "hf" | "ollama" | "openai_like" | "mock"
    hf_model: str
    hf_load_in_4bit: bool
    ollama_model: str
    openai_like_base_url: str | None
    openai_like_model: str | None
    openai_like_api_key: str
    embed_model: str
    rerank_model: str

    # Offline mode: no downloads, no API calls, mock models.
    # Useful to check the plumbing on any machine in seconds.
    offline: bool

    @property
    def catalogue_path(self) -> Path:
        return self.home / "catalogue.json"

    @property
    def chroma_dir(self) -> Path:
        return self.home / "chroma"

    @property
    def cache_dir(self) -> Path:
        return self.home / "cache"

    def ensure_dirs(self) -> None:
        for d in (self.home, self.chroma_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)

    def summary(self) -> str:
        llm = {
            "hf": f"hf:{self.hf_model}{' (4-bit)' if self.hf_load_in_4bit else ''}",
            "ollama": f"ollama:{self.ollama_model}",
            "openai_like": f"openai_like:{self.openai_like_model} @ {self.openai_like_base_url}",
            "mock": "mock",
        }.get(self.llm_backend, self.llm_backend)
        return "\n".join(
            [
                f"bookmarks     : {self.bookmarks_path}",
                f"state dir     : {self.home}",
                f"LLM           : {llm}",
                f"embeddings    : {self.embed_model}",
                f"reranker      : {self.rerank_model}",
                f"TMDb key      : {'yes' if (self.tmdb_api_key or self.tmdb_read_token) else 'MISSING'}",
                f"YouTube key   : {'yes' if self.youtube_api_key else 'MISSING'}",
                f"watch region  : {self.watch_region}",
                f"offline mode  : {self.offline}",
            ]
        )


@lru_cache(maxsize=1)
def get_config() -> Config:
    offline = _env_bool("MOVIERAG_OFFLINE", False)
    cfg = Config(
        bookmarks_path=Path(_env("MOVIERAG_BOOKMARKS", str(PROJECT_DIR / "movies-reels.json"))),
        home=Path(_env("MOVIERAG_HOME", str(Path.home() / ".movierag"))),
        tmdb_api_key=_env("TMDB_API_KEY"),
        tmdb_read_token=_env("TMDB_READ_TOKEN"),
        youtube_api_key=_env("YOUTUBE_API_KEY"),
        watch_region=_env("WATCH_REGION", "CH"),
        language=_env("TMDB_LANGUAGE", "en-US"),
        llm_backend="mock" if offline else _env("LLM_BACKEND", "hf"),
        hf_model=_env("HF_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
        hf_load_in_4bit=_env_bool("HF_LOAD_IN_4BIT", True),
        ollama_model=_env("OLLAMA_MODEL", "qwen3.5:4b"),
        openai_like_base_url=_env("OPENAI_LIKE_BASE_URL"),
        openai_like_model=_env("OPENAI_LIKE_MODEL"),
        openai_like_api_key=_env("OPENAI_LIKE_API_KEY", "not-needed"),
        embed_model=_env("EMBED_MODEL", "BAAI/bge-small-en-v1.5"),
        rerank_model=_env("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"),
        offline=offline,
    )
    cfg.ensure_dirs()
    return cfg
