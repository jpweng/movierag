"""Wire all the pieces together in one call (used by notebooks 03-04 and the CLI)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .catalogue import load_catalogue
from .config import Config, get_config
from .index import MovieIndex
from .models import load_embed_model, load_llm, load_reranker
from .recommender import MovieRecommender
from .schemas import Title
from .tmdb import TMDbClient
from .youtube import YouTubeClient


@dataclass
class Components:
    cfg: Config
    titles: list[Title]
    embed_model: Any
    index: MovieIndex
    reranker: Any
    tmdb: TMDbClient
    youtube: YouTubeClient
    llm: Any
    recommender: MovieRecommender


def load_components(cfg: Config | None = None, with_llm: bool = True, with_reranker: bool = True,
                    llm_backend: str | None = None) -> Components:
    cfg = cfg or get_config()
    if not cfg.catalogue_path.exists():
        raise FileNotFoundError(f"{cfg.catalogue_path} not found: run notebooks/01_bookmarks_to_catalogue.ipynb first.")
    titles = load_catalogue(cfg.catalogue_path)
    embed_model = load_embed_model(cfg)
    index = MovieIndex(titles, embed_model, persist_dir=cfg.chroma_dir)
    reranker = load_reranker(cfg) if with_reranker else None
    tmdb = TMDbClient.from_config(cfg)
    youtube = YouTubeClient.from_config(cfg)
    llm = load_llm(cfg, backend=llm_backend) if with_llm else None
    recommender = MovieRecommender(index, llm=llm, reranker=reranker, tmdb=tmdb, youtube=youtube)
    return Components(cfg, titles, embed_model, index, reranker, tmdb, youtube, llm, recommender)
