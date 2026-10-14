"""movierag: a small, readable RAG system that recommends what to watch
from your saved Facebook/Instagram reels, topped up with TMDb and YouTube."""

from .config import Config, get_config
from .schemas import Bookmark, QueryPlan, Title

__all__ = ["Config", "get_config", "Bookmark", "QueryPlan", "Title"]
