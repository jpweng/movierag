"""Data shapes used everywhere. Pydantic gives us validation + JSON for free."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Kind = Literal["film", "series", "documentary", "short", "youtube"]
Source = Literal["my_list", "tmdb_discovery", "youtube"]

# One shared genre vocabulary. TMDb uses slightly different genres for movies
# and TV ("Sci-Fi & Fantasy" vs "Science Fiction"); we normalise to these.
GENRES: list[str] = [
    "Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary",
    "Drama", "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
    "Romance", "Science Fiction", "Thriller", "War", "Western",
]


class Bookmark(BaseModel):
    """One saved reel/post, as found in movies-reels.json."""

    id: str
    platform: str
    reel_url: Optional[str] = None
    creator: Optional[str] = None
    caption: Optional[str] = None
    title: Optional[str] = None
    year: Optional[int] = None
    kind: Optional[str] = None
    imdb_url: Optional[str] = None
    tmdb_url: Optional[str] = None
    match_status: Optional[str] = None
    evidence: Optional[str] = None


class Title(BaseModel):
    """A watchable thing: one film, series, documentary, short or video.

    `item_id` is our primary key, e.g. "movie:27205", "tv:1399", "yt:dQw4w9WgXcQ"
    or "imdb:tt0047216" when TMDb could not resolve the item.
    """

    item_id: str
    source: Source = "my_list"
    kind: Kind = "film"
    title: str
    original_title: Optional[str] = None
    year: Optional[int] = None
    runtime_min: Optional[int] = None
    genres: list[str] = Field(default_factory=list)
    overview: str = ""
    tagline: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)
    directors: list[str] = Field(default_factory=list)
    cast: list[str] = Field(default_factory=list)
    countries: list[str] = Field(default_factory=list)
    original_language: Optional[str] = None
    seasons: Optional[int] = None

    tmdb_id: Optional[int] = None
    tmdb_rating: Optional[float] = None
    tmdb_votes: Optional[int] = None
    imdb_id: Optional[str] = None
    imdb_rating: Optional[float] = None
    imdb_votes: Optional[int] = None
    providers: list[str] = Field(default_factory=list)  # streaming in WATCH_REGION

    url: Optional[str] = None  # TMDb / YouTube page
    channel: Optional[str] = None  # YouTube channel
    views: Optional[int] = None

    # Personal signal: which saved reels point to this title, and how sure we are.
    bookmarks: list[Bookmark] = Field(default_factory=list)
    confidence: Literal["confirmed", "candidate", "unknown"] = "unknown"
    enriched: bool = False  # True when TMDb (or YouTube) metadata was fetched

    # ------------------------------------------------------------------ text
    def embedding_text(self) -> str:
        """What gets embedded and BM25-indexed: the *meaning* of the title."""
        lines = [f"{self.title}" + (f" ({self.year})" if self.year else "") + f" - {self.kind}"]
        if self.original_title and self.original_title != self.title:
            lines.append(f"Original title: {self.original_title}")
        if self.genres:
            lines.append("Genres: " + ", ".join(self.genres))
        if self.tagline:
            lines.append(f"Tagline: {self.tagline}")
        if self.overview:
            lines.append(f"Plot: {self.overview}")
        if self.keywords:
            lines.append("Keywords: " + ", ".join(self.keywords[:20]))
        if self.directors:
            lines.append("By: " + ", ".join(self.directors[:3]))
        if self.cast:
            lines.append("Cast: " + ", ".join(self.cast[:5]))
        if self.channel:
            lines.append(f"YouTube channel: {self.channel}")
        why = self.saved_because()
        if why:
            lines.append(f"Saved reel said: {why}")
        return "\n".join(lines)

    def saved_because(self, max_chars: int = 300) -> str:
        """Short summary of the reels' captions (often in French/other languages)."""
        bits = []
        for b in self.bookmarks:
            text = (b.caption or "").strip()
            if text and text.lower() != (self.title or "").lower():
                bits.append(text)
        joined = " | ".join(dict.fromkeys(bits))  # dedupe, keep order
        return joined[:max_chars]

    def card(self) -> str:
        """Compact one-block description for the LLM prompt."""
        facts = [self.kind]
        if self.year:
            facts.append(str(self.year))
        if self.runtime_min:
            facts.append(f"{self.runtime_min} min" + (" per episode" if self.kind == "series" else ""))
        if self.genres:
            facts.append("/".join(self.genres[:3]))
        if self.imdb_rating:
            facts.append(f"IMDb {self.imdb_rating:.1f}")
        elif self.tmdb_rating:
            facts.append(f"TMDb {self.tmdb_rating:.1f}")
        if self.providers:
            facts.append("streaming: " + ", ".join(self.providers[:3]))
        if self.channel:
            facts.append(f"channel: {self.channel}")
        body = self.overview or "(no plot available)"
        return f"{self.title} ({', '.join(facts)})\n{body[:600]}"


class QueryPlan(BaseModel):
    """What the user *means*, as structured data (the 'self-query' step)."""

    search_text: str = Field(
        description="The themes, mood, plot or style the user wants, as a short "
        "description. Leave out length, year and format constraints."
    )
    genres: list[str] = Field(
        default_factory=list,
        description="Zero or more genres, chosen ONLY from: " + ", ".join(GENRES),
    )
    kinds: list[Kind] = Field(
        default_factory=list,
        description="Formats explicitly requested: film, series, documentary, short, youtube. "
        "Empty if the user did not say.",
    )
    max_runtime_min: Optional[int] = Field(
        default=None, description="Maximum length in minutes, if the user gave one."
    )
    min_year: Optional[int] = Field(default=None, description="Earliest release year, if given.")
    max_year: Optional[int] = Field(default=None, description="Latest release year, if given.")
    similar_to: Optional[str] = Field(
        default=None, description="A title the user mentioned as a reference ('like X')."
    )
    wants_new: bool = Field(
        default=False,
        description="True only if the user explicitly wants things they have NOT saved / new discoveries.",
    )

    def constraints_text(self) -> str:
        parts = []
        if self.kinds:
            parts.append("format: " + "/".join(self.kinds))
        if self.genres:
            parts.append("genres: " + "/".join(self.genres))
        if self.max_runtime_min:
            parts.append(f"<= {self.max_runtime_min} min")
        if self.min_year:
            parts.append(f">= {self.min_year}")
        if self.max_year:
            parts.append(f"<= {self.max_year}")
        if self.similar_to:
            parts.append(f"like '{self.similar_to}'")
        if self.wants_new:
            parts.append("new discoveries")
        return ", ".join(parts) or "none"
