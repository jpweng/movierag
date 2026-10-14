# Architecture

How a question becomes a recommendation, and why each step is there. File references point
into the `movierag/` package.

## The big picture

```
                OFFLINE (notebook 01, once)                           ONLINE (per question)
┌──────────────────────────────────────────────────┐   ┌─────────────────────────────────────────────┐
│ movies-reels.json                                │   │ "something like Devs but a movie, < 2h"     │
│   │ group reels by IMDb/TMDb id                  │   │   │ 1. parse   (LLM + regex rules)          │
│   ▼                                              │   │   ▼                                         │
│ distinct titles ──TMDb──▶ plot, genres, runtime, │   │ QueryPlan(search_text, kinds, genres,       │
│                           cast, keywords,        │   │           max_runtime_min, years,           │
│                           streaming providers    │   │           similar_to, wants_new)            │
│                 ──IMDb dump──▶ rating, votes     │   │   │ 2. filter  (plain Python)               │
│   ▼                                              │   │   │ 3. retrieve: vector + BM25 → RRF        │
│ catalogue.json  (single source of truth)         │   │   │             → cross-encoder rerank      │
│   │                                              │   │   │ 4. discover (TMDb / YouTube) if needed  │
│   ├──▶ Chroma vector index ──────────────────────┼───┼──▶│ 5. generate (LLM, cites [L1] / [N2])    │
│   └──▶ BM25 keyword index  ──────────────────────┼───┼──▶│ 6. check citations                      │
└──────────────────────────────────────────────────┘   └─────────────────────────────────────────────┘
```

## 1. Catalogue: reels → titles (`catalogue.py`, `tmdb.py`, `imdb.py`)

The input is a list of saved reels (`Bookmark`). Several reels can point to the same film, and
a saved TV episode should count as its series, so reels are **grouped** by IMDb id (or TMDb URL).
Unresolved reels are skipped.

Each group is then **enriched** through TMDb: `/find` maps the IMDb id to a TMDb movie or TV
record, and `details` fetches plot, runtime, genres, keywords, directors, cast, countries and
streaming providers for `WATCH_REGION`. TMDb's TV genres ("Sci-Fi & Fantasy") are normalised to
one shared vocabulary (`schemas.GENRES`). If TMDb can't resolve a title, a fallback `Title` is
built from the reel's own fields so nothing is lost.

**IMDb ratings** come from IMDb's public `title.ratings.tsv.gz` dump (`imdb.py`), streamed and
filtered down to the ids in the catalogue. It is re-downloaded at most weekly.

The result, `catalogue.json` in `MOVIERAG_HOME`, is the single source of truth. The indexes are
derived from it and can be deleted and rebuilt at any time.

All API responses go through `http.JsonCache`, a SHA-1-keyed JSON-on-disk cache, so re-running a
notebook makes no network calls and the pipeline is reproducible.

## 2. Indexing and retrieval (`index.py`, `models.py`)

One document per title (texts are short, so no chunking). `Title.embedding_text()` decides what
gets indexed: title, year, kind, genres, tagline, plot, keywords, directors, cast, and the
captions of the saved reels ("Saved reel said: ...").

`MovieIndex` holds two retrievers over the same nodes:

- **Vector search** (Chroma, cosine, `BAAI/bge-small-en-v1.5` by default) is good at meaning:
  *"a slow, eerie film about loneliness"*.
- **BM25** is good at names and rare words: *"Shaolin"*, *"Paganini"*, *"Nuremberg"*.

**Hybrid search** runs both and merges them with **Reciprocal Rank Fusion**
(`score = Σ 1 / (60 + rank)`), which uses only ranks, so a cosine similarity and a BM25 score can
be merged without calibration.

An optional **cross-encoder reranker** (`ms-marco-MiniLM-L6-v2`) then reads each
(query, candidate) pair together and re-orders the top results. It is slower than a bi-encoder
but sharper.

The Chroma collection is stored with a **fingerprint** of the node texts and embedding model
name. If the catalogue or embedder changes, the stale collection is dropped and rebuilt;
otherwise it is reused.

Hard filters (runtime, years, kind, genres) are applied as an `allowed_ids` set: a Chroma
metadata `IN` filter for vectors, and a small throwaway BM25 index over the allowed nodes.

## 3. Answering a question (`query_parser.py`, `recommender.py`)

`MovieRecommender.recommend()` runs the loop and times each stage.

**Parse (self-query).** Embeddings are bad at hard constraints: "under 90 minutes" is a number
to filter on, not a vibe to match. The LLM fills a `QueryPlan` via `structured_predict`. A
regex-based parser (`heuristic_plan`) runs as well, and its explicit numbers (runtime, years)
override the LLM's; it also adds `documentary` / `youtube` kinds and `wants_new` if the LLM
missed them. If the LLM returns broken JSON, the rules are used alone. `Recommendation.parsed_by`
records which happened: `llm`, `llm+rules` or `rules`.

**Filter.** Titles are kept if they match kind, runtime, years and genres. Unknown runtimes pass
(better shown than hidden). If nothing survives, the softest filters are relaxed in order (genres,
then years) and the relaxation is reported to the user and the LLM.

**Retrieve.** Hybrid search + rerank over the allowed titles. For *"like X"* requests where X is
in the list, X's plot is appended to the query and X itself is excluded.

**Discover.** New items are fetched when the user asks for them, when the list has too few
matches, or when the request is for documentaries or YouTube. TMDb candidates come from, in
priority order: item-to-item recommendations for the reference title, recommendations seeded by
the best matches from the list, and `/discover` queries built from the plan's genres, runtime,
years and topic keywords. Candidates already in the list are excluded, the hard filters are
re-applied, and the survivors are ranked against the query with a throwaway in-memory vector
index (`rank_in_memory`) and the reranker. YouTube search honours `max_runtime_min`.

**Generate.** The candidates are rendered as compact cards (`Title.card()`) labelled `[L1]..`
(from the list) and `[N1]..` (new), and the LLM writes the answer citing those labels.

**Check.** Every `[Lx]` / `[Nx]` in the answer must exist among the candidates; any that don't
are flagged in `Recommendation.bad_citations`. It's a cheap guard against invented titles.

## 4. Models (`models.py`, `config.py`)

The LLM sits behind LlamaIndex's `LLM` interface, so the backend is a config switch:

| `LLM_BACKEND` | What runs |
|---|---|
| `hf` | Qwen3-4B-Instruct via `transformers`, 4-bit with bitsandbytes (~2.5 GB VRAM) |
| `ollama` | any Ollama model on `localhost:11434` |
| `openai_like` | any OpenAI-compatible server, e.g. `deploy/modal_vllm.py` |
| `mock` | no model; set automatically when `MOVIERAG_OFFLINE=true` |

In offline mode, the embedder is a `MockEmbedding`, the reranker is disabled and the API clients
make no network calls (TMDb still answers from its on-disk cache, if there is one). That's enough to exercise the plumbing anywhere in seconds.

## 5. Evaluation (`evaluate.py`, `eval/questions.json`, notebook 04)

Three layers, cheapest first:

1. **Retrieval:** for each question with a known answer key, hit@k, recall@k and MRR. Notebook
   04 compares vector, BM25, hybrid and hybrid + rerank.
2. **Parsing:** per-field accuracy of the `QueryPlan` against expected values.
3. **Generation:** citation validity, plus an **LLM-as-judge** faithfulness check that asks
   whether every claim in the answer is supported by the candidate cards the model saw.

## 6. Tests (`tests/test_offline.py`)

An end-to-end test that needs no GPU and no network: a deterministic hashed bag-of-words
embedding, a fake TMDb client that answers from the reels' own titles, and the mock LLM. It
covers grouping and merging, the three search modes, filtering and relaxation, RRF, the full
`recommend()` loop, "like X" exclusion, the rule-based parser, the metrics, and YouTube duration
parsing.

```bash
uv run pytest -q
```
