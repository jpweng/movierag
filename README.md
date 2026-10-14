# movierag: what should I watch tonight?

A small, readable **retrieval-augmented generation (RAG)** system, built as a learning project.
It starts from ~90 movie reels saved on Facebook and Instagram and answers questions like
*"a war movie under 2 hours"* or *"something like Devs, but a film"*.

It recommends from **your saved list** first, then tops up with **new** films, series,
documentaries (via TMDb) and YouTube videos. Every answer cites its sources (`[L1]` = from your
list, `[N2]` = new), and the citations are checked automatically.

```
movies-reels.json ─▶ catalogue (TMDb + IMDb ratings) ─▶ Chroma (vectors) + BM25 (keywords)
                                                                 │
question ─▶ LLM parses filters ─▶ filter ─▶ hybrid search ─▶ rerank ─▶ (TMDb/YouTube discovery)
                                                                 ─▶ LLM answer with [L1]/[N2] citations ─▶ citation check
```

**What it covers:** LlamaIndex, sentence embeddings, Chroma, BM25, Reciprocal Rank Fusion,
a cross-encoder reranker, self-querying (natural language → structured filters), a 4-bit
quantized Qwen on a consumer GPU, the TMDb and YouTube APIs, evaluation (retrieval metrics,
parser accuracy, LLM-as-judge), tracing with Arize Phoenix, and optionally vLLM on Modal.

See [docs/architecture.md](docs/architecture.md) for how the pieces fit together.

## Example

```console
$ uv run python -m movierag "a war movie under 2 hours"
Q: a war movie under 2 hours
plan (llm+rules): search='war' | genres: War, <= 120 min

From your list:
  [L1] ...
New:
  [N1] ...

<the LLM's answer, citing [L1], [N1], ...>

parse 0.8s | retrieve 0.1s | discover 1.2s | generate 4.0s | total 6.1s
```

## Repository layout

| Path | What |
|---|---|
| `notebooks/01_bookmarks_to_catalogue.ipynb` | reels → enriched catalogue (TMDb, IMDb ratings) |
| `notebooks/02_embeddings_and_search.ipynb` | embeddings, vector vs BM25, RRF hybrid, reranking, filters |
| `notebooks/03_ask_for_movies.ipynb` | 4-bit Qwen, self-query parsing, discovery, grounded answers |
| `notebooks/04_evaluate.ipynb` | retrieval ablation, parser accuracy, citation check, LLM-as-judge |
| `movierag/` | the library the notebooks call, one file per concept (see below) |
| `movies-reels.json` | the input: saved reels, each matched to an IMDb/TMDb title |
| `movies-reels.html` | the same list as a browsable web page |
| `eval/questions.json` | evaluation set: retrieval questions, parsing cases, end-to-end prompts |
| `tests/test_offline.py` | end-to-end test with fake TMDb and toy embeddings: no GPU, no network |
| `deploy/modal_vllm.py` | serve a 30B Qwen with vLLM on Modal (OpenAI-compatible API) |

### The `movierag` package

| Module | Role |
|---|---|
| `config.py` | all settings, read once from environment variables / `.env` |
| `schemas.py` | Pydantic models: `Bookmark` (a saved reel), `Title` (a watchable thing), `QueryPlan` (parsed request) |
| `catalogue.py` | step 1: group reels by title, enrich via TMDb, attach IMDb ratings |
| `tmdb.py`, `youtube.py`, `imdb.py` | data-source clients (TMDb details/discover/recommendations, YouTube search, IMDb ratings dump) |
| `http.py` | JSON-on-disk cache for API calls, so notebook re-runs are instant and free |
| `index.py` | step 2: Chroma vector index + BM25, RRF fusion, cross-encoder reranking |
| `models.py` | loads the embedding model, the reranker and the LLM (`hf` / `ollama` / `openai_like` / `mock`) |
| `query_parser.py` | step 3a: LLM + rule-based parsing of a request into a `QueryPlan` |
| `recommender.py` | step 3: the full loop (parse → filter → retrieve → discover → generate → check) |
| `evaluate.py` | step 4: hit@k, recall@k, MRR, per-field parser accuracy, LLM-as-judge |
| `tracing.py` | optional OpenTelemetry tracing into Arize Phoenix |
| `app.py`, `__main__.py` | wire everything together; command-line entry point |

## Requirements

- Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/)
- An NVIDIA GPU for the default local LLM (Qwen3-4B in 4-bit uses ~2.5 GB of VRAM).
  Without one, use Ollama, a remote OpenAI-compatible server, or offline mode.
- Free API keys for TMDb and (optionally) the YouTube Data API

## Setup

**1. (Optional) keep the virtual environment out of cloud-synced folders.** If the project
lives in Google Drive, Dropbox or OneDrive, put the environment elsewhere so tens of thousands
of files aren't synced. Set this in each new terminal before running `uv`:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/movierag"           # Linux / macOS
```
```powershell
$env:UV_PROJECT_ENVIRONMENT = "$HOME\.venvs\movierag"          # Windows PowerShell
```

**2. Install.**

```bash
uv sync --extra dev
```

On Linux and Windows this pulls PyTorch built for CUDA 12.8 from PyTorch's own index
(needed for Blackwell / RTX 50-series GPUs). Check the GPU is visible:

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**3. API keys.** Copy `.env.example` to `.env` and fill in:

- `TMDB_API_KEY` (or `TMDB_READ_TOKEN`): free at themoviedb.org → Settings → API
- `YOUTUBE_API_KEY`: free in the Google Cloud console → enable *YouTube Data API v3* → Credentials

`.env` is gitignored. Never commit it.

**4. Check the plumbing** (no GPU, no network, about 5 seconds):

```bash
uv run pytest -q
```

**5. Run the notebooks in order** (01 → 02 → 03 → 04). In VS Code, pick the project's
virtual environment as the kernel, or run:

```bash
uv run --with jupyter jupyter lab
```

Notebook 01 builds the catalogue; everything else depends on it. The first run of notebook 03
downloads Qwen3-4B (~8 GB in bf16) into the Hugging Face cache; it is quantized to 4-bit at
load time.

Generated state goes to `~/.movierag` (override with `MOVIERAG_HOME`): `catalogue.json`, the
Chroma index and the API cache. Keep it outside synced folders too, since Chroma uses SQLite.

## Command line

```bash
uv run python -m movierag "a war movie under 2 hours"   # one question
uv run python -m movierag                               # interactive
```

## Configuration

All settings are environment variables (or lines in `.env`); see `.env.example` for the full list.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BACKEND` | `hf` | `hf`, `ollama` or `openai_like` |
| `HF_MODEL` | `Qwen/Qwen3-4B-Instruct-2507` | model for the `hf` backend |
| `HF_LOAD_IN_4BIT` | `true` | 4-bit quantization with bitsandbytes |
| `OLLAMA_MODEL` | `qwen3.5:4b` | model for the `ollama` backend (`ollama pull` it first) |
| `OPENAI_LIKE_BASE_URL`, `OPENAI_LIKE_MODEL`, `OPENAI_LIKE_API_KEY` | | any OpenAI-compatible server, e.g. the Modal deployment |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | embedding model |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L6-v2` | cross-encoder reranker |
| `WATCH_REGION` | `CH` | country for streaming-provider lookups |
| `TMDB_LANGUAGE` | `en-US` | language for TMDb metadata |
| `MOVIERAG_HOME` | `~/.movierag` | where the catalogue, index and caches go |
| `MOVIERAG_BOOKMARKS` | `movies-reels.json` | input file |
| `MOVIERAG_OFFLINE` | `false` | mock models and no API calls (plumbing checks only) |

## Optional extras

- **Tracing:** `uv sync --extra tracing`, start the UI with `uv run phoenix serve`, then call
  `movierag.tracing.enable_tracing()` (shown in notebook 03). Every retrieve, rerank and LLM call
  shows up at http://localhost:6006 with inputs, outputs, latency and token counts.
- **A bigger model on a cloud GPU:** `deploy/modal_vllm.py` serves Qwen3-30B-A3B (FP8) with vLLM on
  a Modal L40S, behind an OpenAI-compatible API that scales to zero. Setup steps are in the
  file's docstring.

## Using your own list

The input is a JSON file with a `reels` array. Each entry needs at least an `id` and a
`platform`; the fields that matter for matching are `title`, `year`, `kind`, `imdb_url` /
`tmdb_url` and `match_status` (`Confirmed` / `Tentative`). See `Bookmark` in
`movierag/schemas.py` and the first entries of `movies-reels.json`. Point `MOVIERAG_BOOKMARKS`
at your file and re-run notebook 01. Remember to update the answer keys in
`eval/questions.json` to match your titles.

## Data sources and terms

- **TMDb:** free for non-commercial use, with attribution: *This product uses the TMDb API but is
  not endorsed or certified by TMDb.*
- **IMDb ratings** come from IMDb's [non-commercial datasets](https://developer.imdb.com/non-commercial-datasets/)
  (personal and non-commercial use only).
- **YouTube Data API:** 10,000 quota units a day; one search costs 100.

## Status

This is a learning project. The offline test suite passes, and the notebooks are committed with
the outputs of a real run (TMDb enrichment, Qwen3-4B in 4-bit on a local GPU). The evaluation answer keys in `eval/questions.json` were drafted automatically
and are worth reviewing before trusting the numbers. The Modal deployment is written against
Modal 1.x and has not been exercised as thoroughly as the local path.
