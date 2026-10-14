"""Step 4: measure it. Without numbers, every change is a guess.

Three layers, cheapest first:
1. Retrieval   : did the right titles come back?  hit@k, recall@k, MRR
2. Parsing     : did the self-query step extract the right filters?  per-field accuracy
3. Generation  : is the answer grounded in the retrieved candidates?  citations + LLM-as-judge
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .schemas import QueryPlan


def load_eval_set(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- retrieval
def retrieval_metrics(ranked_ids: list[str], relevant: set[str], k: int) -> dict[str, float]:
    """ranked_ids: what the system returned, best first. relevant: the answer key."""
    top = ranked_ids[:k]
    hits = [i for i in top if i in relevant]
    first = next((r for r, i in enumerate(ranked_ids, 1) if i in relevant), None)
    return {
        f"hit@{k}": 1.0 if hits else 0.0,  # at least one right answer in the top k?
        f"recall@{k}": len(set(hits)) / len(relevant),  # share of right answers found
        "mrr": 1.0 / first if first else 0.0,  # 1 if the first result is right, 1/2 if second...
    }


def evaluate_retrieval(
    questions: list[dict],
    search: Callable[[str], list[str]],
    k: int = 5,
) -> tuple[dict[str, float], list[dict]]:
    """search(question) -> list of IMDb ids, best first."""
    rows = []
    for q in questions:
        ranked = search(q["question"])
        m = retrieval_metrics(ranked, set(q["relevant_imdb"]), k)
        rows.append({"question": q["question"], **m, "top": ranked[:k]})
    keys = [key for key in rows[0] if key not in ("question", "top")] if rows else []
    avg = {key: sum(r[key] for r in rows) / len(rows) for key in keys}
    return avg, rows


# ---------------------------------------------------------------- parsing
PLAN_FIELDS = ["genres", "kinds", "max_runtime_min", "min_year", "max_year", "similar_to", "wants_new"]


def plan_field_matches(plan: QueryPlan, expected: dict) -> dict[str, bool]:
    """Only fields present in `expected` are scored; lists compare as sets."""
    out = {}
    for f in PLAN_FIELDS:
        if f not in expected:
            continue
        got, want = getattr(plan, f), expected[f]
        if isinstance(want, list):
            out[f] = set(got or []) == set(want)
        elif isinstance(want, str) and isinstance(got, str):
            out[f] = got.strip().lower() == want.strip().lower()
        else:
            out[f] = got == want
    return out


# ---------------------------------------------------------------- generation
JUDGE_PROMPT = """You are grading a movie recommendation for faithfulness.

CANDIDATES (the only allowed source of facts):
{context}

ANSWER TO GRADE:
{answer}

Does every recommended title appear in the candidates, and is every claim about
a title supported by its candidate description? Reply with exactly one word,
PASS or FAIL, then one short sentence explaining why."""


def judge_faithfulness(llm, answer: str, context: str) -> tuple[bool, str]:
    """LLM-as-judge. Ideally use a *stronger* model than the one being graded."""
    verdict = llm.complete(JUDGE_PROMPT.format(context=context, answer=answer)).text.strip()
    return verdict.upper().startswith("PASS"), verdict
