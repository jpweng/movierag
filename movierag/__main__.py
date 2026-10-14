"""Command line:  uv run python -m movierag "a war movie under 2 hours"
Interactive:     uv run python -m movierag
"""

import sys

from .app import load_components


def main() -> None:
    c = load_components()
    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else None
    while True:
        q = questions.pop() if questions else input("\nWhat do you feel like watching? (empty = quit) > ").strip()
        if not q:
            break
        c.recommender.recommend(q).show()
        if len(sys.argv) > 1:
            break


if __name__ == "__main__":
    main()
