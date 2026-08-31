"""CI gate: validate the gold set and resolve every label against the committed corpus.

Usage: ``uv run python scripts/validate_dataset.py [--dataset PATH] [--committed-dir PATH]``

Exit codes: 0 = dataset valid and fully resolvable (or not yet authored — the gold set
lands in a later wave, so absence is a note, not a failure); 1 = any validation or
resolution error (printed to stderr).
"""

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from hedis_copilot.config import get_settings
from hedis_copilot.corpus.chunk import chunk_document
from hedis_copilot.corpus.normalize import load_committed
from hedis_copilot.evals.dataset import (
    DatasetError,
    LabelResolutionError,
    load_dataset,
    resolve_labels,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evals/dataset/questions.jsonl"),
        help="gold-set JSONL path",
    )
    parser.add_argument(
        "--committed-dir",
        type=Path,
        default=Path("corpus/committed"),
        help="Tier-A normalized corpus directory",
    )
    args = parser.parse_args(argv)
    dataset_path: Path = args.dataset
    committed_dir: Path = args.committed_dir

    if not dataset_path.exists():
        print(
            f"NOTE: {dataset_path} not found - the gold set is authored in a later wave; "
            "nothing to validate yet."
        )
        return 0

    try:
        items = load_dataset(dataset_path)
    except DatasetError as exc:
        print(f"DATASET INVALID: {exc}", file=sys.stderr)
        return 1

    settings = get_settings()
    try:
        docs = load_committed(committed_dir)
    except FileNotFoundError as exc:
        print(f"CORPUS MISSING: {exc}", file=sys.stderr)
        return 1
    chunks = [
        chunk
        for doc in docs
        for chunk in chunk_document(
            doc,
            max_tokens=settings.chunk_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
    ]

    try:
        resolved = resolve_labels(items, chunks)
    except LabelResolutionError as exc:
        print(f"LABELS UNRESOLVABLE: {exc}", file=sys.stderr)
        return 1

    answerable = [item for item in items if not item.is_refusal]
    by_split = Counter(item.split for item in items)
    by_category = Counter(item.category for item in items)
    gold_counts = [len(resolved[item.item_id]) for item in answerable]
    print(
        f"dataset OK: {len(items)} items "
        f"({len(answerable)} answerable, {len(items) - len(answerable)} refusal traps)"
    )
    print("  splits: " + ", ".join(f"{k}={v}" for k, v in sorted(by_split.items())))
    print("  categories: " + ", ".join(f"{k}={v}" for k, v in sorted(by_category.items())))
    if gold_counts:
        print(
            "  resolved gold chunks per answerable item: "
            f"min={min(gold_counts)} max={max(gold_counts)} total={sum(gold_counts)}"
        )
    print(f"  corpus: {len(docs)} docs -> {len(chunks)} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
