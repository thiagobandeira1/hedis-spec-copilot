"""Normalize fetched documents into the committed Tier-A corpus."""

from pathlib import Path

from hedis_copilot.corpus.fetch import cache_path
from hedis_copilot.corpus.manifest import Manifest
from hedis_copilot.corpus.models import NormalizedDoc
from hedis_copilot.corpus.parse import parse_document


def normalize_all(manifest: Manifest, cache_dir: Path, committed_dir: Path) -> list[Path]:
    """Parse every commit_normalized doc and write ``corpus/committed/{doc_id}.json``."""
    committed_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for doc in manifest.documents:
        if doc.commit_policy != "commit_normalized":
            continue
        normalized = parse_document(cache_path(cache_dir, doc), doc)
        out = committed_dir / f"{doc.doc_id}.json"
        out.write_text(normalized.model_dump_json(indent=2) + "\n", encoding="utf-8")
        written.append(out)
    return written


def load_committed(committed_dir: Path) -> list[NormalizedDoc]:
    docs = [
        NormalizedDoc.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(committed_dir.glob("*.json"))
    ]
    if not docs:
        raise FileNotFoundError(
            f"no committed corpus under {committed_dir} — run `hedis fetch` then `hedis build`"
        )
    return docs
