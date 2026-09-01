"""``hedis`` CLI — fetch | build | ask | eval | review | report | version.

Command bodies import the index/retrieval/answer stack lazily: importing this module (or
running ``hedis version --help``) must never pay the chromadb/fastembed import cost.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, NoReturn

import typer

from hedis_copilot import __version__
from hedis_copilot.config import ConfigError, Settings, get_settings

if TYPE_CHECKING:
    from hedis_copilot.retrieval.types import RetrieverLike

app = typer.Typer(
    name="hedis",
    help="RAG copilot over public CMS Star Ratings / HEDIS measure documents.",
    add_completion=False,
    no_args_is_help=True,
)


def main() -> NoReturn:
    """Console-script entrypoint.

    Chroma's rust binding keeps compacting the HNSW index in a background (non-Python)
    thread after large writes; on Windows that blocks interpreter shutdown for many minutes
    after the command's work — including its WAL commits — has durably finished. The console
    entrypoint therefore flushes and hard-exits with the command's real exit code. Tests
    invoke ``app`` in-process via CliRunner and never pass through here.
    """
    import os
    import sys

    code = 0
    try:
        app()
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)

MANIFEST_FILENAME = "manifest.yaml"
#: Module-level so tests (and the eval-content wave) can point at alternative locations.
DATASET_PATH = Path("evals/dataset/questions.jsonl")
BASELINE_PATH = Path("evals/baseline.json")
RESULTS_DIR = Path("evals/results")


def _fail(message: str) -> NoReturn:
    typer.secho(message, err=True, fg=typer.colors.RED)
    raise typer.Exit(1)


def _open_retriever(settings: Settings) -> "RetrieverLike":
    """Open the stamped index as a HybridRetriever; exit 1 with a rebuild hint if stale."""
    from hedis_copilot.index.store import StaleIndexError
    from hedis_copilot.retrieval.hybrid import HybridRetriever

    try:
        return HybridRetriever.from_settings(settings)
    except StaleIndexError as exc:
        _fail(str(exc))


@app.command()
def fetch(
    force: Annotated[
        bool, typer.Option("--force", help="Redownload even when a cached copy exists.")
    ] = False,
) -> None:
    """Download every whitelisted corpus document into corpus/cache (sha256-verified)."""
    from hedis_copilot.corpus import fetch as fetch_mod
    from hedis_copilot.corpus.manifest import load_manifest

    settings = get_settings()
    manifest = load_manifest(settings.corpus_dir / MANIFEST_FILENAME)
    cache_dir = settings.corpus_dir / "cache"
    paths = fetch_mod.fetch_all(manifest, cache_dir, force=force)

    typer.echo(f"{'doc_id':<24} {'bytes':>12}  sha256")
    for doc, path in zip(manifest.documents, paths, strict=True):
        size = path.stat().st_size if path.exists() else 0
        typer.echo(f"{doc.doc_id:<24} {size:>12}  ok")
    typer.echo(f"{len(paths)} document(s) cached under {cache_dir}")


@app.command()
def build(
    source: Annotated[
        Path,
        typer.Option(
            "--source",
            help="Normalized corpus directory to index (CI: `--source corpus/committed`, offline).",
        ),
    ] = Path("corpus/committed"),
    renormalize: Annotated[
        bool,
        typer.Option(
            "--renormalize", help="Re-parse cached documents into --source even if it exists."
        ),
    ] = False,
) -> None:
    """Build the Chroma index + chunk sidecar + stamp from the normalized corpus.

    When --source already holds normalized docs, fetch and normalize are skipped entirely —
    the committed Tier-A corpus is enough to build offline (keyless CI relies on this).
    """
    settings = get_settings()
    has_normalized = source.is_dir() and any(source.glob("*.json"))
    if not has_normalized or renormalize:
        from hedis_copilot.corpus import fetch as fetch_mod
        from hedis_copilot.corpus.fetch import cache_path
        from hedis_copilot.corpus.manifest import load_manifest
        from hedis_copilot.corpus.normalize import normalize_all

        manifest = load_manifest(settings.corpus_dir / MANIFEST_FILENAME)
        cache_dir = settings.corpus_dir / "cache"
        missing = [doc for doc in manifest.documents if not cache_path(cache_dir, doc).exists()]
        if missing:
            typer.echo(f"fetching {len(missing)} uncached document(s)…")
            fetch_mod.fetch_all(manifest, cache_dir)
        typer.echo(f"normalizing into {source}…")
        normalize_all(manifest, cache_dir, source)
    else:
        typer.echo(f"using existing normalized corpus at {source} (no fetch, no normalize)")

    from hedis_copilot.index.build import build_index

    stamp = build_index(settings, source)
    typer.echo(f"indexed {stamp.chunk_count} chunks into {settings.index_dir}")
    typer.echo("stamp:")
    for key, value in stamp.model_dump().items():
        typer.echo(f"  {key}: {value}")


@app.command()
def ask(
    query: Annotated[
        str, typer.Argument(help="Question about the public HEDIS / Star Ratings corpus.")
    ],
    year: Annotated[
        int | None, typer.Option("--year", help="Plan year filter (default: all years).")
    ] = None,
    retrieval_only: Annotated[
        bool,
        typer.Option(
            "--retrieval-only", help="Skip generation; render retrieved citation cards only."
        ),
    ] = False,
) -> None:
    """Ask a question; renders the cited answer (or citations only in keyless mode)."""
    settings = get_settings()
    retriever = _open_retriever(settings)

    keyless = settings.anthropic_api_key is None
    use_model = not retrieval_only and not keyless
    if not use_model:
        reason = "--retrieval-only" if retrieval_only else "no HEDIS_ANTHROPIC_API_KEY configured"
        typer.secho(
            f"[keyless mode] retrieval-only ({reason}) — citation cards render, "
            "no generated answer",
            fg=typer.colors.YELLOW,
        )

    from hedis_copilot.answer.service import AnswerService

    if use_model:
        from langchain_core.language_models.chat_models import BaseChatModel

        from hedis_copilot.llm import answer_model

        def factory() -> BaseChatModel:
            return answer_model(settings)

        service = AnswerService(retriever=retriever, model_factory=factory, settings=settings)
    else:
        service = AnswerService(retriever=retriever, model_factory=None, settings=settings)

    try:
        answer = service.ask(query, plan_year=year)
    except ConfigError as exc:
        _fail(str(exc))

    typer.echo("")
    typer.echo(answer.text)
    if answer.citations:
        typer.echo("")
        typer.echo("Citations:")
        for cit in answer.citations:
            measure = (
                f" | {cit.measure_id} {cit.measure_name}"
                if cit.measure_id and cit.measure_name
                else ""
            )
            page = f" | p.{cit.page}" if cit.page is not None else ""
            typer.echo(f"  [{cit.marker}] {cit.doc_title}{measure} | {cit.section}{page}")
            typer.echo(f"      {cit.source_url} (plan year {cit.plan_year})")


def _dataset_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _gated_baseline(raw: object) -> dict[str, float]:
    """Extract flat metric floats from evals/baseline.json (flat, metrics, or overall shape)."""
    candidate: object = raw
    if isinstance(candidate, dict) and isinstance(candidate.get("metrics"), dict):
        candidate = candidate["metrics"]
    if isinstance(candidate, dict) and isinstance(candidate.get("overall"), dict):
        candidate = candidate["overall"]
    if not isinstance(candidate, dict):
        _fail(f"{BASELINE_PATH} does not contain a metric mapping")
    return {
        str(key): float(value)
        for key, value in candidate.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


@app.command("eval")
def evaluate(
    retrieval: Annotated[
        bool,
        typer.Option("--retrieval", help="Run the keyless retrieval-metric tier."),
    ] = False,
    gate: Annotated[
        bool,
        typer.Option("--gate", help="Exit 1 when gated metrics regress vs evals/baseline.json."),
    ] = False,
) -> None:
    """Run the eval harness (v1: --retrieval tier; --full arrives with the eval-content wave)."""
    if not retrieval:
        typer.secho(
            "nothing to run: pass --retrieval (--full arrives with the eval-content wave)",
            err=True,
        )
        raise typer.Exit(2)

    if not DATASET_PATH.exists():
        typer.echo(f"no dataset yet at {DATASET_PATH} — arrives with the eval-content wave")
        raise typer.Exit(0)

    settings = get_settings()
    retriever = _open_retriever(settings)

    from hedis_copilot.evals.dataset import DatasetError, LabelResolutionError, load_dataset
    from hedis_copilot.evals.dataset import resolve_labels as resolve
    from hedis_copilot.evals.report import write_artifact
    from hedis_copilot.evals.runner import compare_to_baseline, run_retrieval_eval
    from hedis_copilot.index.store import chunks_path, load_chunks_jsonl

    try:
        items = load_dataset(DATASET_PATH)
        chunks = list(load_chunks_jsonl(chunks_path(settings.index_dir)).values())
        resolved = resolve(items, chunks)
    except (DatasetError, LabelResolutionError) as exc:
        _fail(str(exc))

    def retriever_fn(question: str) -> list[str]:
        return [scored.chunk.chunk_id for scored in retriever.retrieve(question).chunks]

    def gate_a_fn(question: str) -> bool:
        # Mirrors AnswerService gate A exactly: dense cosine similarity, not RRF rank score.
        result = retriever.retrieve(question)
        similarity = result.best_dense_similarity
        return not result.chunks or (
            similarity is not None and similarity < settings.refusal_score_floor
        )

    report = run_retrieval_eval(retriever_fn, items, resolved, gate_a_fn=gate_a_fn)

    date = datetime.now(UTC).date().isoformat()
    artifact_path = RESULTS_DIR / f"retrieval-{date}.json"
    write_artifact(
        artifact_path,
        {
            "tier": "retrieval",
            "date": date,
            "config_hash": settings.config_hash(),
            "embedding_model": settings.embedding_model,
            "dataset_hash": _dataset_hash(DATASET_PATH),
            "item_count": len(items),
            "metrics": {
                "overall": report.overall,
                "per_category": report.per_category,
                "refusal_trap_accuracy": report.refusal_trap_accuracy,
            },
        },
    )

    typer.echo("overall (answerable items):")
    for name, value in sorted(report.overall.items()):
        typer.echo(f"  {name:<14} {value:.4f}")
    if report.refusal_trap_accuracy is not None:
        typer.echo(f"  {'refusal_acc':<14} {report.refusal_trap_accuracy:.4f}")
    typer.echo("per category:")
    for category, metrics in sorted(report.per_category.items()):
        rendered = "  ".join(f"{name}={value:.3f}" for name, value in sorted(metrics.items()))
        typer.echo(f"  {category:<24} {rendered}")
    typer.echo(f"wrote {artifact_path}")

    if gate:
        if not BASELINE_PATH.exists():
            typer.echo(
                f"no baseline at {BASELINE_PATH} — gate passes vacuously "
                "(the first measured run ratchets it)"
            )
            return
        regressions = compare_to_baseline(
            report, _gated_baseline(json.loads(BASELINE_PATH.read_text(encoding="utf-8")))
        )
        if regressions:
            for regression in regressions:
                typer.secho(f"REGRESSION: {regression}", err=True, fg=typer.colors.RED)
            raise typer.Exit(1)
        typer.echo("ratchet gate: PASS")


@app.command()
def review() -> None:
    """Human spot-check of judged samples — arrives with the eval-content wave."""
    typer.echo("hedis review arrives with the eval-content wave")


@app.command()
def report() -> None:
    """Regenerate the README eval table from the latest artifact — eval-content wave."""
    typer.echo("hedis report arrives with the eval-content wave")


@app.command()
def version() -> None:
    """Print the installed hedis-spec-copilot version."""
    typer.echo(f"hedis-spec-copilot {__version__}")


if __name__ == "__main__":  # pragma: no cover
    app()
