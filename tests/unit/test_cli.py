"""CLI wiring tests — no network, no index, no API key.

Heavy paths (embedding, Chroma, real fetches) are exercised by integration tests and CI's
build step; here we pin the wiring: exit codes, hints, and that `fetch` delegates to
``corpus.fetch.fetch_all`` without ever touching the network.
"""

import contextlib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hedis_copilot import __version__, cli
from hedis_copilot.corpus.manifest import Manifest

runner = CliRunner()


def _all_output(result: object) -> str:
    """stdout + stderr across click versions (>=8.2 splits them, older mixes)."""
    out = str(getattr(result, "output", ""))
    # Older click (mix_stderr=True) raises ValueError on .stderr access.
    with contextlib.suppress(ValueError):
        out += str(getattr(result, "stderr", ""))
    return out


def test_version_prints_package_version() -> None:
    result = runner.invoke(cli.app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_ask_without_index_exits_nonzero_with_rebuild_hint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HEDIS_INDEX_DIR", str(tmp_path / "no-index-here"))
    result = runner.invoke(cli.app, ["ask", "What are the exclusions for COL?"])
    assert result.exit_code != 0
    assert "hedis build" in _all_output(result)


def test_eval_retrieval_without_dataset_exits_zero_with_note(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(cli, "DATASET_PATH", tmp_path / "absent" / "questions.jsonl")
    result = runner.invoke(cli.app, ["eval", "--retrieval"])
    assert result.exit_code == 0
    assert "no dataset yet" in _all_output(result)


def test_eval_retrieval_gate_without_dataset_still_exits_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """--gate cannot fail before a dataset exists (nothing to regress against)."""
    monkeypatch.setattr(cli, "DATASET_PATH", tmp_path / "absent" / "questions.jsonl")
    result = runner.invoke(cli.app, ["eval", "--retrieval", "--gate"])
    assert result.exit_code == 0
    assert "no dataset yet" in _all_output(result)


def test_eval_without_mode_flag_is_a_usage_error() -> None:
    result = runner.invoke(cli.app, ["eval"])
    assert result.exit_code == 2
    assert "--retrieval" in _all_output(result)


def test_fetch_delegates_to_fetch_all_and_prints_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[Manifest, Path, bool]] = []

    def fake_fetch_all(manifest: Manifest, cache_dir: Path, *, force: bool = False) -> list[Path]:
        calls.append((manifest, cache_dir, force))
        paths: list[Path] = []
        for doc in manifest.documents:
            p = tmp_path / f"{doc.doc_id}.pdf"
            p.write_bytes(b"%PDF fake")
            paths.append(p)
        return paths

    monkeypatch.setattr("hedis_copilot.corpus.fetch.fetch_all", fake_fetch_all)
    result = runner.invoke(cli.app, ["fetch"])
    assert result.exit_code == 0, _all_output(result)

    assert len(calls) == 1
    manifest, cache_dir, force = calls[0]
    assert cache_dir == Path("corpus") / "cache"
    assert force is False
    assert len(manifest.documents) >= 4  # the real committed manifest was loaded

    out = _all_output(result)
    for doc in manifest.documents:
        assert doc.doc_id in out
    assert "ok" in out  # sha column: fetch_all raises on mismatch, so every row is ok


def test_fetch_force_flag_is_forwarded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, bool] = {}

    def fake_fetch_all(manifest: Manifest, cache_dir: Path, *, force: bool = False) -> list[Path]:
        seen["force"] = force
        paths: list[Path] = []
        for doc in manifest.documents:
            p = tmp_path / f"{doc.doc_id}.pdf"
            p.write_bytes(b"%PDF fake")
            paths.append(p)
        return paths

    monkeypatch.setattr("hedis_copilot.corpus.fetch.fetch_all", fake_fetch_all)
    result = runner.invoke(cli.app, ["fetch", "--force"])
    assert result.exit_code == 0
    assert seen["force"] is True


@pytest.mark.parametrize("command", ["review", "report"])
def test_stub_commands_exit_zero_with_wave_note(command: str) -> None:
    result = runner.invoke(cli.app, [command])
    assert result.exit_code == 0
    assert "arrives with the eval-content wave" in result.output
