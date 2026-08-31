"""License hygiene as tested behavior (SPEC section 5): what is committed must be
committable, what is fetch-only must never be tracked, and secrets never leak."""

import json
import subprocess
from pathlib import Path

import pytest
from pydantic import SecretStr

from hedis_copilot.config import Settings
from hedis_copilot.corpus.manifest import Manifest, load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_DIR = REPO_ROOT / "corpus" / "committed"


def _git_ls_files(*pathspecs: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--", *pathspecs],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


@pytest.fixture(scope="module")
def manifest() -> Manifest:
    return load_manifest(REPO_ROOT / "corpus" / "manifest.yaml")


class TestCommittedCorpusIsCommittable:
    def test_every_committed_doc_maps_to_a_committable_manifest_entry(
        self, manifest: Manifest
    ) -> None:
        committed = sorted(COMMITTED_DIR.glob("*.json"))
        assert committed, "committed corpus is empty"
        for path in committed:
            doc_id = json.loads(path.read_text(encoding="utf-8"))["doc_id"]
            entry = manifest.by_id(doc_id)  # KeyError here = untracked committed doc
            assert entry.license_posture.committable, (
                f"{path.name}: committed but manifest posture "
                f"{entry.license_posture.value} is not committable"
            )
            assert entry.commit_policy == "commit_normalized", path.name

    def test_committed_filenames_match_their_doc_ids(self) -> None:
        for path in sorted(COMMITTED_DIR.glob("*.json")):
            doc_id = json.loads(path.read_text(encoding="utf-8"))["doc_id"]
            assert path.stem == doc_id, f"{path.name} contains doc_id {doc_id!r}"


class TestNothingFetchedIsTracked:
    def test_fetch_dirs_and_data_are_not_in_git(self) -> None:
        tracked = _git_ls_files("corpus/cache", "corpus/fetched", "data")
        assert tracked == [], f"raw/fetched artifacts tracked by git: {tracked}"


class TestNoSecretsTracked:
    def test_dot_env_is_not_in_git(self) -> None:
        tracked = set(_git_ls_files())
        assert ".env" not in tracked
        assert not any(name.endswith("/.env") for name in tracked)


class TestSecretNeverLeaks:
    def test_api_key_is_masked_in_repr_and_str(self) -> None:
        secret = "sk-ant-test-never-leak-me"
        settings = Settings(anthropic_api_key=SecretStr(secret))
        assert secret not in repr(settings)
        assert secret not in str(settings)
        assert secret not in str(settings.anthropic_api_key)
        assert secret not in repr(settings.anthropic_api_key)

    def test_secret_is_still_retrievable_on_purpose(self) -> None:
        settings = Settings(anthropic_api_key=SecretStr("sk-ant-test-never-leak-me"))
        assert settings.anthropic_api_key is not None
        assert settings.anthropic_api_key.get_secret_value() == "sk-ant-test-never-leak-me"
