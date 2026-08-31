"""Manifest loading, the commit-policy/license gate, and the LicensePosture truth table."""

from pathlib import Path
from typing import Any

import pytest
import yaml

from hedis_copilot.corpus.manifest import LicensePosture, load_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "corpus" / "manifest.yaml"


def _doc_entry(**overrides: Any) -> dict[str, Any]:
    """A minimal valid manifest document entry, override fields per test."""
    entry: dict[str, Any] = {
        "doc_id": "fake-doc",
        "title": "Fake Document",
        "publisher": "Centers for Medicare & Medicaid Services",
        "source_url": "https://www.cms.gov/files/document/fake.pdf",
        "sha256": "0" * 64,
        "retrieval_date": "2026-08-30",
        "plan_year": 2026,
        "doc_type": "technical_notes",
        "license_posture": "us_gov_public_domain",
        "commit_policy": "commit_normalized",
        "parser_id": "technical_notes",
    }
    entry.update(overrides)
    return entry


class TestRealManifest:
    def test_loads_and_has_four_documents(self) -> None:
        manifest = load_manifest(MANIFEST_PATH)
        assert len(manifest.documents) == 4

    def test_all_documents_are_us_gov_public_domain(self) -> None:
        manifest = load_manifest(MANIFEST_PATH)
        for doc in manifest.documents:
            assert doc.license_posture is LicensePosture.US_GOV_PUBLIC_DOMAIN, doc.doc_id

    def test_all_documents_commit_normalized(self) -> None:
        manifest = load_manifest(MANIFEST_PATH)
        for doc in manifest.documents:
            assert doc.commit_policy == "commit_normalized", doc.doc_id

    def test_by_id_round_trip_and_missing_id(self) -> None:
        manifest = load_manifest(MANIFEST_PATH)
        assert manifest.by_id("cms-tn-2026").plan_year == 2026
        with pytest.raises(KeyError):
            manifest.by_id("no-such-doc")


class TestCommitPolicyGate:
    def test_fetch_at_build_only_with_commit_normalized_raises(self, tmp_path: Path) -> None:
        """The load-time gate: a non-committable posture must never pair with committing."""
        raw = {
            "version": "test",
            "documents": [
                _doc_entry(
                    doc_id="bad-doc",
                    license_posture="fetch_at_build_only",
                    commit_policy="commit_normalized",
                )
            ],
        }
        path = tmp_path / "manifest.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        with pytest.raises(ValueError, match="committable"):
            load_manifest(path)

    def test_fetch_at_build_only_with_fetch_only_is_valid(self, tmp_path: Path) -> None:
        raw = {
            "version": "test",
            "documents": [
                _doc_entry(
                    doc_id="ok-doc",
                    license_posture="fetch_at_build_only",
                    commit_policy="fetch_only",
                )
            ],
        }
        path = tmp_path / "manifest.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")
        manifest = load_manifest(path)
        assert manifest.by_id("ok-doc").license_posture is LicensePosture.FETCH_AT_BUILD_ONLY


@pytest.mark.parametrize(
    ("posture", "committable"),
    [
        (LicensePosture.US_GOV_PUBLIC_DOMAIN, True),
        (LicensePosture.REDISTRIBUTABLE, True),
        (LicensePosture.PUBLIC_WEB_CITE_ONLY, False),
        (LicensePosture.FETCH_AT_BUILD_ONLY, False),
    ],
)
def test_license_posture_committable_truth_table(
    posture: LicensePosture, committable: bool
) -> None:
    assert posture.committable is committable


def test_truth_table_covers_every_posture() -> None:
    """If a new posture is added, the truth table above must be extended deliberately."""
    assert set(LicensePosture) == {
        LicensePosture.US_GOV_PUBLIC_DOMAIN,
        LicensePosture.REDISTRIBUTABLE,
        LicensePosture.PUBLIC_WEB_CITE_ONLY,
        LicensePosture.FETCH_AT_BUILD_ONLY,
    }
