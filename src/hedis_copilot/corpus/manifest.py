"""Typed corpus manifest — the explicit whitelist of everything this project may fetch."""

from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, HttpUrl


class LicensePosture(StrEnum):
    US_GOV_PUBLIC_DOMAIN = "us_gov_public_domain"
    PUBLIC_WEB_CITE_ONLY = "public_web_cite_only"
    REDISTRIBUTABLE = "redistributable"
    FETCH_AT_BUILD_ONLY = "fetch_at_build_only"

    @property
    def committable(self) -> bool:
        return self in (self.US_GOV_PUBLIC_DOMAIN, self.REDISTRIBUTABLE)


class ManifestDoc(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    title: str
    publisher: str
    source_url: HttpUrl
    sha256: str
    retrieval_date: str
    plan_year: int
    doc_type: str
    license_posture: LicensePosture
    commit_policy: Literal["commit_normalized", "fetch_only"]
    parser_id: Literal["technical_notes", "generic_pdf", "ncqa_summary_html"]


class Manifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: str
    documents: list[ManifestDoc]

    def by_id(self, doc_id: str) -> ManifestDoc:
        for doc in self.documents:
            if doc.doc_id == doc_id:
                return doc
        raise KeyError(doc_id)


def load_manifest(path: Path) -> Manifest:
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    manifest = Manifest.model_validate(raw)
    # A non-committable posture must never be paired with a committing policy.
    for doc in manifest.documents:
        if doc.commit_policy == "commit_normalized" and not doc.license_posture.committable:
            raise ValueError(
                f"{doc.doc_id}: commit_normalized requires a committable license posture"
            )
    return manifest
