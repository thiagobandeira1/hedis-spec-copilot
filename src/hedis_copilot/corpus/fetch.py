"""Whitelist-only corpus fetching with sha256 pinning."""

import hashlib
from pathlib import Path

import httpx

from hedis_copilot.corpus.manifest import Manifest, ManifestDoc

#: cms.gov (and most .gov CDNs) 403 generic bot user agents; a browser UA is required.
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)


class FetchIntegrityError(RuntimeError):
    """Downloaded content does not match the manifest's pinned sha256.

    CMS revises documents in place; the pin makes that a loud, reviewable event (update the
    manifest in a PR) instead of silent corpus drift.
    """


def cache_path(cache_dir: Path, doc: ManifestDoc) -> Path:
    return cache_dir / f"{doc.doc_id}{Path(str(doc.source_url)).suffix or '.bin'}"


def fetch_document(doc: ManifestDoc, cache_dir: Path, *, force: bool = False) -> Path:
    """Download one whitelisted document into the (gitignored) cache; verify its pin."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_path(cache_dir, doc)
    if target.exists() and not force:
        _verify(target, doc)
        return target
    with httpx.Client(headers={"User-Agent": _UA}, follow_redirects=True, timeout=60) as client:
        response = client.get(str(doc.source_url))
        response.raise_for_status()
        target.write_bytes(response.content)
    _verify(target, doc)
    return target


def _verify(path: Path, doc: ManifestDoc) -> None:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != doc.sha256:
        raise FetchIntegrityError(
            f"{doc.doc_id}: sha256 mismatch (got {digest[:12]}…, pinned {doc.sha256[:12]}…). "
            "The source likely changed — review it and update the manifest deliberately."
        )


def fetch_all(manifest: Manifest, cache_dir: Path, *, force: bool = False) -> list[Path]:
    return [fetch_document(doc, cache_dir, force=force) for doc in manifest.documents]
