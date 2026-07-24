#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
from pypdf import PdfReader

APP_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from portfolio_core import (  # noqa: E402
    load_portfolio_data,
    resolve_project_file,
    verify_index_manifest,
)

STALE_TERMS = (
    "Expected June 2026",
    "Expected graduation: 2026",
    "Beklenen Mezuniyet: Haziran 2026",
    "graduating in 3 months",
)


def validate_pdf(path: Path, expected_text: str) -> None:
    reader = PdfReader(path)
    if len(reader.pages) != 1:
        raise ValueError(f"{path.name} must contain exactly one page.")
    page = reader.pages[0]
    text = page.extract_text() or ""
    if expected_text not in text:
        raise ValueError(f"{path.name} does not contain {expected_text!r}.")
    for stale_term in STALE_TERMS:
        if stale_term in text:
            raise ValueError(f"{path.name} contains stale text: {stale_term!r}.")

    linked_urls = []
    for annotation_reference in page.get("/Annots", []):
        action = annotation_reference.get_object().get("/A")
        if action and action.get("/URI"):
            linked_urls.append(str(action["/URI"]))

    required_link_fragments = (
        "github.com/meliherenn",
        "linkedin.com/in/meliheren",
        "mailto:meliheren2834@gmail.com",
    )
    if len(linked_urls) < 8:
        raise ValueError(f"{path.name} must contain at least eight clickable links.")
    for fragment in required_link_fragments:
        if not any(fragment in url for url in linked_urls):
            raise ValueError(f"{path.name} is missing a clickable {fragment!r} link.")


def validate_retrieval_artifacts(index_dir: Path) -> None:
    index_ok, reason = verify_index_manifest(index_dir)
    if not index_ok:
        raise ValueError(f"FAISS index verification failed: {reason}.")

    if (index_dir / "index.pkl").exists():
        raise ValueError("Obsolete pickle metadata must not be present.")

    index = faiss.read_index(str(index_dir / "index.faiss"))
    with (index_dir / "documents.json").open("r", encoding="utf-8") as stream:
        documents = json.load(stream)

    if not isinstance(documents, list) or not documents:
        raise ValueError("Retrieval metadata must be a non-empty JSON array.")
    if index.ntotal != len(documents):
        raise ValueError("FAISS vector count does not match retrieval metadata.")
    if index.d <= 0:
        raise ValueError("FAISS index has an invalid vector dimension.")

    for position, document in enumerate(documents):
        if not isinstance(document, dict) or not document.get("text"):
            raise ValueError(f"Invalid retrieval document at position {position}.")
        if document.get("language") not in {"tr", "en"}:
            raise ValueError(f"Invalid retrieval language at position {position}.")


def main() -> int:
    data = load_portfolio_data(APP_ROOT / "data.json")
    profile = data["profile"]

    tr_pdf = resolve_project_file(APP_ROOT, profile["cv_pdf_tr"])
    en_pdf = resolve_project_file(APP_ROOT, profile["cv_pdf_en"])
    if not tr_pdf or not en_pdf:
        raise FileNotFoundError("Configured CV PDFs are missing or unsafe.")

    validate_pdf(tr_pdf, "Ocak 2027")
    validate_pdf(en_pdf, "January 2027")

    validate_retrieval_artifacts(APP_ROOT / "faiss_index")

    print("Project data, CV PDFs and FAISS index passed validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
