import json
import os
import tempfile
from pathlib import Path

import faiss
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from portfolio_core import (
    EMBEDDING_MODEL,
    load_portfolio_data,
    resolve_project_file,
    write_index_manifest,
)

APP_ROOT = Path(__file__).resolve().parent
DATA_PATH = APP_ROOT / "data.json"
INDEX_PATH = APP_ROOT / "faiss_index"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 120


def get_configured_pdfs(data):
    """Read active CV PDF paths from validated portfolio data."""
    profile = data["profile"]
    return [
        (profile["cv_pdf_tr"], "tr"),
        (profile["cv_pdf_en"], "en"),
    ]


def get_structured_records(data):
    """Create searchable bilingual records from the verified JSON data."""
    records = []
    for lang in ("tr", "en"):
        lang_data = data[lang]
        project_lines = [
            f"{project['name']}: {project['description']} "
            f"Stack: {', '.join(project['stack'])}. URL: {project['url']}"
            for project in lang_data["projects"]
        ]
        skill_lines = [f"{name}: {value}" for name, value in lang_data["skills"].items()]
        content = "\n".join(
            [
                lang_data["prompts"]["identity_a"],
                f"Education: {lang_data['education']}",
                f"Strengths: {lang_data['prompts']['strengths']}",
                "Experience:",
                *lang_data["experience"],
                "Projects:",
                *project_lines,
                "Skills:",
                *skill_lines,
            ]
        )
        records.append({"text": content, "language": lang, "source_file": "data.json"})
    return records


def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks while preferring natural boundaries."""
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not normalized:
        return []

    chunks = []
    start = 0
    while start < len(normalized):
        target_end = min(start + chunk_size, len(normalized))
        end = target_end
        if target_end < len(normalized):
            search_floor = start + chunk_size // 2
            candidates = [
                normalized.rfind(separator, search_floor, target_end) for separator in ("\n", ". ", "; ", " ")
            ]
            natural_end = max(candidates)
            if natural_end > start:
                end = natural_end + 1

        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(end - overlap, start + 1)

    return chunks


def create_vector_db():
    """Build a pickle-free FAISS index from verified JSON data and both CV PDFs."""
    data = load_portfolio_data(DATA_PATH)
    source_records = get_structured_records(data)

    for pdf_name, lang in get_configured_pdfs(data):
        pdf_path = resolve_project_file(APP_ROOT, pdf_name)
        if not pdf_path:
            raise FileNotFoundError(f"Missing or unsafe configured PDF: {pdf_name}")

        print(f"Reading: {pdf_path.name}")
        reader = PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                source_records.append(
                    {
                        "text": text,
                        "language": lang,
                        "source_file": pdf_path.name,
                        "source_page": page_number,
                    }
                )
        print(f"  Loaded {len(reader.pages)} page(s).")

    documents = []
    for record in source_records:
        for chunk in split_text(record["text"]):
            documents.append({**record, "text": chunk})

    if not documents:
        raise RuntimeError("No verified portfolio documents were found.")

    print(f"Created {len(documents)} searchable chunk(s).")
    print(f"Creating embeddings with {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    vectors = model.encode(
        [document["text"] for document in documents],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    print("Writing FAISS index and JSON metadata...")
    INDEX_PATH.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".faiss-build-", dir=APP_ROOT) as temp_dir:
        temp_path = Path(temp_dir)
        faiss.write_index(index, str(temp_path / "index.faiss"))
        with (temp_path / "documents.json").open("w", encoding="utf-8") as stream:
            json.dump(documents, stream, ensure_ascii=False, indent=2)
            stream.write("\n")

        for filename in ("index.faiss", "documents.json"):
            os.replace(temp_path / filename, INDEX_PATH / filename)

    manifest_path = write_index_manifest(INDEX_PATH)
    print(f"Index and checksum manifest updated: {manifest_path.relative_to(APP_ROOT)}")


if __name__ == "__main__":
    create_vector_db()
