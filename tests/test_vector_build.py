from pathlib import Path

from build_vector_db import get_structured_records, split_text
from portfolio_core import load_portfolio_data

APP_ROOT = Path(__file__).resolve().parents[1]


def test_structured_records_are_bilingual():
    data = load_portfolio_data(APP_ROOT / "data.json")

    records = get_structured_records(data)

    assert {record["language"] for record in records} == {"tr", "en"}
    assert all(record["source_file"] == "data.json" for record in records)
    assert "January 2027" in next(record["text"] for record in records if record["language"] == "en")


def test_split_text_creates_bounded_overlapping_chunks():
    text = "\n".join(f"Sentence {index} contains verified portfolio information." for index in range(60))

    chunks = split_text(text, chunk_size=240, overlap=40)

    assert len(chunks) > 1
    assert all(1 <= len(chunk) <= 240 for chunk in chunks)
    assert all(
        set(chunks[index][-20:].split()) & set(chunks[index + 1][:80].split())
        for index in range(len(chunks) - 1)
    )
