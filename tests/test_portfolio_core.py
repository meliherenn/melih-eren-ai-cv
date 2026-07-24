import copy
import json
from pathlib import Path

from portfolio_core import (
    EMBEDDING_MODEL,
    bounded_int,
    load_portfolio_data,
    parse_bool,
    resolve_project_file,
    safe_external_url,
    safe_llm_base_url,
    save_portfolio_data,
    verify_index_manifest,
    write_index_manifest,
)

APP_ROOT = Path(__file__).resolve().parents[1]


def test_repository_data_matches_public_schema():
    data = load_portfolio_data(APP_ROOT / "data.json")

    assert data["profile"]["name"] == "Melih Eren"
    assert "January 2027" in data["en"]["education"]
    assert "Ocak 2027" in data["tr"]["education"]


def test_safe_external_url_accepts_public_links_and_email():
    assert safe_external_url("https://github.com/meliherenn") == "https://github.com/meliherenn"
    assert safe_external_url("mailto:meliheren2834@gmail.com") == "mailto:meliheren2834@gmail.com"


def test_safe_external_url_rejects_unsafe_values():
    unsafe_values = (
        "javascript:alert(1)",
        "data:text/html,unsafe",
        "https://user:password@example.com",
        "https://example.com/\nheader",
        "/local/path",
    )

    assert all(safe_external_url(value) is None for value in unsafe_values)


def test_safe_llm_base_url_requires_tls_except_localhost():
    default = "https://api.example.com/v1"

    assert safe_llm_base_url("https://secure.example.com/v1", default) == "https://secure.example.com/v1"
    assert safe_llm_base_url("http://localhost:11434/v1", default) == "http://localhost:11434/v1"
    assert safe_llm_base_url("http://public.example.com/v1", default) == default


def test_bounded_configuration_parsing():
    assert parse_bool("YES")
    assert not parse_bool("off", True)
    assert parse_bool("unknown", True)
    assert bounded_int("40", 20, 1, 100) == 40
    assert bounded_int("400", 20, 1, 100) == 100
    assert bounded_int("invalid", 20, 1, 100) == 20


def test_portfolio_save_is_validated_and_atomic(tmp_path):
    source = load_portfolio_data(APP_ROOT / "data.json")
    candidate = copy.deepcopy(source)
    candidate["profile"]["name"] = "Test Candidate"
    destination = tmp_path / "data.json"

    save_portfolio_data(destination, candidate)

    assert load_portfolio_data(destination)["profile"]["name"] == "Test Candidate"
    assert not list(tmp_path.glob("*.tmp"))


def test_index_manifest_detects_tampering(tmp_path):
    (tmp_path / "index.faiss").write_bytes(b"faiss")
    (tmp_path / "documents.json").write_text("[]\n", encoding="utf-8")

    manifest_path = write_index_manifest(tmp_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["embedding_model"] == EMBEDDING_MODEL
    assert verify_index_manifest(tmp_path) == (True, "verified")

    (tmp_path / "documents.json").write_text('[{"tampered": true}]\n', encoding="utf-8")
    index_ok, reason = verify_index_manifest(tmp_path)
    assert not index_ok
    assert "checksum mismatch" in reason


def test_resolve_project_file_blocks_escape(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    in_root = project_root / "public.pdf"
    in_root.write_bytes(b"pdf")
    outside = tmp_path / "private.txt"
    outside.write_text("private", encoding="utf-8")

    assert resolve_project_file(project_root, "public.pdf") == in_root.resolve()
    assert resolve_project_file(project_root, "../private.txt") is None
