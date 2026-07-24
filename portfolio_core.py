from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_FILENAMES = ("index.faiss", "documents.json")
INDEX_MANIFEST_FILENAME = "checksums.json"


class PortfolioDataError(ValueError):
    """Raised when portfolio data does not match the expected public schema."""


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(parsed, maximum))


def safe_external_url(value: Any, *, allow_mailto: bool = True) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or any(ord(char) < 32 for char in candidate):
        return None

    parsed = urlsplit(candidate)
    if parsed.scheme in {"https", "http"}:
        if not parsed.netloc or parsed.username or parsed.password:
            return None
        return candidate

    if allow_mailto and parsed.scheme == "mailto":
        address = parsed.path
        if "@" not in address or any(char.isspace() for char in address):
            return None
        return candidate

    return None


def safe_llm_base_url(value: Any, default: str) -> str:
    candidate = safe_external_url(value, allow_mailto=False)
    if not candidate:
        return default
    parsed = urlsplit(candidate)
    if parsed.scheme == "https":
        return candidate
    if parsed.hostname in {"127.0.0.1", "localhost", "::1"}:
        return candidate
    return default


def resolve_project_file(root: Path, path_value: Any) -> Path | None:
    if not path_value:
        return None
    try:
        candidate = (root / str(path_value)).resolve()
        candidate.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return candidate if candidate.is_file() else None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PortfolioDataError(message)


def validate_portfolio_data(data: Any) -> dict[str, Any]:
    _require(isinstance(data, dict), "Portfolio data must be a JSON object.")

    profile = data.get("profile")
    _require(isinstance(profile, dict), "Missing profile object.")
    for field in ("name", "title_tr", "title_en", "cv_pdf_tr", "cv_pdf_en"):
        _require(isinstance(profile.get(field), str) and profile[field].strip(), f"Invalid profile.{field}.")

    contact = profile.get("contact")
    _require(isinstance(contact, dict), "Missing profile.contact object.")
    for field in ("github", "linkedin", "email"):
        _require(
            safe_external_url(contact.get(field)) is not None,
            f"Invalid public contact URL: profile.contact.{field}.",
        )

    for lang_code in ("tr", "en"):
        lang_data = data.get(lang_code)
        _require(isinstance(lang_data, dict), f"Missing {lang_code} data.")

        prompts = lang_data.get("prompts")
        _require(isinstance(prompts, dict), f"Missing {lang_code}.prompts.")
        for field in ("identity_a", "career_goals", "strengths", "style_rules"):
            _require(prompts.get(field), f"Missing {lang_code}.prompts.{field}.")

        education = lang_data.get("education")
        _require(isinstance(education, str) and education.strip(), f"Missing {lang_code}.education.")

        experience = lang_data.get("experience")
        _require(
            isinstance(experience, list)
            and all(isinstance(item, str) and item.strip() for item in experience),
            f"Invalid {lang_code}.experience.",
        )

        projects = lang_data.get("projects")
        _require(isinstance(projects, list) and projects, f"Invalid {lang_code}.projects.")
        for index, project in enumerate(projects):
            _require(isinstance(project, dict), f"Invalid {lang_code}.projects[{index}].")
            for field in ("name", "description", "url", "stack"):
                _require(project.get(field), f"Missing {lang_code}.projects[{index}].{field}.")
            _require(
                safe_external_url(project["url"], allow_mailto=False) is not None,
                f"Invalid {lang_code}.projects[{index}].url.",
            )
            _require(
                isinstance(project["stack"], list)
                and all(isinstance(item, str) and item.strip() for item in project["stack"]),
                f"Invalid {lang_code}.projects[{index}].stack.",
            )

        skills = lang_data.get("skills")
        _require(isinstance(skills, dict) and skills, f"Invalid {lang_code}.skills.")
        _require(
            all(
                isinstance(key, str) and key.strip() and isinstance(value, str) and value.strip()
                for key, value in skills.items()
            ),
            f"Invalid {lang_code}.skills values.",
        )

        certificates = lang_data.get("certificates")
        _require(isinstance(certificates, list), f"Invalid {lang_code}.certificates.")
        for index, certificate in enumerate(certificates):
            _require(isinstance(certificate, dict), f"Invalid {lang_code}.certificates[{index}].")
            _require(certificate.get("name"), f"Missing {lang_code}.certificates[{index}].name.")
            _require(
                safe_external_url(certificate.get("url"), allow_mailto=False) is not None,
                f"Invalid {lang_code}.certificates[{index}].url.",
            )

        ui = lang_data.get("ui")
        _require(isinstance(ui, dict), f"Missing {lang_code}.ui.")
        buttons = ui.get("buttons")
        hidden_prompts = ui.get("hidden_prompts")
        _require(isinstance(buttons, list) and buttons, f"Invalid {lang_code}.ui.buttons.")
        _require(
            isinstance(hidden_prompts, list) and len(buttons) == len(hidden_prompts),
            f"{lang_code}.ui buttons and hidden prompts must have equal length.",
        )

    return data


def load_portfolio_data(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except json.JSONDecodeError as exc:
        raise PortfolioDataError(f"Invalid JSON in {path.name}: {exc.msg}.") from exc
    return validate_portfolio_data(data)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temp_path = Path(stream.name)
        json.dump(payload, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


def save_portfolio_data(path: Path, data: dict[str, Any]) -> None:
    validate_portfolio_data(data)
    _write_json_atomic(path, data)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_index_manifest(index_dir: Path, embedding_model: str = EMBEDDING_MODEL) -> Path:
    files = {}
    for filename in INDEX_FILENAMES:
        path = index_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing FAISS index file: {filename}")
        files[filename] = sha256_file(path)

    manifest_path = index_dir / INDEX_MANIFEST_FILENAME
    _write_json_atomic(
        manifest_path,
        {
            "version": 1,
            "embedding_model": embedding_model,
            "files": files,
        },
    )
    return manifest_path


def verify_index_manifest(
    index_dir: Path,
    expected_model: str = EMBEDDING_MODEL,
) -> tuple[bool, str]:
    manifest_path = index_dir / INDEX_MANIFEST_FILENAME
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return False, "missing or invalid checksum manifest"

    if manifest.get("version") != 1:
        return False, "unsupported checksum manifest version"
    if manifest.get("embedding_model") != expected_model:
        return False, "embedding model mismatch"

    files = manifest.get("files")
    if not isinstance(files, dict):
        return False, "invalid checksum file map"

    for filename in INDEX_FILENAMES:
        expected_hash = files.get(filename)
        path = index_dir / filename
        if not isinstance(expected_hash, str) or len(expected_hash) != 64 or not path.is_file():
            return False, f"missing checksum or file for {filename}"
        if path.is_symlink():
            return False, f"symbolic links are not allowed for {filename}"
        if sha256_file(path) != expected_hash:
            return False, f"checksum mismatch for {filename}"

    return True, "verified"
