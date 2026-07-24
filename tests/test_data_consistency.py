from pathlib import Path

from portfolio_core import load_portfolio_data

APP_ROOT = Path(__file__).resolve().parents[1]


def test_bilingual_project_links_are_consistent():
    data = load_portfolio_data(APP_ROOT / "data.json")
    tr_links = {project["name"]: project["url"] for project in data["tr"]["projects"]}
    en_links = {project["name"]: project["url"] for project in data["en"]["projects"]}

    assert tr_links == en_links


def test_quick_actions_have_matching_prompts():
    data = load_portfolio_data(APP_ROOT / "data.json")

    for lang_code in ("tr", "en"):
        ui = data[lang_code]["ui"]
        assert len(ui["buttons"]) == len(ui["hidden_prompts"]) == 6


def test_no_stale_graduation_claims_in_structured_data():
    raw_data = (APP_ROOT / "data.json").read_text(encoding="utf-8")

    assert "June 2026" not in raw_data
    assert "Haziran 2026" not in raw_data
    assert "graduating in 3 months" not in raw_data
