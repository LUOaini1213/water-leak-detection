"""Every number in README.md must come from results/: re-render and compare, and
make sure the template itself contains no hand-typed numbers."""
import re

from conftest import RESULTS, ROOT
from render_readme import README, TEMPLATE, render

ALLOWED = [
    r"\b(2018|2019|2020)\b",  # dataset years
    r"BattLeDIM 2020",
    r"CC BY 4\.0",
    r"EUPL-1\.2",
    r"Python 3\.11",
    r"\bm3(/h)?\b",
    r"\b[pnRT]\d+\b",  # network element IDs (p227, n54, T1, R1)
    r"PUMP_1",
    r"N1\b",
    r"\bF1\b",
    r"LUOaini1213",
    r"10\.5281/zenodo\.\d+",  # dataset DOI
]


def _strip(text: str) -> str:
    text = re.sub(r"```.*?```", "", text, flags=re.S)  # code blocks
    text = re.sub(r"`[^`]*`", "", text)  # inline code
    text = re.sub(r"\{\{.+?\}\}", "", text)  # placeholders
    text = re.sub(r"\]\([^)]*\)", "]", text)  # link targets
    text = re.sub(r"https?://\S+", "", text)
    for pat in ALLOWED:
        text = re.sub(pat, "", text)
    return text


def test_readme_matches_rendered_results():
    expected = render(TEMPLATE.read_text(encoding="utf-8"), RESULTS)
    assert README.read_text(encoding="utf-8") == expected, "README.md is stale: run scripts/render_readme.py"


def test_template_has_no_hand_typed_numbers():
    leftover = _strip(TEMPLATE.read_text(encoding="utf-8"))
    digits = [line.strip() for line in leftover.splitlines() if re.search(r"\d", line)]
    assert digits == [], f"hand-typed numbers in template: {digits[:5]}"


def test_digit_check_catches_a_typed_number():
    assert re.search(r"\d", _strip("The detector found 17 leaks."))
    assert not re.search(r"\d", _strip("Pipe p227 in 2019 had {{x.json:y}} m3/h."))


def test_every_table_has_rows():
    from render_readme import TABLES

    for name, build in TABLES.items():
        lines = build(RESULTS).splitlines()
        assert len(lines) >= 3, f"table {name} has no data rows"
        assert all(line.count("|") == lines[0].count("|") for line in lines), name


def test_readme_links_exist():
    text = README.read_text(encoding="utf-8")
    for target in re.findall(r"\]\(([^)#]+)\)", text):
        if target.startswith("http"):
            continue
        assert (ROOT / target).exists(), target
