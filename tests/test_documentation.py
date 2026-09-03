from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTATION_FILES = (
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    *sorted((ROOT / "docs").glob("*.md")),
)


@pytest.mark.parametrize("path", DOCUMENTATION_FILES, ids=lambda path: path.name)
def test_github_display_math_uses_balanced_double_dollar_fences(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    standalone_delimiters = [line.strip() for line in lines if line.strip() in {"$", "$$"}]

    assert "$" not in standalone_delimiters, (
        f"{path.relative_to(ROOT)} uses a standalone single-dollar display-math fence; "
        "GitHub display mathematics must use $$ fences"
    )
    assert standalone_delimiters.count("$$") % 2 == 0, (
        f"{path.relative_to(ROOT)} has unbalanced $$ display-math fences"
    )


def test_contributor_math_guidance_is_not_duplicated_or_corrupted() -> None:
    text = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

    assert text.count("# Contributing") == 1
    assert ("use `$...$` for inline mathematics and `$$...$$`\nfor display mathematics") in text
