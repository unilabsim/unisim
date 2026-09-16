"""Repository-level invariants for bilingual docs and maintained trees."""

from __future__ import annotations

import re
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
EN_DOCS = DOCS / "en"
ZH_DOCS = DOCS / "zh"
STRUCTURAL = re.compile(r"^(?:#{1,6}\s|\||\s*(?:[-*+]|\d+\.)\s|>(?:\s|$))")


def markdown_files(directory: Path) -> set[Path]:
    return {path.relative_to(directory) for path in directory.glob("*.md")}


def outside_code_lines(path: Path) -> list[str]:
    lines: list[str] = []
    in_code = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
            continue
        if not in_code:
            lines.append(line)
    return lines


def structural_signature(path: Path) -> tuple[list[int], int, int, int, int]:
    lines = outside_code_lines(path)
    headings = [len(match.group(1)) for line in lines if (match := re.match(r"^(#{1,6})\s", line))]
    tables = sum(line.startswith("|") for line in lines)
    bullets = sum(bool(re.match(r"^\s*[-*+]\s", line)) for line in lines)
    ordered = sum(bool(re.match(r"^\s*\d+\.\s", line)) for line in lines)
    ordinary = [line for line in lines if line.strip() and not STRUCTURAL.match(line)]
    return headings, tables, bullets, ordered, len(ordinary)


def code_block_count(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    return sum(line.lstrip().startswith("```") for line in lines) // 2


def markdown_outside_code(path: Path) -> str:
    return "\n".join(outside_code_lines(path))


def relative_markdown_links_exist(path: Path) -> None:
    text = markdown_outside_code(path)
    for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
        if target.startswith(("#", "http://", "https://", "mailto:")):
            continue
        link_path = urllib.parse.unquote(target.split("#", 1)[0])
        if link_path:
            assert (path.parent / link_path).resolve().is_file(), f"{path}: {target}"


def test_documentation_language_trees_have_matching_files_and_structure() -> None:
    english = markdown_files(EN_DOCS)
    chinese = markdown_files(ZH_DOCS)
    assert english
    assert english == chinese
    assert not {path.name for path in DOCS.glob("*.md")}

    for relative in english:
        assert structural_signature(EN_DOCS / relative) == structural_signature(ZH_DOCS / relative)
        assert code_block_count(EN_DOCS / relative) == code_block_count(ZH_DOCS / relative)
        assert f"[中文](../zh/{relative.as_posix()})" in (EN_DOCS / relative).read_text(
            encoding="utf-8"
        )
        assert f"[English](../en/{relative.as_posix()})" in (ZH_DOCS / relative).read_text(
            encoding="utf-8"
        )


def test_markdown_relative_links_resolve() -> None:
    paths = [
        ROOT / "README.md",
        ROOT / "README_zh.md",
        ROOT / "AGENTS.md",
        ROOT / "CONTRIBUTING.md",
        *DOCS.rglob("*.md"),
    ]
    for path in paths:
        relative_markdown_links_exist(path)


def test_readme_languages_preserve_structure() -> None:
    assert structural_signature(ROOT / "README.md") == structural_signature(ROOT / "README_zh.md")
    assert code_block_count(ROOT / "README.md") == code_block_count(ROOT / "README_zh.md")


def test_markdown_prose_is_not_hard_wrapped() -> None:
    markdown_paths = [
        *ROOT.glob("*.md"),
        ROOT / "scripts" / "README.md",
        ROOT / "tests" / "README.md",
        *DOCS.rglob("*.md"),
    ]
    for path in markdown_paths:
        previous_was_ordinary = False
        in_code = False
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.lstrip().startswith("```"):
                in_code = not in_code
                previous_was_ordinary = False
                continue
            if in_code or not line.strip() or STRUCTURAL.match(line):
                previous_was_ordinary = False
                continue
            assert not previous_was_ordinary, f"Hard-wrapped prose in {path}: {line}"
            previous_was_ordinary = True


def test_tests_are_grouped_by_ownership() -> None:
    tests = ROOT / "tests"
    assert {path.name for path in tests.glob("*.py")} == {"__init__.py"}
    expected = {"core", "contract", "factory", "adapters"}
    directories = {
        path.name
        for path in tests.iterdir()
        if path.is_dir() and not path.name.startswith(".") and path.name != "__pycache__"
    }
    assert directories == expected


def test_scripts_are_explicit_maintainer_entry_points() -> None:
    scripts = ROOT / "scripts"
    actual = {path.relative_to(scripts).as_posix() for path in scripts.rglob("*") if path.is_file()}
    expected = {
        "README.md",
        "benchmarks/superdex_scene_step.py",
        "diagnostics/check_newton_runtime.py",
        "diagnostics/check_support.py",
    }
    assert actual == expected
