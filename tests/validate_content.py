"""Validate public Markdown metadata, links, and the no-tracking content boundary."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def check_metadata(path: Path, text: str) -> list[str]:
    errors: list[str] = []
    if not text.startswith("---\n"):
        errors.append(f"{path}: missing YAML front matter")
        return errors
    front_matter = text.split("---\n", 2)[1]
    if "title:" not in front_matter:
        errors.append(f"{path}: missing title metadata")
    if "last_reviewed:" not in front_matter:
        errors.append(f"{path}: missing last_reviewed metadata")
    if re.search(r"^status:", front_matter, re.MULTILINE):
        errors.append(f"{path}: course/progress status metadata is not allowed")
    return errors


def check_links(path: Path, text: str) -> list[str]:
    errors: list[str] = []
    for target in LINK.findall(text):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        clean = target.split("#", 1)[0]
        if not clean:
            continue
        destination = (path.parent / clean).resolve()
        if destination.suffix == "":
            destination = destination / "index.md"
        if not destination.exists():
            errors.append(f"{path}: broken local link {target}")
    return errors


def main() -> None:
    errors: list[str] = []
    markdown_files = sorted(DOCS.rglob("*.md"))
    if not markdown_files:
        raise SystemExit("No Markdown content found")
    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        errors.extend(check_metadata(path, text))
        errors.extend(check_links(path, text))
    forbidden_course_dirs = [
        DOCS / "courses" / "techniques-ecosystem",
        DOCS / "courses" / "advanced-architectures-deployment",
    ]
    for directory in forbidden_course_dirs:
        if directory.exists() and any(directory.rglob("*.md")):
            errors.append(f"{directory}: unpublished future-course placeholders are not allowed")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Validated {len(markdown_files)} Markdown files")


if __name__ == "__main__":
    main()

