"""Validate metadata, links, guide shape, and the no-tracking boundary."""

from __future__ import annotations

import re
from pathlib import Path
import yaml


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
    expected_primary_pages = {
        DOCS / "index.md",
        DOCS / "guides" / "fundamentals" / "core-workflow.md",
        DOCS / "guides" / "fundamentals" / "vision-real-data.md",
        DOCS / "projects" / "index.md",
        DOCS / "reference" / "index.md",
        DOCS / "about" / "index.md",
    }
    for path in expected_primary_pages:
        if not path.exists():
            errors.append(f"{path}: required primary page is missing")

    retired_directories = ["start", "collections", "concepts", "courses"]
    for name in retired_directories:
        directory = DOCS / name
        if directory.exists() and any(directory.rglob("*.md")):
            errors.append(f"{directory}: retired small-page collection still contains Markdown")

    guides_root = DOCS / "guides"
    for guide_directory in sorted(path for path in guides_root.iterdir() if path.is_dir()):
        pages = sorted(guide_directory.glob("*.md"))
        if len(pages) != 2:
            errors.append(
                f"{guide_directory}: published guides require exactly two substantial pages, found {len(pages)}"
            )

    # Navigation is the published inventory; every source page must be reachable.
    config = yaml.load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    def nav_paths(value):
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [item for child in value for item in nav_paths(child)]
        if isinstance(value, dict):
            return [item for child in value.values() for item in nav_paths(child)]
        return []
    listed = nav_paths(config["nav"])
    if len(listed) != len(set(listed)):
        errors.append("Navigation lists a page more than once")
    published = {(DOCS / item).resolve() for item in listed}
    actual = {path.resolve() for path in markdown_files}
    for path in sorted(published - actual):
        errors.append(f"Navigation source is missing: {path}")
    for path in sorted(actual - published):
        errors.append(f"Public page is absent from navigation: {path}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Validated {len(markdown_files)} Markdown files")


if __name__ == "__main__":
    main()
