"""Check generated local links and fragments after the strict MkDocs build."""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        element_id = values.get("id")
        if element_id:
            self.ids.add(element_id)
        href = values.get("href")
        if href:
            self.links.append(href)


def target_file(source: Path, path: str) -> Path:
    decoded = unquote(path)
    if decoded.startswith("/"):
        target = SITE / decoded.lstrip("/")
    else:
        target = source.parent / decoded
    if decoded.endswith("/") or not target.suffix:
        target /= "index.html"
    return target.resolve()


def main() -> None:
    pages: dict[Path, PageParser] = {}
    errors: list[str] = []

    for html_path in SITE.rglob("*.html"):
        parser = PageParser()
        parser.feed(html_path.read_text(encoding="utf-8"))
        pages[html_path.resolve()] = parser

    for source, parser in pages.items():
        for href in parser.links:
            parsed = urlparse(href)
            if parsed.scheme or parsed.netloc or href.startswith(("mailto:", "javascript:")):
                continue
            destination = source if not parsed.path else target_file(source, parsed.path)
            if not destination.exists():
                errors.append(f"{source.relative_to(SITE)}: missing generated target {href}")
                continue
            if parsed.fragment and destination.suffix == ".html":
                target_parser = pages.get(destination)
                if target_parser is None or unquote(parsed.fragment) not in target_parser.ids:
                    errors.append(f"{source.relative_to(SITE)}: missing generated anchor {href}")

    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Validated generated links and anchors across {len(pages)} HTML pages")


if __name__ == "__main__":
    main()
