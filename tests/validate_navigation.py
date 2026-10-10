"""Check the published reading route across navigation, cards and learning pages."""
from __future__ import annotations

from pathlib import Path
import re
import yaml

ROOT = Path(__file__).resolve().parents[1]


def check_navigation(config: dict, docs: Path = ROOT / "docs") -> list[str]:
    errors: list[str] = []
    guide_pages: list[tuple[str, str]] = []
    project_pages: list[tuple[str, str]] = []
    nav = config.get("nav", [])
    groups = [item for item in nav if isinstance(item, dict)
              and any(re.match(r"^[1-6] · ", key) for key in item)]
    if len(groups) != 6:
        errors.append("Navigation needs six numbered guide groups")
    for n, group in enumerate(groups, 1):
        label, children = next(iter(group.items()))
        if not label.startswith(f"{n} · ") or not isinstance(children, list) or len(children) != 2:
            errors.append(f"Guide group {n} needs its number and two ordered pages")
            continue
        for j, child in enumerate(children, 1):
            page_label, target = next(iter(child.items()))
            if not page_label.startswith(f"{n}.{j} · "):
                errors.append(f"Guide page must start with {n}.{j} · : {page_label}")
            guide_pages.append((f"{n}.{j}", target))
    project_groups = [item["Practice with projects"] for item in nav
                      if isinstance(item, dict) and "Practice with projects" in item]
    if len(project_groups) != 1:
        errors.append("Navigation needs one project group")
    else:
        project_pages = [(label.split(" · ")[0], target)
                         for item in project_groups[0] for label, target in item.items()
                         if label != "Choose a project"]
        expected_order = ["regression", "emnist", "robust-image-pipeline", "nature-cnn",
                          "training-comparison", "vision-head", "text-bags", "model-lifecycle"]
        if [ident for ident, _ in project_pages] != [f"P{i}" for i in range(1, 9)]:
            errors.append("Projects must be numbered P1 through P8 in order")
        if [Path(target).stem for _, target in project_pages] != expected_order:
            errors.append("Projects must follow foundations, training, vision, text and lifecycle")

    home = (docs / "index.md").read_text(encoding="utf-8")
    gallery = (docs / "projects/index.md").read_text(encoding="utf-8")
    for group in groups:
        for child in next(iter(group.values())):
            if not isinstance(child, dict):
                continue
            label, target = next(iter(child.items()))
            if f"[{label}]({target})" not in home:
                errors.append(f"Home link label disagrees with navigation: {label}")
    if re.findall(r'data-topic="(\d+)"', home) != [str(i) for i in range(1, 7)]:
        errors.append("Home topic cards must match guide group order")
    if re.findall(r'data-project="(P\d+)"', gallery) != [ident for ident, _ in project_pages]:
        errors.append("Project gallery must match navigation order")
    if "Continue into Course 3 topics" in home or "## Practice as you learn" not in home:
        errors.append("Home must present one integrated guide and practice route")
    for anchor in ["continue-into-course-3-topics", "selected-course-2-examples"]:
        if anchor not in home:
            errors.append(f"Home lost legacy anchor: {anchor}")
    for anchor in ["foundation-projects", "selected-course-2-workflows", "course-3-preserve-and-reuse-a-model"]:
        if f"{{#{anchor}}}" not in gallery:
            errors.append(f"Project gallery lost legacy anchor: {anchor}")

    def text_for(target):
        path = docs / target
        if not path.is_file():
            errors.append(f"Reading route source missing: {target}")
            return ""
        return path.read_text(encoding="utf-8")

    guide_texts = {target: text_for(target) for _, target in guide_pages}
    project_texts = {target: text_for(target) for _, target in project_pages}
    for ident, target in guide_pages:
        text = guide_texts[target]
        if not re.search(rf'^title: {re.escape(ident)} · ', text, re.M) or not re.search(rf'^# {re.escape(ident)} · ', text, re.M):
            errors.append(f"Guide title/heading lacks its navigation number: {target}")
        if not re.search(rf'^guide_id: ["\']?{re.escape(ident)}["\']?$', text, re.M):
            errors.append(f"Guide metadata disagrees with navigation: {target}")
        if f'data-guide-id="{ident}"' not in text or "**Before you start:**" not in text:
            errors.append(f"Guide lacks numbered orientation/prerequisites: {target}")
        if f"]({target})" not in home:
            errors.append(f"Home card does not link to guide: {target}")
        if ident.startswith("5."):
            if "#remember-and-apply" not in text:
                errors.append(f"Architecture guide lacks a local visual/self-check route: {target}")
        elif not re.search(r'\]\(\.\./\.\./projects/[\w-]+\.md\)', text):
            errors.append(f"Guide lacks related project: {target}")
        if "**Continue reading:**" not in text and "**Apply the route:**" not in text:
            errors.append(f"Guide lacks its reading-route continuation: {target}")

    for ident, target in project_pages:
        text = project_texts[target]
        if not re.search(rf'^title: {ident} · ', text, re.M) or not re.search(rf'^# {ident} · ', text, re.M):
            errors.append(f"Project title/heading lacks its navigation number: {target}")
        if f"project_id: {ident}" not in text or f'data-project-id="{ident}"' not in text:
            errors.append(f"Project identity disagrees with navigation: {target}")
        if "**Read first:**" not in text or "**Data and setup:**" not in text or "**Inspect after running:**" not in text:
            errors.append(f"Project needs prerequisites, data and output interpretation: {target}")
        if f"]({Path(target).name})" not in gallery:
            errors.append(f"Gallery does not link to project: {target}")
        linked_guides = re.findall(r'\]\(\.\./(guides/[\w/-]+\.md)(?:#[^)]*)?\)', text)
        if not linked_guides:
            errors.append(f"Project has no guide connection: {target}")
        for guide in linked_guides:
            if guide not in guide_texts or f"../../{target}" not in guide_texts[guide]:
                errors.append(f"Guide/project connection is not reciprocal: {guide} ↔ {target}")
    return errors


if __name__ == "__main__":
    config = yaml.load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"), Loader=yaml.BaseLoader)
    errors = check_navigation(config)
    if errors:
        raise SystemExit("\n".join(errors))
    print("Validated guide numbering, project order, orientation and reciprocal learning links")
