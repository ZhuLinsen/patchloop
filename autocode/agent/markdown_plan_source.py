"""Markdown 开发计划输入源。"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from agent.plan_source import PlanItem, PlanSource

_OPEN_CHECKBOX_RE = re.compile(r"^\s*[-*]\s+\[\s\]\s+(.+?)\s*$")
_HEADING_RE = re.compile(r"^(#+)\s+(.+?)\s*$")


class MarkdownPlanSource(PlanSource):
    """从 Markdown 开发计划文件中提取未完成条目。"""

    def __init__(
        self,
        *,
        repo_path: str,
        file_paths: list[str],
        default_labels: list[str],
        title_prefix: str,
        max_items_per_cycle: int,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.file_paths = [item.strip() for item in file_paths if item.strip()]
        self.default_labels = [item.strip() for item in default_labels if item.strip()]
        self.title_prefix = title_prefix.strip()
        self.max_items_per_cycle = max_items_per_cycle

    def collect_items(self) -> list[PlanItem]:
        items: list[PlanItem] = []
        for rel_path in self.file_paths:
            if len(items) >= self.max_items_per_cycle:
                break
            resolved = self._resolve_path(rel_path)
            if resolved is None or not resolved.exists() or not resolved.is_file():
                continue
            items.extend(self._parse_file(resolved, rel_path))
            if len(items) >= self.max_items_per_cycle:
                break
        return items[: self.max_items_per_cycle]

    def has_item_key(self, item_key: str) -> bool:
        return self.get_item_by_key(item_key) is not None

    def get_item_by_key(self, item_key: str) -> PlanItem | None:
        if not item_key:
            return None
        for rel_path in self.file_paths:
            resolved = self._resolve_path(rel_path)
            if resolved is None or not resolved.exists() or not resolved.is_file():
                continue
            item = self._file_item_by_key(resolved, rel_path, item_key)
            if item is not None:
                return item
        return None

    def _resolve_path(self, rel_path: str) -> Path | None:
        candidate = (self.repo_path / rel_path).resolve()
        try:
            candidate.relative_to(self.repo_path)
        except ValueError:
            return None
        return candidate

    def _file_item_by_key(self, path: Path, rel_path: str, item_key: str) -> PlanItem | None:
        for item in self._parse_file(path, rel_path):
            if item.key == item_key:
                return item
        return None

    def _parse_file(self, path: Path, rel_path: str) -> list[PlanItem]:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return []

        candidates: list[tuple[int, str, str]] = []
        headings: list[str] = []
        for index, line in enumerate(lines, start=1):
            heading_match = _HEADING_RE.match(line)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                headings = headings[: level - 1]
                headings.append(title)
                continue

            match = _OPEN_CHECKBOX_RE.match(line)
            if not match:
                continue

            text = " ".join(match.group(1).strip().split())
            if not text:
                continue
            section = " / ".join(headings)
            candidates.append((index, section, text))

        duplicate_suffix_counts = self._build_duplicate_suffix_counts(candidates)
        items: list[PlanItem] = []
        for index, section, text in candidates:
            item_key = self._build_item_key(
                rel_path,
                section,
                text,
                duplicate_suffix_count=duplicate_suffix_counts[(index, section, text)],
            )
            title = self._format_title(text)
            items.append(
                PlanItem(
                    key=item_key,
                    title=title,
                    body=self._format_body(rel_path=rel_path, line_number=index, section=section),
                    source_path=rel_path,
                    line_number=index,
                    labels=list(self.default_labels),
                )
            )
        return items

    def _build_duplicate_suffix_counts(self, candidates: list[tuple[int, str, str]]) -> dict[tuple[int, str, str], int]:
        counts: dict[tuple[str, str], int] = {}
        suffix_counts: dict[tuple[int, str, str], int] = {}
        for index, section, text in reversed(candidates):
            signature = (section, text)
            duplicate_suffix_count = counts.get(signature, 0)
            suffix_counts[(index, section, text)] = duplicate_suffix_count
            counts[signature] = duplicate_suffix_count + 1
        return suffix_counts

    def _build_item_key(
        self,
        rel_path: str,
        section: str,
        text: str,
        *,
        duplicate_suffix_count: int = 0,
    ) -> str:
        digest = hashlib.sha256(
            f"{rel_path}\n{section}\n{text}\n{int(duplicate_suffix_count)}".encode("utf-8")
        ).hexdigest()[:16]
        return f"plan:{digest}"

    def _format_title(self, text: str) -> str:
        normalized = text.strip()
        if self.title_prefix:
            normalized = f"{self.title_prefix} {normalized}"
        return normalized[:120].rstrip()

    def _format_body(self, *, rel_path: str, line_number: int, section: str) -> str:
        lines: list[str] = []
        if section:
            lines.append(f"归属：{section}")
        lines.append(f"来源：`{rel_path}:{line_number}`")
        return "\n\n".join(lines).strip()
