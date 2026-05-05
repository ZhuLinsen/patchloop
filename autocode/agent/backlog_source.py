"""本地 backlog 文件读取与 issue 同步输入。"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from agent.plan_source import PlanItem, PlanSource


_OPEN_CHECKBOX_RE = re.compile(r"^\s*[-*]\s+\[\s\]\s+(.+?)\s*$")
_HEADING_RE = re.compile(r"^(#+)\s+(.+?)\s*$")
_ISSUE_LINK_RE = re.compile(r"^\s{2,}issue:\s+(.*)$", re.IGNORECASE)
_ISSUE_MARKDOWN_LINK_RE = re.compile(r"\[#(?P<number>\d+)\]\((?P<url>[^)]+)\)")
_ISSUE_PLAIN_RE = re.compile(r"#(?P<number>\d+)")


BacklogItem = PlanItem


class BacklogSource(PlanSource):
    """从本地 Markdown backlog/todo 文件中提取未完成条目。"""

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

    def annotate_item_issue(self, item_key: str, *, issue_number: int, issue_url: str) -> bool:
        if not item_key or issue_number <= 0:
            return False
        for rel_path in self.file_paths:
            resolved = self._resolve_path(rel_path)
            if resolved is None or not resolved.exists() or not resolved.is_file():
                continue
            if self._annotate_file_issue(resolved, rel_path, item_key=item_key, issue_number=issue_number, issue_url=issue_url):
                return True
        return False

    def get_item_issue_reference(self, item_key: str) -> dict[str, str | int] | None:
        if not item_key:
            return None
        for rel_path in self.file_paths:
            resolved = self._resolve_path(rel_path)
            if resolved is None or not resolved.exists() or not resolved.is_file():
                continue
            reference = self._file_issue_reference_by_key(resolved, rel_path, item_key)
            if reference is not None:
                return reference
        return None

    def _parse_file(self, path: Path, rel_path: str) -> list[PlanItem]:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return []

        candidates = self._collect_candidates(lines)

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
                BacklogItem(
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
        return f"backlog:{digest}"

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

    def _collect_candidates(self, lines: list[str]) -> list[tuple[int, str, str]]:
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
        return candidates

    def _annotate_file_issue(self, path: Path, rel_path: str, *, item_key: str, issue_number: int, issue_url: str) -> bool:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return False

        candidates = self._collect_candidates(lines)
        duplicate_suffix_counts = self._build_duplicate_suffix_counts(candidates)
        matched_line_number = 0
        for index, section, text in candidates:
            candidate_key = self._build_item_key(
                rel_path,
                section,
                text,
                duplicate_suffix_count=duplicate_suffix_counts[(index, section, text)],
            )
            if candidate_key == item_key:
                matched_line_number = index
                break
        if matched_line_number <= 0:
            return False

        start_idx, end_idx = self._find_item_block(lines, matched_line_number)

        annotation = f"  issue: [#{issue_number}]({issue_url})" if issue_url else f"  issue: #{issue_number}"
        issue_line_idx = None
        for pos in range(start_idx + 1, end_idx):
            if _ISSUE_LINK_RE.match(lines[pos]):
                issue_line_idx = pos
                break

        if issue_line_idx is not None:
            if lines[issue_line_idx] == annotation:
                return False
            lines[issue_line_idx] = annotation
        else:
            insert_idx = end_idx
            while insert_idx > start_idx + 1 and not lines[insert_idx - 1].strip():
                insert_idx -= 1
            lines.insert(insert_idx, annotation)

        try:
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        except OSError:
            return False
        return True

    def _file_issue_reference_by_key(self, path: Path, rel_path: str, item_key: str) -> dict[str, str | int] | None:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return None

        candidates = self._collect_candidates(lines)
        duplicate_suffix_counts = self._build_duplicate_suffix_counts(candidates)
        matched_line_number = 0
        for index, section, text in candidates:
            candidate_key = self._build_item_key(
                rel_path,
                section,
                text,
                duplicate_suffix_count=duplicate_suffix_counts[(index, section, text)],
            )
            if candidate_key == item_key:
                matched_line_number = index
                break
        if matched_line_number <= 0:
            return None

        start_idx, end_idx = self._find_item_block(lines, matched_line_number)
        for pos in range(start_idx + 1, end_idx):
            match = _ISSUE_LINK_RE.match(lines[pos])
            if not match:
                continue
            return self._parse_issue_reference(match.group(1))
        return None

    def _find_item_block(self, lines: list[str], matched_line_number: int) -> tuple[int, int]:
        start_idx = matched_line_number - 1
        end_idx = len(lines)
        for pos in range(start_idx + 1, len(lines)):
            if _HEADING_RE.match(lines[pos]) or _OPEN_CHECKBOX_RE.match(lines[pos]):
                end_idx = pos
                break
        return start_idx, end_idx

    def _parse_issue_reference(self, value: str) -> dict[str, str | int] | None:
        text = str(value or "").strip()
        if not text:
            return None
        markdown_match = _ISSUE_MARKDOWN_LINK_RE.search(text)
        if markdown_match:
            return {
                "issue_number": int(markdown_match.group("number")),
                "issue_url": markdown_match.group("url").strip(),
            }
        plain_match = _ISSUE_PLAIN_RE.search(text)
        if plain_match:
            return {
                "issue_number": int(plain_match.group("number")),
                "issue_url": "",
            }
        return None
