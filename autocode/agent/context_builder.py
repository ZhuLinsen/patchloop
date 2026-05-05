"""
本地代码上下文构建器

职责:
1. 基于本地仓库目录生成结构概览
2. 为 Issue 按关键词匹配相关文件并提取代码片段
3. 为 PR 基于 changed files 提取本地文件快照
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import re

logger = logging.getLogger(__name__)

_IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "venv",
    ".venv",
    "node_modules",
    "dist",
    "build",
    "__pycache__",
}

_TEXT_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".java", ".go", ".rs", ".c", ".cc", ".cpp",
    ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt", ".kts", ".scala", ".sql",
    ".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf", ".md", ".txt", ".sh",
}
_MAX_SCAN_FILE_BYTES = 200_000
_CONTENT_MATCH_WINDOW = 2
_CONTENT_MATCH_LIMIT = 3
_DOC_EXTENSIONS = {".md", ".txt"}
_CONFIG_EXTENSIONS = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf"}
_USAGE_HINTS = (
    "怎么", "如何", "在哪", "哪里", "配置", "文档", "说明", "readme", "docs",
    "env", "环境变量", "参数", "设置", "开关", "选项", "命令", "启动", "使用",
    "部署", "how", "where", "usage", "config", "configure", "setting", "option",
)
_FILE_MENTION_RE = re.compile(
    r"(?<![\w/.-])(?:[\w.-]+/)*[\w.-]+\."
    r"(?:py|js|ts|tsx|jsx|go|rs|java|kt|kts|md|mdx|rst|yml|yaml|json|toml|ini|cfg|conf|sh|sql|css|scss|html|vue)"
    r"(?::\d+)?"
)
_STACK_FILE_RE = re.compile(r"File [\"']([^\"']+)[\"'], line \d+")
_ERROR_SYMBOL_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:Error|Exception)\b|\b[A-Z][A-Z0-9_]{3,}\b")


@dataclass(frozen=True)
class LocalFileMatch:
    path: Path
    score: int
    matched_terms: tuple[str, ...]
    snippet: str


class LocalContextBuilder:
    """从本地代码仓库构建可注入到 prompt 的上下文。"""

    def __init__(
        self,
        repo_path: str,
        max_context_files: int = 6,
        max_file_chars: int = 1800,
        max_total_chars: int = 12000,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.max_context_files = max_context_files
        self.max_file_chars = max_file_chars
        self.max_total_chars = max_total_chars

    def build_issue_context(self, title: str, body: str) -> str:
        """为 Issue 构建本地代码上下文。"""
        issue_text = f"{title}\n{body}".strip()
        sections = [
            "## 本地仓库结构概览\n" + self._build_repo_outline(),
        ]
        issue_guidance = self._build_agents_guidance("issue")
        if issue_guidance:
            sections.append(issue_guidance)

        repo_instructions = self.build_repo_instructions()
        if repo_instructions:
            sections.append(repo_instructions)

        relevant_matches = self._find_issue_matches(issue_text)
        evidence_pack = self._build_evidence_pack(
            issue_text,
            seed_paths=[match.path for match in relevant_matches[: self.max_context_files]],
        )
        if evidence_pack:
            sections.append("## 自动证据包\n" + evidence_pack)
        if relevant_matches:
            sections.append("## 可能相关的本地代码位置\n" + self._format_issue_matches(relevant_matches))

        return "\n\n".join(section for section in sections if section.strip())

    def build_pr_context(self, changed_files: list[str] | None) -> str:
        """为 PR 构建本地代码上下文。"""
        sections = [
            "## 本地仓库结构概览\n" + self._build_repo_outline(),
        ]
        pr_guidance = self._build_agents_guidance("pr")
        if pr_guidance:
            sections.append(pr_guidance)

        repo_instructions = self.build_repo_instructions()
        if repo_instructions:
            sections.append(repo_instructions)

        scored_files: list[tuple[int, str, Path]] = []
        changed_rel_paths: set[str] = set()
        for rel_path in changed_files or []:
            resolved = self._resolve_file(rel_path)
            if resolved is not None:
                normalized_rel_path = resolved.relative_to(self.repo_path).as_posix()
                changed_rel_paths.add(normalized_rel_path)
                scored_files.append((self._score_pr_file(normalized_rel_path), normalized_rel_path, resolved))

        local_files = [
            path for _, _, path in sorted(scored_files, key=lambda item: (-item[0], len(item[1]), item[1]))[
                : self.max_context_files
            ]
        ]

        if local_files:
            sections.append("## PR 对应的本地文件快照\n" + self._format_files(local_files))
        evidence_pack = self._build_evidence_pack(
            "\n".join(changed_files or []),
            seed_paths=[path for _, _, path in scored_files],
            exclude_paths=changed_rel_paths,
        )
        if evidence_pack:
            sections.append("## 自动证据包\n" + evidence_pack)

        return "\n\n".join(section for section in sections if section.strip())

    def build_idle_context(self, candidate_paths: list[str] | None = None) -> str:
        """为 idle 巡检候选构建仓库上下文。"""
        sections = [
            "## 本地仓库结构概览\n" + self._build_repo_outline(),
        ]
        idle_guidance = self._build_agents_guidance("issue")
        if idle_guidance:
            sections.append(idle_guidance)

        files: list[Path] = []
        for rel_path in candidate_paths or []:
            resolved = self._resolve_file(rel_path)
            if resolved is not None and resolved not in files:
                files.append(resolved)
            if len(files) >= self.max_context_files:
                break

        if files:
            sections.append("## Idle 巡检候选相关文件\n" + self._format_files(files))

        return "\n\n".join(section for section in sections if section.strip())

    def build_repo_instructions(self) -> str:
        """加载 .github/instructions/*.md 并返回合并后的编码规范文本。"""
        instructions_dir = self.repo_path / ".github" / "instructions"
        if not instructions_dir.is_dir():
            return ""
        collected: list[str] = []
        total_chars = 0
        budget = self.max_total_chars
        for md_path in sorted(instructions_dir.iterdir()):
            if not md_path.is_file() or md_path.suffix.lower() not in (".md", ".txt"):
                continue
            try:
                content = md_path.read_text(encoding="utf-8", errors="ignore").strip()
            except OSError as exc:
                logger.warning("读取 instructions 文件失败 %s: %s", md_path, exc)
                continue
            if not content:
                continue
            if total_chars + len(content) > budget:
                remaining = max(budget - total_chars - 50, 0)
                if remaining > 200:
                    content = content[:remaining] + "\n\n... [截断]"
                else:
                    break
            collected.append(f"### {md_path.name}\n\n{content}")
            total_chars += len(content)
        if not collected:
            return ""
        return "## 目标仓库编码规范 (.github/instructions)\n\n" + "\n\n---\n\n".join(collected)

    def _build_agents_guidance(self, review_type: str) -> str:
        agents_path = self.repo_path / "AGENTS.md"
        if not agents_path.exists() or not agents_path.is_file():
            return ""
        try:
            content = agents_path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            logger.warning("读取 AGENTS.md 失败 %s: %s", agents_path, exc)
            return ""

        if review_type == "pr":
            headings = ["PR 分析原则", "建议评审输出格式"]
        else:
            headings = ["Issue 分析原则", "建议评审输出格式"]
        extracted = self._extract_markdown_sections(content, headings)
        if not extracted:
            return ""
        if len(extracted) > self.max_file_chars:
            extracted = extracted[: self.max_file_chars] + "\n\n... [AGENTS.md 内容过长，已截断]"
        return "## 仓库 AGENTS 规范\n" + extracted

    def _extract_markdown_sections(self, content: str, headings: list[str]) -> str:
        lines = content.splitlines()
        collected: list[str] = []
        current: list[str] = []
        capture = False
        targets = {self._normalize_markdown_heading(heading) for heading in headings}

        def flush():
            nonlocal current
            if current:
                collected.append("\n".join(current).strip())
                current = []

        for line in lines:
            if line.startswith("## "):
                if capture:
                    flush()
                    capture = False
                if self._normalize_markdown_heading(line) in targets:
                    capture = True
                    current = [line]
                continue
            if capture:
                current.append(line)

        if capture:
            flush()
        return "\n\n".join(section for section in collected if section.strip())

    def _normalize_markdown_heading(self, heading: str) -> str:
        normalized = heading.strip()
        if normalized.startswith("#"):
            normalized = normalized.lstrip("#").strip()
        return re.sub(r"^\d+(?:\.\d+)*\.\s*", "", normalized).strip()

    def _build_repo_outline(self, max_entries: int = 30) -> str:
        if not self.repo_path.exists():
            return f"仓库目录不存在: {self.repo_path}"

        lines: list[str] = []
        count = 0
        for path in sorted(self.repo_path.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
            if self._should_skip(path):
                continue
            suffix = "/" if path.is_dir() else ""
            lines.append(f"- {path.name}{suffix}")
            count += 1
            if path.is_dir():
                for child in sorted(path.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
                    if self._should_skip(child):
                        continue
                    child_suffix = "/" if child.is_dir() else ""
                    lines.append(f"  - {child.name}{child_suffix}")
                    count += 1
                    if count >= max_entries:
                        lines.append("- ...")
                        return "\n".join(lines)
            if count >= max_entries:
                lines.append("- ...")
                return "\n".join(lines)

        return "\n".join(lines) if lines else "- (仓库为空)"

    def _find_issue_matches(self, text: str) -> list[LocalFileMatch]:
        if not text.strip() or not self.repo_path.exists():
            return []

        text_lower = text.lower()
        search_terms = self._extract_search_terms(text)
        usage_focused = self._looks_like_usage_question(text)
        scored: list[LocalFileMatch] = []
        for path in self._iter_candidate_files():
            rel_path = path.relative_to(self.repo_path).as_posix().lower()
            file_name = path.name.lower()
            stem = path.stem.lower()

            score = 0
            path_hits: list[str] = []
            if rel_path in text_lower:
                score += 10
                path_hits.append(rel_path)
            if file_name in text_lower:
                score += 6
                path_hits.append(file_name)
            if len(stem) >= 4 and stem in text_lower:
                score += 3
                path_hits.append(stem)

            content_score, content_hits, snippet = self._score_file_content(path, search_terms)
            total_score = score + content_score + self._score_usage_bias(
                path,
                usage_focused,
                text_lower=text_lower,
                has_path_hits=bool(path_hits),
                has_content_hits=bool(content_hits),
            )
            if total_score <= 0:
                continue

            matched_terms = tuple(dict.fromkeys(path_hits + content_hits))
            if not snippet:
                snippet = self._read_file_snippet(path)
            if not snippet:
                continue
            scored.append(
                LocalFileMatch(
                    path=path,
                    score=total_score,
                    matched_terms=matched_terms,
                    snippet=snippet,
                )
            )

        scored.sort(
            key=lambda item: (
                -item.score,
                len(item.path.relative_to(self.repo_path).as_posix()),
                item.path.relative_to(self.repo_path).as_posix(),
            )
        )
        return scored[: self.max_context_files]

    def _iter_candidate_files(self):
        for path in self.repo_path.rglob("*"):
            if not path.is_file():
                continue
            if self._should_skip(path):
                continue
            if not self._is_text_candidate(path):
                continue
            yield path

    def _build_evidence_pack(
        self,
        text: str,
        *,
        seed_paths: list[Path] | None = None,
        exclude_paths: set[str] | None = None,
    ) -> str:
        if not self.repo_path.exists():
            return ""
        exclude_paths = exclude_paths or set()
        explicit_files = self._resolve_explicit_file_mentions(text)
        evidence_terms = self._extract_evidence_terms(text)
        related_tests = self._find_related_tests(
            [*explicit_files, *(seed_paths or [])],
            terms=evidence_terms,
            exclude_paths=exclude_paths,
        )

        blocks: list[str] = []
        if explicit_files:
            blocks.append("### 明确提到的文件\n" + self._format_path_bullets(explicit_files, limit=8))
        if evidence_terms:
            blocks.append("### 关键错误/标识符线索\n" + "\n".join(f"- `{term}`" for term in evidence_terms[:10]))
        if related_tests:
            blocks.append("### 可能相关测试\n" + self._format_path_bullets(related_tests, limit=8))
        return "\n\n".join(blocks)

    def _resolve_explicit_file_mentions(self, text: str) -> list[Path]:
        paths: list[Path] = []
        seen: set[str] = set()
        candidates = [match.group(0) for match in _FILE_MENTION_RE.finditer(str(text or ""))]
        candidates.extend(match.group(1) for match in _STACK_FILE_RE.finditer(str(text or "")))
        for candidate in candidates:
            resolved = self._resolve_mentioned_file(candidate)
            if resolved is None:
                continue
            rel_path = resolved.relative_to(self.repo_path).as_posix()
            if rel_path in seen:
                continue
            seen.add(rel_path)
            paths.append(resolved)
            if len(paths) >= self.max_context_files:
                break
        return paths

    def _resolve_mentioned_file(self, value: str) -> Path | None:
        normalized = str(value or "").strip().strip("`'\"")
        normalized = re.sub(r":\d+(?::\d+)?$", "", normalized).replace("\\", "/")
        if not normalized:
            return None
        candidate = Path(normalized)
        if candidate.is_absolute():
            try:
                rel = candidate.resolve().relative_to(self.repo_path)
            except (OSError, ValueError):
                parts = candidate.parts
                for index in range(max(len(parts) - 6, 0), len(parts)):
                    resolved = self._resolve_file("/".join(parts[index:]))
                    if resolved is not None:
                        return resolved
                return None
            return self._resolve_file(rel.as_posix())
        return self._resolve_file(normalized)

    def _extract_evidence_terms(self, text: str) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()
        for raw in _ERROR_SYMBOL_RE.findall(str(text or "")):
            term = str(raw or "").strip()
            lowered = term.lower()
            if not term or lowered in seen:
                continue
            seen.add(lowered)
            terms.append(term)
        for term in self._extract_search_terms(str(text or ""), max_terms=16):
            if not any(token in term for token in ("_", ".", "-", "/")) and not term[:1].isupper():
                continue
            lowered = term.lower()
            if lowered in seen:
                continue
            seen.add(lowered)
            terms.append(term)
            if len(terms) >= 12:
                break
        return terms

    def _find_related_tests(
        self,
        seed_paths: list[Path],
        *,
        terms: list[str] | None = None,
        exclude_paths: set[str] | None = None,
        limit: int = 8,
    ) -> list[Path]:
        exclude_paths = exclude_paths or set()
        stems: set[str] = set()
        for path in seed_paths:
            if not path:
                continue
            stem = path.stem.lower()
            if stem and not stem.startswith("test"):
                stems.add(stem)
        for term in terms or []:
            normalized = re.sub(r"[^A-Za-z0-9_]+", "_", term).strip("_").lower()
            if len(normalized) >= 4:
                stems.add(normalized)
        if not stems:
            return []

        matches: list[Path] = []
        seen: set[str] = set()
        for path in self._iter_candidate_files():
            rel_path = path.relative_to(self.repo_path).as_posix()
            if rel_path in exclude_paths or not self._is_test_file(path):
                continue
            rel_lower = rel_path.lower()
            name_lower = path.stem.lower()
            if not any(stem in rel_lower or stem in name_lower for stem in stems):
                continue
            if rel_path in seen:
                continue
            seen.add(rel_path)
            matches.append(path)
            if len(matches) >= limit:
                break
        return matches

    def _is_test_file(self, path: Path) -> bool:
        rel_path = path.relative_to(self.repo_path).as_posix().lower()
        name = path.name.lower()
        return (
            rel_path.startswith("tests/")
            or "/tests/" in rel_path
            or "__tests__/" in rel_path
            or name.startswith("test_")
            or name.endswith("_test.py")
            or ".test." in name
            or ".spec." in name
        )

    def _format_path_bullets(self, paths: list[Path], *, limit: int) -> str:
        lines = [f"- `{path.relative_to(self.repo_path).as_posix()}`" for path in paths[:limit]]
        if len(paths) > limit:
            lines.append(f"- ... 还有 {len(paths) - limit} 个候选未展开")
        return "\n".join(lines)

    def _format_files(self, files: list[Path]) -> str:
        blocks: list[str] = []
        remaining_chars = self.max_total_chars

        rendered_files = 0

        for path in files:
            snippet = self._read_file_snippet(path)
            if not snippet:
                continue

            rel_path = path.relative_to(self.repo_path).as_posix()
            block = (
                f"### 文件: {rel_path}\n"
                f"```\n{snippet}\n```"
            )
            if len(block) > remaining_chars:
                if remaining_chars <= 0:
                    break
                if rendered_files > 0:
                    break
                block = block[:remaining_chars]
            blocks.append(block)
            rendered_files += 1
            remaining_chars -= len(block)
            if remaining_chars <= 0:
                break

        omitted_files = max(len(files) - rendered_files, 0)
        if omitted_files > 0:
            blocks.append(f"... [其余 {omitted_files} 个文件未展开，完整候选列表已在上文给出]")

        return "\n\n".join(blocks)

    def _format_issue_matches(self, matches: list[LocalFileMatch]) -> str:
        blocks: list[str] = []
        remaining_chars = self.max_total_chars

        for match in matches:
            rel_path = match.path.relative_to(self.repo_path).as_posix()
            matched_terms = ", ".join(match.matched_terms[:6]) if match.matched_terms else "无"
            block = (
                f"### 文件: {rel_path}\n"
                f"- 命中线索: {matched_terms}\n"
                f"```text\n{match.snippet}\n```"
            )
            if len(block) > remaining_chars:
                if remaining_chars <= 0:
                    break
                block = block[:remaining_chars]
            blocks.append(block)
            remaining_chars -= len(block)
            if remaining_chars <= 0:
                break

        return "\n\n".join(blocks)

    def _read_file_snippet(self, path: Path) -> str:
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            logger.warning("读取本地文件失败 %s: %s", path, exc)
            return ""

        content = content.strip()
        if not content:
            return ""

        if len(content) > self.max_file_chars:
            return content[: self.max_file_chars] + "\n\n... [本地文件内容过长，已截断]"
        return content

    def _score_file_content(self, path: Path, terms: list[str]) -> tuple[int, list[str], str]:
        if not terms:
            return 0, [], ""

        try:
            if path.stat().st_size > _MAX_SCAN_FILE_BYTES:
                return 0, [], ""
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            logger.warning("扫描本地文件失败 %s: %s", path, exc)
            return 0, [], ""

        if not content.strip():
            return 0, [], ""

        content_lower = content.lower()
        matched_terms: list[str] = []
        score = 0
        for term in terms:
            normalized = term.lower()
            if normalized not in content_lower:
                continue
            matched_terms.append(term)
            occurrences = content_lower.count(normalized)
            score += min(occurrences, 3) * 2
            if any(token in term for token in ("_", ".", "/", "-")):
                score += 1

        if score <= 0:
            return 0, [], ""
        return score, matched_terms, self._extract_match_snippet(content, terms)

    def _extract_match_snippet(self, content: str, terms: list[str]) -> str:
        lines = content.splitlines()
        if not lines:
            return ""

        normalized_terms = [term.lower() for term in terms if term.strip()]
        hit_indexes: list[int] = []
        for idx, line in enumerate(lines):
            lowered = line.lower()
            if any(term in lowered for term in normalized_terms):
                hit_indexes.append(idx)
                if len(hit_indexes) >= _CONTENT_MATCH_LIMIT:
                    break

        if not hit_indexes:
            return self._truncate_text_block("\n".join(lines[: min(len(lines), 20)]))

        selected_indexes: list[int] = []
        seen_indexes: set[int] = set()
        for hit_idx in hit_indexes:
            start = max(0, hit_idx - _CONTENT_MATCH_WINDOW)
            end = min(len(lines), hit_idx + _CONTENT_MATCH_WINDOW + 1)
            for line_idx in range(start, end):
                if line_idx in seen_indexes:
                    continue
                seen_indexes.add(line_idx)
                selected_indexes.append(line_idx)

        snippet_lines: list[str] = []
        previous_idx = -1
        for line_idx in selected_indexes:
            if previous_idx >= 0 and line_idx > previous_idx + 1:
                snippet_lines.append("...")
            snippet_lines.append(f"{line_idx + 1}: {lines[line_idx]}")
            previous_idx = line_idx

        return self._truncate_text_block("\n".join(snippet_lines))

    def _truncate_text_block(self, text: str) -> str:
        normalized = text.strip()
        if len(normalized) > self.max_file_chars:
            return normalized[: self.max_file_chars] + "\n\n... [本地文件内容过长，已截断]"
        return normalized

    def _extract_search_terms(self, text: str, *, max_terms: int = 10) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()
        for raw in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z0-9_./-]{3,}", text):
            term = raw.strip().replace("\"", "")
            if not term:
                continue
            lowered = term.lower()
            if lowered in seen:
                continue
            if term.isdigit():
                continue
            seen.add(lowered)
            terms.append(term)
            if len(terms) >= max_terms:
                break
        return terms

    def _looks_like_usage_question(self, text: str) -> bool:
        lowered = text.lower()
        return any(token in lowered for token in _USAGE_HINTS)

    def _score_usage_bias(
        self,
        path: Path,
        usage_focused: bool,
        *,
        text_lower: str,
        has_path_hits: bool,
        has_content_hits: bool,
    ) -> int:
        if not usage_focused:
            return 0

        rel_path = path.relative_to(self.repo_path).as_posix().lower()
        file_name = path.name.lower()
        suffix = path.suffix.lower()
        aligned = has_path_hits or has_content_hits or self._path_matches_usage_intent(
            rel_path=rel_path,
            file_name=file_name,
            suffix=suffix,
            text_lower=text_lower,
        )
        if not aligned:
            return 0

        score = 0

        if file_name == "readme.md":
            score += 4
        elif rel_path.startswith("docs/") and suffix in _DOC_EXTENSIONS:
            score += 4

        if file_name.startswith(".env"):
            score += 4
        if "config" in file_name or "settings" in file_name:
            score += 3
        if suffix in _CONFIG_EXTENSIONS:
            score += 2
        if rel_path.startswith(".github/workflows/"):
            score += 1

        return score

    def _path_matches_usage_intent(
        self,
        *,
        rel_path: str,
        file_name: str,
        suffix: str,
        text_lower: str,
    ) -> bool:
        wants_docs = any(token in text_lower for token in ("readme", "docs", "文档", "说明", "usage", "how"))
        wants_env = any(token in text_lower for token in ("env", "环境变量"))
        wants_config = any(token in text_lower for token in ("config", "setting", "option", "配置", "参数", "设置"))
        wants_workflow = any(token in text_lower for token in ("workflow", "actions", "ci", "github action"))

        if wants_docs and (file_name == "readme.md" or rel_path.startswith("docs/")):
            return True
        if wants_env and file_name.startswith(".env"):
            return True
        if wants_config and (
            "config" in file_name
            or "settings" in file_name
            or suffix in _CONFIG_EXTENSIONS
        ):
            return True
        if wants_workflow and rel_path.startswith(".github/workflows/"):
            return True
        return False

    def _resolve_file(self, rel_path: str) -> Path | None:
        normalized = rel_path.strip().replace("\\", "/")
        if not normalized:
            return None

        path = (self.repo_path / normalized).resolve()
        try:
            path.relative_to(self.repo_path)
        except ValueError:
            return None

        if not path.exists() or not path.is_file() or self._should_skip(path):
            return None
        return path

    def _is_text_candidate(self, path: Path) -> bool:
        return path.suffix.lower() in _TEXT_EXTENSIONS or path.name.lower().startswith(".env")

    def _should_skip(self, path: Path) -> bool:
        parts = set(path.parts)
        if _IGNORED_DIRS & parts:
            return True
        return path.name.startswith(".") and not (
            path.name.lower().startswith(".env") or path.name == ".gitignore"
        )

    def _score_pr_file(self, rel_path: str) -> int:
        rel_lower = rel_path.lower()
        score = 0
        if rel_lower.startswith("tests/") or "/tests/" in rel_lower or rel_lower.endswith("_test.py"):
            score += 4
        if rel_lower.startswith("docs/"):
            score += 1
        return score
