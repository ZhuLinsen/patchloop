"""空闲巡检器。"""
from __future__ import annotations

import ast
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from adapters.base import BaseLLMAdapter, LLMError
from agent.context_builder import LocalContextBuilder
from agent.json_utils import extract_json_object
from agent.plan_source import PlanItem, PlanSource
from prompts import build_idle_candidate_prompt

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
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".rb",
    ".php",
    ".swift",
    ".kt",
    ".kts",
    ".scala",
    ".sql",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    ".md",
    ".txt",
    ".sh",
}

_FINDING_RE = re.compile(r"(?i)\b(TODO|FIXME)\b[:\s-]*(.+)?")
_CODE_SCOPE_RE = re.compile(r"^\s*(async\s+def|def|class|interface|enum|type|func|function|module)\b")
_MARKDOWN_SCOPE_RE = re.compile(r"^\s*#{1,6}\s+\S")
_PYTHON_EXCEPT_RE = re.compile(r"^\s*except(?:\s+.+)?\s*:\s*$")

_CATEGORY_LABELS = {
    "bug_suspect": "idle-bug-suspect",
    "code_improvement": "idle-code-improvement",
    "usability": "idle-usability",
    "reference_inspired": "idle-reference",
}

_RULE_METADATA = {
    "swallowed_exception": {
        "category": "bug_suspect",
        "title": "异常处理分支可能吞掉错误",
        "suggestion": "补充显式错误处理、日志或上抛路径，避免静默回退。",
    },
    "requests_without_timeout": {
        "category": "bug_suspect",
        "title": "网络请求缺少 timeout",
        "suggestion": "为 requests 调用补充 timeout，并明确失败后的回退策略。",
    },
    "subprocess_shell_true": {
        "category": "bug_suspect",
        "title": "subprocess 调用使用 shell=True",
        "suggestion": "评估是否可改为参数数组调用，并收敛命令拼接风险。",
    },
    "long_function": {
        "category": "code_improvement",
        "title": "函数过长，职责可能过于集中",
        "suggestion": "拆分函数职责，把校验、IO 和业务判断分离，降低维护成本。",
    },
    "argparse_without_help": {
        "category": "usability",
        "title": "命令行参数缺少 help 文案",
        "suggestion": "补充 help 文案和默认值提示，提升命令行可用性。",
    },
}


@dataclass(frozen=True)
class IdleAnalysisCandidate:
    key: str
    rule_id: str
    category: str
    source_path: str
    line_number: int
    scope_hint: str
    evidence: str
    snippet: str
    suggestion_hint: str


@dataclass(frozen=True)
class IdleAnalysisNarrative:
    summary: str
    impact: str
    suggestion: str
    reference_hint: str
    confidence: float


class IdleScanner(PlanSource):
    """扫描本地仓库中的 TODO/FIXME 与高价值改进候选，并转成统一计划项。"""

    def __init__(
        self,
        *,
        repo_path: str,
        default_labels: list[str],
        title_prefix: str,
        max_items_per_cycle: int,
        analyzer: BaseLLMAdapter | None = None,
        context_builder: LocalContextBuilder | None = None,
        repo_name: str = "",
        enable_heuristic_analysis: bool = True,
        enable_llm_enrichment: bool = True,
        advanced_only_when_todo_clear: bool = True,
        long_function_lines: int = 80,
        llm_max_context_files: int = 4,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.default_labels = [item.strip() for item in default_labels if item.strip()]
        self.title_prefix = title_prefix.strip()
        self.max_items_per_cycle = max_items_per_cycle
        self.analyzer = analyzer
        self.context_builder = context_builder
        self.repo_name = repo_name.strip()
        self.enable_heuristic_analysis = enable_heuristic_analysis
        self.enable_llm_enrichment = enable_llm_enrichment
        self.advanced_only_when_todo_clear = advanced_only_when_todo_clear
        self.long_function_lines = max(int(long_function_lines or 0), 1)
        self.llm_max_context_files = max(int(llm_max_context_files or 0), 1)
        # Cache LLM-enriched narratives by candidate key to avoid re-calling
        # LLM for the same candidate across idle scan cycles.
        self._narrative_cache: dict[str, IdleAnalysisNarrative] = {}

    def collect_items(self) -> list[PlanItem]:
        if not self.repo_path.exists():
            return []

        todo_items = self._collect_todo_items(limit=self.max_items_per_cycle)
        if todo_items and self.advanced_only_when_todo_clear:
            return todo_items[: self.max_items_per_cycle]

        items = todo_items[: self.max_items_per_cycle]
        if len(items) >= self.max_items_per_cycle:
            return items

        advanced_items = self._collect_advanced_items()
        if todo_items:
            items.extend(
                item
                for item in advanced_items
                if item.key not in {existing.key for existing in items}
            )
            return items[: self.max_items_per_cycle]
        return advanced_items[: self.max_items_per_cycle]

    def has_item_key(self, item_key: str) -> bool:
        return self.get_item_by_key(item_key) is not None

    def get_item_by_key(self, item_key: str) -> PlanItem | None:
        if not item_key or not self.repo_path.exists():
            return None
        for item in self._collect_todo_items(limit=None):
            if item.key == item_key:
                return item
        for candidate in self._collect_heuristic_candidates(limit=None):
            if candidate.key == item_key:
                return self._plan_item_from_candidate(candidate)
        return None

    def _collect_todo_items(self, *, limit: int | None) -> list[PlanItem]:
        effective_limit = max(int(limit or 0), 1) if limit is not None else None
        items: list[PlanItem] = []
        for path in self._iter_candidate_files():
            if effective_limit is not None and len(items) >= effective_limit:
                break
            if self._should_skip(path):
                continue
            if path.suffix.lower() not in _TEXT_EXTENSIONS:
                continue
            items.extend(self._scan_todo_file(path))
            if effective_limit is not None and len(items) >= effective_limit:
                break
        return items[:effective_limit] if effective_limit is not None else items

    def _collect_advanced_items(self) -> list[PlanItem]:
        if not self.enable_heuristic_analysis:
            return []
        return [self._plan_item_from_candidate(candidate) for candidate in self._collect_heuristic_candidates(limit=self.max_items_per_cycle)]

    def _collect_heuristic_candidates(self, *, limit: int | None) -> list[IdleAnalysisCandidate]:
        effective_limit = max(int(limit or 0), 1) if limit is not None else None
        candidates: list[IdleAnalysisCandidate] = []
        for path in self._iter_candidate_files():
            if self._should_skip(path):
                continue
            if path.suffix.lower() != ".py":
                continue
            candidates.extend(self._scan_python_file(path))

        candidates = self._stabilize_heuristic_candidates(candidates)
        deduped: list[IdleAnalysisCandidate] = []
        seen_keys: set[str] = set()
        for candidate in candidates:
            if candidate.key in seen_keys:
                continue
            seen_keys.add(candidate.key)
            deduped.append(candidate)

        deduped.sort(
            key=lambda item: (
                0 if item.category == "bug_suspect" else 1 if item.category == "usability" else 2,
                item.source_path,
                item.line_number,
                item.rule_id,
            )
        )
        return deduped[:effective_limit] if effective_limit is not None else deduped

    def _scan_todo_file(self, path: Path) -> list[PlanItem]:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return []

        rel_path = path.relative_to(self.repo_path).as_posix()
        findings: list[dict[str, object]] = []
        for index, line in enumerate(lines, start=1):
            match = _FINDING_RE.search(line)
            if not match:
                continue
            marker = match.group(1).upper()
            detail = " ".join((match.group(2) or "").strip().split())
            if not detail:
                continue
            normalized_line = " ".join(str(line or "").strip().split())
            scope_hint = self._scope_hint(path, lines, index - 1)
            findings.append(
                {
                    "index": index,
                    "marker": marker,
                    "detail": detail,
                    "normalized_line": normalized_line,
                    "scope_hint": scope_hint,
                }
            )

        duplicate_suffix_counts = self._build_duplicate_suffix_counts(findings)
        items: list[PlanItem] = []
        for finding in findings:
            index = int(finding["index"])
            marker = str(finding["marker"])
            detail = str(finding["detail"])
            normalized_line = str(finding["normalized_line"])
            scope_hint = str(finding["scope_hint"])
            duplicate_suffix_count = duplicate_suffix_counts[(index, marker, detail, normalized_line, scope_hint)]
            item_key = self._build_todo_item_key(
                rel_path,
                marker,
                detail,
                line_text=normalized_line,
                scope_hint=scope_hint,
                duplicate_suffix_count=duplicate_suffix_count,
            )
            title = self._format_title(detail)
            body = self._format_todo_body(
                rel_path=rel_path,
                line_number=index,
                marker=marker,
                detail=detail,
                scope_hint=scope_hint,
            )
            items.append(
                PlanItem(
                    key=item_key,
                    title=title,
                    body=body,
                    source_path=rel_path,
                    line_number=index,
                    labels=list(self.default_labels),
                )
            )
        return items

    def _scan_python_file(self, path: Path) -> list[IdleAnalysisCandidate]:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return []
        lines = text.splitlines()
        rel_path = path.relative_to(self.repo_path).as_posix()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            tree = None
        candidates: list[IdleAnalysisCandidate] = []
        candidates.extend(self._scan_swallowed_exceptions(rel_path, lines, path))
        if tree is None:
            return candidates
        candidates.extend(self._scan_requests_without_timeout(rel_path, text, lines, path, tree))
        candidates.extend(self._scan_subprocess_shell_true(rel_path, text, lines, path, tree))
        candidates.extend(self._scan_long_functions(rel_path, lines, tree))
        candidates.extend(self._scan_argparse_without_help(rel_path, text, lines, path, tree))
        return candidates

    def _scan_swallowed_exceptions(self, rel_path: str, lines: list[str], path: Path) -> list[IdleAnalysisCandidate]:
        candidates: list[IdleAnalysisCandidate] = []
        for index, line in enumerate(lines):
            if not _PYTHON_EXCEPT_RE.match(line):
                continue
            indent = len(line) - len(line.lstrip(" "))
            next_line = self._next_meaningful_line(lines, index + 1)
            if next_line is None:
                continue
            next_index, next_text = next_line
            next_indent = len(next_text) - len(next_text.lstrip(" "))
            if next_indent <= indent:
                continue
            action = next_text.strip()
            if not (
                action == "pass"
                or action.startswith("return None")
                or action.startswith("return []")
                or action.startswith("return {}")
                or action.startswith('return ""')
                or action.startswith("return False")
                or action.startswith("continue")
                or action.startswith("break")
            ):
                continue
            scope_hint = self._scope_hint(path, lines, index)
            evidence = f"{line.strip()} -> {action}"
            candidates.append(
                self._build_candidate(
                    rule_id="swallowed_exception",
                    source_path=rel_path,
                    line_number=index + 1,
                    scope_hint=scope_hint,
                    evidence=evidence,
                    snippet=self._build_snippet(lines, index),
                )
            )
        return candidates

    def _scan_requests_without_timeout(
        self,
        rel_path: str,
        text: str,
        lines: list[str],
        path: Path,
        tree: ast.AST,
    ) -> list[IdleAnalysisCandidate]:
        candidates: list[IdleAnalysisCandidate] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not self._is_module_call(node, module_name="requests", function_names={"get", "post", "put", "patch", "delete", "head", "options"}):
                continue
            if self._has_keyword(node, "timeout"):
                continue
            line_number = int(getattr(node, "lineno", 0) or 0)
            if line_number <= 0:
                continue
            scope_hint = self._scope_hint(path, lines, line_number - 1)
            evidence = self._call_evidence(text, node)
            candidates.append(
                self._build_candidate(
                    rule_id="requests_without_timeout",
                    source_path=rel_path,
                    line_number=line_number,
                    scope_hint=scope_hint,
                    evidence=evidence[:240],
                    snippet=self._build_snippet(lines, line_number - 1),
                )
            )
        return candidates

    def _scan_subprocess_shell_true(
        self,
        rel_path: str,
        text: str,
        lines: list[str],
        path: Path,
        tree: ast.AST,
    ) -> list[IdleAnalysisCandidate]:
        candidates: list[IdleAnalysisCandidate] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not self._is_module_call(
                node,
                module_name="subprocess",
                function_names={"run", "Popen", "call", "check_call", "check_output"},
            ):
                continue
            if not self._keyword_is_true(node, "shell"):
                continue
            line_number = int(getattr(node, "lineno", 0) or 0)
            if line_number <= 0:
                continue
            scope_hint = self._scope_hint(path, lines, line_number - 1)
            evidence = self._call_evidence(text, node)
            candidates.append(
                self._build_candidate(
                    rule_id="subprocess_shell_true",
                    source_path=rel_path,
                    line_number=line_number,
                    scope_hint=scope_hint,
                    evidence=evidence[:240],
                    snippet=self._build_snippet(lines, line_number - 1),
                )
            )
        return candidates

    def _scan_long_functions(self, rel_path: str, lines: list[str], tree: ast.AST) -> list[IdleAnalysisCandidate]:
        candidates: list[IdleAnalysisCandidate] = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            start_line = int(getattr(node, "lineno", 0) or 0)
            end_line = int(getattr(node, "end_lineno", 0) or 0)
            if start_line <= 0 or end_line <= 0:
                continue
            length = end_line - start_line + 1
            if length < self.long_function_lines:
                continue
            signature = self._safe_line(lines, start_line - 1).strip()
            scope_hint = signature or getattr(node, "name", "")
            evidence = f"{scope_hint} 约 {length} 行"
            candidates.append(
                self._build_candidate(
                    rule_id="long_function",
                    source_path=rel_path,
                    line_number=start_line,
                    scope_hint=scope_hint,
                    evidence=evidence,
                    snippet=self._build_snippet(lines, start_line - 1),
                )
            )
        return candidates

    def _scan_argparse_without_help(
        self,
        rel_path: str,
        text: str,
        lines: list[str],
        path: Path,
        tree: ast.AST,
    ) -> list[IdleAnalysisCandidate]:
        candidates: list[IdleAnalysisCandidate] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute) or str(node.func.attr or "") != "add_argument":
                continue
            if self._has_keyword(node, "help"):
                continue
            line_number = int(getattr(node, "lineno", 0) or 0)
            if line_number <= 0:
                continue
            scope_hint = self._scope_hint(path, lines, line_number - 1)
            evidence = self._call_evidence(text, node)
            candidates.append(
                self._build_candidate(
                    rule_id="argparse_without_help",
                    source_path=rel_path,
                    line_number=line_number,
                    scope_hint=scope_hint,
                    evidence=evidence[:240],
                    snippet=self._build_snippet(lines, line_number - 1),
                )
            )
        return candidates

    def _build_candidate(
        self,
        *,
        rule_id: str,
        source_path: str,
        line_number: int,
        scope_hint: str,
        evidence: str,
        snippet: str,
    ) -> IdleAnalysisCandidate:
        metadata = _RULE_METADATA[rule_id]
        return IdleAnalysisCandidate(
            key="",
            rule_id=rule_id,
            category=str(metadata["category"]),
            source_path=source_path,
            line_number=line_number,
            scope_hint=scope_hint,
            evidence=evidence,
            snippet=snippet,
            suggestion_hint=str(metadata["suggestion"]),
        )

    def _plan_item_from_candidate(self, candidate: IdleAnalysisCandidate) -> PlanItem:
        title = self._format_analysis_title(candidate)
        body = self._format_analysis_body(candidate)
        labels = list(dict.fromkeys(self.default_labels))
        return PlanItem(
            key=candidate.key,
            title=title,
            body=body,
            source_path=candidate.source_path,
            line_number=candidate.line_number,
            labels=labels,
        )

    def _format_analysis_title(self, candidate: IdleAnalysisCandidate) -> str:
        metadata = _RULE_METADATA[candidate.rule_id]
        title = f"{candidate.source_path}: {metadata['title']}"
        return self._format_title(title)

    def _format_analysis_body(self, candidate: IdleAnalysisCandidate) -> str:
        narrative = self._build_candidate_narrative(candidate)
        lines = [
            f"在 `{candidate.source_path}:{candidate.line_number}` 发现了一个值得确认的问题。",
        ]
        if candidate.scope_hint:
            lines.extend(["", f"上下文：`{candidate.scope_hint}`"])
        lines.extend(
            [
                "",
                "问题说明",
                narrative.summary,
                "",
                "触发线索",
                candidate.evidence,
                "",
                "影响",
                narrative.impact,
                "",
                "建议",
                narrative.suggestion,
            ]
        )
        if narrative.reference_hint:
            lines.extend(
                [
                    "",
                    "参考",
                    narrative.reference_hint,
                ]
            )
        lines.extend(
            [
                "",
                f"置信度：`{narrative.confidence:.2f}`",
                "",
                "代码片段",
                "```text",
                candidate.snippet,
                "```",
            ]
        )
        return "\n".join(lines).strip()

    def _build_candidate_narrative(self, candidate: IdleAnalysisCandidate) -> IdleAnalysisNarrative:
        # Return cached result if this candidate was already enriched.
        if candidate.key and candidate.key in self._narrative_cache:
            return self._narrative_cache[candidate.key]
        fallback = self._fallback_narrative(candidate)
        if (
            not self.enable_llm_enrichment
            or self.analyzer is None
            or self.context_builder is None
            or not self.repo_name
        ):
            return fallback

        try:
            local_context = self.context_builder.build_idle_context(
                candidate_paths=[candidate.source_path][: self.llm_max_context_files]
            )
            prompt = build_idle_candidate_prompt(
                repo_name=self.repo_name,
                category=candidate.category,
                rule_id=candidate.rule_id,
                source_path=candidate.source_path,
                line_number=candidate.line_number,
                scope_hint=candidate.scope_hint,
                evidence=candidate.evidence,
                suggestion_hint=candidate.suggestion_hint,
                local_code_context=local_context,
            )
            raw = self.analyzer.analyze(prompt, system="你是保守的 AutoCode idle 巡检说明生成器，只输出 JSON。")
            result = self._parse_narrative(raw, fallback=fallback)
        except (LLMError, ValueError) as exc:
            logger.warning("Idle 巡检 LLM 增强失败，回退到静态说明: %s", exc)
            result = fallback
        if candidate.key:
            self._narrative_cache[candidate.key] = result
        return result

    def _fallback_narrative(self, candidate: IdleAnalysisCandidate) -> IdleAnalysisNarrative:
        metadata = _RULE_METADATA[candidate.rule_id]
        summary = f"该位置命中了 `{candidate.rule_id}` 巡检规则，建议作为后续改进候选确认。"
        impact = "如果这里确实存在隐藏的异常路径、稳定性风险或职责膨胀，后续维护成本会继续上升。"
        if candidate.category == "usability":
            impact = "这类问题通常不会立刻阻断功能，但会持续增加使用门槛和排障成本。"
        suggestion = candidate.suggestion_hint or str(metadata["suggestion"])
        return IdleAnalysisNarrative(
            summary=summary,
            impact=impact,
            suggestion=suggestion,
            reference_hint="",
            confidence=0.55,
        )

    def _parse_narrative(self, raw: str, *, fallback: IdleAnalysisNarrative) -> IdleAnalysisNarrative:
        data = extract_json_object(raw, context="idle narrative")

        summary = str(data.get("summary", "") or "").strip() or fallback.summary
        impact = str(data.get("impact", "") or "").strip() or fallback.impact
        suggestion = str(data.get("suggestion", "") or "").strip() or fallback.suggestion
        reference_hint = str(data.get("reference_hint", "") or "").strip()
        confidence = float(data.get("confidence", fallback.confidence) or fallback.confidence)
        confidence = min(max(confidence, 0.0), 1.0)
        return IdleAnalysisNarrative(
            summary=summary,
            impact=impact,
            suggestion=suggestion,
            reference_hint=reference_hint,
            confidence=confidence,
        )

    def _build_todo_item_key(
        self,
        rel_path: str,
        marker: str,
        detail: str,
        *,
        line_text: str,
        scope_hint: str,
        duplicate_suffix_count: int = 0,
    ) -> str:
        digest = hashlib.sha256(
            f"{rel_path}\n{marker}\n{detail}\n{line_text}\n{scope_hint}\n{int(duplicate_suffix_count)}".encode("utf-8")
        ).hexdigest()[:16]
        return f"idle:{digest}"

    def _build_analysis_key(
        self,
        *,
        rule_id: str,
        source_path: str,
        evidence: str,
        scope_hint: str,
        duplicate_suffix_count: int = 0,
    ) -> str:
        digest = hashlib.sha256(
            f"{rule_id}\n{source_path}\n{evidence}\n{scope_hint}\n{int(duplicate_suffix_count)}".encode("utf-8")
        ).hexdigest()[:16]
        return f"idle-analysis:{digest}"

    def _stabilize_heuristic_candidates(
        self,
        candidates: list[IdleAnalysisCandidate],
    ) -> list[IdleAnalysisCandidate]:
        if not candidates:
            return []
        duplicate_suffix_counts = self._build_analysis_duplicate_suffix_counts(candidates)
        stabilized: list[IdleAnalysisCandidate] = []
        for candidate in candidates:
            duplicate_suffix_count = duplicate_suffix_counts[
                (
                    candidate.line_number,
                    candidate.rule_id,
                    candidate.source_path,
                    candidate.scope_hint,
                    candidate.evidence,
                )
            ]
            stabilized.append(
                IdleAnalysisCandidate(
                    key=self._build_analysis_key(
                        rule_id=candidate.rule_id,
                        source_path=candidate.source_path,
                        evidence=candidate.evidence,
                        scope_hint=candidate.scope_hint,
                        duplicate_suffix_count=duplicate_suffix_count,
                    ),
                    rule_id=candidate.rule_id,
                    category=candidate.category,
                    source_path=candidate.source_path,
                    line_number=candidate.line_number,
                    scope_hint=candidate.scope_hint,
                    evidence=candidate.evidence,
                    snippet=candidate.snippet,
                    suggestion_hint=candidate.suggestion_hint,
                )
            )
        return stabilized

    def _build_analysis_duplicate_suffix_counts(
        self,
        candidates: list[IdleAnalysisCandidate],
    ) -> dict[tuple[int, str, str, str, str], int]:
        counts: dict[tuple[str, str, str, str], int] = {}
        suffix_counts: dict[tuple[int, str, str, str, str], int] = {}
        for candidate in reversed(candidates):
            signature = (
                candidate.rule_id,
                candidate.source_path,
                candidate.scope_hint,
                candidate.evidence,
            )
            duplicate_suffix_count = counts.get(signature, 0)
            suffix_counts[
                (
                    candidate.line_number,
                    candidate.rule_id,
                    candidate.source_path,
                    candidate.scope_hint,
                    candidate.evidence,
                )
            ] = duplicate_suffix_count
            counts[signature] = duplicate_suffix_count + 1
        return suffix_counts

    def _build_duplicate_suffix_counts(
        self,
        findings: list[dict[str, object]],
    ) -> dict[tuple[int, str, str, str, str], int]:
        counts: dict[tuple[str, str, str, str], int] = {}
        suffix_counts: dict[tuple[int, str, str, str, str], int] = {}
        for finding in reversed(findings):
            signature = (
                str(finding["marker"]),
                str(finding["detail"]),
                str(finding["normalized_line"]),
                str(finding["scope_hint"]),
            )
            duplicate_suffix_count = counts.get(signature, 0)
            suffix_counts[
                (
                    int(finding["index"]),
                    str(finding["marker"]),
                    str(finding["detail"]),
                    str(finding["normalized_line"]),
                    str(finding["scope_hint"]),
                )
            ] = duplicate_suffix_count
            counts[signature] = duplicate_suffix_count + 1
        return suffix_counts

    def _scope_hint(self, path: Path, lines: list[str], line_index: int) -> str:
        allow_markdown_headings = path.suffix.lower() in {".md", ".markdown", ".txt"}
        for index in range(line_index - 1, -1, -1):
            candidate = str(lines[index] or "").strip()
            if not candidate:
                continue
            if _CODE_SCOPE_RE.match(candidate):
                return " ".join(candidate.split())
            if allow_markdown_headings and _MARKDOWN_SCOPE_RE.match(candidate):
                return " ".join(candidate.split())
        return ""

    def _is_module_call(self, node: ast.Call, *, module_name: str, function_names: set[str]) -> bool:
        func = node.func
        if not isinstance(func, ast.Attribute):
            return False
        if str(func.attr or "") not in function_names:
            return False
        return isinstance(func.value, ast.Name) and str(func.value.id or "") == module_name

    def _has_keyword(self, node: ast.Call, keyword_name: str) -> bool:
        return any(str(keyword.arg or "") == keyword_name for keyword in node.keywords)

    def _keyword_is_true(self, node: ast.Call, keyword_name: str) -> bool:
        for keyword in node.keywords:
            if str(keyword.arg or "") != keyword_name:
                continue
            return isinstance(keyword.value, ast.Constant) and keyword.value.value is True
        return False

    def _call_evidence(self, text: str, node: ast.Call) -> str:
        segment = ast.get_source_segment(text, node) or ""
        if not segment.strip():
            return ""
        return " ".join(part.strip() for part in segment.splitlines() if part.strip())

    def _build_snippet(self, lines: list[str], center_index: int, *, window: int = 2) -> str:
        start = max(0, center_index - window)
        end = min(len(lines), center_index + window + 1)
        snippet_lines = [f"{index + 1}: {lines[index]}" for index in range(start, end)]
        return "\n".join(snippet_lines)

    def _next_meaningful_line(self, lines: list[str], start_index: int) -> tuple[int, str] | None:
        for index in range(start_index, len(lines)):
            candidate = str(lines[index] or "")
            stripped = candidate.strip()
            if not stripped or stripped.startswith("#"):
                continue
            return index, candidate
        return None

    def _safe_line(self, lines: list[str], index: int) -> str:
        if 0 <= index < len(lines):
            return str(lines[index] or "")
        return ""

    def _should_skip(self, path: Path) -> bool:
        relative = path.relative_to(self.repo_path)
        return any(part in _IGNORED_DIRS for part in relative.parts)

    def _iter_candidate_files(self):
        for root, dirnames, filenames in os.walk(self.repo_path):
            dirnames[:] = sorted(name for name in dirnames if name not in _IGNORED_DIRS)
            for filename in sorted(filenames):
                yield Path(root) / filename

    def _format_title(self, detail: str) -> str:
        normalized = detail.strip()
        if self.title_prefix:
            normalized = f"{self.title_prefix} {normalized}"
        return normalized[:120].rstrip()

    def _format_todo_body(
        self,
        *,
        rel_path: str,
        line_number: int,
        marker: str,
        detail: str,
        scope_hint: str,
    ) -> str:
        lines = [
            f"在 `{rel_path}:{line_number}` 发现了一条 `{marker}` 注释。",
            "",
            detail,
        ]
        if scope_hint:
            lines.extend(["", f"上下文：`{scope_hint}`"])
        return "\n".join(lines).strip()
