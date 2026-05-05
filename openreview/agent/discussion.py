"""GitHub Issue / PR 讨论上下文辅助工具。"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re

_EN_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "that", "this", "there", "have", "when",
    "what", "where", "which", "while", "after", "before", "issue", "error", "failed",
    "failure", "cannot", "could", "would", "should", "please", "need", "help", "does",
    "doesnt", "cant", "it's", "its", "been", "using", "used", "use", "about", "more",
}
_ZH_STOPWORDS = {
    "问题", "报错", "无法", "不能", "怎么", "是否", "已经", "需要", "没有", "这里", "这个", "那个",
    "相关", "出现", "一个", "一下", "支持", "修复", "还是", "以及", "目前", "希望",
}
_IDEMPOTENCY_MARKER_RE = re.compile(r"<!--\s*(?:openreview:idempotency|autocode:)[^>]*-->")


@dataclass(frozen=True)
class DiscussionActivity:
    kind: str
    activity_id: int
    author_login: str
    author_type: str
    body: str
    updated_at: str
    created_at: str = ""
    state: str = ""
    url: str = ""


def is_ignored_actor(user: dict, ignored_logins: set[str] | None = None) -> bool:
    """忽略 bot/self 等不应触发新分析的参与者。"""
    ignored_logins = {login.lower() for login in (ignored_logins or set()) if login}
    login = str(user.get("login", "") or "").strip().lower()
    user_type = str(user.get("type", "") or "").strip().lower()
    if user_type == "bot" or "[bot]" in login:
        return True
    return bool(login) and login in ignored_logins


def has_openreview_idempotency_marker(body: str) -> bool:
    """判断内容里是否包含 openreview 或 autocode 写入的幂等标记。"""
    return bool(_IDEMPOTENCY_MARKER_RE.search(str(body or "")))


def is_ignored_activity(
    user: dict,
    body: str = "",
    ignored_logins: set[str] | None = None,
) -> bool:
    """忽略 bot 活动，以及带 openreview 标记的同账号自发讨论。"""
    if is_ignored_actor(user):
        return True
    ignored_logins = {login.lower() for login in (ignored_logins or set()) if login}
    login = str(user.get("login", "") or "").strip().lower()
    return bool(login and login in ignored_logins and has_openreview_idempotency_marker(body))


def collect_discussion_entries(
    *,
    issue_comments: list[dict] | None = None,
    reviews: list[dict] | None = None,
    review_comments: list[dict] | None = None,
) -> list[DiscussionActivity]:
    """把不同来源的 GitHub 讨论统一为可排序的条目。"""
    entries: list[DiscussionActivity] = []

    for comment in issue_comments or []:
        body = str(comment.get("body", "") or "").strip()
        if not body:
            continue
        entries.append(
            DiscussionActivity(
                kind="issue_comment",
                activity_id=int(comment.get("id", 0) or 0),
                author_login=str(comment.get("user", {}).get("login", "") or ""),
                author_type=str(comment.get("user", {}).get("type", "") or ""),
                body=body,
                updated_at=str(comment.get("updated_at", "") or comment.get("created_at", "") or ""),
                created_at=str(comment.get("created_at", "") or ""),
                url=str(comment.get("html_url", "") or ""),
            )
        )

    for review in reviews or []:
        body = str(review.get("body", "") or "").strip()
        if not body:
            continue
        entries.append(
            DiscussionActivity(
                kind="review",
                activity_id=int(review.get("id", 0) or 0),
                author_login=str(review.get("user", {}).get("login", "") or ""),
                author_type=str(review.get("user", {}).get("type", "") or ""),
                body=body,
                updated_at=str(review.get("submitted_at", "") or review.get("updated_at", "") or review.get("created_at", "") or ""),
                created_at=str(review.get("submitted_at", "") or review.get("created_at", "") or ""),
                state=str(review.get("state", "") or ""),
                url=str(review.get("html_url", "") or ""),
            )
        )

    for comment in review_comments or []:
        body = str(comment.get("body", "") or "").strip()
        if not body:
            continue
        entries.append(
            DiscussionActivity(
                kind="review_comment",
                activity_id=int(comment.get("id", 0) or 0),
                author_login=str(comment.get("user", {}).get("login", "") or ""),
                author_type=str(comment.get("user", {}).get("type", "") or ""),
                body=body,
                updated_at=str(comment.get("updated_at", "") or comment.get("created_at", "") or ""),
                created_at=str(comment.get("created_at", "") or ""),
                url=str(comment.get("html_url", "") or ""),
            )
        )

    return sorted(entries, key=_activity_sort_key)


def pick_latest_external_activity(
    entries: list[DiscussionActivity],
    *,
    ignored_logins: set[str] | None = None,
) -> DiscussionActivity | None:
    """从讨论中挑出最新一条外部活动。"""
    for entry in reversed(entries):
        if is_ignored_activity(
            {"login": entry.author_login, "type": entry.author_type},
            body=entry.body,
            ignored_logins=ignored_logins,
        ):
            continue
        return entry
    return None


def format_discussion_context(entries: list[DiscussionActivity], *, limit: int = 8) -> str:
    """把讨论条目格式化为 prompt 可读文本。"""
    selected = entries[-limit:] if limit > 0 else entries
    lines: list[str] = []
    for entry in selected:
        state_suffix = f", state={entry.state}" if entry.state else ""
        lines.append(
            f"- [{entry.kind}] @{entry.author_login} ({entry.updated_at or entry.created_at}{state_suffix})"
        )
        lines.append(f"  {truncate_text(entry.body, 320)}")
    return "\n".join(lines).strip()


def build_issue_search_query(
    repo_full_name: str,
    title: str,
    body: str = "",
    *,
    max_terms: int = 6,
) -> str:
    """为仓库内相似 Issue 搜索构造查询语句。"""
    terms = _extract_search_terms(title, body, max_terms=max_terms)
    query_parts = [f"repo:{repo_full_name}", "is:issue"]
    if terms:
        query_parts.extend(terms)
    elif title.strip():
        query_parts.append(f"\"{title.strip()[:60]}\"")
    return " ".join(query_parts)


def build_pr_search_query(
    repo_full_name: str,
    title: str,
    body: str = "",
    *,
    max_terms: int = 6,
) -> str:
    """为仓库内相关 PR 搜索构造查询语句。"""
    terms = _extract_search_terms(title, body, max_terms=max_terms)
    query_parts = [f"repo:{repo_full_name}", "is:pr"]
    if terms:
        query_parts.extend(terms)
    elif title.strip():
        query_parts.append(f"\"{title.strip()[:60]}\"")
    return " ".join(query_parts)


def build_issue_reference_pr_query(repo_full_name: str, issue_number: int) -> str:
    """搜索可能直接引用当前 Issue 的 PR。"""
    if issue_number <= 0:
        return ""
    return f'repo:{repo_full_name} is:pr "#{issue_number}"'


def format_related_issues_context(
    issues: list[dict],
    *,
    current_issue_number: int = 0,
    limit: int = 6,
) -> str:
    """把相似 Issue 搜索结果整理成 prompt 上下文。"""
    lines: list[str] = []
    kept = 0
    for issue in issues:
        number = int(issue.get("number", 0) or 0)
        if number <= 0 or number == current_issue_number:
            continue
        title = str(issue.get("title", "") or "").strip()
        if not title:
            continue
        state = str(issue.get("state", "") or "unknown")
        labels = issue.get("labels", []) or []
        label_names = ", ".join(
            str(label.get("name", "") or "").strip()
            for label in labels
            if str(label.get("name", "") or "").strip()
        )
        body = str(issue.get("body", "") or "").strip()
        updated_at = str(issue.get("updated_at", "") or "")
        lines.append(f"- #{number} [{state}] {title} (updated_at={updated_at or 'unknown'})")
        if label_names:
            lines.append(f"  labels: {label_names}")
        if body:
            lines.append(f"  摘要: {truncate_text(body, 220)}")
        kept += 1
        if kept >= limit:
            break
    return "\n".join(lines).strip()


def detect_duplicate_issues(
    current_title: str,
    current_body: str,
    issues: list[dict],
    *,
    current_issue_number: int = 0,
    limit: int = 3,
) -> list[dict]:
    """从相似 issue 结果里挑出高置信的重复候选。"""
    current_title_terms = set(_extract_search_terms(current_title, "", max_terms=8))
    current_terms = set(_extract_search_terms(current_title, current_body, max_terms=12))
    current_title_norm = _normalize_text_for_match(current_title)
    candidates: list[dict] = []

    for issue in issues:
        number = int(issue.get("number", 0) or 0)
        if number <= 0 or number == current_issue_number:
            continue

        title = str(issue.get("title", "") or "").strip()
        if not title:
            continue
        body = str(issue.get("body", "") or "").strip()
        candidate_title_terms = set(_extract_search_terms(title, "", max_terms=8))
        candidate_terms = set(_extract_search_terms(title, body, max_terms=12))
        title_overlap = sorted(current_title_terms & candidate_title_terms)
        all_overlap = sorted(current_terms & candidate_terms)
        distinctive_title_overlap = [term for term in title_overlap if _is_distinctive_duplicate_term(term)]
        title_ratio = _overlap_ratio(current_title_terms, candidate_title_terms)
        title_exact = bool(current_title_norm) and current_title_norm == _normalize_text_for_match(title)

        is_high_confidence = (
            title_exact
            or (len(title_overlap) >= 3 and title_ratio >= 0.60 and len(distinctive_title_overlap) >= 2)
            or (len(title_overlap) >= 2 and title_ratio >= 0.75 and len(distinctive_title_overlap) >= 1)
            or (
                len(title_overlap) >= 2
                and title_ratio >= 0.50
                and len(distinctive_title_overlap) >= 2
                and len(all_overlap) >= 5
            )
        )
        if not is_high_confidence:
            continue

        enriched = dict(issue)
        enriched["duplicate_terms"] = tuple((distinctive_title_overlap or title_overlap or all_overlap)[:4])
        enriched["duplicate_score"] = _duplicate_confidence_score(
            title_exact=title_exact,
            title_overlap=title_overlap,
            all_overlap=all_overlap,
            state=str(issue.get("state", "") or ""),
        )
        candidates.append(enriched)

    candidates.sort(
        key=lambda item: (
            int(item.get("duplicate_score", 0) or 0),
            str(item.get("state", "") or "").lower() == "closed",
            _parse_dt(str(item.get("updated_at", "") or "")),
            int(item.get("number", 0) or 0),
        ),
        reverse=True,
    )
    return candidates[:limit]


def format_duplicate_issues_context(
    issues: list[dict],
    *,
    limit: int = 3,
) -> str:
    """把高置信重复 issue 候选格式化为 prompt 上下文。"""
    lines: list[str] = []
    kept = 0
    for issue in issues:
        number = int(issue.get("number", 0) or 0)
        if number <= 0:
            continue
        title = str(issue.get("title", "") or "").strip()
        if not title:
            continue
        state = str(issue.get("state", "") or "unknown")
        updated_at = str(issue.get("updated_at", "") or "")
        lines.append(f"- #{number} [{state}] {title} (updated_at={updated_at or 'unknown'})")
        duplicate_terms = issue.get("duplicate_terms", ()) or ()
        if duplicate_terms:
            lines.append(f"  重合线索: {', '.join(str(term) for term in duplicate_terms)}")
        body = str(issue.get("body", "") or "").strip()
        if body:
            lines.append(f"  摘要: {truncate_text(body, 180)}")
        kept += 1
        if kept >= limit:
            break
    return "\n".join(lines).strip()


def format_related_prs_context(
    prs: list[dict],
    *,
    limit: int = 6,
) -> str:
    """把相关 PR 搜索结果整理成 prompt 上下文。"""
    lines: list[str] = []
    kept = 0
    for pr in prs:
        number = int(pr.get("number", 0) or 0)
        if number <= 0:
            continue
        title = str(pr.get("title", "") or "").strip()
        if not title:
            continue
        state = _summarize_pr_state(pr)
        updated_at = str(pr.get("updated_at", "") or "")
        mergeable_state = str(pr.get("mergeable_state", "") or "").strip()
        lines.append(f"- PR #{number} [{state}] {title} (updated_at={updated_at or 'unknown'})")
        if mergeable_state:
            lines.append(f"  mergeable_state: {mergeable_state}")
        merged_sha = _extract_short_merged_sha(pr)
        if merged_sha:
            lines.append(f"  merged_sha: {merged_sha}")
        body = str(pr.get("body", "") or "").strip()
        if body:
            lines.append(f"  摘要: {truncate_text(body, 220)}")
        kept += 1
        if kept >= limit:
            break
    return "\n".join(lines).strip()


def format_resolution_signals_context(
    prs: list[dict],
    *,
    limit: int = 4,
) -> str:
    """提炼“是否可能已修复 / 已有在途 PR”的线索。"""
    lines: list[str] = []
    kept = 0
    for pr in prs:
        number = int(pr.get("number", 0) or 0)
        if number <= 0:
            continue
        title = str(pr.get("title", "") or "").strip()
        if not title:
            continue
        merged_at = str(pr.get("merged_at", "") or pr.get("pull_request", {}).get("merged_at", "") or "")
        state = str(pr.get("state", "") or "").strip().lower()
        merged_sha = _extract_short_merged_sha(pr)
        commit_hint = f"，合入后 SHA: {merged_sha}" if merged_sha else ""
        if merged_at:
            lines.append(f"- PR #{number} 已合并，可能已经包含修复：{title}{commit_hint}")
        elif state == "open":
            lines.append(f"- PR #{number} 仍在进行中，可能正在处理这个问题：{title}")
        else:
            lines.append(f"- PR #{number} 已关闭但未见合并记录，是否真正修复还需要结合改动确认：{title}")
        kept += 1
        if kept >= limit:
            break
    return "\n".join(lines).strip()


def truncate_text(text: str, max_chars: int) -> str:
    normalized = re.sub(r"\s+", " ", text).strip()
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _extract_search_terms(title: str, body: str, *, max_terms: int) -> list[str]:
    text = "\n".join(part for part in (title, body) if part.strip())
    cjk_terms = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    latin_terms = re.findall(r"[A-Za-z0-9_./-]{3,}", text)
    terms: list[str] = []
    seen: set[str] = set()

    for raw in cjk_terms + latin_terms:
        term = raw.strip().replace("\"", "")
        if not term:
            continue
        lower = term.lower()
        if lower in seen:
            continue
        if lower in _EN_STOPWORDS or term in _ZH_STOPWORDS:
            continue
        if term.isdigit():
            continue
        seen.add(lower)
        terms.append(f"\"{term}\"" if re.search(r"[\u4e00-\u9fff]", term) else term)
        if len(terms) >= max_terms:
            break
    return terms


def _activity_sort_key(entry: DiscussionActivity) -> tuple[datetime, int, str]:
    return (_parse_dt(entry.updated_at or entry.created_at), entry.activity_id, entry.kind)


def _summarize_pr_state(pr: dict) -> str:
    merged_at = str(pr.get("merged_at", "") or pr.get("pull_request", {}).get("merged_at", "") or "")
    if merged_at:
        return "merged"

    state = str(pr.get("state", "") or "unknown").strip().lower()
    draft = bool(pr.get("draft"))
    if draft:
        return f"{state or 'open'} draft"
    return state or "unknown"


def _duplicate_confidence_score(
    *,
    title_exact: bool,
    title_overlap: list[str],
    all_overlap: list[str],
    state: str,
) -> int:
    score = 0
    if title_exact:
        score += 10
    score += len(title_overlap) * 3
    score += len(all_overlap)
    if state.strip().lower() == "closed":
        score += 1
    return score


def _extract_short_merged_sha(pr: dict) -> str:
    sha = str(pr.get("merge_commit_sha", "") or "").strip()
    if not sha:
        return ""
    return sha[:7]


def _is_distinctive_duplicate_term(term: str) -> bool:
    normalized = term.strip().lower()
    if not normalized:
        return False
    if normalized in {
        "config",
        "configuration",
        "setting",
        "settings",
        "path",
        "readme",
        "docs",
        "usage",
        "option",
        "question",
        "issue",
        "error",
        "problem",
        "配置",
        "文档",
        "说明",
        "路径",
        "使用",
        "问题",
    }:
        return False
    if re.search(r"[_./-]", normalized):
        return True
    if re.search(r"\d", normalized):
        return True
    if re.search(r"[\u4e00-\u9fff]", normalized):
        return len(normalized) >= 4
    return len(normalized) >= 6


def _normalize_text_for_match(text: str) -> str:
    lowered = text.strip().lower()
    if not lowered:
        return ""
    return re.sub(r"[\W_]+", "", lowered)


def _overlap_ratio(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(len(left), len(right), 1)


def _parse_dt(value: str) -> datetime:
    if not value:
        return datetime.min
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min
