"""Shared helpers for identifying owner-issued implement commands on issues."""
from __future__ import annotations

import re

_BOT_BODY_MARKERS = (
    "<!-- openreview:idempotency:issue:",
    "<!-- autocode:idempotency:issue:",
    "<!-- openreview:",
    "<!-- autocode:",
)
_BOT_FOOTER_RE = re.compile(r"此回复由\s+.*?bot\s+自动生成", re.IGNORECASE)
_IMPLEMENT_NEGATION_RE = re.compile(
    r"(?:未实现|已实现|难实现|不好实现|实现不了|无法实现|不能实现|不要实现|不用实现|无需实现|不必实现|"
    r"暂不实现|暂时不实现|先别实现|别实现|不实现)",
    re.IGNORECASE,
)
_IMPLEMENT_TAIL_RE = r"(?:[，,；;。.!！]\s*.+)?"
_IMPLEMENT_APPROVAL_PATTERNS = (
    re.compile(
        rf"^(?:请|麻烦|辛苦)?(?:现在|直接|尽快|继续)?(?:开始|继续|进行|着手|安排|推进)?"
        rf"(?:实现|implement|落地)(?:一下|下)?(?:了|吧)?{_IMPLEMENT_TAIL_RE}$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^(?:可以|可)(?:现在|直接|开始|继续)?(?:实现|implement|落地)(?:了|吧)?{_IMPLEMENT_TAIL_RE}$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^(?:就)?按(?:这个|此|该|上面|上述|前述)?(?:方案|思路|方向|计划)?(?:来|去)?"
        rf"(?:实现|做|改|落地)(?:吧|一下|下)?{_IMPLEMENT_TAIL_RE}$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^(?:按|照)(?:这个|此|该|上面|上述|前述)?(?:方案|思路|方向|计划)?(?:来|去)?"
        rf"(?:实现|做|改|落地)(?:吧|一下|下)?{_IMPLEMENT_TAIL_RE}$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^(?:可以|可)(?:按|照)(?:这个|此|该|上面|上述|前述)?(?:方案|思路|方向|计划)?(?:来|去)?"
        rf"(?:实现|做|改|落地)(?:了|吧)?{_IMPLEMENT_TAIL_RE}$",
        re.IGNORECASE,
    ),
)
_DIRECT_IMPLEMENT_PATTERNS = (
    re.compile(r"(?:你|bot|autocode)?\s*(?:直接|继续|重新|再|帮我|帮忙|麻烦|辛苦).{0,12}(?:修|改|做|处理|实现|落地)", re.IGNORECASE),
    re.compile(r"(?:按|照).{0,20}(?:方案|思路|方向|计划|你说的|上面|上述|前述).{0,12}(?:做|改|修|处理|实现|落地)", re.IGNORECASE),
    re.compile(r"(?:开|提)\s*(?:个)?\s*PR", re.IGNORECASE),
    re.compile(r"(?:go ahead|please fix|fix it|ship it|open a pr|create a pr|send a pr)", re.IGNORECASE),
)
_NOT_RESOLVED_PATTERNS = (
    re.compile(r"(?:还是|仍然|依然|依旧).{0,12}(?:不行|失败|报错|没好|没有解决)", re.IGNORECASE),
    re.compile(r"(?:没|没有|未).{0,8}(?:解决|修好|生效)", re.IGNORECASE),
    re.compile(r"(?:不是|不对).{0,12}(?:这个|这个问题|这个方向|根因|原因)", re.IGNORECASE),
    re.compile(r"(?:继续|重新|再).{0,8}(?:修|改|试|处理|跑)", re.IGNORECASE),
    re.compile(r"(?:not fixed|still failing|still broken|does not work|try again)", re.IGNORECASE),
)
_MARKDOWN_NOISE_RE = re.compile(r"[`*_~]")
_WHITESPACE_RE = re.compile(r"\s+")
_MAX_SEMANTIC_COMMENT_LENGTH = 500


def is_bot_generated_issue_comment(comment_or_body: dict | str) -> bool:
    body = comment_or_body if isinstance(comment_or_body, str) else str(comment_or_body.get("body", "") or "")
    lowered = body.lower()
    return any(marker in lowered for marker in _BOT_BODY_MARKERS) or bool(_BOT_FOOTER_RE.search(body))


def normalize_issue_comment_text(body: str) -> str:
    text = str(body or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = _MARKDOWN_NOISE_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def issue_comment_requests_implementation(body: str) -> bool:
    raw = str(body or "").strip()
    if not raw or is_bot_generated_issue_comment(raw):
        return False

    text = normalize_issue_comment_text(raw)
    if not text or len(text) > _MAX_SEMANTIC_COMMENT_LENGTH:
        return False
    if _IMPLEMENT_NEGATION_RE.search(text):
        return False
    nonempty_lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(nonempty_lines) == 1 and any(pattern.fullmatch(text) for pattern in _IMPLEMENT_APPROVAL_PATTERNS):
        return True
    if any(pattern.search(text) for pattern in _DIRECT_IMPLEMENT_PATTERNS):
        return True
    if any(pattern.search(text) for pattern in _NOT_RESOLVED_PATTERNS):
        return True
    return False


def is_repo_owner_human_issue_comment(comment: dict, repo_owner: str) -> bool:
    login = str((comment.get("user", {}) or {}).get("login", "") or "").strip().lower()
    owner = str(repo_owner or "").strip().lower()
    user_type = str((comment.get("user", {}) or {}).get("type", "") or "").strip().lower()
    if not login or not owner or login != owner:
        return False
    if user_type == "bot":
        return False
    return not is_bot_generated_issue_comment(comment)
