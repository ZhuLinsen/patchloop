"""PR title normalization helpers for AutoCode-created pull requests."""
from __future__ import annotations

import re

TASK_TYPE_PREFIX_MAP: dict[str, str] = {
    "bug_fix": "fix",
    "small_feature": "feat",
    "high_risk_feature": "feat",
    "todo_refactor": "refactor",
}

PR_TITLE_SUBJECT_MAX_LEN = 36

_ISSUE_TITLE_PREFIX_RE = re.compile(r"^\[(bug|feature|docs|question)\]\s*", re.IGNORECASE)
_CONVENTIONAL_TITLE_PREFIX_RE = re.compile(
    r"^(?:fix|feat|feature|bug|docs?|chore|refactor|test|improve|improvement)\s*[:：]\s*",
    re.IGNORECASE,
)
_INFORMAL_TITLE_RE = re.compile(
    r"[？?吗呢吧嘛]\s*$|[？?吗]|^(?:为什么|为啥|怎么|如何|能否|能不能|能新增|能增加|能加|可以|可否|请问|是否|有没有|是不是)",
)
_PLACEHOLDER_TITLE_RE = re.compile(r"^[\s(#\d)]*$")
_LOW_SIGNAL_TITLE_PREFIX_RE = re.compile(
    r"^(?:今天|昨天|刚刚|现在|目前|突然|又|还是|这边|这里|那边|这个|这块|好像|似乎|建议)"
)
_LOW_SIGNAL_TITLE_SUFFIX_RE = re.compile(
    r"(?:跑不动(?:了)?|起不来(?:了)?|用不了(?:了)?|不工作(?:了)?|有问题(?:了)?|出问题(?:了)?|挂了|坏了|不行了|崩了|炸了|略作调整|略做调整|调整一下)$"
)
_PR_TITLE_ISSUE_RE = re.compile(r"\(#(\d+)\)\s*$")
_API_PATH_RE = re.compile(r"(/api/[^\s，,。；;]+)")


def normalize_pr_title_issue_text(title: str) -> str:
    normalized = _ISSUE_TITLE_PREFIX_RE.sub("", str(title or "").strip()).strip()
    normalized = _strip_conventional_prefixes(normalized)
    if len(normalized) < 5:
        return ""
    return normalized


def is_usable_pr_title(text: str) -> bool:
    if not text or _PLACEHOLDER_TITLE_RE.match(text):
        return False
    if _INFORMAL_TITLE_RE.search(text):
        return False
    compact = re.sub(r"\s+", "", text)
    if _LOW_SIGNAL_TITLE_PREFIX_RE.match(compact):
        return False
    if _LOW_SIGNAL_TITLE_SUFFIX_RE.search(compact):
        return False
    return True


def compact_pr_title_subject(text: str, task_type: str) -> str:
    subject = _clean_title_subject(text)
    if not subject:
        return ""

    api_subject = _compact_api_result_subject(subject)
    if api_subject:
        return api_subject

    packaged_data_subject = _compact_packaged_data_subject(subject)
    if packaged_data_subject:
        return packaged_data_subject

    related_sector_layout_subject = _compact_related_sector_layout_subject(subject)
    if related_sector_layout_subject:
        return related_sector_layout_subject

    subject = _drop_trailing_explanation(subject)
    subject = _strip_conventional_prefixes(subject)
    subject = _strip_task_action_prefix(subject, task_type)
    subject = _normalize_title_spacing(subject)
    return subject.strip("，。、,. ")


def truncate_pr_title_subject(text: str, max_len: int = PR_TITLE_SUBJECT_MAX_LEN) -> str:
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    for sep in ("，", "、", "；", ",", " "):
        head, _, _ = cut.rpartition(sep)
        if head and len(head) >= max_len // 2:
            cut = head
            break
    return cut.rstrip("，。、,. ") + "…"


def build_pr_title(
    task_type: str,
    issue_title: str,
    issue_number: int,
    *,
    plan_goal: str = "",
    prefix_map: dict[str, str] | None = None,
) -> str:
    prefix = (prefix_map or TASK_TYPE_PREFIX_MAP).get(task_type, "chore")
    issue_subject = normalize_pr_title_issue_text(issue_title)
    goal_subject = _first_sentence(plan_goal)

    subject = issue_subject
    if not is_usable_pr_title(subject):
        subject = goal_subject
    subject = compact_pr_title_subject(subject, task_type)

    if not is_usable_pr_title(subject) and goal_subject:
        subject = compact_pr_title_subject(goal_subject, task_type)
    if not subject:
        subject = f"resolve issue #{issue_number}"

    subject = truncate_pr_title_subject(subject)
    return f"{prefix}: {subject} (#{issue_number})"


def should_keep_existing_pr_title(
    existing_title: str,
    task_type: str,
    issue_number: int,
    *,
    prefix_map: dict[str, str] | None = None,
) -> bool:
    title = str(existing_title or "").strip()
    if not title:
        return False

    expected_prefix = f"{(prefix_map or TASK_TYPE_PREFIX_MAP).get(task_type, 'chore')}: "
    if not title.lower().startswith(expected_prefix):
        return False

    match = _PR_TITLE_ISSUE_RE.search(title)
    if not match or int(match.group(1)) != issue_number:
        return False
    if match.start() == 0 or not title[match.start() - 1].isspace():
        return False

    subject = title[len(expected_prefix) : match.start()].strip()
    cleaned = normalize_pr_title_issue_text(subject)
    if len(cleaned) < 5 or not is_usable_pr_title(cleaned):
        return False
    if "…" in subject or len(subject) > PR_TITLE_SUBJECT_MAX_LEN:
        return False
    if _strip_conventional_prefixes(subject) != subject:
        return False
    return True


def _first_sentence(text: str) -> str:
    return str(text or "").split("。")[0].split("\n")[0].strip()


def _clean_title_subject(text: str) -> str:
    subject = str(text or "").strip()
    subject = re.sub(r"`([^`]+)`", r"\1", subject)
    subject = re.sub(r"\bv\d+(?:\.\d+){1,3}\b", "", subject)
    subject = re.sub(r"返回\s*null", "返回 null", subject, flags=re.IGNORECASE)
    subject = subject.replace("永远都会", "").replace("永远都", "")
    subject = re.sub(r"\s*/\s*目录", " 目录", subject)
    return _normalize_title_spacing(subject)


def _normalize_title_spacing(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    normalized = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", normalized)
    normalized = re.sub(r"\s+([，。；：、])", r"\1", normalized)
    return normalized.strip()


def _strip_conventional_prefixes(text: str) -> str:
    subject = str(text or "").strip()
    while True:
        updated = _CONVENTIONAL_TITLE_PREFIX_RE.sub("", subject).strip()
        if updated == subject:
            return subject
        subject = updated


def _strip_task_action_prefix(text: str, task_type: str) -> str:
    prefixes_by_type = {
        "bug_fix": ("修复", "解决", "处理", "避免", "防止", "恢复"),
        "small_feature": ("新增", "增加", "支持", "接入", "实现", "提升", "调整", "优化"),
        "high_risk_feature": ("新增", "增加", "支持", "接入", "实现", "提升", "调整", "优化"),
        "todo_refactor": ("补齐", "补充", "调整", "统一", "清理", "移除", "删除"),
    }
    prefixes = prefixes_by_type.get(task_type, ())
    for prefix in prefixes:
        if text.startswith(prefix) and len(text) - len(prefix) >= 5:
            return text[len(prefix) :].strip()
    return text


def _drop_trailing_explanation(text: str) -> str:
    for marker in ("，这让", ", 这让", "，导致", ", 导致", "，影响", ", 影响"):
        head, sep, _ = text.partition(marker)
        if sep and len(head.strip()) >= 8:
            return head.strip()
    return text


def _compact_api_result_subject(text: str) -> str:
    path_match = _API_PATH_RE.search(text)
    if not path_match:
        return ""
    normalized = text.lower()
    if "result" not in normalized:
        return ""
    if not re.search(r"返回\s*null|为空|空返回", normalized):
        return ""

    label = _api_path_label(path_match.group(1))
    suffix = "result 返回 null" if re.search(r"返回\s*null", normalized) else "result 为空"
    return f"{label} {suffix}".strip()


def _api_path_label(path: str) -> str:
    parts = [part for part in path.strip("/").split("/") if part]
    if parts and parts[0] == "api":
        parts = parts[1:]
    if parts and re.fullmatch(r"v\d+", parts[0], re.IGNORECASE):
        parts = parts[1:]
    parts = [part.strip("{}") for part in parts if not (part.startswith("{") and part.endswith("}"))]
    if len(parts) >= 2:
        parts = parts[-2:]
    return " ".join(parts) or path.strip("/")


def _compact_packaged_data_subject(text: str) -> str:
    if "数据文件" not in text and "随包" not in text and "PyInstaller" not in text:
        return ""
    match = re.search(r"(?:将|把)([^，。；;]+?)(?:作为数据文件|随包|打包|包含|纳入)", text)
    if not match:
        return ""
    item = match.group(1).strip()
    if not item:
        return ""
    return _normalize_title_spacing(f"打包 {item} 数据文件")


def _compact_related_sector_layout_subject(text: str) -> str:
    if "关联板块" not in text:
        return ""
    if not any(marker in text for marker in ("布局", "展示位置", "操作建议", "tag", "标签")):
        return ""
    return "个股分析页关联板块布局"
