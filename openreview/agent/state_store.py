"""
本地状态存储

职责:
1. 记录已分析过的 Issue 和 PR
2. 为 polling 模式持久化“只处理新对象”的时间基线
3. 通过原子写入避免状态文件损坏
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

_MAX_STORED_KEYS = 32
logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    """返回 UTC ISO8601 时间字符串。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_activity_fingerprint(
    activity_type: str = "",
    activity_id: int | str = "",
    body: str = "",
    updated_at: str = "",
) -> str:
    """为一次评论/评审活动构造稳定指纹。"""
    normalized_type = activity_type.strip().lower()
    normalized_id = str(activity_id).strip()
    normalized_body = body.strip()
    normalized_updated_at = updated_at.strip()
    if not any((normalized_type, normalized_id, normalized_body, normalized_updated_at)):
        return ""
    digest = hashlib.sha256(
        f"{normalized_type}\n{normalized_id}\n{normalized_updated_at}\n\n{normalized_body}".encode("utf-8")
    ).hexdigest()[:16]
    prefix = normalized_type or "activity"
    suffix = normalized_id or normalized_updated_at or "unknown"
    return f"{prefix}:{suffix}:{digest}"


def build_issue_processing_key(title: str, body: str, activity_fingerprint: str = "") -> str:
    """为一次 Issue 分析构造稳定 key，覆盖正文和最新讨论活动。"""
    normalized_title = title.strip()
    normalized_body = body.strip()
    normalized_activity = activity_fingerprint.strip()
    digest = hashlib.sha256(
        f"{normalized_title}\n\n{normalized_body}\n\n{normalized_activity}".encode("utf-8")
    ).hexdigest()[:16]
    return digest


def build_pr_review_key(
    head_sha: str,
    title: str,
    body: str,
    base_ref: str = "",
    discussion_fingerprint: str = "",
) -> str:
    """为一次 PR review 构造稳定 key，覆盖代码、描述和目标分支变更。"""
    digest = build_pr_review_metadata(title, body, base_ref, discussion_fingerprint)["digest"]
    return f"{head_sha}:{digest}" if head_sha else digest


def build_pr_review_metadata(
    title: str,
    body: str,
    base_ref: str = "",
    discussion_fingerprint: str = "",
) -> dict[str, str]:
    """构造 PR review key 的可观测元数据指纹，不保存完整标题或正文。"""
    normalized_title = title.strip()
    normalized_body = body.strip()
    normalized_base_ref = base_ref.strip()
    normalized_discussion = discussion_fingerprint.strip()
    digest = hashlib.sha256(
        f"{normalized_base_ref}\n{normalized_title}\n\n{normalized_body}\n\n{normalized_discussion}".encode("utf-8")
    ).hexdigest()[:16]
    return {
        "base_ref": normalized_base_ref,
        "title_hash": hashlib.sha256(normalized_title.encode("utf-8")).hexdigest()[:12],
        "body_hash": hashlib.sha256(normalized_body.encode("utf-8")).hexdigest()[:12],
        "discussion_hash": hashlib.sha256(normalized_discussion.encode("utf-8")).hexdigest()[:12]
        if normalized_discussion
        else "",
        "digest": digest,
    }


def _parse_iso_datetime(value: str) -> datetime | None:
    """解析状态文件中的 ISO 时间。"""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _rewind_iso_datetime(value: str, seconds: int) -> str:
    """将 ISO 时间向前回退固定秒数，用于 polling 游标重叠窗口。"""
    dt = _parse_iso_datetime(value)
    if dt is None:
        return value
    shifted = (dt - timedelta(seconds=max(seconds, 0))).replace(microsecond=0)
    return shifted.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _max_iso_datetime_value(current: str, candidate: str) -> str:
    """返回两个 ISO 时间里更晚的那个。"""
    candidate_dt = _parse_iso_datetime(candidate)
    if candidate_dt is None:
        return current
    current_dt = _parse_iso_datetime(current)
    if current_dt is None or candidate_dt > current_dt:
        return candidate
    return current


def _processed_pr_updated_at(pr: dict[str, Any]) -> str:
    """返回一次已处理 PR 对应的更新时间，用于兼容旧状态。"""
    return str(pr.get("last_processed_updated_at") or pr.get("updated_at") or "")


def _legacy_issue_cursor_seed(issue: dict[str, Any]) -> str:
    """从旧 issue 状态中提取可复用的 cursor 时间。"""
    return str(issue.get("updated_at") or issue.get("created_at") or "")


def _legacy_pr_cursor_seed(pr: dict[str, Any]) -> str:
    """从旧 PR 状态中提取可复用的对象 cursor 时间。"""
    return str(pr.get("last_processed_updated_at") or pr.get("updated_at") or pr.get("created_at") or "")


def _seed_with_latest_iso(*candidates: str) -> str:
    """从多个 ISO 时间候选里选出最晚的一个。"""
    latest = ""
    for candidate in candidates:
        latest = _max_iso_datetime_value(latest, candidate)
    return latest


def _latest_pr_activity_at(pr: dict[str, Any]) -> datetime | None:
    candidates = [
        str(pr.get("last_seen_at", "") or ""),
        str(pr.get("last_processed_at", "") or ""),
        str(pr.get("last_blocked_at", "") or ""),
    ]
    parsed = [_parse_iso_datetime(value) for value in candidates]
    present = [dt for dt in parsed if dt is not None]
    return max(present) if present else None


def _latest_pr_source_activity_at(pr: dict[str, Any]) -> datetime | None:
    candidates = [
        str(pr.get("updated_at", "") or ""),
        str(pr.get("created_at", "") or ""),
        str(pr.get("last_processed_updated_at", "") or ""),
        str(pr.get("last_review_submission_cursor_at", "") or ""),
    ]
    parsed = [_parse_iso_datetime(value) for value in candidates]
    present = [dt for dt in parsed if dt is not None]
    return max(present) if present else None


def _latest_iso_from_fields(item: dict[str, Any], *fields: str) -> str:
    latest = ""
    for field in fields:
        latest = _max_iso_datetime_value(latest, str(item.get(field, "") or ""))
    return latest


def _issue_state_marker(issue: dict[str, Any]) -> str:
    return _latest_iso_from_fields(issue, "processed_at", "updated_at", "created_at")


def _pr_state_marker(pr: dict[str, Any]) -> str:
    return _latest_iso_from_fields(
        pr,
        "last_processed_at",
        "last_processed_updated_at",
        "updated_at",
        "last_seen_at",
        "last_review_scan_at",
        "last_review_submission_cursor_at",
        "last_blocked_at",
        "created_at",
    )


def _mapping_richness(item: dict[str, Any]) -> int:
    score = 0
    for value in item.values():
        if isinstance(value, list):
            score += len([entry for entry in value if str(entry or "").strip()])
        elif value not in ("", None, [], {}, False):
            score += 1
    return score


def _prefer_newer_record(
    primary: dict[str, Any],
    runtime: dict[str, Any],
    *,
    marker_getter,
) -> dict[str, Any]:
    primary_marker = marker_getter(primary)
    runtime_marker = marker_getter(runtime)
    primary_dt = _parse_iso_datetime(primary_marker)
    runtime_dt = _parse_iso_datetime(runtime_marker)

    if primary_dt is not None and runtime_dt is not None:
        if runtime_dt > primary_dt:
            return copy.deepcopy(runtime)
        if primary_dt > runtime_dt:
            return copy.deepcopy(primary)
    elif runtime_dt is not None and primary_dt is None:
        return copy.deepcopy(runtime)
    elif primary_dt is not None and runtime_dt is None:
        return copy.deepcopy(primary)

    if _mapping_richness(runtime) >= _mapping_richness(primary):
        return copy.deepcopy(runtime)
    return copy.deepcopy(primary)


class LocalStateStore:
    """OpenReview 的本地 JSON 状态仓库。"""

    def __init__(self, path: str):
        self.path = Path(path).resolve()
        self.runtime_path = self.path.with_name(f"{self.path.stem}.runtime{self.path.suffix}")
        self._lock = threading.Lock()
        self._inflight_pr_reviews: set[int] = set()
        self._inflight_issues: set[int] = set()
        self._data = self._load()

    def ensure_polling_baseline(self) -> bool:
        """
        初始化 polling 基线。

        首次启用 polling 时，将当前时刻记为 issue/pr 的基线时间，
        只处理之后新创建的对象。
        """
        with self._lock:
            polling = self._data.setdefault("polling", {})
            now = utc_now_iso()
            changed = False
            if not polling.get("bootstrapped_at"):
                polling["bootstrapped_at"] = now
                changed = True
            if not polling.get("issue_created_after"):
                polling["issue_created_after"] = now
                changed = True
            if not polling.get("pr_created_after"):
                polling["pr_created_after"] = now
                changed = True
            if not polling.get("issue_updated_after"):
                polling["issue_updated_after"] = polling.get("issue_created_after") or now
                changed = True
            if not polling.get("pr_updated_after"):
                polling["pr_updated_after"] = polling.get("pr_created_after") or now
                changed = True
            if not polling.get("issue_comment_updated_after"):
                polling["issue_comment_updated_after"] = (
                    polling.get("issue_updated_after")
                    or polling.get("issue_created_after")
                    or polling.get("bootstrapped_at")
                    or now
                )
                changed = True
            if not polling.get("review_comment_updated_after"):
                polling["review_comment_updated_after"] = (
                    polling.get("pr_updated_after")
                    or polling.get("pr_created_after")
                    or polling.get("bootstrapped_at")
                    or now
                )
                changed = True
            if changed:
                self._save_unlocked()
            return changed

    def issue_baseline(self) -> str:
        with self._lock:
            return self._data["polling"].get("issue_created_after", "")

    def pr_baseline(self) -> str:
        with self._lock:
            return self._data["polling"].get("pr_created_after", "")

    def issue_cursor(self) -> str:
        with self._lock:
            polling = self._data["polling"]
            return polling.get("issue_updated_after", "") or polling.get("issue_created_after", "")

    def pr_cursor(self) -> str:
        with self._lock:
            polling = self._data["polling"]
            return polling.get("pr_updated_after", "") or polling.get("pr_created_after", "")

    def issue_comment_cursor(self) -> str:
        with self._lock:
            return self._data["polling"].get("issue_comment_updated_after", "")

    def review_comment_cursor(self) -> str:
        with self._lock:
            return self._data["polling"].get("review_comment_updated_after", "")

    def issue_min_number(self) -> int:
        with self._lock:
            return int(self._data["thresholds"].get("issue_min_number", 0) or 0)

    def pr_min_number(self) -> int:
        with self._lock:
            return int(self._data["thresholds"].get("pr_min_number", 0) or 0)

    def should_skip_issue_number(self, issue_number: int) -> bool:
        return issue_number <= self.issue_min_number()

    def should_skip_pr_number(self, pr_number: int) -> bool:
        return pr_number <= self.pr_min_number()

    def has_processed_issue(self, issue_number: int) -> bool:
        with self._lock:
            issue = self._data["issues"].get(str(issue_number), {})
            return bool(issue.get("processed_at"))

    def has_processed_pr(self, pr_number: int) -> bool:
        with self._lock:
            pr = self._data["pull_requests"].get(str(pr_number), {})
            return bool(pr.get("last_processed_at"))

    def has_tracked_issue(self, issue_number: int) -> bool:
        with self._lock:
            return str(issue_number) in self._data["issues"]

    def has_issue_processing_keys(self, issue_number: int) -> bool:
        with self._lock:
            issue = self._data["issues"].get(str(issue_number), {})
            return bool(self._processed_keys(issue, "processed_keys", "last_processed_key"))

    def issue_processed_at(self, issue_number: int) -> str:
        with self._lock:
            issue = self._data["issues"].get(str(issue_number), {})
            return str(issue.get("processed_at", "") or "")

    def issue_processed_updated_at(self, issue_number: int) -> str:
        with self._lock:
            issue = self._data["issues"].get(str(issue_number), {})
            return str(issue.get("updated_at", "") or "")

    def pr_processed_head_sha(self, pr_number: int) -> str:
        with self._lock:
            pr = self._data["pull_requests"].get(str(pr_number), {})
            return str(pr.get("last_processed_head_sha", "") or "")

    def pr_processed_updated_at(self, pr_number: int) -> str:
        with self._lock:
            pr = self._data["pull_requests"].get(str(pr_number), {})
            return _processed_pr_updated_at(pr)

    def pr_processed_at(self, pr_number: int) -> str:
        with self._lock:
            pr = self._data["pull_requests"].get(str(pr_number), {})
            return str(pr.get("last_processed_at", "") or "")

    def pr_processed_review_key(self, pr_number: int) -> str:
        with self._lock:
            pr = self._data["pull_requests"].get(str(pr_number), {})
            return str(pr.get("last_processed_review_key", "") or "")

    def pr_processed_review_metadata(self, pr_number: int) -> dict[str, str]:
        with self._lock:
            pr = self._data["pull_requests"].get(str(pr_number), {})
            metadata = pr.get("last_processed_review_metadata", {})
            if not isinstance(metadata, dict):
                return {}
            return {str(key): str(value or "") for key, value in metadata.items()}

    def has_pr_review_processing_keys(self, pr_number: int) -> bool:
        with self._lock:
            pr = self._data["pull_requests"].get(str(pr_number), {})
            raw_keys = pr.get("processed_review_keys", [])
            return isinstance(raw_keys, list) and any(str(raw or "").strip() for raw in raw_keys)

    def is_pr_same_version_processed(
        self,
        pr_number: int,
        *,
        head_sha: str = "",
        updated_at: str = "",
    ) -> bool:
        with self._lock:
            pr = self._data["pull_requests"].get(str(pr_number), {})
            if not pr.get("last_processed_at"):
                return False
            if bool(head_sha) and str(pr.get("last_processed_head_sha", "") or "") != head_sha:
                return False

            processed_updated_at = _processed_pr_updated_at(pr)
            if not processed_updated_at or not updated_at:
                return False

            processed_dt = _parse_iso_datetime(processed_updated_at)
            updated_dt = _parse_iso_datetime(updated_at)
            if processed_dt is not None and updated_dt is not None:
                return updated_dt <= processed_dt
            return processed_updated_at == updated_at

    def is_issue_processed(
        self,
        issue_number: int,
        processing_key: str,
        *,
        updated_at: str = "",
    ) -> bool:
        with self._lock:
            issue = self._data["issues"].get(str(issue_number), {})
            processed_keys = self._processed_keys(issue, "processed_keys", "last_processed_key")
            legacy_processed = (
                bool(issue.get("processed_at"))
                and bool(updated_at)
                and str(issue.get("updated_at", "")) == updated_at
            )
            if processing_key:
                if processing_key in processed_keys:
                    return True
                if not processed_keys:
                    return legacy_processed
                return False
            return legacy_processed

    def mark_issue_processed(
        self,
        issue_number: int,
        *,
        processing_key: str = "",
        extra_processing_keys: list[str] | None = None,
        source: str,
        created_at: str = "",
        updated_at: str = "",
    ):
        with self._lock:
            issue = self._data["issues"].setdefault(str(issue_number), {})
            processed_keys = self._processed_keys(issue, "processed_keys", "last_processed_key")
            processed_keys = self._append_processed_key(processed_keys, processing_key)
            for extra_key in extra_processing_keys or []:
                processed_keys = self._append_processed_key(processed_keys, extra_key)
            issue.update(
                {
                    "processed_at": utc_now_iso(),
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "source": source,
                    "last_processed_key": processing_key,
                    "processed_keys": processed_keys,
                }
            )
            self._save_unlocked()

    def is_pr_review_processed(
        self,
        pr_number: int,
        review_key: str,
        *,
        head_sha: str = "",
        updated_at: str = "",
    ) -> bool:
        with self._lock:
            pr = self._data["pull_requests"].get(str(pr_number), {})
            processed_keys = self._processed_keys(pr, "processed_review_keys", "last_processed_review_key")
            if review_key and review_key in processed_keys:
                return True
            last_review_key = str(pr.get("last_processed_review_key", "") or "")
            if review_key and last_review_key:
                return last_review_key == review_key
            if not review_key:
                return bool(head_sha) and pr.get("last_processed_head_sha") == head_sha
            return (
                bool(head_sha)
                and pr.get("last_processed_head_sha") == head_sha
                and bool(updated_at)
                and _processed_pr_updated_at(pr) == updated_at
            )

    def try_claim_pr_review(
        self,
        pr_number: int,
        review_key: str,
        *,
        head_sha: str = "",
        updated_at: str = "",
    ) -> str:
        """
        原子地尝试占用一次 PR review 的处理权。

        返回:
            claimed: 成功占用，可继续处理
            processed: 该相同版本（review_key）已成功评审过
            inflight: 当前已有协程在处理该 PR
        """
        with self._lock:
            pr = self._data["pull_requests"].get(str(pr_number), {})
            processed_keys = self._processed_keys(pr, "processed_review_keys", "last_processed_review_key")
            if review_key and review_key in processed_keys:
                return "processed"
            last_review_key = str(pr.get("last_processed_review_key", "") or "")
            if review_key and last_review_key == review_key:
                return "processed"
            if (
                not review_key
                and bool(head_sha)
                and pr.get("last_processed_head_sha") == head_sha
            ):
                return "processed"
            if (
                review_key
                and not last_review_key
                and bool(head_sha)
                and pr.get("last_processed_head_sha") == head_sha
                and bool(updated_at)
                and _processed_pr_updated_at(pr) == updated_at
            ):
                return "processed"

            if pr_number in self._inflight_pr_reviews:
                return "inflight"

            self._inflight_pr_reviews.add(pr_number)
            return "claimed"

    def release_pr_review(self, pr_number: int, review_key: str, *, head_sha: str = ""):
        """释放一次 PR review 的处理中占位。"""
        with self._lock:
            self._inflight_pr_reviews.discard(pr_number)

    def is_pr_head_processed(self, pr_number: int, head_sha: str) -> bool:
        """兼容旧逻辑，默认只按 head 去重。"""
        with self._lock:
            pr = self._data["pull_requests"].get(str(pr_number), {})
            return bool(head_sha) and pr.get("last_processed_head_sha") == head_sha

    def try_claim_pr_head(self, pr_number: int, head_sha: str) -> str:
        """兼容旧逻辑，默认只按 head 去重。"""
        with self._lock:
            pr = self._data["pull_requests"].get(str(pr_number), {})
            if bool(head_sha) and pr.get("last_processed_head_sha") == head_sha:
                return "processed"

            if pr_number in self._inflight_pr_reviews:
                return "inflight"

            self._inflight_pr_reviews.add(pr_number)
            return "claimed"

    def release_pr_head(self, pr_number: int, head_sha: str):
        """兼容旧逻辑，默认只按 head 去重。"""
        with self._lock:
            self._inflight_pr_reviews.discard(pr_number)

    def try_claim_issue(self, issue_number: int) -> str:
        """
        原子地尝试占用一个 Issue 的处理权。

        返回:
            claimed: 成功占用，可继续处理
            processed: 该 Issue 已成功回复过
            inflight: 当前已有协程在处理该 Issue
        """
        return self.try_claim_issue_key(issue_number, "")

    def try_claim_issue_key(
        self,
        issue_number: int,
        processing_key: str,
        *,
        updated_at: str = "",
    ) -> str:
        """
        原子地尝试占用一个 Issue 指纹的处理权。

        返回:
            claimed: 成功占用，可继续处理
            processed: 该相同内容已成功回复过
            inflight: 当前已有协程在处理该 Issue
        """
        with self._lock:
            issue = self._data["issues"].get(str(issue_number), {})
            processed_keys = self._processed_keys(issue, "processed_keys", "last_processed_key")
            legacy_processed = (
                bool(issue.get("processed_at"))
                and bool(updated_at)
                and str(issue.get("updated_at", "")) == updated_at
            )
            if processing_key and processing_key in processed_keys:
                return "processed"
            if processing_key and not processed_keys and legacy_processed:
                return "processed"
            if not processing_key and legacy_processed:
                return "processed"
            if issue_number in self._inflight_issues:
                return "inflight"
            self._inflight_issues.add(issue_number)
            return "claimed"

    def release_issue(self, issue_number: int, processing_key: str = ""):
        """释放一个 Issue 的处理中占位。"""
        with self._lock:
            self._inflight_issues.discard(issue_number)

    def should_notify_pr_blocked(
        self,
        pr_number: int,
        *,
        head_sha: str,
        reason: str,
        cooldown_seconds: int = 1800,
    ) -> bool:
        with self._lock:
            pr = self._data["pull_requests"].get(str(pr_number), {})
            if pr.get("last_blocked_notified_head_sha") != head_sha:
                return True
            if pr.get("last_blocked_notified_reason") != reason:
                return True

            last_notified_at = _parse_iso_datetime(str(pr.get("last_blocked_notified_at", "")))
            if last_notified_at is None:
                return True

            elapsed = (datetime.now(timezone.utc) - last_notified_at).total_seconds()
            return elapsed >= cooldown_seconds

    def has_tracked_pr(self, pr_number: int) -> bool:
        with self._lock:
            return str(pr_number) in self._data["pull_requests"]

    def pending_pr_numbers_for_polling(
        self,
        *,
        max_age_days: int = 30,
        limit: int = 50,
    ) -> list[int]:
        """返回 polling 模式下仍需继续观察的已跟踪 PR。"""
        with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(days=max(max_age_days, 1))
            candidates: list[tuple[datetime, int]] = []
            for raw_number, pr in self._data["pull_requests"].items():
                try:
                    pr_number = int(raw_number)
                except (TypeError, ValueError):
                    continue

                if int(pr.get("unreachable_count", 0) or 0) >= 3:
                    continue

                activity_at = _latest_pr_source_activity_at(pr)
                if activity_at is None or activity_at < cutoff:
                    continue

                last_seen_head_sha = str(pr.get("last_seen_head_sha", "") or "")
                last_processed_head_sha = str(pr.get("last_processed_head_sha", "") or "")
                has_processed = bool(pr.get("last_processed_at"))

                if not has_processed:
                    candidates.append((activity_at, pr_number))
                    continue
                if last_seen_head_sha and last_seen_head_sha != last_processed_head_sha:
                    candidates.append((activity_at, pr_number))
                    continue
            candidates.sort(key=lambda item: (-item[0].timestamp(), -item[1]))
            return [pr_number for _, pr_number in candidates[: max(limit, 1)]]

    def mark_pr_unreachable(self, pr_number: int):
        """标记 PR 为不可达（如 404），后续 polling 跳过。"""
        with self._lock:
            pr = self._data["pull_requests"].setdefault(str(pr_number), {})
            count = int(pr.get("unreachable_count", 0) or 0) + 1
            pr["unreachable_count"] = count
            pr["unreachable_at"] = utc_now_iso()
            self._save_unlocked()
            logger.info("PR #%d 标记为不可达 (累计 %d 次)", pr_number, count)

    def is_pr_unreachable(self, pr_number: int, threshold: int = 3) -> bool:
        """PR 连续不可达次数超过阈值时返回 True。"""
        with self._lock:
            pr = self._data["pull_requests"].get(str(pr_number), {})
            return int(pr.get("unreachable_count", 0) or 0) >= threshold

    def clear_pr_unreachable(self, pr_number: int):
        """成功访问 PR 后清除不可达标记。"""
        with self._lock:
            pr = self._data["pull_requests"].get(str(pr_number))
            if pr and pr.get("unreachable_count"):
                pr.pop("unreachable_count", None)
                pr.pop("unreachable_at", None)
                self._save_unlocked()

    def is_pr_head_stable(self, pr_number: int, head_sha: str, stabilize_seconds: int = 300) -> bool:
        """
        检查指定 head SHA 是否已稳定（首次观测至今超过 stabilize_seconds）。

        用于替代指数冷却：开发者频繁 push 时只 review 最后一个稳定版本，
        而不是每个中间 commit 都触发评审。首次评审不受此限制。
        """
        with self._lock:
            pr = self._data["pull_requests"].get(str(pr_number), {})
            if str(pr.get("last_seen_head_sha", "") or "") != head_sha:
                return False
            first_seen_at = _parse_iso_datetime(str(pr.get("head_sha_first_seen_at", "") or ""))
            if first_seen_at is None:
                return True
            elapsed = (datetime.now(timezone.utc) - first_seen_at).total_seconds()
            return elapsed >= stabilize_seconds

    def pr_numbers_for_review_polling(
        self,
        *,
        limit: int = 20,
        max_age_days: int = 21,
    ) -> list[int]:
        """返回需要额外扫描 review submissions 的近期 PR。"""
        with self._lock:
            cutoff = datetime.now(timezone.utc) - timedelta(days=max(max_age_days, 1))
            candidates: list[tuple[datetime, datetime, int]] = []
            for raw_number, pr in self._data["pull_requests"].items():
                try:
                    pr_number = int(raw_number)
                except (TypeError, ValueError):
                    continue

                if int(pr.get("unreachable_count", 0) or 0) >= 3:
                    continue

                activity_at = _latest_pr_source_activity_at(pr)
                if activity_at is None or activity_at < cutoff:
                    continue
                last_review_scan_at = _parse_iso_datetime(str(pr.get("last_review_scan_at", "") or ""))
                if last_review_scan_at is None:
                    last_review_scan_at = datetime.min.replace(tzinfo=timezone.utc)
                elif last_review_scan_at >= activity_at:
                    continue
                candidates.append((last_review_scan_at, activity_at, pr_number))

            candidates.sort(key=lambda item: (item[0], -item[1].timestamp(), -item[2]))
            return [pr_number for _, _, pr_number in candidates[: max(limit, 1)]]

    def pr_review_polling_cutoff(self, pr_number: int) -> str:
        """返回 review submission 跟进的时间下界。"""
        with self._lock:
            pr = self._data["pull_requests"].get(str(pr_number), {})
            review_cursor = str(pr.get("last_review_submission_cursor_at", "") or "")
            if review_cursor:
                return review_cursor
            last_processed_updated_at = _processed_pr_updated_at(pr)
            if last_processed_updated_at:
                return last_processed_updated_at

            polling = self._data["polling"]
            return str(
                polling.get("pr_created_after", "")
                or polling.get("bootstrapped_at", "")
                or ""
            )

    def mark_pr_review_polling_scanned(
        self,
        pr_number: int,
        *,
        scanned_at: str = "",
        latest_review_updated_at: str = "",
    ):
        """记录某个 PR 的 review submissions 已完成一次扫描。"""
        with self._lock:
            pr = self._data["pull_requests"].setdefault(str(pr_number), {})
            pr["last_review_scan_at"] = scanned_at or utc_now_iso()
            current_review_cursor = str(pr.get("last_review_submission_cursor_at", "") or "")
            next_review_cursor = _max_iso_datetime_value(current_review_cursor, latest_review_updated_at)
            if next_review_cursor:
                pr["last_review_submission_cursor_at"] = next_review_cursor
            self._save_unlocked()

    def mark_pr_seen(
        self,
        pr_number: int,
        *,
        head_sha: str,
        ci_state: str,
        created_at: str = "",
        updated_at: str = "",
    ):
        with self._lock:
            pr = self._data["pull_requests"].setdefault(str(pr_number), {})
            if pr.get("last_processed_head_sha") and not pr.get("last_processed_updated_at"):
                pr["last_processed_updated_at"] = pr.get("updated_at", "")
            old_seen_sha = str(pr.get("last_seen_head_sha", "") or "")
            if head_sha != old_seen_sha:
                pr["head_sha_first_seen_at"] = utc_now_iso()
            pr.update(
                {
                    "created_at": created_at or pr.get("created_at", ""),
                    "updated_at": updated_at,
                    "last_seen_head_sha": head_sha,
                    "last_seen_ci_state": ci_state,
                    "last_seen_at": utc_now_iso(),
                }
            )
            self._save_unlocked()

    def mark_pr_processed(
        self,
        pr_number: int,
        *,
        review_key: str,
        extra_review_keys: list[str] | None = None,
        head_sha: str,
        source: str,
        ci_state: str = "",
        created_at: str = "",
        updated_at: str = "",
        review_metadata: dict[str, str] | None = None,
    ):
        with self._lock:
            pr = self._data["pull_requests"].setdefault(str(pr_number), {})
            processed_updated_at = updated_at or pr.get("updated_at", "")
            review_submission_cursor_at = str(pr.get("last_review_submission_cursor_at", "") or "")
            if not review_submission_cursor_at and processed_updated_at:
                review_submission_cursor_at = processed_updated_at
            processed_keys = self._processed_keys(pr, "processed_review_keys", "last_processed_review_key")
            processed_keys = self._append_processed_key(processed_keys, review_key)
            for extra_key in extra_review_keys or []:
                processed_keys = self._append_processed_key(processed_keys, extra_key)
            updates = {
                "created_at": created_at or pr.get("created_at", ""),
                "updated_at": updated_at,
                "last_seen_head_sha": head_sha,
                "last_seen_ci_state": ci_state,
                "last_seen_at": utc_now_iso(),
                "last_processed_head_sha": head_sha,
                "last_processed_review_key": review_key,
                "processed_review_keys": processed_keys,
                "last_review_submission_cursor_at": review_submission_cursor_at,
                "last_processed_updated_at": processed_updated_at,
                "last_processed_at": utc_now_iso(),
                "source": source,
            }
            if review_metadata is not None:
                updates["last_processed_review_metadata"] = dict(review_metadata)
            pr.update(updates)
            self._save_unlocked()

    def mark_pr_blocked(
        self,
        pr_number: int,
        *,
        head_sha: str,
        source: str,
        reason: str,
        notified: bool,
        ci_state: str = "",
        created_at: str = "",
        updated_at: str = "",
    ):
        with self._lock:
            pr = self._data["pull_requests"].setdefault(str(pr_number), {})
            now = utc_now_iso()
            pr.update(
                {
                    "created_at": created_at or pr.get("created_at", ""),
                    "updated_at": updated_at,
                    "last_seen_head_sha": head_sha,
                    "last_seen_ci_state": ci_state,
                    "last_seen_at": now,
                    "last_blocked_head_sha": head_sha,
                    "last_blocked_at": now,
                    "last_blocked_reason": reason,
                    "last_blocked_source": source,
                }
            )
            if notified:
                pr.update(
                    {
                        "last_blocked_notified_head_sha": head_sha,
                        "last_blocked_notified_reason": reason,
                        "last_blocked_notified_at": now,
                    }
                )
            self._save_unlocked()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state_file_path": str(self.path),
                "issues_processed": sum(
                    1 for issue in self._data["issues"].values() if issue.get("processed_at")
                ),
                "prs_tracked": len(self._data["pull_requests"]),
                "prs_processed": sum(
                    1 for pr in self._data["pull_requests"].values() if pr.get("last_processed_at")
                ),
                "issue_baseline": self._data["polling"].get("issue_created_after", ""),
                "pr_baseline": self._data["polling"].get("pr_created_after", ""),
                "issue_cursor": self._data["polling"].get("issue_updated_after", ""),
                "pr_cursor": self._data["polling"].get("pr_updated_after", ""),
                "issue_comment_cursor": self._data["polling"].get("issue_comment_updated_after", ""),
                "review_comment_cursor": self._data["polling"].get("review_comment_updated_after", ""),
                "issue_min_number": self._data["thresholds"].get("issue_min_number", 0),
                "pr_min_number": self._data["thresholds"].get("pr_min_number", 0),
            }

    def advance_issue_cursor(self, latest_updated_at: str, *, overlap_seconds: int = 5):
        self._advance_polling_cursor("issue_updated_after", latest_updated_at, overlap_seconds)

    def advance_pr_cursor(self, latest_updated_at: str, *, overlap_seconds: int = 5):
        self._advance_polling_cursor("pr_updated_after", latest_updated_at, overlap_seconds)

    def advance_issue_comment_cursor(self, latest_updated_at: str, *, overlap_seconds: int = 5):
        self._advance_polling_cursor("issue_comment_updated_after", latest_updated_at, overlap_seconds)

    def advance_review_comment_cursor(self, latest_updated_at: str, *, overlap_seconds: int = 5):
        self._advance_polling_cursor("review_comment_updated_after", latest_updated_at, overlap_seconds)

    def _load(self) -> dict[str, Any]:
        primary_data = self._read_state_file(self.path)
        runtime_data = self._read_state_file(self.runtime_path)

        if primary_data is None and runtime_data is None:
            return self._default_data()
        if primary_data is None:
            return self._normalize(runtime_data or {})
        if runtime_data is None:
            return self._normalize(primary_data)

        normalized_primary = self._normalize(primary_data)
        normalized_runtime = self._normalize(runtime_data)
        merged = self._merge_runtime_state(normalized_primary, normalized_runtime)
        if merged != normalized_primary:
            logger.info(
                "检测到更新的 runtime state 影子文件，已恢复较新的本地进度: primary=%s, runtime=%s",
                self.path,
                self.runtime_path,
            )
        return merged

    def _read_state_file(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _merge_runtime_state(
        self,
        primary: dict[str, Any],
        runtime: dict[str, Any],
    ) -> dict[str, Any]:
        merged = self._default_data()

        merged["thresholds"].update(runtime.get("thresholds", {}))
        merged["thresholds"].update(primary.get("thresholds", {}))

        primary_polling = primary.get("polling", {})
        runtime_polling = runtime.get("polling", {})
        for field in merged["polling"]:
            merged["polling"][field] = _seed_with_latest_iso(
                str(primary_polling.get(field, "") or ""),
                str(runtime_polling.get(field, "") or ""),
            )

        for issue_number in set(primary.get("issues", {})) | set(runtime.get("issues", {})):
            primary_issue = primary.get("issues", {}).get(issue_number, {})
            runtime_issue = runtime.get("issues", {}).get(issue_number, {})
            merged["issues"][issue_number] = _prefer_newer_record(
                primary_issue,
                runtime_issue,
                marker_getter=_issue_state_marker,
            )

        for pr_number in set(primary.get("pull_requests", {})) | set(runtime.get("pull_requests", {})):
            primary_pr = primary.get("pull_requests", {}).get(pr_number, {})
            runtime_pr = runtime.get("pull_requests", {}).get(pr_number, {})
            merged["pull_requests"][pr_number] = _prefer_newer_record(
                primary_pr,
                runtime_pr,
                marker_getter=_pr_state_marker,
            )

        return merged

    def _normalize(self, data: dict[str, Any]) -> dict[str, Any]:
        normalized = self._default_data()
        source_schema_version = 0
        try:
            source_schema_version = int(data.get("schema_version", 0) or 0)
        except (TypeError, ValueError):
            source_schema_version = 0
        normalized["polling"].update(data.get("polling", {}))
        normalized["thresholds"].update(data.get("thresholds", {}))
        normalized["issues"].update(data.get("issues", {}))
        normalized["pull_requests"].update(data.get("pull_requests", {}))
        self._migrate_legacy_state(normalized, source_schema_version)
        return normalized

    def _default_data(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "polling": {
                "bootstrapped_at": "",
                "issue_created_after": "",
                "pr_created_after": "",
                "issue_updated_after": "",
                "pr_updated_after": "",
                "issue_comment_updated_after": "",
                "review_comment_updated_after": "",
            },
            "thresholds": {
                "issue_min_number": 603,
                "pr_min_number": 603,
            },
            "issues": {},
            "pull_requests": {},
        }

    def _migrate_legacy_state(self, data: dict[str, Any], source_schema_version: int):
        """把旧状态补齐为当前 schema，尽量沿用已有进度而不是回退重扫。"""
        data["schema_version"] = self._default_data()["schema_version"]
        polling = data["polling"]

        latest_issue_cursor = ""
        latest_pr_cursor = ""
        latest_review_cursor = ""

        for issue in data["issues"].values():
            latest_issue_cursor = _max_iso_datetime_value(
                latest_issue_cursor,
                _legacy_issue_cursor_seed(issue),
            )

        for pr in data["pull_requests"].values():
            processed_updated_at = _processed_pr_updated_at(pr)
            if pr.get("last_processed_head_sha") and not pr.get("last_processed_updated_at") and processed_updated_at:
                pr["last_processed_updated_at"] = processed_updated_at

            review_cursor_seed = str(
                pr.get("last_review_submission_cursor_at")
                or pr.get("last_processed_updated_at")
                or pr.get("updated_at")
                or pr.get("created_at")
                or ""
            )
            if pr.get("last_processed_head_sha") and not pr.get("last_review_submission_cursor_at") and review_cursor_seed:
                pr["last_review_submission_cursor_at"] = review_cursor_seed
            if pr.get("last_processed_head_sha") and not pr.get("last_review_scan_at"):
                last_review_scan_at = str(
                    pr.get("last_seen_at")
                    or pr.get("last_processed_at")
                    or review_cursor_seed
                    or ""
                )
                if last_review_scan_at:
                    pr["last_review_scan_at"] = last_review_scan_at

            latest_pr_cursor = _max_iso_datetime_value(latest_pr_cursor, _legacy_pr_cursor_seed(pr))
            latest_review_cursor = _max_iso_datetime_value(latest_review_cursor, review_cursor_seed)

        if source_schema_version < 3:
            if not polling.get("issue_updated_after"):
                polling["issue_updated_after"] = _seed_with_latest_iso(
                    latest_issue_cursor,
                    str(polling.get("issue_created_after", "") or ""),
                    str(polling.get("bootstrapped_at", "") or ""),
                )
            if not polling.get("pr_updated_after"):
                polling["pr_updated_after"] = _seed_with_latest_iso(
                    latest_pr_cursor,
                    str(polling.get("pr_created_after", "") or ""),
                    str(polling.get("bootstrapped_at", "") or ""),
                )
            if not polling.get("issue_comment_updated_after"):
                polling["issue_comment_updated_after"] = _seed_with_latest_iso(
                    latest_issue_cursor,
                    str(polling.get("issue_updated_after", "") or ""),
                    str(polling.get("issue_created_after", "") or ""),
                    str(polling.get("bootstrapped_at", "") or ""),
                )
            if not polling.get("review_comment_updated_after"):
                polling["review_comment_updated_after"] = _seed_with_latest_iso(
                    latest_review_cursor,
                    str(polling.get("pr_updated_after", "") or ""),
                    str(polling.get("pr_created_after", "") or ""),
                    str(polling.get("bootstrapped_at", "") or ""),
                )

    def _processed_keys(
        self,
        item: dict[str, Any],
        list_field: str,
        legacy_field: str,
    ) -> list[str]:
        keys: list[str] = []
        raw_keys = item.get(list_field, [])
        if isinstance(raw_keys, list):
            for raw in raw_keys:
                key = str(raw or "").strip()
                if key and key not in keys:
                    keys.append(key)
        legacy_key = str(item.get(legacy_field, "") or "").strip()
        if legacy_key and legacy_key not in keys:
            keys.append(legacy_key)
        return keys[-_MAX_STORED_KEYS:]

    def _append_processed_key(self, keys: list[str], key: str) -> list[str]:
        normalized = key.strip()
        if not normalized:
            return keys[-_MAX_STORED_KEYS:]
        merged = [existing for existing in keys if existing != normalized]
        merged.append(normalized)
        return merged[-_MAX_STORED_KEYS:]

    def _advance_polling_cursor(self, field: str, latest_updated_at: str, overlap_seconds: int):
        normalized = latest_updated_at.strip()
        if not normalized:
            return

        with self._lock:
            polling = self._data.setdefault("polling", {})
            current_value = str(polling.get(field, "") or "")
            candidate_value = _rewind_iso_datetime(normalized, overlap_seconds)
            current_dt = _parse_iso_datetime(current_value)
            candidate_dt = _parse_iso_datetime(candidate_value)
            if candidate_dt is None:
                return
            if current_dt is not None and candidate_dt <= current_dt:
                return
            polling[field] = candidate_value
            self._save_unlocked()

    def _save_unlocked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_state_file(self.path, self._data)
        self._write_state_file(self.runtime_path, self._data)

    def _write_state_file(self, path: Path, data: dict[str, Any]):
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)
