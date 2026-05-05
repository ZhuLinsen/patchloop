"""AutoCode 本地状态存储。"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_RECENT_EVENT_LIMIT = 30
_MAX_STORED_KEYS = 16


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_issue_plan_key(issue_number: int, updated_at: str, task_type: str) -> str:
    digest = hashlib.sha256(f"{issue_number}\n{updated_at}\n{task_type}".encode("utf-8")).hexdigest()[:16]
    return f"issue:{issue_number}:{digest}"


def build_feedback_item_fingerprint(item: dict[str, Any]) -> str:
    item_id = int(item.get("id", 0) or 0)
    kind = "review_comment" if str(item.get("path", "") or "").strip() else "review"
    body = str(item.get("body", "") or "").strip()
    marker = str(item.get("updated_at") or item.get("submitted_at") or item.get("created_at") or "").strip()
    user_login = str(item.get("user", {}).get("login", "") or "").strip().lower()
    path = str(item.get("path", "") or "").strip()
    state = str(item.get("state", "") or "").strip().upper()
    subject_type = str(item.get("subject_type", "") or "").strip()
    side = str(item.get("side", "") or "").strip()
    line = int(item.get("line", 0) or 0)
    original_line = int(item.get("original_line", 0) or 0)
    review_id = int(item.get("pull_request_review_id", 0) or 0)
    digest = hashlib.sha256(
        (
            f"{kind}\n{item_id}\n{marker}\n{user_login}\n{state}\n{path}\n{subject_type}\n"
            f"{side}\n{line}\n{original_line}\n{review_id}\n\n{body}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    suffix = item_id if item_id > 0 else digest[:8]
    return f"{kind}:{suffix}:{digest}"


def build_feedback_key(
    pr_number: int,
    comment_ids: list[int],
    updated_at_values: list[str],
    *,
    item_fingerprints: list[str] | None = None,
) -> str:
    joined_ids = ",".join(str(item) for item in sorted(comment_ids))
    joined_times = ",".join(sorted(value for value in updated_at_values if value))
    joined_fingerprints = "\n".join(sorted(item for item in (item_fingerprints or []) if str(item).strip()))
    digest = hashlib.sha256(
        f"{pr_number}\n{joined_ids}\n{joined_times}\n\n{joined_fingerprints}".encode("utf-8")
    ).hexdigest()[:16]
    return f"pr:{pr_number}:{digest}"


@dataclass(frozen=True)
class AutoCodeRunRecord:
    run_id: str
    status: str
    kind: str
    target_number: int
    branch_name: str = ""
    pr_number: int = 0
    pr_url: str = ""
    failure_reason: str = ""
    started_at: str = ""
    finished_at: str = ""


class AutoCodeStateStore:
    """为 AutoCode pipeline 记录计划、执行、polling 与 PR 队列状态。"""

    def __init__(self, path: str):
        self.path = Path(path).resolve()
        self._lock = threading.Lock()
        self._data = self._load()
        self._inflight_issue_keys: set[str] = set()
        self._inflight_feedback_keys: dict[str, str] = {}
        self._inflight_issue_events: set[int] = set()
        self._inflight_issue_comments: set[int] = set()
        self._inflight_tracked_prs: set[int] = set()

    def ensure_polling_baseline(self) -> bool:
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
            if "issue_event_after_id" not in polling:
                polling["issue_event_after_id"] = 0
                changed = True
            if not polling.get("issue_comment_after"):
                polling["issue_comment_after"] = now
                changed = True
            if "issue_comment_after_id" not in polling:
                polling["issue_comment_after_id"] = 0
                changed = True
            if changed:
                self._save_unlocked()
            return changed

    def issue_baseline(self) -> str:
        with self._lock:
            return str(self._data["polling"].get("issue_created_after", "") or "")

    def issue_comment_baseline(self) -> str:
        with self._lock:
            return str(self._data["polling"].get("issue_comment_after", "") or "")

    def issue_event_cursor(self) -> tuple[str, int]:
        with self._lock:
            return (
                str(self._data["polling"].get("issue_created_after", "") or ""),
                int(self._data["polling"].get("issue_event_after_id", 0) or 0),
            )

    def issue_comment_cursor(self) -> tuple[str, int]:
        with self._lock:
            return (
                str(self._data["polling"].get("issue_comment_after", "") or ""),
                int(self._data["polling"].get("issue_comment_after_id", 0) or 0),
            )

    def advance_issue_baseline(self, created_at: str, event_id: int):
        with self._lock:
            polling = self._data.setdefault("polling", {})
            polling["issue_created_after"] = created_at
            polling["issue_event_after_id"] = int(event_id or 0)
            self._save_unlocked()

    def advance_issue_comment_baseline(self, updated_at: str, comment_id: int):
        with self._lock:
            polling = self._data.setdefault("polling", {})
            polling["issue_comment_after"] = updated_at
            polling["issue_comment_after_id"] = int(comment_id or 0)
            self._save_unlocked()

    def try_claim_issue_event(self, event_id: int) -> str:
        with self._lock:
            if event_id in self._inflight_issue_events:
                return "inflight"
            if str(event_id) in self._data["issue_events"]:
                return "processed"
            self._inflight_issue_events.add(event_id)
            return "claimed"

    def has_processed_issue_event(self, event_id: int) -> bool:
        with self._lock:
            return str(event_id) in self._data["issue_events"]

    def release_issue_event(self, event_id: int):
        with self._lock:
            self._inflight_issue_events.discard(event_id)

    def mark_issue_event_processed(
        self,
        *,
        event_id: int,
        issue_number: int,
        action: str,
        created_at: str,
        source: str,
    ):
        with self._lock:
            self._data["issue_events"][str(event_id)] = {
                "issue_number": issue_number,
                "action": action,
                "created_at": created_at,
                "source": source,
                "processed_at": utc_now_iso(),
            }
            self._save_unlocked()

    def has_processed_issue_number(self, issue_number: int) -> bool:
        with self._lock:
            return str(issue_number) in self._data["issues"]

    def has_processed_issue_version(self, issue_number: int, updated_at: str) -> bool:
        with self._lock:
            issue_record = self._data["issues"].get(str(issue_number), {})
            if not isinstance(issue_record, dict):
                return False
            normalized_updated_at = str(updated_at or "")
            processed_updated_ats = _recent_string_values(
                issue_record,
                "processed_updated_ats",
                "updated_at",
            )
            return normalized_updated_at in processed_updated_ats

    def try_claim_issue_plan(self, issue_key: str) -> str:
        with self._lock:
            if issue_key in self._inflight_issue_keys:
                return "inflight"
            if issue_key in self._data["issue_runs"]:
                return "processed"
            self._inflight_issue_keys.add(issue_key)
            return "claimed"

    def release_issue_plan(self, issue_key: str):
        with self._lock:
            self._inflight_issue_keys.discard(issue_key)

    def mark_issue_plan(
        self,
        *,
        issue_key: str,
        issue_number: int,
        task_type: str,
        updated_at: str,
        summary: str,
        last_outcome: str = "",
    ):
        with self._lock:
            planned_at = utc_now_iso()
            self._data["issue_runs"][issue_key] = {
                "issue_number": issue_number,
                "task_type": task_type,
                "updated_at": updated_at,
                "summary": summary,
                "planned_at": planned_at,
            }
            existing = dict(self._data["issues"].get(str(issue_number), {}))
            processed_issue_keys = _append_recent_value(
                _recent_string_values(existing, "processed_issue_keys", "last_issue_key"),
                issue_key,
            )
            processed_updated_ats = _append_recent_value(
                _recent_string_values(existing, "processed_updated_ats", "updated_at"),
                updated_at,
            )
            existing.update(
                {
                    "issue_number": issue_number,
                    "task_type": task_type,
                    "updated_at": updated_at,
                    "summary": summary,
                    "planned_at": planned_at,
                    "last_issue_key": issue_key,
                    "processed_issue_keys": processed_issue_keys,
                    "processed_updated_ats": processed_updated_ats,
                    "last_outcome": last_outcome or "",
                }
            )
            self._data["issues"][str(issue_number)] = existing
            self._save_unlocked()

    def get_issue_last_outcome(self, issue_number: int) -> str:
        with self._lock:
            issue_record = self._data["issues"].get(str(issue_number), {})
            if not isinstance(issue_record, dict):
                return ""
            outcome = str(issue_record.get("last_outcome", "") or "")
            if outcome:
                return outcome
            # Fallback for legacy records: check runs data for the latest run
            # targeting this issue to determine its outcome.
            latest_status = ""
            latest_finished = ""
            for run in self._data.get("runs", {}).values():
                if not isinstance(run, dict):
                    continue
                if int(run.get("target_number", 0) or 0) != issue_number:
                    continue
                if str(run.get("kind", "") or "") != "issue":
                    continue
                finished = str(run.get("finished_at", "") or "")
                if finished > latest_finished:
                    latest_finished = finished
                    latest_status = str(run.get("status", "") or "")
            return latest_status

    def save_issue_plan_snapshot(self, issue_number: int, snapshot: dict[str, Any]):
        with self._lock:
            self._data["issue_plans"][str(issue_number)] = dict(snapshot)
            self._save_unlocked()

    def get_issue_plan_snapshot(self, issue_number: int) -> dict[str, Any] | None:
        with self._lock:
            snapshot = self._data["issue_plans"].get(str(issue_number))
            return dict(snapshot) if isinstance(snapshot, dict) else None

    def try_claim_issue_comment(self, comment_id: int, updated_at: str = "") -> str:
        with self._lock:
            if comment_id in self._inflight_issue_comments:
                return "inflight"
            existing = self._data["issue_comments"].get(str(comment_id), {})
            existing_updated_at = str(existing.get("updated_at", "") or "") if isinstance(existing, dict) else ""
            if str(comment_id) in self._data["issue_comments"] and (
                not updated_at or existing_updated_at == str(updated_at or "")
            ):
                return "processed"
            self._inflight_issue_comments.add(comment_id)
            return "claimed"

    def has_processed_issue_comment(self, comment_id: int, updated_at: str = "") -> bool:
        with self._lock:
            existing = self._data["issue_comments"].get(str(comment_id))
            if not isinstance(existing, dict):
                return False
            if not updated_at:
                return True
            return str(existing.get("updated_at", "") or "") == str(updated_at or "")

    def release_issue_comment(self, comment_id: int):
        with self._lock:
            self._inflight_issue_comments.discard(comment_id)

    def mark_issue_comment_processed(
        self,
        *,
        comment_id: int,
        issue_number: int,
        updated_at: str,
        command: str,
        source: str,
    ):
        with self._lock:
            self._data["issue_comments"][str(comment_id)] = {
                "issue_number": issue_number,
                "updated_at": updated_at,
                "command": command,
                "source": source,
                "processed_at": utc_now_iso(),
            }
            self._save_unlocked()

    def set_issue_paused(self, issue_number: int, *, paused: bool, reason: str = ""):
        with self._lock:
            key = str(issue_number)
            if paused:
                self._data["paused_issues"][key] = {
                    "issue_number": issue_number,
                    "reason": reason,
                    "updated_at": utc_now_iso(),
                }
            else:
                self._data["paused_issues"].pop(key, None)
            self._save_unlocked()

    def is_issue_paused(self, issue_number: int) -> bool:
        with self._lock:
            return str(issue_number) in self._data["paused_issues"]

    def get_issue_pause(self, issue_number: int) -> dict[str, Any] | None:
        with self._lock:
            record = self._data["paused_issues"].get(str(issue_number))
            return dict(record) if isinstance(record, dict) else None

    def try_claim_feedback(
        self,
        feedback_key: str,
        *,
        extra_feedback_keys: list[str] | None = None,
        owner: str = "",
    ) -> str:
        with self._lock:
            candidate_keys = _candidate_keys(feedback_key, extra_feedback_keys)
            if any(key in self._inflight_feedback_keys for key in candidate_keys):
                return "inflight"
            if any(key in self._data["review_feedback"] for key in candidate_keys):
                return "processed"
            self._inflight_feedback_keys[feedback_key] = str(owner or "").strip()
            return "claimed"

    def set_feedback_claim_owner(self, feedback_key: str, *, owner: str):
        with self._lock:
            key = str(feedback_key or "").strip()
            if not key or key not in self._inflight_feedback_keys:
                return
            self._inflight_feedback_keys[key] = str(owner or "").strip()

    def feedback_claim_owned_by(
        self,
        feedback_key: str,
        *,
        owner: str,
        extra_feedback_keys: list[str] | None = None,
    ) -> bool:
        with self._lock:
            expected_owner = str(owner or "").strip()
            if not expected_owner:
                return False
            return any(
                str(self._inflight_feedback_keys.get(key, "") or "").strip() == expected_owner
                for key in _candidate_keys(feedback_key, extra_feedback_keys)
            )

    def release_feedback(self, feedback_key: str, *, owner: str = ""):
        with self._lock:
            key = str(feedback_key or "").strip()
            if not key:
                return
            if owner:
                current_owner = str(self._inflight_feedback_keys.get(key, "") or "").strip()
                expected_owner = str(owner or "").strip()
                if current_owner and current_owner != expected_owner:
                    return
            self._inflight_feedback_keys.pop(key, None)

    def mark_feedback_processed(
        self,
        *,
        feedback_key: str,
        pr_number: int,
        summary: str,
        extra_feedback_keys: list[str] | None = None,
    ):
        with self._lock:
            record = {
                "pr_number": pr_number,
                "summary": summary,
                "processed_at": utc_now_iso(),
                "canonical_feedback_key": feedback_key,
            }
            for key in _candidate_keys(feedback_key, extra_feedback_keys):
                self._data["review_feedback"][key] = dict(record)
                self._inflight_feedback_keys.pop(key, None)
            self._save_unlocked()

    def try_claim_tracked_pr(self, pr_number: int) -> str:
        with self._lock:
            if pr_number in self._inflight_tracked_prs:
                return "inflight"
            self._inflight_tracked_prs.add(pr_number)
            return "claimed"

    def release_tracked_pr(self, pr_number: int):
        with self._lock:
            self._inflight_tracked_prs.discard(pr_number)

    def upsert_tracked_pr(
        self,
        *,
        pr_number: int,
        branch_name: str,
        base_ref: str,
        head_sha: str,
        author_login: str,
        pr_url: str,
        queue_source: str,
        pr_updated_at: str = "",
    ):
        with self._lock:
            key = str(pr_number)
            existing = dict(self._data["tracked_pull_requests"].get(key, {}))
            if not existing.get("first_seen_at"):
                existing["first_seen_at"] = utc_now_iso()
            existing.update(
                {
                    "pr_number": pr_number,
                    "branch_name": branch_name,
                    "base_ref": base_ref,
                    "head_sha": head_sha,
                    "author_login": author_login,
                    "pr_url": pr_url,
                    "queue_source": queue_source,
                    "last_seen_at": utc_now_iso(),
                }
            )
            if pr_updated_at:
                existing["pr_updated_at"] = pr_updated_at
            if not existing.get("status"):
                existing["status"] = "queued"
            self._data["tracked_pull_requests"][key] = existing
            self._save_unlocked()

    def mark_tracked_pr_status(
        self,
        pr_number: int,
        *,
        status: str,
        summary: str = "",
        review_state: str = "",
    ):
        with self._lock:
            key = str(pr_number)
            existing = dict(self._data["tracked_pull_requests"].get(key, {}))
            if not existing.get("first_seen_at"):
                existing["first_seen_at"] = utc_now_iso()
            existing["pr_number"] = pr_number
            existing["status"] = status
            existing["last_seen_at"] = utc_now_iso()
            if summary:
                existing["last_summary"] = summary
            if review_state:
                existing["review_state"] = review_state
            self._data["tracked_pull_requests"][key] = existing
            self._save_unlocked()

    def increment_tracked_pr_repair_count(self, pr_number: int, feedback_count: int = 0) -> int:
        """递增 PR 修复计数并返回新值。同时记录每轮 feedback 数量用于收敛性判断。"""
        with self._lock:
            key = str(pr_number)
            existing = self._data["tracked_pull_requests"].get(key)
            if not isinstance(existing, dict):
                return 0
            count = int(existing.get("repair_count", 0) or 0) + 1
            existing["repair_count"] = count
            history = existing.get("repair_feedback_counts", [])
            if not isinstance(history, list):
                history = []
            history.append(feedback_count)
            existing["repair_feedback_counts"] = history
            self._save_unlocked()
            return count

    def get_tracked_pr_repair_count(self, pr_number: int) -> int:
        with self._lock:
            record = self._data["tracked_pull_requests"].get(str(pr_number))
            if not isinstance(record, dict):
                return 0
            return int(record.get("repair_count", 0) or 0)

    def is_repair_diverging(self, pr_number: int, consecutive_threshold: int = 2) -> bool:
        """连续 N 轮 feedback 数量递增则判定为发散。"""
        with self._lock:
            record = self._data["tracked_pull_requests"].get(str(pr_number))
            if not isinstance(record, dict):
                return False
            history = record.get("repair_feedback_counts", [])
            if not isinstance(history, list) or len(history) < consecutive_threshold + 1:
                return False
            tail = history[-(consecutive_threshold + 1):]
            return all(tail[i] > 0 and tail[i + 1] > tail[i] for i in range(len(tail) - 1))

    def reset_tracked_pr_repair_count(self, pr_number: int):
        """将 PR 修复计数归零（用于 resume 后重新允许自动修复）。"""
        with self._lock:
            key = str(pr_number)
            existing = self._data["tracked_pull_requests"].get(key)
            if isinstance(existing, dict) and int(existing.get("repair_count", 0) or 0) > 0:
                existing["repair_count"] = 0
                existing.pop("repair_feedback_counts", None)
                self._save_unlocked()

    def list_tracked_prs(self) -> list[dict[str, Any]]:
        with self._lock:
            records = [dict(item) for item in self._data["tracked_pull_requests"].values()]
        records.sort(key=lambda item: (str(item.get("first_seen_at", "")), int(item.get("pr_number", 0) or 0)))
        return records

    def get_tracked_pr(self, pr_number: int) -> dict[str, Any] | None:
        with self._lock:
            record = self._data["tracked_pull_requests"].get(str(pr_number))
            return dict(record) if isinstance(record, dict) else None

    def mark_tracked_pr_rebase_failure(self, pr_number: int, *, head_sha: str, summary: str = ""):
        with self._lock:
            key = str(pr_number)
            existing = dict(self._data["tracked_pull_requests"].get(key, {}))
            if not existing.get("first_seen_at"):
                existing["first_seen_at"] = utc_now_iso()
            existing["pr_number"] = pr_number
            existing["last_seen_at"] = utc_now_iso()
            existing["last_rebase_failed_head_sha"] = str(head_sha or "")
            existing["last_rebase_failed_at"] = utc_now_iso()
            if summary:
                existing["last_rebase_failed_summary"] = str(summary)
            self._data["tracked_pull_requests"][key] = existing
            self._save_unlocked()

    def clear_tracked_pr_rebase_failure(self, pr_number: int):
        with self._lock:
            key = str(pr_number)
            existing = self._data["tracked_pull_requests"].get(key)
            if not isinstance(existing, dict):
                return
            changed = False
            for field in ("last_rebase_failed_head_sha", "last_rebase_failed_at", "last_rebase_failed_summary"):
                if field in existing:
                    existing.pop(field, None)
                    changed = True
            if changed:
                existing["last_seen_at"] = utc_now_iso()
                self._save_unlocked()

    def set_tracked_pr_paused(self, pr_number: int, *, paused: bool, reason: str = ""):
        with self._lock:
            key = str(pr_number)
            if paused:
                self._data["paused_tracked_prs"][key] = {
                    "pr_number": pr_number,
                    "reason": reason,
                    "updated_at": utc_now_iso(),
                }
            else:
                self._data["paused_tracked_prs"].pop(key, None)
                # resume 时重置修复计数和历史，允许自动修复重新开始
                existing = self._data["tracked_pull_requests"].get(key)
                if isinstance(existing, dict) and int(existing.get("repair_count", 0) or 0) > 0:
                    existing["repair_count"] = 0
                    existing.pop("repair_feedback_counts", None)
            self._save_unlocked()

    def is_tracked_pr_paused(self, pr_number: int) -> bool:
        with self._lock:
            return str(pr_number) in self._data["paused_tracked_prs"]

    def get_tracked_pr_pause(self, pr_number: int) -> dict[str, Any] | None:
        with self._lock:
            record = self._data["paused_tracked_prs"].get(str(pr_number))
            return dict(record) if isinstance(record, dict) else None

    def update_tracked_pr_feedback_cursor(
        self,
        pr_number: int,
        *,
        updated_at: str,
        item_ids: list[int],
    ):
        with self._lock:
            key = str(pr_number)
            existing = dict(self._data["tracked_pull_requests"].get(key, {}))
            if not existing.get("first_seen_at"):
                existing["first_seen_at"] = utc_now_iso()
            existing["pr_number"] = pr_number
            existing["last_feedback_updated_at"] = updated_at
            existing["last_feedback_item_ids"] = list(item_ids)
            existing["last_seen_at"] = utc_now_iso()
            self._data["tracked_pull_requests"][key] = existing
            self._save_unlocked()

    def reset_tracked_pr_feedback_cursor(self, pr_number: int):
        with self._lock:
            key = str(pr_number)
            existing = dict(self._data["tracked_pull_requests"].get(key, {}))
            if not existing.get("first_seen_at"):
                existing["first_seen_at"] = utc_now_iso()
            existing["pr_number"] = pr_number
            existing.pop("last_feedback_updated_at", None)
            existing.pop("last_feedback_item_ids", None)
            existing["last_seen_at"] = utc_now_iso()
            self._data["tracked_pull_requests"][key] = existing
            self._save_unlocked()

    def mark_tracked_pr_stale_feedback_retry(self, pr_number: int, *, feedback_updated_at: str):
        with self._lock:
            key = str(pr_number)
            existing = dict(self._data["tracked_pull_requests"].get(key, {}))
            if not existing.get("first_seen_at"):
                existing["first_seen_at"] = utc_now_iso()
            existing["pr_number"] = pr_number
            existing["last_stale_feedback_retry_at"] = utc_now_iso()
            existing["last_stale_feedback_retry_updated_at"] = str(feedback_updated_at or "")
            existing["last_seen_at"] = utc_now_iso()
            self._data["tracked_pull_requests"][key] = existing
            self._save_unlocked()

    def mark_backlog_item_synced(
        self,
        *,
        item_key: str,
        source_name: str,
        source_path: str,
        line_number: int,
        title: str,
        issue_number: int,
        issue_url: str,
        issue_title: str = "",
        dry_run: bool = False,
        sync_status: str | None = None,
        sync_reason: str = "",
    ):
        with self._lock:
            effective_sync_status = str(sync_status or "").strip() or ("dry_run" if dry_run else "created")
            source_title = str(title or "")
            effective_issue_title = str(issue_title or "").strip() or source_title
            self._data["backlog_items"][item_key] = {
                "item_key": item_key,
                "source_name": str(source_name or ""),
                "source_path": source_path,
                "line_number": int(line_number or 0),
                "title": source_title,
                "source_title": source_title,
                "issue_title": effective_issue_title,
                "issue_number": issue_number,
                "issue_url": issue_url,
                "dry_run": bool(dry_run),
                "sync_status": effective_sync_status,
                "sync_reason": str(sync_reason or ""),
                "created_at": utc_now_iso(),
                "last_seen_at": utc_now_iso(),
            }
            self._save_unlocked()

    def has_backlog_item(self, item_key: str, *, require_real_issue: bool = False) -> bool:
        with self._lock:
            record = self._data["backlog_items"].get(item_key)
            if not isinstance(record, dict):
                return False
            if require_real_issue and bool(record.get("dry_run", False)):
                return False
            return True

    def touch_backlog_issue_check(self, item_key: str) -> None:
        """记录 backlog 条目最近一次 issue 状态检查的时间。"""
        with self._lock:
            record = self._data["backlog_items"].get(item_key)
            if isinstance(record, dict):
                record["last_issue_check_at"] = utc_now_iso()
                self._save_unlocked()

    def get_backlog_item(self, item_key: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._data["backlog_items"].get(item_key)
            return dict(record) if isinstance(record, dict) else None

    def list_backlog_items(self, *, require_real_issue: bool = False) -> list[dict[str, Any]]:
        with self._lock:
            records = []
            for record in self._data["backlog_items"].values():
                if not isinstance(record, dict):
                    continue
                if require_real_issue and bool(record.get("dry_run", False)):
                    continue
                records.append(dict(record))
        records.sort(
            key=lambda item: (
                str(item.get("source_name", "") or ""),
                str(item.get("source_path", "") or ""),
                int(item.get("line_number", 0) or 0),
                str(item.get("created_at", "") or ""),
                str(item.get("item_key", "") or ""),
            )
        )
        return records

    def mark_run(self, record: AutoCodeRunRecord):
        with self._lock:
            self._data["runs"][record.run_id] = asdict(record)
            self._save_unlocked()

    def enqueue_task(
        self,
        *,
        task_id: str,
        task_type: str,
        target_number: int,
        source: str,
        priority: str = "normal",
        command: str = "",
        payload: dict[str, Any] | None = None,
        max_attempts: int = 3,
        replace_existing: bool = False,
    ) -> tuple[dict[str, Any], bool]:
        with self._lock:
            tasks = self._data["task_queue"]
            existing = tasks.get(task_id)
            if isinstance(existing, dict):
                if not replace_existing:
                    return dict(existing), False
                if str(existing.get("status", "") or "") == "running":
                    return dict(existing), False
                if bool(existing.get("permanent_failure", False)):
                    return dict(existing), False
                now = utc_now_iso()
                existing["source"] = str(source or existing.get("source", "") or "")
                existing["priority"] = _merge_task_priority(
                    str(existing.get("priority", "") or ""),
                    str(priority or "normal"),
                )
                if command:
                    existing["command"] = str(command)
                if payload is not None:
                    existing["payload"] = _json_safe(payload)
                existing["max_attempts"] = max(
                    int(existing.get("max_attempts", 0) or 0),
                    max(int(max_attempts or 0), 1),
                )
                existing["updated_at"] = now
                existing["available_at"] = now
                if str(existing.get("status", "") or "") in {"retry_waiting", "dead_letter", "done"}:
                    existing["status"] = "queued"
                    existing["attempt_count"] = 0
                    existing["last_error"] = ""
                    existing["result_summary"] = ""
                    existing["lease_owner"] = ""
                    existing["lease_acquired_at"] = ""
                self._save_unlocked()
                return dict(existing), False
            active_records = [
                record
                for record in tasks.values()
                if isinstance(record, dict)
                and str(record.get("task_type", "") or "") == task_type
                and int(target_number or 0) > 0
                and int(record.get("target_number", 0) or 0) == int(target_number or 0)
                and str(record.get("status", "") or "") in {"queued", "running", "retry_waiting"}
            ]
            if active_records:
                if replace_existing:
                    pending_records = [
                        record for record in active_records if str(record.get("status", "") or "") != "running"
                    ]
                    if pending_records:
                        record = sorted(pending_records, key=_task_sort_key)[0]
                        now = utc_now_iso()
                        record["source"] = str(source or record.get("source", "") or "")
                        record["priority"] = _merge_task_priority(
                            str(record.get("priority", "") or ""),
                            str(priority or "normal"),
                        )
                        if command:
                            record["command"] = str(command)
                        if payload is not None:
                            record["payload"] = _json_safe(payload)
                        record["max_attempts"] = max(
                            int(record.get("max_attempts", 0) or 0),
                            max(int(max_attempts or 0), 1),
                        )
                        record["updated_at"] = now
                        record["available_at"] = now
                        if str(record.get("status", "") or "") == "retry_waiting":
                            record["status"] = "queued"
                            record["attempt_count"] = 0
                            record["last_error"] = ""
                            record["lease_owner"] = ""
                            record["lease_acquired_at"] = ""
                        self._save_unlocked()
                        return dict(record), False
                else:
                    return dict(sorted(active_records, key=_task_sort_key)[0]), False

            now = utc_now_iso()
            record = {
                "task_id": task_id,
                "task_type": task_type,
                "target_number": int(target_number or 0),
                "source": str(source or ""),
                "priority": str(priority or "normal"),
                "command": str(command or ""),
                "payload": _json_safe(payload or {}),
                "status": "queued",
                "attempt_count": 0,
                "max_attempts": max(int(max_attempts or 0), 1),
                "available_at": now,
                "lease_owner": "",
                "lease_acquired_at": "",
                "created_at": now,
                "updated_at": now,
                "last_started_at": "",
                "last_finished_at": "",
                "last_error": "",
                "result_summary": "",
            }
            tasks[task_id] = record
            self._save_unlocked()
            return dict(record), True

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._data["task_queue"].get(task_id)
            return dict(record) if isinstance(record, dict) else None

    def list_tasks(
        self,
        *,
        task_type: str = "",
        statuses: set[str] | None = None,
        target_number: int = 0,
    ) -> list[dict[str, Any]]:
        with self._lock:
            records = []
            for record in self._data["task_queue"].values():
                if not isinstance(record, dict):
                    continue
                if task_type and str(record.get("task_type", "") or "") != task_type:
                    continue
                if statuses and str(record.get("status", "") or "") not in statuses:
                    continue
                if target_number > 0 and int(record.get("target_number", 0) or 0) != int(target_number):
                    continue
                records.append(dict(record))
        records.sort(
            key=lambda item: (
                _task_priority_rank(str(item.get("priority", "") or "")),
                str(item.get("available_at", "") or ""),
                str(item.get("created_at", "") or ""),
                str(item.get("task_id", "") or ""),
            )
        )
        return records

    def claim_next_task(self, *, task_type: str, owner: str = "task-queue") -> dict[str, Any] | None:
        with self._lock:
            now = utc_now_iso()
            candidates = [
                record
                for record in self._data["task_queue"].values()
                if isinstance(record, dict)
                and str(record.get("task_type", "") or "") == task_type
                and str(record.get("status", "") or "") in {"queued", "retry_waiting"}
                and _is_iso_not_after(str(record.get("available_at", "") or ""), now)
            ]
            if not candidates:
                return None
            candidates.sort(
                key=lambda item: (
                    _task_priority_rank(str(item.get("priority", "") or "")),
                    str(item.get("available_at", "") or ""),
                    str(item.get("created_at", "") or ""),
                    str(item.get("task_id", "") or ""),
                )
            )
            record = candidates[0]
            record["status"] = "running"
            record["attempt_count"] = int(record.get("attempt_count", 0) or 0) + 1
            record["lease_owner"] = owner
            record["lease_acquired_at"] = now
            record["last_started_at"] = now
            record["updated_at"] = now
            self._save_unlocked()
            return dict(record)

    def mark_task_succeeded(self, task_id: str, *, result_summary: str = "") -> dict[str, Any] | None:
        with self._lock:
            record = self._data["task_queue"].get(task_id)
            if not isinstance(record, dict):
                return None
            now = utc_now_iso()
            record["status"] = "done"
            record["result_summary"] = str(result_summary or "")
            record["last_error"] = ""
            record["lease_owner"] = ""
            record["lease_acquired_at"] = ""
            record["last_finished_at"] = now
            record["updated_at"] = now
            self._save_unlocked()
            return dict(record)

    def mark_task_failed(
        self,
        task_id: str,
        *,
        error: str,
        retry_delay_seconds: float = 30.0,
        permanent: bool = False,
    ) -> dict[str, Any] | None:
        with self._lock:
            record = self._data["task_queue"].get(task_id)
            if not isinstance(record, dict):
                return None
            now = utc_now_iso()
            attempt_count = int(record.get("attempt_count", 0) or 0)
            max_attempts = max(int(record.get("max_attempts", 0) or 0), 1)
            record["last_error"] = str(error or "")
            record["lease_owner"] = ""
            record["lease_acquired_at"] = ""
            record["last_finished_at"] = now
            record["updated_at"] = now
            if permanent or attempt_count >= max_attempts:
                record["status"] = "dead_letter"
                record["available_at"] = ""
                if permanent:
                    record["permanent_failure"] = True
            else:
                record["status"] = "retry_waiting"
                record["available_at"] = _shift_utc_iso(now, retry_delay_seconds)
            self._save_unlocked()
            return dict(record)

    def is_permanent_failure(self, task_id: str) -> bool:
        with self._lock:
            record = self._data["task_queue"].get(task_id)
            if not isinstance(record, dict):
                return False
            return bool(record.get("permanent_failure", False))

    def defer_task_queue_item(
        self,
        task_id: str,
        *,
        reason: str,
        delay_seconds: float = 300.0,
    ) -> dict[str, Any] | None:
        with self._lock:
            record = self._data["task_queue"].get(task_id)
            if not isinstance(record, dict):
                return None
            now = utc_now_iso()
            record["status"] = "retry_waiting"
            record["available_at"] = _shift_utc_iso(now, delay_seconds)
            record["lease_owner"] = ""
            record["lease_acquired_at"] = ""
            record["last_error"] = str(reason or "")
            record["last_finished_at"] = now
            record["updated_at"] = now
            self._save_unlocked()
            return dict(record)

    def retry_task_queue_item(self, task_id: str, *, task_type: str = "") -> dict[str, Any]:
        with self._lock:
            record = self._data["task_queue"].get(task_id)
            if not isinstance(record, dict):
                raise KeyError(task_id)
            if task_type and str(record.get("task_type", "") or "") != task_type:
                raise KeyError(task_id)
            if str(record.get("status", "") or "") == "running":
                raise RuntimeError(f"任务正在运行中，无法恢复: {task_id}")
            now = utc_now_iso()
            record["status"] = "queued"
            record["attempt_count"] = 0
            record["available_at"] = now
            record["updated_at"] = now
            record["last_error"] = ""
            record["lease_owner"] = ""
            record["lease_acquired_at"] = ""
            record.pop("permanent_failure", None)
            self._save_unlocked()
            return dict(record)

    def recover_running_tasks(self, *, task_type: str = "") -> int:
        with self._lock:
            recovered = 0
            now = utc_now_iso()
            for record in self._data["task_queue"].values():
                if not isinstance(record, dict):
                    continue
                if task_type and str(record.get("task_type", "") or "") != task_type:
                    continue
                if str(record.get("status", "") or "") != "running":
                    continue
                record["status"] = "queued"
                record["lease_owner"] = ""
                record["lease_acquired_at"] = ""
                record["available_at"] = now
                record["updated_at"] = now
                recovered += 1
            if recovered:
                self._save_unlocked()
            return recovered

    def increment_counter(self, name: str, amount: int = 1):
        with self._lock:
            counters = self._observability_unlocked()["counters"]
            counters[name] = int(counters.get(name, 0) or 0) + int(amount or 0)
            self._save_unlocked()

    def record_event(
        self,
        *,
        category: str,
        message: str,
        level: str = "info",
        source: str = "",
        target_kind: str = "",
        target_number: int = 0,
        details: dict[str, Any] | None = None,
    ):
        with self._lock:
            recent_events = self._observability_unlocked()["recent_events"]
            event = {
                "timestamp": utc_now_iso(),
                "category": str(category or "").strip(),
                "level": str(level or "info").strip().lower(),
                "message": str(message or "").strip(),
            }
            if source:
                event["source"] = str(source)
            if target_kind:
                event["target_kind"] = str(target_kind)
            if target_number > 0:
                event["target_number"] = int(target_number)
            if details:
                event["details"] = _json_safe(details)
            recent_events.append(event)
            if len(recent_events) > _RECENT_EVENT_LIMIT:
                del recent_events[:-_RECENT_EVENT_LIMIT]
            self._save_unlocked()

    def mark_cycle_started(self, cycle_name: str):
        with self._lock:
            cycle = dict(self._observability_unlocked()["cycles"].get(cycle_name, {}))
            cycle["name"] = cycle_name
            cycle["last_status"] = "running"
            cycle["last_started_at"] = utc_now_iso()
            self._observability_unlocked()["cycles"][cycle_name] = cycle
            self._save_unlocked()

    def record_cycle_result(
        self,
        cycle_name: str,
        *,
        status: str,
        stats: dict[str, Any] | None = None,
        error: str = "",
    ):
        with self._lock:
            observability = self._observability_unlocked()
            cycles = observability["cycles"]
            cycle = dict(cycles.get(cycle_name, {}))
            cycle["name"] = cycle_name
            cycle["last_status"] = str(status or "unknown")
            cycle["last_finished_at"] = utc_now_iso()
            cycle["run_count"] = int(cycle.get("run_count", 0) or 0) + 1
            cycle["last_error"] = str(error or "")
            if stats is not None:
                cycle["last_stats"] = _json_safe(stats)
            if cycle.get("last_started_at"):
                duration = _duration_seconds(str(cycle.get("last_started_at", "")), cycle["last_finished_at"])
                if duration is not None:
                    cycle["last_duration_seconds"] = duration
            if cycle["last_status"] == "succeeded":
                cycle["success_count"] = int(cycle.get("success_count", 0) or 0) + 1
                cycle["consecutive_failures"] = 0
                cycle["last_success_at"] = cycle["last_finished_at"]
            elif cycle["last_status"] == "skipped":
                cycle["skip_count"] = int(cycle.get("skip_count", 0) or 0) + 1
            else:
                cycle["failure_count"] = int(cycle.get("failure_count", 0) or 0) + 1
                cycle["consecutive_failures"] = int(cycle.get("consecutive_failures", 0) or 0) + 1
            cycles[cycle_name] = cycle
            self._save_unlocked()

    def latest_run_for_target(self, kind: str, target_number: int) -> dict[str, Any] | None:
        with self._lock:
            candidates = [
                dict(item)
                for item in self._data["runs"].values()
                if str(item.get("kind", "") or "") == kind and int(item.get("target_number", 0) or 0) == target_number
            ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda item: (
                str(item.get("finished_at", "") or ""),
                str(item.get("started_at", "") or ""),
                str(item.get("run_id", "") or ""),
            )
        )
        return candidates[-1]

    def list_failed_runs_for_target(self, kind: str, target_number: int, *, limit: int = 5) -> list[dict[str, Any]]:
        """返回指定目标最近的失败 run 记录（含 failure_reason），用于重试时提供上下文。"""
        with self._lock:
            candidates = [
                dict(item)
                for item in self._data["runs"].values()
                if (
                    str(item.get("kind", "") or "") == kind
                    and int(item.get("target_number", 0) or 0) == target_number
                    and str(item.get("status", "") or "") == "failed"
                    and str(item.get("failure_reason", "") or "").strip()
                )
            ]
        candidates.sort(
            key=lambda item: (
                str(item.get("finished_at", "") or ""),
                str(item.get("started_at", "") or ""),
            )
        )
        return candidates[-limit:]

    def _pending_tracked_pr_count_unlocked(self) -> int:
        pending_statuses = {"queued", "running", "review_feedback_pending"}
        return sum(
            1
            for item in self._data["tracked_pull_requests"].values()
            if isinstance(item, dict) and str(item.get("status", "") or "") in pending_statuses
        )

    def summary(self) -> dict[str, Any]:
        with self._lock:
            tracked = list(self._data["tracked_pull_requests"].values())
            return {
                "state_file_path": str(self.path),
                "issue_baseline": str(self._data["polling"].get("issue_created_after", "") or ""),
                "issue_baseline_id": int(self._data["polling"].get("issue_event_after_id", 0) or 0),
                "issue_comment_baseline": str(self._data["polling"].get("issue_comment_after", "") or ""),
                "issue_comment_baseline_id": int(self._data["polling"].get("issue_comment_after_id", 0) or 0),
                "planned_issues": len(self._data["issue_runs"]),
                "frozen_issue_plans": len(self._data["issue_plans"]),
                "processed_issue_events": len(self._data["issue_events"]),
                "processed_issue_comments": len(self._data["issue_comments"]),
                "processed_feedback_items": _unique_feedback_record_count(self._data["review_feedback"]),
                "paused_issues": len(self._data["paused_issues"]),
                "paused_tracked_prs": len(self._data["paused_tracked_prs"]),
                "backlog_items": len(self._data["backlog_items"]),
                "tracked_prs": len(tracked),
                "pending_tracked_prs": self._pending_tracked_pr_count_unlocked(),
                "runs": len(self._data["runs"]),
                "queued_tasks": sum(
                    1 for item in self._data["task_queue"].values() if str(item.get("status", "") or "") == "queued"
                ),
                "running_tasks": sum(
                    1 for item in self._data["task_queue"].values() if str(item.get("status", "") or "") == "running"
                ),
                "retry_waiting_tasks": sum(
                    1
                    for item in self._data["task_queue"].values()
                    if str(item.get("status", "") or "") == "retry_waiting"
                ),
                "dead_letter_tasks": sum(
                    1
                    for item in self._data["task_queue"].values()
                    if str(item.get("status", "") or "") == "dead_letter"
                ),
                "observed_cycles": len(self._observability_unlocked()["cycles"]),
                "recent_events": len(self._observability_unlocked()["recent_events"]),
            }

    def count_active_tasks(
        self,
        *,
        include_types: frozenset[str] | set[str] | None = None,
        exclude_types: frozenset[str] | set[str] | None = None,
    ) -> int:
        """Count tasks in active statuses, optionally filtered by task_type.

        Returns the number of tasks with status in {queued, running,
        retry_waiting} whose task_type matches the filter criteria.
        """
        active_statuses = {"queued", "running", "retry_waiting"}
        with self._lock:
            count = 0
            for item in self._data["task_queue"].values():
                if not isinstance(item, dict):
                    continue
                if str(item.get("status", "") or "") not in active_statuses:
                    continue
                tt = str(item.get("task_type", "") or "")
                if include_types is not None and tt not in include_types:
                    continue
                if exclude_types is not None and tt in exclude_types:
                    continue
                count += 1
            return count

    def has_active_tasks_for_target(self, target_number: int, task_type: str | None = None) -> bool:
        """Check if there are active tasks for a specific target (issue/PR).

        Returns True if any task with matching target_number (and optionally
        task_type) is in queued/running/retry_waiting status.
        """
        active_statuses = {"queued", "running", "retry_waiting"}
        with self._lock:
            for item in self._data["task_queue"].values():
                if not isinstance(item, dict):
                    continue
                if str(item.get("status", "") or "") not in active_statuses:
                    continue
                tn = item.get("target_number")
                if tn is not None and int(tn) == target_number:
                    if task_type is None or str(item.get("task_type", "") or "") == task_type:
                        return True
            return False

    def cleanup_completed_tasks(self, *, max_age_hours: int = 168, keep_recent: int = 50) -> int:
        """Remove old completed/dead-letter tasks to prevent unbounded growth.

        Keeps the most recent *keep_recent* completed tasks and removes any
        older than *max_age_hours*.  Returns the number of pruned tasks.
        """
        from datetime import datetime, timezone, timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat()
        removable_statuses = {"done", "dead_letter"}
        with self._lock:
            tasks = self._data["task_queue"]
            candidates: list[tuple[str, str]] = []
            for task_id, item in tasks.items():
                if not isinstance(item, dict):
                    continue
                if str(item.get("status", "") or "") not in removable_statuses:
                    continue
                finished = str(
                    item.get("finished_at", "")
                    or item.get("updated_at", "")
                    or item.get("available_at", "")
                    or ""
                )
                candidates.append((task_id, finished))
            candidates.sort(key=lambda x: x[1], reverse=True)
            to_remove: list[str] = []
            for i, (task_id, finished) in enumerate(candidates):
                if i < keep_recent:
                    continue
                if finished and finished < cutoff:
                    to_remove.append(task_id)
            for task_id in to_remove:
                del tasks[task_id]
            if to_remove:
                self._save_unlocked()
            return len(to_remove)

    def observability_summary(self) -> dict[str, Any]:
        with self._lock:
            observability = self._observability_unlocked()
            cycles = [
                dict(item)
                for _, item in sorted(
                    observability["cycles"].items(),
                    key=lambda entry: str(entry[0] or ""),
                )
            ]
            recent_events = [dict(item) for item in observability["recent_events"]]
            return {
                "counters": dict(sorted(observability["counters"].items(), key=lambda item: item[0])),
                "cycles": cycles,
                "recent_events": recent_events,
                "last_event_at": str(recent_events[-1].get("timestamp", "") or "") if recent_events else "",
            }

    def export_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "snapshot_version": 1,
                "exported_at": utc_now_iso(),
                "state": json.loads(json.dumps(self._data, ensure_ascii=False)),
            }

    def restore_snapshot(
        self,
        snapshot: dict[str, Any],
        *,
        create_backup: bool = True,
        recover_running_tasks: bool = True,
    ) -> dict[str, Any]:
        payload = snapshot.get("state") if isinstance(snapshot.get("state"), dict) else snapshot
        if not isinstance(payload, dict):
            raise ValueError("Invalid snapshot payload")
        with self._lock:
            backup_path = ""
            if create_backup and self.path.exists():
                backup_path = str(self._backup_path_unlocked())
                Path(backup_path).parent.mkdir(parents=True, exist_ok=True)
                Path(backup_path).write_text(
                    json.dumps(self._data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            normalized = self._normalize_data(payload)
            recovered = 0
            if recover_running_tasks:
                recovered = self._recover_running_tasks_unlocked(normalized)
            recovered_runs = self._recover_stale_runs_unlocked(normalized)
            self._data = normalized
            self._inflight_issue_keys.clear()
            self._inflight_feedback_keys.clear()
            self._inflight_issue_events.clear()
            self._inflight_issue_comments.clear()
            self._inflight_tracked_prs.clear()
            self._save_unlocked()
            summary = {
                "state_file_path": str(self.path),
                "issue_baseline": str(self._data["polling"].get("issue_created_after", "") or ""),
                "issue_baseline_id": int(self._data["polling"].get("issue_event_after_id", 0) or 0),
                "issue_comment_baseline": str(self._data["polling"].get("issue_comment_after", "") or ""),
                "issue_comment_baseline_id": int(self._data["polling"].get("issue_comment_after_id", 0) or 0),
                "planned_issues": len(self._data["issue_runs"]),
                "frozen_issue_plans": len(self._data["issue_plans"]),
                "processed_issue_events": len(self._data["issue_events"]),
                "processed_issue_comments": len(self._data["issue_comments"]),
                "processed_feedback_items": _unique_feedback_record_count(self._data["review_feedback"]),
                "paused_issues": len(self._data["paused_issues"]),
                "paused_tracked_prs": len(self._data["paused_tracked_prs"]),
                "backlog_items": len(self._data["backlog_items"]),
                "tracked_prs": len(self._data["tracked_pull_requests"]),
                "pending_tracked_prs": self._pending_tracked_pr_count_unlocked(),
                "runs": len(self._data["runs"]),
                "queued_tasks": sum(
                    1 for item in self._data["task_queue"].values() if str(item.get("status", "") or "") == "queued"
                ),
                "running_tasks": sum(
                    1 for item in self._data["task_queue"].values() if str(item.get("status", "") or "") == "running"
                ),
                "retry_waiting_tasks": sum(
                    1
                    for item in self._data["task_queue"].values()
                    if str(item.get("status", "") or "") == "retry_waiting"
                ),
                "dead_letter_tasks": sum(
                    1
                    for item in self._data["task_queue"].values()
                    if str(item.get("status", "") or "") == "dead_letter"
                ),
                "observed_cycles": len(self._observability_unlocked()["cycles"]),
                "recent_events": len(self._observability_unlocked()["recent_events"]),
            }
            return {
                "backup_path": backup_path,
                "recovered_running_tasks": recovered,
                "recovered_stale_runs": recovered_runs,
                "summary": summary,
            }

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return self._default_data()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self._default_data()
        normalized = self._normalize_data(data)
        recovered = self._recover_stale_runs_unlocked(normalized)
        if recovered > 0:
            logger.info("state-store: recovered %d stale running run record(s) on load", recovered)
        return normalized

    def _default_data(self) -> dict[str, Any]:
        return {
            "schema_version": 6,
            "polling": {
                "bootstrapped_at": "",
                "issue_created_after": "",
                "issue_event_after_id": 0,
                "issue_comment_after": "",
                "issue_comment_after_id": 0,
            },
            "issue_runs": {},
            "issue_plans": {},
            "issues": {},
            "issue_events": {},
            "issue_comments": {},
            "review_feedback": {},
            "paused_issues": {},
            "paused_tracked_prs": {},
            "backlog_items": {},
            "tracked_pull_requests": {},
            "runs": {},
            "task_queue": {},
            "observability": {
                "counters": {},
                "cycles": {},
                "recent_events": [],
            },
        }

    def _save_unlocked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._data, ensure_ascii=False, indent=2)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f"{self.path.name}.",
            suffix=".tmp",
            dir=self.path.parent,
            text=True,
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
            os.replace(tmp_path, self.path)
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    def _backup_path_unlocked(self) -> Path:
        timestamp = utc_now_iso().replace(":", "").replace("-", "")
        suffix = self.path.suffix or ".json"
        return self.path.with_name(f"{self.path.stem}.backup-{timestamp}{suffix}")

    def _observability_unlocked(self) -> dict[str, Any]:
        observability = self._data.setdefault("observability", {})
        observability.setdefault("counters", {})
        observability.setdefault("cycles", {})
        observability.setdefault("recent_events", [])
        return observability

    def _normalize_data(self, data: dict[str, Any]) -> dict[str, Any]:
        normalized = self._default_data()
        normalized["polling"].update(data.get("polling", {}))
        normalized["issue_runs"].update(data.get("issue_runs", {}))
        normalized["issue_plans"].update(data.get("issue_plans", {}))
        normalized["issues"].update(data.get("issues", {}))
        normalized["issue_events"].update(data.get("issue_events", {}))
        normalized["issue_comments"].update(data.get("issue_comments", {}))
        normalized["review_feedback"].update(data.get("review_feedback", {}))
        normalized["paused_issues"].update(data.get("paused_issues", {}))
        normalized["paused_tracked_prs"].update(data.get("paused_tracked_prs", {}))
        normalized["backlog_items"].update(data.get("backlog_items", {}))
        for record in normalized["backlog_items"].values():
            if not isinstance(record, dict):
                continue
            source_title = str(record.get("source_title", "") or record.get("title", "") or "")
            issue_title = str(record.get("issue_title", "") or record.get("title", "") or source_title)
            record["title"] = source_title
            record["source_title"] = source_title
            record["issue_title"] = issue_title
        normalized["tracked_pull_requests"].update(data.get("tracked_pull_requests", {}))
        normalized["runs"].update(data.get("runs", {}))
        normalized["task_queue"].update(data.get("task_queue", {}))
        loaded_observability = data.get("observability", {})
        if isinstance(loaded_observability, dict):
            normalized["observability"]["counters"].update(loaded_observability.get("counters", {}))
            normalized["observability"]["cycles"].update(loaded_observability.get("cycles", {}))
            recent_events = loaded_observability.get("recent_events", [])
            if isinstance(recent_events, list):
                normalized["observability"]["recent_events"].extend(recent_events[-_RECENT_EVENT_LIMIT:])
        return normalized

    def _recover_running_tasks_unlocked(self, data: dict[str, Any]) -> int:
        recovered = 0
        now = utc_now_iso()
        for record in data.get("task_queue", {}).values():
            if not isinstance(record, dict):
                continue
            if str(record.get("status", "") or "") != "running":
                continue
            record["status"] = "queued"
            record["lease_owner"] = ""
            record["lease_acquired_at"] = ""
            record["available_at"] = now
            record["updated_at"] = now
            recovered += 1
        return recovered

    def _recover_stale_runs_unlocked(self, data: dict[str, Any], max_age_hours: int = 4) -> int:
        """Mark stale 'running' run records as failed on startup."""
        recovered = 0
        now = utc_now_iso()
        now_dt = _parse_utc_iso(now)
        if now_dt is None:
            return 0
        for run_id, record in data.get("runs", {}).items():
            if not isinstance(record, dict):
                continue
            if str(record.get("status", "") or "") != "running":
                continue
            started_at = str(record.get("started_at", "") or "")
            started_dt = _parse_utc_iso(started_at)
            if started_dt is not None and (now_dt - started_dt).total_seconds() > max_age_hours * 3600:
                record["status"] = "failed"
                record["failure_reason"] = "orphaned_on_restart"
                record["finished_at"] = now
                recovered += 1
        return recovered


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _duration_seconds(started_at: str, finished_at: str) -> float | None:
    start = _parse_utc_iso(started_at)
    finish = _parse_utc_iso(finished_at)
    if start is None or finish is None:
        return None
    return max(round((finish - start).total_seconds(), 3), 0.0)


def _parse_utc_iso(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _shift_utc_iso(value: str, delta_seconds: float) -> str:
    parsed = _parse_utc_iso(value)
    if parsed is None:
        return utc_now_iso()
    return (parsed + timedelta(seconds=max(delta_seconds, 0.1))).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _is_iso_not_after(candidate: str, now: str) -> bool:
    candidate_dt = _parse_utc_iso(candidate)
    now_dt = _parse_utc_iso(now)
    if candidate_dt is not None and now_dt is not None:
        return candidate_dt <= now_dt
    return candidate <= now


def _task_priority_rank(priority: str) -> int:
    normalized = str(priority or "normal").strip().lower()
    if normalized == "urgent":
        return 0
    if normalized == "background":
        return 2
    return 1


def _merge_task_priority(existing: str, incoming: str) -> str:
    existing_normalized = str(existing or "normal").strip().lower() or "normal"
    incoming_normalized = str(incoming or "normal").strip().lower() or "normal"
    return existing_normalized if _task_priority_rank(existing_normalized) <= _task_priority_rank(incoming_normalized) else incoming_normalized


def _task_sort_key(item: dict[str, Any]) -> tuple[bool, str, str, str]:
    return (
        str(item.get("status", "") or "") != "queued",
        str(item.get("available_at", "") or ""),
        str(item.get("created_at", "") or ""),
        str(item.get("task_id", "") or ""),
    )


def _recent_string_values(record: dict[str, Any], list_key: str, legacy_key: str = "") -> list[str]:
    values: list[str] = []
    raw_values = record.get(list_key, [])
    if isinstance(raw_values, list):
        for item in raw_values:
            normalized = str(item or "").strip()
            if normalized and normalized not in values:
                values.append(normalized)
    if legacy_key:
        legacy_value = str(record.get(legacy_key, "") or "").strip()
        if legacy_value and legacy_value not in values:
            values.append(legacy_value)
    return values[-_MAX_STORED_KEYS:]


def _append_recent_value(values: list[str], value: str) -> list[str]:
    normalized = str(value or "").strip()
    if not normalized:
        return list(values[-_MAX_STORED_KEYS:])
    filtered = [item for item in values if str(item or "").strip() and str(item) != normalized]
    filtered.append(normalized)
    return filtered[-_MAX_STORED_KEYS:]


def _candidate_keys(primary_key: str, extra_keys: list[str] | None = None) -> list[str]:
    keys: list[str] = []
    for raw_key in [primary_key, *(extra_keys or [])]:
        normalized = str(raw_key or "").strip()
        if normalized and normalized not in keys:
            keys.append(normalized)
    return keys


def _unique_feedback_record_count(records: dict[str, Any]) -> int:
    canonical_keys: set[str] = set()
    for raw_key, record in records.items():
        if not isinstance(record, dict):
            continue
        canonical = str(record.get("canonical_feedback_key", "") or "").strip() or str(raw_key or "").strip()
        if canonical:
            canonical_keys.add(canonical)
    return len(canonical_keys)
