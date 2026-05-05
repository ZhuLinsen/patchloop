"""持久化任务队列。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.autocode_state_store import AutoCodeStateStore


@dataclass(frozen=True)
class QueueTask:
    task_id: str
    task_type: str
    target_number: int
    source: str
    priority: str
    command: str
    payload: dict[str, Any]
    status: str
    attempt_count: int
    max_attempts: int
    available_at: str
    lease_owner: str
    lease_acquired_at: str
    created_at: str
    updated_at: str
    last_started_at: str
    last_finished_at: str
    last_error: str
    result_summary: str

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> "QueueTask":
        return cls(
            task_id=str(record.get("task_id", "") or ""),
            task_type=str(record.get("task_type", "") or ""),
            target_number=int(record.get("target_number", 0) or 0),
            source=str(record.get("source", "") or ""),
            priority=str(record.get("priority", "normal") or "normal"),
            command=str(record.get("command", "") or ""),
            payload=dict(record.get("payload", {}) or {}),
            status=str(record.get("status", "") or ""),
            attempt_count=int(record.get("attempt_count", 0) or 0),
            max_attempts=int(record.get("max_attempts", 0) or 0),
            available_at=str(record.get("available_at", "") or ""),
            lease_owner=str(record.get("lease_owner", "") or ""),
            lease_acquired_at=str(record.get("lease_acquired_at", "") or ""),
            created_at=str(record.get("created_at", "") or ""),
            updated_at=str(record.get("updated_at", "") or ""),
            last_started_at=str(record.get("last_started_at", "") or ""),
            last_finished_at=str(record.get("last_finished_at", "") or ""),
            last_error=str(record.get("last_error", "") or ""),
            result_summary=str(record.get("result_summary", "") or ""),
        )


class TaskQueue:
    """对状态存储上的任务队列做轻量封装。"""

    def __init__(self, *, state_store: AutoCodeStateStore, task_type: str, owner: str):
        self.state_store = state_store
        self.task_type = task_type
        self.owner = owner

    def enqueue(
        self,
        *,
        task_id: str,
        target_number: int,
        source: str,
        priority: str = "normal",
        command: str = "",
        payload: dict[str, Any] | None = None,
        max_attempts: int = 3,
        replace_existing: bool = False,
    ) -> tuple[QueueTask, bool]:
        record, created = self.state_store.enqueue_task(
            task_id=task_id,
            task_type=self.task_type,
            target_number=target_number,
            source=source,
            priority=priority,
            command=command,
            payload=payload,
            max_attempts=max_attempts,
            replace_existing=replace_existing,
        )
        return QueueTask.from_record(record), created

    def claim_next(self) -> QueueTask | None:
        record = self.state_store.claim_next_task(task_type=self.task_type, owner=self.owner)
        return QueueTask.from_record(record) if record is not None else None

    def mark_succeeded(self, task_id: str, *, result_summary: str = "") -> QueueTask | None:
        record = self.state_store.mark_task_succeeded(task_id, result_summary=result_summary)
        return QueueTask.from_record(record) if record is not None else None

    def mark_failed(self, task_id: str, *, error: str, retry_delay_seconds: float = 30.0, permanent: bool = False) -> QueueTask | None:
        record = self.state_store.mark_task_failed(task_id, error=error, retry_delay_seconds=retry_delay_seconds, permanent=permanent)
        return QueueTask.from_record(record) if record is not None else None

    def is_permanent_failure(self, task_id: str) -> bool:
        return self.state_store.is_permanent_failure(task_id)

    def defer(self, task_id: str, *, reason: str, delay_seconds: float = 300.0) -> QueueTask | None:
        record = self.state_store.defer_task_queue_item(task_id, reason=reason, delay_seconds=delay_seconds)
        return QueueTask.from_record(record) if record is not None else None

    def retry_task(self, task_id: str) -> QueueTask:
        return QueueTask.from_record(self.state_store.retry_task_queue_item(task_id, task_type=self.task_type))

    def get_task(self, task_id: str) -> QueueTask | None:
        record = self.state_store.get_task(task_id)
        return QueueTask.from_record(record) if record is not None else None

    def list_tasks(self, *, statuses: set[str] | None = None, target_number: int = 0) -> list[QueueTask]:
        return [
            QueueTask.from_record(record)
            for record in self.state_store.list_tasks(
                task_type=self.task_type,
                statuses=statuses,
                target_number=target_number,
            )
        ]

    def recover_running_tasks(self) -> int:
        return self.state_store.recover_running_tasks(task_type=self.task_type)
