"""后台任务调度器。"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable


TaskHandler = Callable[[], Awaitable[None]]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class TaskLease:
    owner: str
    acquired_at: str


@dataclass(frozen=True)
class TaskRecord:
    name: str
    status: str
    priority: str
    interval_seconds: float
    run_count: int
    failure_count: int
    consecutive_failures: int
    max_retries: int
    next_run_at: str
    lease_owner: str
    lease_acquired_at: str
    last_started_at: str
    last_finished_at: str
    last_error: str
    last_success_at: str


@dataclass
class ScheduledTask:
    name: str
    handler: TaskHandler
    interval_seconds: float
    run_immediately: bool = True
    priority: str = "normal"
    max_retries: int = 3
    backoff_base_seconds: float = 30.0
    max_backoff_seconds: float = 900.0
    next_run_at: datetime | None = None
    lease: TaskLease | None = None
    run_count: int = 0
    failure_count: int = 0
    consecutive_failures: int = 0
    last_started_at: str = ""
    last_finished_at: str = ""
    last_status: str = "idle"
    last_error: str = ""
    last_success_at: str = ""

    def schedule_next_success(self, now: datetime):
        self.next_run_at = now + timedelta(seconds=max(self.interval_seconds, 0.1))

    def schedule_next_retry(self, now: datetime):
        exponent = max(self.consecutive_failures - 1, 0)
        delay = min(self.backoff_base_seconds * (2 ** exponent), self.max_backoff_seconds)
        self.next_run_at = now + timedelta(seconds=max(delay, 0.1))

    def to_record(self) -> TaskRecord:
        return TaskRecord(
            name=self.name,
            status=self.last_status,
            priority=self.priority,
            interval_seconds=self.interval_seconds,
            run_count=self.run_count,
            failure_count=self.failure_count,
            consecutive_failures=self.consecutive_failures,
            max_retries=self.max_retries,
            next_run_at=_format_dt(self.next_run_at),
            lease_owner=self.lease.owner if self.lease else "",
            lease_acquired_at=self.lease.acquired_at if self.lease else "",
            last_started_at=self.last_started_at,
            last_finished_at=self.last_finished_at,
            last_error=self.last_error,
            last_success_at=self.last_success_at,
        )


class TaskScheduler:
    """按固定周期执行后台任务。

    不同 cycle 允许并发运行，但同一个 cycle 仍通过 lease 防止重入。
    """

    def __init__(
        self,
        *,
        logger: logging.Logger | None = None,
        owner: str = "task-scheduler",
        stop_wait_seconds: float = 30.0,
    ):
        self.logger = logger or logging.getLogger(__name__)
        self.owner = owner
        self.stop_wait_seconds = max(stop_wait_seconds, 0.0)
        self._tasks: dict[str, ScheduledTask] = {}
        self._loop_task: asyncio.Task | None = None
        self._active_runs: set[asyncio.Task] = set()
        self._stopping = False

    def register_task(
        self,
        *,
        name: str,
        handler: TaskHandler,
        interval_seconds: float,
        run_immediately: bool = True,
        priority: str = "normal",
        max_retries: int = 3,
        backoff_base_seconds: float = 30.0,
        max_backoff_seconds: float = 900.0,
    ):
        if name in self._tasks:
            raise ValueError(f"任务已存在: {name}")
        now = datetime.now(timezone.utc)
        task = ScheduledTask(
            name=name,
            handler=handler,
            interval_seconds=max(interval_seconds, 0.1),
            run_immediately=run_immediately,
            priority=priority,
            max_retries=max(max_retries, 0),
            backoff_base_seconds=max(backoff_base_seconds, 0.1),
            max_backoff_seconds=max(max_backoff_seconds, 0.1),
        )
        task.next_run_at = now if run_immediately else now + timedelta(seconds=task.interval_seconds)
        self._tasks[name] = task

    def start(self) -> asyncio.Task:
        if self._loop_task is not None and not self._loop_task.done():
            return self._loop_task
        self._stopping = False
        self._loop_task = asyncio.create_task(self._run_loop(), name="autocode-task-scheduler")
        return self._loop_task

    async def stop(self):
        self._stopping = True
        if self._loop_task is not None:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
            self._loop_task = None
        if self._active_runs:
            active_runs = list(self._active_runs)
            # 先给已启动的后台任务一个自然收尾窗口，尽量避免 to_thread 中的
            # 底层线程在事件循环关闭后仍尝试回调；若超时，再显式取消剩余任务，
            # 防止 shutdown 被长期阻塞。
            done, pending = await asyncio.wait(active_runs, timeout=self.stop_wait_seconds)
            if pending:
                self.logger.warning(
                    "后台调度任务停机等待超时，取消剩余任务: pending=%s timeout=%.1fs",
                    len(pending),
                    self.stop_wait_seconds,
                )
                for run in pending:
                    run.cancel()
            await asyncio.gather(*active_runs, return_exceptions=True)
            self._active_runs.clear()

    async def run_pending(self):
        now = datetime.now(timezone.utc)
        due_tasks = [
            task
            for task in self._tasks.values()
            if task.next_run_at is not None and task.next_run_at <= now and task.last_status != "dead_letter"
        ]
        due_tasks.sort(key=lambda item: (_priority_rank(item.priority), item.next_run_at or now, item.name))
        started_any = False
        for task in due_tasks:
            started_any = self._start_task(task) or started_any
        if started_any:
            # 让刚启动的短任务有机会在当前 tick 内完成，保持现有可观测行为。
            await asyncio.sleep(0)

    def summary(self) -> dict:
        records = [task.to_record() for task in sorted(self._tasks.values(), key=lambda item: item.name)]
        tasks = [record.__dict__.copy() for record in records]
        return {
            "registered": len(tasks),
            "running": sum(1 for item in tasks if item["status"] == "running"),
            "failed": sum(1 for item in tasks if item["status"] == "failed"),
            "retry_waiting": sum(1 for item in tasks if item["status"] == "retry_waiting"),
            "dead_letter": sum(1 for item in tasks if item["status"] == "dead_letter"),
            "tasks": tasks,
        }

    def get_task_record(self, name: str) -> TaskRecord | None:
        task = self._tasks.get(name)
        return task.to_record() if task is not None else None

    def retry_task(self, name: str) -> TaskRecord:
        task = self._tasks.get(name)
        if task is None:
            raise KeyError(name)
        if task.lease is not None:
            raise RuntimeError(f"任务正在运行中，无法恢复: {name}")
        task.consecutive_failures = 0
        task.last_status = "queued"
        task.last_error = ""
        task.next_run_at = datetime.now(timezone.utc)
        return task.to_record()

    async def _run_loop(self):
        while not self._stopping:
            await self.run_pending()
            await asyncio.sleep(self._sleep_seconds())

    def _start_task(self, task: ScheduledTask) -> bool:
        if task.lease is not None or task.last_status == "dead_letter":
            return False
        now = datetime.now(timezone.utc)
        task.lease = TaskLease(owner=self.owner, acquired_at=utc_now_iso())
        task.run_count += 1
        task.last_started_at = utc_now_iso()
        task.last_status = "running"
        task.last_error = ""
        runner = asyncio.create_task(self._run_task(task, started_at=now), name=f"autocode-scheduled-{task.name}")
        self._active_runs.add(runner)
        runner.add_done_callback(self._finalize_active_run)
        return True

    async def _run_task(self, task: ScheduledTask, *, started_at: datetime):
        try:
            await task.handler()
        except asyncio.CancelledError:
            task.last_status = "cancelled"
            task.last_finished_at = utc_now_iso()
            task.schedule_next_success(started_at)
            raise
        except Exception as exc:
            task.failure_count += 1
            task.consecutive_failures += 1
            task.last_error = str(exc)
            task.last_finished_at = utc_now_iso()
            if task.consecutive_failures > task.max_retries:
                task.last_status = "dead_letter"
                task.next_run_at = None
            else:
                task.last_status = "retry_waiting"
                task.schedule_next_retry(datetime.now(timezone.utc))
            self.logger.exception("后台调度任务失败 %s: %s", task.name, exc)
        else:
            task.consecutive_failures = 0
            task.last_status = "succeeded"
            task.last_finished_at = utc_now_iso()
            task.last_success_at = task.last_finished_at
            task.schedule_next_success(datetime.now(timezone.utc))
        finally:
            task.lease = None

    def _finalize_active_run(self, run: asyncio.Task):
        self._active_runs.discard(run)
        try:
            run.result()
        except asyncio.CancelledError:
            pass
        except Exception:
            self.logger.exception("后台调度任务收尾失败")

    def _sleep_seconds(self) -> float:
        next_due = min(
            (
                task.next_run_at
                for task in self._tasks.values()
                if task.next_run_at is not None and task.last_status != "dead_letter"
            ),
            default=None,
        )
        if next_due is None:
            return 1.0
        remaining = (next_due - datetime.now(timezone.utc)).total_seconds()
        if remaining <= 0:
            return 0.05
        return min(max(remaining, 0.05), 1.0)


def _format_dt(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _priority_rank(priority: str) -> int:
    normalized = str(priority or "normal").strip().lower()
    if normalized == "urgent":
        return 0
    if normalized == "background":
        return 2
    return 1
