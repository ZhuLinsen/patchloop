"""
AutoCode 入口

功能:
1. 支持 GitHub Webhook 与 polling 双模式接入
2. 为 bug/feature 生成计划，并按策略执行自动编码
3. 严格限制发布权限：只创建 PR，禁止直接向 main/默认分支推代码，禁止 merge
4. 定期扫描受跟踪 PR 的检视意见，并按队列串行尝试修复
"""
from __future__ import annotations

import asyncio
from collections import Counter
import hashlib
import hmac
import json
import logging
import re
import sys
import time
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn

from adapters import create_adapter_from_profiles
from adapters.base import LLMError
from agent.backlog_source import BacklogSource
from agent.context_builder import LocalContextBuilder
from agent.idle_scanner import IdleScanner
from agent.markdown_plan_source import MarkdownPlanSource
from agent.autocode_state_store import AutoCodeStateStore
from agent.execution_controller import ExecutionController
from agent.execution_policy import ExecutionPolicyEngine
from agent.github_client import GitHubClient
from agent.github_write_client import GitHubWriteClient
from agent.issue_comment_intent import (
    is_repo_owner_human_issue_comment,
    issue_comment_requests_implementation,
)
from agent.json_utils import extract_json_object
from agent.patch_inspector import PatchInspector
from agent.task_queue import TaskQueue
from agent.task_scheduler import TaskScheduler
from agent.validation_runner import ValidationRunner
from agent.workspace_manager import WorkspaceManager
from config import AppConfig, load_autocode_config
from prompts import build_source_issue_prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("autocode")

config: AppConfig | None = None
gh: GitHubClient | None = None
gh_writer: GitHubWriteClient | None = None
controller: ExecutionController | None = None
state_store: AutoCodeStateStore | None = None
backlog_source: BacklogSource | None = None
plan_source: MarkdownPlanSource | None = None
idle_scanner: IdleScanner | None = None
scheduler: TaskScheduler | None = None
scheduler_task: asyncio.Task | None = None
issue_task_queue: TaskQueue | None = None
pr_task_queue: TaskQueue | None = None
source_task_queue: TaskQueue | None = None
_source_issue_check_cache: dict[int, float] = {}
_SOURCE_ISSUE_CHECK_COOLDOWN_SECONDS = 3600.0
# Issue outcomes that mean "done, don't re-process this version".
_ISSUE_TERMINAL_SUCCESS_OUTCOMES = frozenset({
    "pr_ready", "pr_opened", "pr_updated", "done", "dry_run", "answered", "manual_action", "resolved_by_user",
})
# Outcomes that reconcile should NOT auto-retry.  "blocked" needs explicit
# human approval (POST /issues/{N}/implement); "failed" typically means a
# structural problem (blocked paths, size limits, test failures) that won't
# resolve with a simple retry.
_ISSUE_RECONCILE_SKIP_OUTCOMES = frozenset({
    "blocked", "failed", "planned", "answered", "manual_action", "resolved_by_user",
})
# Scheduler cycle names that represent routine PR maintenance work and should
# NOT prevent idle-scan from running.  Idle-scan's intent is "no pending
# *issue* work", not "no PR repair in progress".
_IDLE_SCAN_EXCLUDED_CYCLES = frozenset({
    "idle-scan-cycle",
    "pr-task-cycle",
    "pr-queue-cycle",
})
_ISSUE_CLUSTER_RULES = (
    ("llm_model_config", ("模型", "deepseek", "kimi", "minimax", "openai/", "litellm", "temperature", "ollama")),
    ("desktop_ui", ("桌面", "windows", "黑屏", "白屏", "设置", "通知", "agent")),
    ("portfolio_data", ("持仓", "筹码", "业绩", "个股", "大盘", "复盘", "数据缺失")),
    ("data_cache", ("cache", "缓存", "过期", "负缓存", "stock_history_cache")),
    ("webhook_docs", ("webhook", "飞书", "钉钉", "通知", "文档")),
)

_SOURCE_ISSUE_SKIP_MARKERS = (
    "不成立",
    "未坐实",
    "没有坐实",
    "not fully validated",
    "不应直接当作已证实",
)
_SOURCE_ISSUE_ALLOWED_LABELS = {"autocode", "bug", "enhancement", "documentation", "question", "invalid"}
_SOURCE_ISSUE_PRIMARY_LABELS = ("bug", "enhancement", "documentation", "question")
_SOURCE_ISSUE_TITLE_PREFIX = {
    "bug": "[Bug]",
    "enhancement": "[Feature]",
    "documentation": "[Docs]",
    "question": "[Question]",
    "invalid": "[Invalid]",
}
_SOURCE_ISSUE_VAGUE_TITLE_MARKERS = (
    "优化一下",
    "补一下",
    "查看一下",
    "看一下",
    "处理一下",
    "完善一下",
    "改一下",
    "调整一下",
    "检查一下",
    "优化",
    "补充",
    "完善",
    "处理",
    "调整",
    "排查一下",
)
_SOURCE_ISSUE_BODY_METADATA_MARKERS = (
    "item key",
    "source file",
    "line number",
    "raw title",
    "raw body",
    "raw labels",
)
_SOURCE_ISSUE_EMERGENCY_MARKERS = (
    "p0",
    "sev1",
    "urgent",
    "critical",
    "blocker",
    "blocking",
    "紧急",
    "阻塞",
    "高危",
    "致命",
)
_SOURCE_ISSUE_SECURITY_MARKERS = (
    "安全",
    "漏洞",
    "缺少签名验证",
    "签名验证",
    "验签",
    "signature",
    "authentication",
    "authorization",
    "auth",
    "鉴权",
    "认证",
    "授权",
    "权限校验",
    "权限控制",
    "越权",
    "伪造",
    "重放",
    "csrf",
    "xss",
    "ssrf",
    "sql 注入",
    "sqli",
    "注入",
)
_SOURCE_ISSUE_BUG_MARKERS = (
    "错误",
    "失败",
    "异常",
    "泄露",
    "泄漏",
    "误判",
    "超时",
    "并发",
    "竞争",
    "锁冲突",
    "回归",
    "bug",
)
_SOURCE_ISSUE_ENHANCEMENT_BLOCK_MARKERS = (
    *_SOURCE_ISSUE_BUG_MARKERS,
    "缺少",
    "风险",
    "锁",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, gh, gh_writer, controller, state_store, backlog_source, plan_source, idle_scanner, scheduler, scheduler_task, issue_task_queue, pr_task_queue, source_task_queue

    logger.info("=== AutoCode 启动中 ===")
    config = load_autocode_config()
    logging.getLogger().setLevel(config.server.log_level)

    planner = create_adapter_from_profiles(
        cli_profiles=config.cli.cli_profiles,
        primary_cli=config.cli.primary_cli,
        fallback_clis=config.cli.fallback_clis,
        cli_timeout=config.cli.planner_cli_timeout,
        working_dir=config.local_repo.path,
    )
    executor = create_adapter_from_profiles(
        cli_profiles=config.cli.cli_profiles,
        primary_cli=config.cli.primary_cli,
        fallback_clis=config.cli.fallback_clis,
        cli_timeout=config.cli.executor_cli_timeout,
        working_dir=config.local_repo.path,
        execution_mode=True,
    )

    gh = GitHubClient(
        token=config.github.token,
        owner=config.github.owner,
        repo=config.github.repo_name,
        dry_run=config.server.dry_run,
    )
    gh_writer = GitHubWriteClient(
        token=config.github.write_token or config.github.token,
        owner=config.github.owner,
        repo=config.github.repo_name,
        dry_run=config.server.dry_run,
        force_draft_pr=config.autocode.force_draft_pr,
        forbid_main_base_branch=config.autocode.forbid_main_base_branch,
        default_branch=config.local_repo.default_branch,
    )
    state_store = AutoCodeStateStore(config.autocode.state_file_path)
    issue_task_queue = TaskQueue(
        state_store=state_store,
        task_type="issue_execution",
        owner="issue-task-queue",
    )
    pr_task_queue = TaskQueue(
        state_store=state_store,
        task_type="pr_feedback_repair",
        owner="pr-task-queue",
    )
    source_task_queue = TaskQueue(
        state_store=state_store,
        task_type="source_issue_create",
        owner="source-task-queue",
    )
    backlog_source = BacklogSource(
        repo_path=config.autocode.backlog_repo_path,
        file_paths=config.autocode.backlog_file_paths,
        default_labels=config.autocode.backlog_labels,
        title_prefix=config.autocode.backlog_title_prefix,
        max_items_per_cycle=config.autocode.backlog_max_items_per_cycle,
    )
    plan_source = MarkdownPlanSource(
        repo_path=config.local_repo.path,
        file_paths=config.autocode.plan_file_paths,
        default_labels=config.autocode.plan_labels,
        title_prefix=config.autocode.plan_title_prefix,
        max_items_per_cycle=config.autocode.plan_max_items_per_cycle,
    )
    idle_scanner = IdleScanner(
        repo_path=config.local_repo.path,
        default_labels=config.autocode.idle_scanner_labels,
        title_prefix=config.autocode.idle_scanner_title_prefix,
        max_items_per_cycle=config.autocode.idle_scanner_max_items_per_cycle,
        analyzer=planner,
        context_builder=LocalContextBuilder(
            repo_path=config.local_repo.path,
            max_context_files=config.autocode.idle_scanner_llm_max_context_files,
            max_file_chars=config.local_repo.max_file_chars,
            max_total_chars=config.local_repo.max_total_context_chars,
        ),
        repo_name=config.github.repo,
        enable_heuristic_analysis=config.autocode.idle_scanner_enable_heuristic_analysis,
        enable_llm_enrichment=config.autocode.idle_scanner_enable_llm_enrichment,
        advanced_only_when_todo_clear=config.autocode.idle_scanner_advanced_only_when_todo_clear,
        long_function_lines=config.autocode.idle_scanner_long_function_lines,
        llm_max_context_files=config.autocode.idle_scanner_llm_max_context_files,
    )
    workspace_manager = WorkspaceManager(
        repo_path=config.local_repo.path,
        worktree_root=config.autocode.worktree_root,
        publish_remote=config.autocode.publish_remote,
        branch_prefix=config.autocode.branch_prefix,
        default_branch=config.local_repo.default_branch,
        commit_user_name=config.autocode.commit_user_name,
        commit_user_email=config.autocode.commit_user_email,
        git_auth_login=config.github.notify_login or gh_writer.get_authenticated_login(),
        git_auth_token=config.github.write_token or config.github.token,
    )
    validation_runner = ValidationRunner(
        format_commands=config.autocode.format_commands,
        lint_commands=config.autocode.lint_commands,
        test_commands=config.autocode.test_commands,
        timeout_seconds=config.autocode.timeout_seconds,
        test_timeout_seconds=config.autocode.test_timeout_seconds,
        fail_fast=config.autocode.validation_fail_fast,
        transient_artifact_dirs=config.autocode.validation_transient_artifact_dirs,
    )
    policy_engine = ExecutionPolicyEngine(config.autocode, config.local_repo.default_branch)
    patch_inspector = PatchInspector(
        policy_engine=policy_engine,
        max_changed_files=config.autocode.max_changed_files,
        max_added_lines=config.autocode.max_added_lines,
        max_deleted_lines=config.autocode.max_deleted_lines,
        documentation_paths=config.autocode.documentation_scope_paths,
        documentation_extensions=config.autocode.documentation_scope_extensions,
    )
    controller = ExecutionController(
        planner=planner,
        executor=executor,
        gh=gh,
        gh_writer=gh_writer,
        state_store=state_store,
        workspace_manager=workspace_manager,
        validation_runner=validation_runner,
        patch_inspector=patch_inspector,
        policy_engine=policy_engine,
        config=config,
    )

    recovered_tasks = issue_task_queue.recover_running_tasks()
    if recovered_tasks > 0:
        logger.warning("issue-task-queue: recovered %d stale running task(s)", recovered_tasks)
        _record_state_event(
            category="task_queue_recovery",
            level="warning",
            message=f"Recovered {recovered_tasks} stale issue task(s) on startup",
            details={"recovered_tasks": recovered_tasks},
        )
    recovered_pr_tasks = pr_task_queue.recover_running_tasks()
    if recovered_pr_tasks > 0:
        logger.warning("pr-task-queue: recovered %d stale running task(s)", recovered_pr_tasks)
        _record_state_event(
            category="task_queue_recovery",
            level="warning",
            message=f"Recovered {recovered_pr_tasks} stale PR task(s) on startup",
            details={"recovered_tasks": recovered_pr_tasks},
        )
    recovered_source_tasks = source_task_queue.recover_running_tasks()
    if recovered_source_tasks > 0:
        logger.warning("source-task-queue: recovered %d stale running task(s)", recovered_source_tasks)
        _record_state_event(
            category="task_queue_recovery",
            level="warning",
            message=f"Recovered {recovered_source_tasks} stale source task(s) on startup",
            details={"recovered_tasks": recovered_source_tasks},
        )
    recovered_dead_letters = _auto_requeue_recoverable_dead_letters(
        limit=config.autocode.dead_letter_auto_requeue_limit,
    )
    if recovered_dead_letters:
        logger.warning("task-queue: requeued %d recoverable dead-letter task(s)", recovered_dead_letters)

    scheduler = _build_background_scheduler()
    if scheduler.summary()["registered"] > 0:
        scheduler_task = scheduler.start()

    logger.info(
        "=== AutoCode 就绪 === repo=%s event_source=%s webhook=%s pr_queue=%s draft=%s base=%s dry_run=%s",
        config.github.repo,
        config.server.event_source,
        config.server.enable_webhook,
        config.autocode.enable_pr_queue,
        config.autocode.force_draft_pr,
        config.autocode.pr_base_branch,
        config.server.dry_run,
    )
    yield

    if scheduler is not None:
        await scheduler.stop()
        scheduler = None
    plan_source = None
    idle_scanner = None
    issue_task_queue = None
    pr_task_queue = None
    source_task_queue = None
    scheduler_task = None
    if gh is not None:
        gh.close()
    if gh_writer is not None:
        gh_writer.close()
    logger.info("=== AutoCode 关闭 ===")


app = FastAPI(title="AutoCode", lifespan=lifespan)


def verify_signature(payload_body: bytes, signature_header: str | None, secret: str) -> bool:
    if not signature_header:
        return False
    expected = "sha256=" + hmac.new(secret.encode("utf-8"), payload_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header)


@app.post("/webhook")
async def github_webhook(request: Request):
    assert config and controller and gh and state_store, "应用未初始化"
    if not config.server.enable_webhook:
        raise HTTPException(status_code=503, detail="Webhook is disabled")

    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")
    if not verify_signature(body, signature, config.github.webhook_secret):
        _increment_state_counter("webhook.invalid_signature")
        _record_state_event(
            category="webhook",
            level="warning",
            message="Rejected webhook because signature verification failed",
        )
        raise HTTPException(status_code=403, detail="Invalid signature")

    event_type = request.headers.get("X-GitHub-Event", "")
    content_type = request.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    _increment_state_counter("webhook.received")
    if content_type == "application/json":
        data = await request.json()
    elif content_type == "application/x-www-form-urlencoded":
        form = await request.form()
        payload = form.get("payload")
        if not isinstance(payload, str):
            _increment_state_counter("webhook.failed")
            _record_state_event(
                category="webhook",
                level="error",
                message="Rejected webhook form payload because payload field is missing",
                details={"event_type": event_type, "content_type": content_type},
            )
            raise HTTPException(status_code=400, detail="Missing payload form field")
        data = json.loads(payload)
    else:
        _increment_state_counter("webhook.failed")
        _record_state_event(
            category="webhook",
            level="error",
            message="Rejected webhook because content type is unsupported",
            details={"event_type": event_type, "content_type": content_type or "unknown"},
        )
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {content_type or 'unknown'}")

    action = data.get("action", "")
    try:
        if event_type == "issues" and action in {"opened", "reopened"}:
            await _handle_issue_event(data["issue"], source="webhook")
        elif event_type == "pull_request" and action in {
            "opened",
            "reopened",
            "synchronize",
            "ready_for_review",
            "converted_to_draft",
            "closed",
        }:
            await _handle_pull_request_event(data["pull_request"], source="webhook", action=action)
        elif event_type == "pull_request_review" and action in {"submitted", "edited", "dismissed"}:
            await _handle_pull_request_event(data["pull_request"], source="webhook", action=action)
        elif event_type == "pull_request_review_comment" and action in {"created", "edited"}:
            await _handle_pull_request_event(data["pull_request"], source="webhook", action=action)
        elif event_type == "issue_comment" and action in {"created", "edited"}:
            await _handle_issue_comment_event(data, source="webhook")
        elif event_type == "ping":
            logger.info("收到 ping 事件")
    except Exception as exc:
        _increment_state_counter("webhook.failed")
        _record_state_event(
            category="webhook",
            level="error",
            message=f"Webhook processing failed for {event_type or 'unknown'}:{action or 'n/a'}",
            details={"event_type": event_type, "action": action, "error": str(exc)},
        )
        logger.exception("处理 AutoCode webhook 失败: %s", exc)
        return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})
    _increment_state_counter("webhook.processed")
    _record_state_event(
        category="webhook",
        message=f"Processed webhook {event_type or 'unknown'}:{action or 'n/a'}",
        details={"event_type": event_type, "action": action},
    )
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "repo": config.github.repo if config else "not initialized",
        "event_source": config.server.event_source if config else "",
        "webhook_enabled": config.server.enable_webhook if config else False,
        "execution_enabled": config.autocode.enabled if config else False,
        "execution_mode": config.autocode.mode if config else "",
        "execution_pr_base_branch": config.autocode.pr_base_branch if config else "",
        "forbid_main_base_branch": config.autocode.forbid_main_base_branch if config else False,
        "force_draft_pr": config.autocode.force_draft_pr if config else False,
        "pr_queue_enabled": config.autocode.enable_pr_queue if config else False,
        "review_feedback_polling": config.autocode.poll_review_feedback if config else False,
        "plan_source_enabled": config.autocode.enable_plan_source if config else False,
        "backlog_source_enabled": config.autocode.enable_backlog_source if config else False,
        "idle_scanner_enabled": config.autocode.enable_idle_scanner if config else False,
        "scheduler": scheduler.summary() if scheduler else {},
        "issue_task_queue": _issue_task_queue_summary(),
        "pr_task_queue": _pr_task_queue_summary(),
        "source_task_queue": _source_task_queue_summary(),
        "state": state_store.summary() if state_store else {},
        "observability": state_store.observability_summary() if state_store else {},
    }


@app.get("/observability")
async def observability():
    return {
        "status": "ok",
        "repo": config.github.repo if config else "not initialized",
        "scheduler": scheduler.summary() if scheduler else {},
        "issue_task_queue": _issue_task_queue_summary(),
        "pr_task_queue": _pr_task_queue_summary(),
        "source_task_queue": _source_task_queue_summary(),
        "state": state_store.summary() if state_store else {},
        "observability": state_store.observability_summary() if state_store else {},
    }


@app.get("/state/snapshot")
async def export_state_snapshot():
    if state_store is None:
        raise HTTPException(status_code=503, detail="State store is not initialized")
    snapshot = state_store.export_snapshot()
    return {
        "status": "ok",
        "snapshot": snapshot,
        "summary": state_store.summary(),
    }


@app.post("/state/restore")
async def restore_state_snapshot(request: Request):
    global scheduler_task
    if state_store is None:
        raise HTTPException(status_code=503, detail="State store is not initialized")
    try:
        payload = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON restore payload") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid restore payload")
    if not bool(payload.get("confirm_restore", False)):
        raise HTTPException(status_code=400, detail="confirm_restore=true is required")
    if _has_active_background_work():
        raise HTTPException(status_code=409, detail="Active background work is running; restore is not allowed")
    snapshot = payload.get("snapshot", payload)
    if not isinstance(snapshot, dict):
        raise HTTPException(status_code=400, detail="Invalid snapshot payload")
    restarted_scheduler = bool(scheduler is not None and scheduler_task is not None and not scheduler_task.done())
    if restarted_scheduler and scheduler is not None:
        await scheduler.stop()
    try:
        result = state_store.restore_snapshot(snapshot)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if restarted_scheduler and scheduler is not None:
            scheduler_task = scheduler.start()
    _increment_state_counter("state.restore")
    _record_state_event(
        category="state_restore",
        message="Restored state snapshot",
        details={
            "backup_path": result.get("backup_path", ""),
            "recovered_running_tasks": result.get("recovered_running_tasks", 0),
        },
    )
    return {
        "status": "ok",
        **result,
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    return PlainTextResponse(_render_metrics(), media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/logs/events")
async def list_log_events(category: str = "", level: str = "", source: str = "", limit: int = 50):
    _validate_task_limit(limit)
    levels = _parse_event_filter(level, allowed={"info", "warning", "error"})
    categories = _parse_event_filter(category)
    sources = _parse_event_filter(source)
    events = _filter_recent_events(categories=categories, levels=levels, sources=sources, limit=limit)
    return {
        "status": "ok",
        "filters": {
            "category": categories or [],
            "level": levels or [],
            "source": sources or [],
            "limit": limit,
        },
        "count": len(events),
        "events": events,
    }


@app.get("/tasks")
async def list_all_tasks():
    queue_summaries = _all_task_queue_summaries()
    return {
        "status": "ok",
        "summary": _aggregate_task_queue_summary(queue_summaries),
        "queues": queue_summaries,
        "active_tasks": _collect_task_items(statuses={"queued", "running", "retry_waiting"}),
        "problem_tasks": _collect_task_items(statuses={"retry_waiting", "dead_letter"}, include_superseded=False),
    }


@app.get("/tasks/dead-letter")
async def export_dead_letter_tasks(queue: str = "", limit: int = 200, include_superseded: bool = True):
    queue_names = _parse_task_queue_names(queue)
    _validate_task_limit(limit)
    tasks = _collect_task_items(
        statuses={"dead_letter"},
        queue_names=queue_names,
        limit=limit,
        include_superseded=include_superseded,
    )
    return {
        "status": "ok",
        "filters": {
            "queue": queue_names or ["issue", "pr", "source"],
            "status": ["dead_letter"],
            "limit": limit,
            "include_superseded": include_superseded,
        },
        "count": len(tasks),
        "action_required_count": sum(1 for task in tasks if bool(task.get("action_required", False))),
        "superseded_count": sum(1 for task in tasks if bool(task.get("superseded", False))),
        "tasks": tasks,
    }


@app.post("/tasks/retry")
async def retry_tasks(queue: str = "", status: str = "retry_waiting,dead_letter", limit: int = 20):
    queue_names = _parse_task_queue_names(queue)
    statuses = _parse_task_statuses(
        status,
        default={"retry_waiting", "dead_letter"},
        allowed={"retry_waiting", "dead_letter"},
    )
    _validate_task_limit(limit)
    matched = _collect_task_items(statuses=statuses, queue_names=queue_names)
    retried = []
    for item in matched[:limit]:
        task = _retry_queue_task(item["queue"], item["task_id"])
        retried.append({"queue": item["queue"], "task": task})
    _increment_state_counter("tasks.batch_retry")
    _record_state_event(
        category="task_operations",
        message=f"Manually re-queued {len(retried)} queued task items",
        details={
            "queues": queue_names or ["issue", "pr", "source"],
            "statuses": sorted(statuses),
            "matched": len(matched),
            "retried": len(retried),
            "limit": limit,
        },
    )
    return {
        "status": "ok",
        "filters": {
            "queue": queue_names or ["issue", "pr", "source"],
            "status": sorted(statuses),
            "limit": limit,
        },
        "matched": len(matched),
        "retried": retried,
    }


def _auto_requeue_recoverable_dead_letters(*, limit: int = 5) -> int:
    """On startup, replay narrow dead-letter classes fixed by newer guard code."""
    if state_store is None or issue_task_queue is None:
        return 0
    if limit <= 0:
        return 0
    markers = list(getattr(config.autocode, "recoverable_dead_letter_markers", []) if config else [])
    if not markers:
        return 0
    try:
        candidates = _collect_task_items(
            statuses={"dead_letter"},
            queue_names=["issue"],
            limit=max(limit * 4, limit),
            include_superseded=False,
        )
    except Exception:
        logger.exception("task-queue: failed to collect recoverable dead-letter tasks")
        return 0

    requeued = 0
    for item in candidates:
        if requeued >= limit:
            break
        task_id = str(item.get("task_id", "") or "")
        if not task_id or issue_task_queue.is_permanent_failure(task_id):
            continue
        reason_text = " ".join(
            str(item.get(key, "") or "")
            for key in ("last_error", "result_summary", "failure_reason", "summary")
        ).lower()
        if not any(marker.lower() in reason_text for marker in markers):
            continue
        try:
            task = _retry_queue_task("issue", task_id)
        except Exception:
            logger.exception("task-queue: failed to auto requeue dead-letter task=%s", task_id)
            continue
        requeued += 1
        _record_state_event(
            category="issue_task_queue_recovery",
            level="warning",
            message=f"Auto re-queued recoverable dead-letter issue task {task_id}",
            target_kind="issue",
            target_number=int(task.get("target_number", 0) or 0),
            details={
                "queue": "issue",
                "task_id": task_id,
                "reason": "recoverable_dead_letter",
            },
        )
    return requeued


@app.post("/scheduler/tasks/{task_name}/retry")
async def retry_scheduler_task(task_name: str):
    if scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler is not initialized")
    previous_record = scheduler.get_task_record(task_name)
    try:
        record = scheduler.retry_task(task_name)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown scheduler task: {task_name}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    _increment_state_counter("scheduler.manual_retry")
    _record_state_event(
        category="scheduler_recovery",
        message=f"Manually re-queued scheduler task {task_name}",
        details={
            "task_name": task_name,
            "previous_status": previous_record.status if previous_record is not None else "",
            "next_run_at": record.next_run_at,
        },
    )
    return {
        "status": "ok",
        "task": record.__dict__.copy(),
    }


@app.get("/tasks/issue")
async def list_issue_tasks(status: str = ""):
    return {
        "status": "ok",
        "tasks": _list_queue_tasks("issue", status=status),
    }


@app.post("/tasks/issue/{task_id}/retry")
async def retry_issue_task(task_id: str):
    return {"status": "ok", "task": _retry_queue_task("issue", task_id)}


@app.post("/issues/{issue_number}/implement")
async def implement_issue(issue_number: int):
    if issue_number <= 0:
        raise HTTPException(status_code=400, detail="Issue number must be greater than 0")
    if gh is None or state_store is None or issue_task_queue is None or config is None:
        raise HTTPException(status_code=503, detail="Issue execution is not initialized")
    issue = await asyncio.to_thread(gh.get_issue, issue_number)
    if "pull_request" in issue:
        raise HTTPException(status_code=400, detail=f"#{issue_number} is a pull request, not an issue")
    if str(issue.get("state", "") or "").lower() != "open":
        raise HTTPException(status_code=409, detail=f"Issue #{issue_number} is not open")

    issue_updated_at = str(issue.get("updated_at", "") or "")
    was_paused = state_store.is_issue_paused(issue_number)
    if was_paused:
        state_store.set_issue_paused(issue_number, paused=False)

    task, created = issue_task_queue.enqueue(
        task_id=_build_manual_issue_task_id(issue_number, issue_updated_at),
        target_number=issue_number,
        source="api",
        priority="urgent",
        command="implement",
        payload={"explicit_command": True, "issue_updated_at": issue_updated_at},
        max_attempts=getattr(config.autocode, "issue_task_max_attempts", 3),
        replace_existing=True,
    )
    _increment_state_counter("issue_task_queue.manual_implement")
    _record_state_event(
        category="issue_task_queue",
        message=f"Queued explicit issue execution for issue #{issue_number}",
        source="api",
        target_kind="issue",
        target_number=issue_number,
        details={
            "task_id": task.task_id,
            "created": created,
            "status": task.status,
            "was_paused": was_paused,
        },
    )
    payload = _issue_task_to_dict(task)
    payload["queue"] = "issue"
    return {
        "status": "ok",
        "created": created,
        "resumed": was_paused,
        "task": payload,
    }


@app.post("/pulls/{pr_number}/recheck")
async def recheck_pull_request_feedback(pr_number: int):
    if pr_number <= 0:
        raise HTTPException(status_code=400, detail="Pull request number must be greater than 0")
    if gh is None or state_store is None or pr_task_queue is None or controller is None or config is None:
        raise HTTPException(status_code=503, detail="PR repair is not initialized")
    pr = await asyncio.to_thread(gh.get_pr, pr_number)
    if str(pr.get("state", "") or "").lower() == "closed":
        raise HTTPException(status_code=409, detail=f"PR #{pr_number} is not open")

    state_store.reset_tracked_pr_feedback_cursor(pr_number)
    summary, task = await _schedule_pull_request_repair(pr, source="api", priority="urgent")
    _record_state_event(
        category="pr_task_queue",
        message=f"Explicitly rechecked PR #{pr_number} review feedback",
        source="api",
        target_kind="pull_request",
        target_number=pr_number,
        details={"summary": summary, "task_id": task.task_id if task is not None else ""},
    )
    return {
        "status": "ok",
        "summary": summary,
        "task": _pr_task_to_dict(task) if task is not None else None,
    }


@app.get("/tasks/pr")
async def list_pr_tasks(status: str = ""):
    return {
        "status": "ok",
        "tasks": _list_queue_tasks("pr", status=status),
    }


@app.post("/tasks/pr/{task_id}/retry")
async def retry_pr_task(task_id: str):
    return {"status": "ok", "task": _retry_queue_task("pr", task_id)}


@app.get("/tasks/source")
async def list_source_tasks(status: str = ""):
    return {
        "status": "ok",
        "tasks": _list_queue_tasks("source", status=status),
    }


@app.post("/tasks/source/{task_id}/retry")
async def retry_source_task(task_id: str):
    return {"status": "ok", "task": _retry_queue_task("source", task_id)}


async def _handle_issue_event(issue: dict, *, source: str):
    """校验一个 issue 事件是否值得进入执行队列。

    这里只做事件级过滤和去重，不直接改代码。真正的计划/实现发生在
    `issue-task-cycle -> controller.execute_issue(...)` 里。
    """
    assert controller and gh and state_store and issue_task_queue
    assert config
    _queue_issue_execution_task(
        issue,
        source=source,
        trigger="issue_open",
        explicit_command=False,
    )


def _mark_issue_comment_processed(
    *,
    comment_id: int,
    issue_number: int,
    updated_at: str,
    command: str,
    source: str,
):
    if state_store is None or comment_id <= 0:
        return
    state_store.mark_issue_comment_processed(
        comment_id=comment_id,
        issue_number=issue_number,
        updated_at=updated_at,
        command=command,
        source=source,
    )


async def _handle_issue_comment_event(data: dict, *, source: str):
    """处理 issue_comment 事件。

    当仓库 owner 在 feature/enhancement issue 上评论"实现"(或 "implement")时，
    以 explicit_command=True 触发该 issue 的自动实现流程。
    PR 上的评论仍由 openreview 处理，此处跳过。
    """
    issue = data.get("issue", {}) or {}
    comment = data.get("comment", {}) or {}
    issue_number = int(issue.get("number", 0) or 0)
    comment_id = int(comment.get("id", 0) or 0)
    comment_updated_at = str(comment.get("updated_at") or comment.get("created_at") or "")
    comment_body = str(comment.get("body", "") or "").strip()
    commenter = str((comment.get("user", {}) or {}).get("login", "") or "").strip()

    # PR 上的评论由 openreview 处理，跳过
    if "pull_request" in issue:
        logger.debug(
            "跳过 PR #%s 上的评论 (comment_id=%s)，由 openreview 处理",
            issue_number, comment_id,
        )
        _mark_issue_comment_processed(
            comment_id=comment_id,
            issue_number=issue_number,
            updated_at=comment_updated_at,
            command="",
            source=source,
        )
        return

    if not issue_number or not comment_body:
        return

    # 只接受语义明确的实现指令评论
    if not issue_comment_requests_implementation(comment_body):
        logger.debug(
            "Issue #%s 评论不包含明确实现意图，跳过 (comment_id=%s)",
            issue_number, comment_id,
        )
        _mark_issue_comment_processed(
            comment_id=comment_id,
            issue_number=issue_number,
            updated_at=comment_updated_at,
            command="",
            source=source,
        )
        return

    # 只接受仓库 owner 的评论
    if config is None:
        return
    if not bool(getattr(getattr(config, "autocode", None), "allow_feature_on_comment", True)):
        logger.info(
            "Issue #%s 收到实现指令，但当前已关闭评论触发执行，跳过",
            issue_number,
        )
        _record_state_event(
            category="issue_comment",
            message=f"Ignored implement command on #{issue_number} because comment approval is disabled",
            source=source,
            target_kind="issue",
            target_number=issue_number,
            details={"comment_id": comment_id, "reason": "comment_approval_disabled"},
        )
        _mark_issue_comment_processed(
            comment_id=comment_id,
            issue_number=issue_number,
            updated_at=comment_updated_at,
            command="implement",
            source=source,
        )
        return
    repo_owner = str(config.github.owner or "").lower()
    if not is_repo_owner_human_issue_comment(comment, repo_owner):
        logger.info(
            "Issue #%s 收到实现指令但评论并非仓库 owner %s 的人工评论，跳过 (commenter=%s)",
            issue_number, commenter, repo_owner,
        )
        _record_state_event(
            category="issue_comment",
            message=f"Ignored implement command on #{issue_number} because the comment was not a human owner approval",
            source=source,
            target_kind="issue",
            target_number=issue_number,
            details={"comment_id": comment_id, "commenter": commenter, "reason": "not_human_owner"},
        )
        _mark_issue_comment_processed(
            comment_id=comment_id,
            issue_number=issue_number,
            updated_at=comment_updated_at,
            command="implement",
            source=source,
        )
        return

    # 检查 issue 状态
    issue_state = str(issue.get("state", "") or "").lower()
    if issue_state != "open":
        logger.info("Issue #%s 不是 open 状态 (%s)，跳过实现指令", issue_number, issue_state)
        _mark_issue_comment_processed(
            comment_id=comment_id,
            issue_number=issue_number,
            updated_at=comment_updated_at,
            command="implement",
            source=source,
        )
        return

    logger.info(
        "Issue #%s 收到 owner (%s) 的实现指令，触发自动实现 (comment_id=%s)",
        issue_number, commenter, comment_id,
    )
    _record_state_event(
        category="issue_comment",
        message=f"Owner triggered implement on issue #{issue_number}",
        source=source,
        target_kind="issue",
        target_number=issue_number,
        details={"comment_id": comment_id, "commenter": commenter, "trigger": "implement_comment"},
    )
    _queue_issue_execution_task(
        issue,
        source=source,
        trigger="implement_comment",
        explicit_command=True,
    )
    _mark_issue_comment_processed(
        comment_id=comment_id,
        issue_number=issue_number,
        updated_at=comment_updated_at,
        command="implement",
        source=source,
    )


def _queue_issue_execution_task(
    issue: dict,
    *,
    source: str,
    trigger: str,
    explicit_command: bool,
    allow_bot_user: bool = False,
) -> dict[str, object]:
    if state_store is None or issue_task_queue is None or config is None:
        return {"queued": False, "created": False, "reason": "queue_unavailable", "task_id": ""}
    issue_number = int(issue.get("number", 0) or 0)
    issue_updated_at = str(issue.get("updated_at") or issue.get("created_at") or "")
    if issue_number > 0 and state_store.is_issue_paused(issue_number):
        logger.info("Issue #%s 已暂停自动处理，跳过事件", issue_number)
        _record_state_event(
            category="issue_event",
            message=f"Issue #{issue_number} skipped because it is paused",
            source=source,
            target_kind="issue",
            target_number=issue_number,
            details={"reason": "paused"},
        )
        return {"queued": False, "created": False, "reason": "paused", "task_id": ""}
    if not explicit_command and not bool(getattr(config.autocode, "auto_plan_on_issue_open", False)):
        logger.info("已关闭 issue opened 自动计划，跳过 Issue #%s", issue.get("number"))
        _record_state_event(
            category="issue_event",
            message=f"Issue #{issue_number} skipped because auto plan is disabled",
            source=source,
            target_kind="issue",
            target_number=issue_number,
            details={"reason": "auto_plan_disabled"},
        )
        return {"queued": False, "created": False, "reason": "auto_plan_disabled", "task_id": ""}
    user = issue.get("user", {}) or {}
    if not allow_bot_user and _is_bot_user(user):
        logger.info(
            "Issue #%s 由 bot 用户 %s 创建，跳过自动入队",
            issue_number,
            str(user.get("login", "") or "") or "unknown",
        )
        _record_state_event(
            category="issue_event",
            message=f"Issue #{issue_number} skipped because it was created by a bot user",
            source=source,
            target_kind="issue",
            target_number=issue_number,
            details={"reason": "bot_user", "login": str(user.get('login', '') or '')},
        )
        return {"queued": False, "created": False, "reason": "bot_user", "task_id": ""}
    if issue_number > 0 and issue_updated_at and state_store.has_processed_issue_version(issue_number, issue_updated_at):
        last_outcome = state_store.get_issue_last_outcome(issue_number)
        if not last_outcome or last_outcome in _ISSUE_TERMINAL_SUCCESS_OUTCOMES:
            logger.debug(
                "Issue #%s 当前版本已处理%s，跳过重复入队",
                issue_number,
                f"且执行成功({last_outcome})" if last_outcome else "",
            )
            _record_state_event(
                category="issue_event",
                message=f"Issue #{issue_number} skipped because this version is already processed successfully",
                source=source,
                target_kind="issue",
                target_number=issue_number,
                details={"reason": "processed_version", "updated_at": issue_updated_at, "last_outcome": last_outcome},
            )
            return {"queued": False, "created": False, "reason": "processed_version", "task_id": ""}
        logger.debug(
            "Issue #%s 当前版本已处理但未成功执行(last_outcome=%s)，允许重新入队",
            issue_number,
            last_outcome or "unknown",
        )
    elif not explicit_command and issue_number > 0:
        # New version (updated_at changed) but not yet recorded as processed.
        # If the previous outcome was already a terminal success, don't auto-reprocess —
        # the update was almost certainly caused by an autocode bot comment
        # changing issue.updated_at, which would create an infinite self-loop.
        # Only explicit_command=True (e.g. owner comment "implement") should re-trigger.
        last_outcome = state_store.get_issue_last_outcome(issue_number)
        if last_outcome in _ISSUE_TERMINAL_SUCCESS_OUTCOMES:
            logger.debug(
                "Issue #%s 新版本但上次执行已成功(%s)，非显式命令跳过以防 autocode bot 评论自循环",
                issue_number,
                last_outcome,
            )
            _record_state_event(
                category="issue_event",
                message=f"Issue #{issue_number} skipped: new version but previous outcome was terminal ({last_outcome})",
                source=source,
                target_kind="issue",
                target_number=issue_number,
                details={"reason": "terminal_outcome_new_version", "updated_at": issue_updated_at, "last_outcome": last_outcome},
            )
            return {"queued": False, "created": False, "reason": "terminal_outcome_new_version", "task_id": ""}
    task_id = _build_issue_task_id(issue_number, issue_updated_at, source, trigger=trigger)
    existing_task = issue_task_queue.get_task(task_id)
    existing_status = str(existing_task.status if existing_task is not None else "" or "")
    if existing_status == "dead_letter" and issue_task_queue.is_permanent_failure(task_id):
        logger.debug(
            "Issue #%s 任务为永久失败状态，跳过自动复活 task=%s",
            issue_number,
            task_id,
        )
        return {"queued": False, "created": False, "reason": "permanent_failure", "task_id": task_id}
    replace_existing = existing_status in {"retry_waiting", "dead_letter"} or explicit_command
    task, created = issue_task_queue.enqueue(
        task_id=task_id,
        target_number=issue_number,
        source=source,
        priority="normal",
        command="issue_open",
        payload={"explicit_command": explicit_command, "issue_updated_at": issue_updated_at},
        max_attempts=getattr(config.autocode, "issue_task_max_attempts", 3),
        replace_existing=replace_existing,
    )
    queue_status = str(task.status or "")
    if created:
        queued = True
        reason = ""
        action = "queued"
    elif replace_existing and queue_status == "queued":
        queued = True
        reason = f"requeued_{existing_status or 'terminal'}"
        action = "requeued"
    elif queue_status in {"queued", "running", "retry_waiting"}:
        queued = True
        reason = (
            "already_running"
            if queue_status == "running"
            else "already_retry_waiting"
            if queue_status == "retry_waiting"
            else "already_queued"
        )
        action = "reused"
    else:
        queued = False
        reason = f"existing_{queue_status or 'unknown'}"
        action = "reused"
    logger.info(
        "issue-task-queue: %s issue #%s source=%s trigger=%s explicit=%s updated_at=%s task=%s",
        action,
        issue_number,
        source,
        trigger,
        explicit_command,
        issue_updated_at or "-",
        task.task_id,
    )
    if created:
        _record_state_event(
            category="issue_task_queue",
            message=f"Queued issue execution for issue #{issue_number}",
            source=source,
            target_kind="issue",
            target_number=issue_number,
            details={
                "task_id": task.task_id,
                "status": task.status,
                "command": task.command,
                "trigger": trigger,
            },
        )
    return {"queued": queued, "created": created, "reason": reason, "task_id": task.task_id}


def _preview_text(text: str, *, limit: int = 180) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(limit - 3, 1)] + "..."


def _extract_json_payload(raw: str) -> dict[str, object]:
    return extract_json_object(str(raw or ""), context="webhook payload")


def _normalize_issue_title(title: str) -> str:
    normalized = re.sub(r"^\[[^\]]+\]\s*", "", str(title or "").strip())
    normalized = normalized.strip().strip("“”\"'").strip()
    return normalized[:120].rstrip()


def _strip_internal_refs(text: str) -> str:
    sanitized = str(text or "")
    sanitized = re.sub(r"`[^`\n]*/[^`\n]*`", "`相关代码位置`", sanitized)
    sanitized = re.sub(r"/(?:[^/\s`]+/)+[^/\s`:]+(?::\d+)?", "相关代码位置", sanitized)
    sanitized = re.sub(
        r"(?<!https://)(?<!http://)\b(?:projects|src|app|apps|api|backend|frontend|tests|docs|scripts|templates|\.github)/[^\s`]+(?::\d+)?",
        "相关代码位置",
        sanitized,
    )
    sanitized = re.sub(r"\s{2,}", " ", sanitized)
    return sanitized.strip()


def _fallback_issue_title(raw_title: str) -> str:
    normalized = _strip_internal_refs(_normalize_issue_title(raw_title))
    lowered = normalized.lower()
    replacements = (
        (("readme",), "精简 README 结构并下沉细节说明到专题文档"),
        (("litellm",), "优化模型接入体验，避免用户直接感知 LiteLLM"),
        (("action", "映射"), "补齐 Actions 配置映射并增加缺失项校验"),
        (("sqlite",), "修复 SQLite 写入链路的低效更新与锁竞争问题"),
        (("断点续传", "交易日"), "修复断点续传按自然日判断数据存在性的逻辑"),
        (("并发", "加锁"), "修复并发执行时共享状态缺少统一加锁的问题"),
        (("notificationservice",), "修复通知链路中共享实例的并发复用问题"),
        (("通知", "并发"), "修复通知链路中共享实例的并发复用问题"),
        (("llm", "日志"), "修复 INFO 日志默认输出 LLM 预览的问题"),
        (("prompt/response",), "修复 INFO 日志默认输出 LLM 预览的问题"),
        (("serpapi",), "收敛 SerpAPI 正文抓取范围，降低搜索链路尾延迟"),
    )
    for needles, replacement in replacements:
        if all(needle in lowered for needle in needles):
            return replacement
    return normalized or "整理自动同步条目为可执行问题"


def _primary_source_issue_label(labels: list[str] | None) -> str:
    for label in labels or []:
        normalized = str(label or "").strip().lower()
        if normalized in _SOURCE_ISSUE_TITLE_PREFIX and normalized != "autocode":
            return normalized
    return "bug"


def _is_vague_issue_title(title: str) -> bool:
    normalized = _normalize_issue_title(title).lower()
    if not normalized:
        return True
    if normalized in _SOURCE_ISSUE_VAGUE_TITLE_MARKERS:
        return True
    if normalized.endswith("一下"):
        return True
    if len(normalized) <= 4:
        return True
    return False


def _format_source_issue_title(*, title: str, labels: list[str], fallback_title_source: str) -> str:
    normalized = _strip_internal_refs(_normalize_issue_title(title))
    if _is_vague_issue_title(normalized):
        normalized = _fallback_issue_title(fallback_title_source)
    prefix = _SOURCE_ISSUE_TITLE_PREFIX.get(_primary_source_issue_label(labels), "[Bug]")
    formatted = f"{prefix} {normalized}".strip()
    return formatted[:120].rstrip()


def _sanitize_source_issue_body(body: str) -> str:
    lines: list[str] = []
    for raw_line in str(body or "").splitlines():
        stripped = raw_line.strip()
        lowered = stripped.lower()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if stripped.startswith("来源：") or stripped.startswith("归属："):
            continue
        if any(marker in lowered for marker in _SOURCE_ISSUE_BODY_METADATA_MARKERS):
            continue
        sanitized = _strip_internal_refs(stripped)
        if sanitized:
            lines.append(sanitized)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines).strip()


def _infer_source_issue_labels(*, source_name: str, raw_title: str, raw_body: str) -> list[str]:
    del source_name
    combined = " ".join(f"{raw_title}\n{raw_body}".split()).lower()
    if any(marker in combined for marker in _SOURCE_ISSUE_SKIP_MARKERS):
        return ["autocode", "invalid"]
    if any(keyword in combined for keyword in _SOURCE_ISSUE_SECURITY_MARKERS):
        return ["autocode", "bug"]
    if any(keyword in combined for keyword in ("readme", "文档", "说明")):
        return ["autocode", "documentation"]
    if any(keyword in combined for keyword in ("优化", "体验", "支持", "重构", "精简", "下沉")) and not any(
        keyword in combined for keyword in _SOURCE_ISSUE_ENHANCEMENT_BLOCK_MARKERS
    ):
        return ["autocode", "enhancement"]
    if any(keyword in combined for keyword in ("排查", "查看", "确认")) and not any(
        keyword in combined for keyword in _SOURCE_ISSUE_ENHANCEMENT_BLOCK_MARKERS
    ):
        return ["autocode", "question"]
    return ["autocode", "bug"]


def _should_force_bug_source_issue(*, raw_title: str, raw_body: str) -> bool:
    combined = " ".join(f"{raw_title}\n{raw_body}".split()).lower()
    if any(keyword in combined for keyword in _SOURCE_ISSUE_SECURITY_MARKERS):
        return True
    return any(keyword in combined for keyword in _SOURCE_ISSUE_BUG_MARKERS)


def _normalize_source_issue_labels(
    labels: list[str] | None,
    *,
    source_name: str,
    raw_title: str,
    raw_body: str,
    for_skip: bool = False,
) -> list[str]:
    normalized = [str(label).strip().lower() for label in (labels or []) if str(label).strip()]
    normalized = [label for label in normalized if label in _SOURCE_ISSUE_ALLOWED_LABELS]
    if _should_force_bug_source_issue(raw_title=raw_title, raw_body=raw_body):
        normalized = [label for label in normalized if label not in _SOURCE_ISSUE_PRIMARY_LABELS]
        normalized.append("bug")
    if "autocode" not in normalized:
        normalized.insert(0, "autocode")
    primary = [label for label in normalized if label in _SOURCE_ISSUE_PRIMARY_LABELS]
    if not primary:
        inferred = _infer_source_issue_labels(
            source_name=source_name,
            raw_title=raw_title,
            raw_body=raw_body,
        )
        normalized = list(dict.fromkeys([*normalized, *inferred]))
        primary = [label for label in normalized if label in _SOURCE_ISSUE_PRIMARY_LABELS]
    if primary:
        keep_primary = primary[0]
        normalized = [label for label in normalized if label == "autocode" or label == keep_primary]
    if for_skip:
        return list(dict.fromkeys(["autocode", "invalid"]))
    return list(dict.fromkeys(normalized))


def _github_issue_label_names(issue: dict) -> list[str]:
    labels: list[str] = []
    for label in issue.get("labels", []) or []:
        if isinstance(label, dict):
            name = str(label.get("name", "") or "").strip()
        else:
            name = str(label or "").strip()
        if name:
            labels.append(name)
    return labels


def _is_emergency_source_issue(*, title: str, body: str, labels: list[str], raw_body: str = "") -> bool:
    if _primary_source_issue_label(labels) != "bug":
        return False
    combined = " ".join(f"{title}\n{body}\n{raw_body}".split()).lower()
    return any(marker in combined for marker in _SOURCE_ISSUE_EMERGENCY_MARKERS)


def _max_open_autocode_issues() -> int:
    if config is None:
        return 20
    return max(int(getattr(config.autocode, "max_open_autocode_issues", 20) or 0), 0)


def _list_open_autocode_issues() -> list[dict]:
    if gh is None:
        return []
    issues: list[dict] = []
    page = 1
    per_page = 100
    while True:
        page_items = gh.list_issues(
            state="open",
            sort="updated",
            direction="desc",
            per_page=per_page,
            page=page,
        )
        if not page_items:
            break
        for issue in page_items:
            if "pull_request" in issue:
                continue
            labels = {label.lower() for label in _github_issue_label_names(issue)}
            if "autocode" not in labels:
                continue
            issues.append(issue)
        if len(page_items) < per_page:
            break
        page += 1
    return issues


def _list_open_reconcile_issues() -> list[dict]:
    if gh is None:
        return []
    issues: list[dict] = []
    page = 1
    per_page = 100
    while True:
        page_items = gh.list_issues(
            state="open",
            sort="updated",
            direction="desc",
            per_page=per_page,
            page=page,
        )
        if not page_items:
            break
        for issue in page_items:
            if "pull_request" in issue:
                continue
            issues.append(issue)
        if len(page_items) < per_page:
            break
        page += 1
    return issues


def _find_matching_open_autocode_issue(open_issues: list[dict], *, title: str, labels: list[str]) -> dict | None:
    candidate_title = _normalize_issue_title(title).casefold()
    candidate_primary = _primary_source_issue_label(labels)
    for issue in open_issues:
        issue_title = _normalize_issue_title(str(issue.get("title", "") or "")).casefold()
        if issue_title != candidate_title:
            continue
        issue_primary = _primary_source_issue_label(_github_issue_label_names(issue))
        if candidate_primary != issue_primary:
            continue
        return issue
    return None


def _source_issue_capacity_block_reason(
    open_issues: list[dict],
    *,
    title: str,
    body: str,
    labels: list[str],
    raw_body: str = "",
) -> str:
    limit = _max_open_autocode_issues()
    if limit <= 0:
        return ""
    if len(open_issues) < limit:
        return ""
    if _is_emergency_source_issue(title=title, body=body, labels=labels, raw_body=raw_body):
        return ""
    return f"当前未关闭的 autocode issue 已达到 {limit} 个，暂缓创建新的非紧急问题单。"


def _fallback_source_issue_draft(
    *,
    source_name: str,
    raw_title: str,
    raw_body: str,
) -> dict[str, object]:
    combined = " ".join(f"{raw_title}\n{raw_body}".split()).lower()
    if any(marker in combined for marker in _SOURCE_ISSUE_SKIP_MARKERS):
        return {
            "action": "skip",
            "title": "",
            "body": "",
            "labels": ["autocode", "invalid"],
            "reason": "条目本身是否定结论或未证实备注，不适合作为 issue",
            "optimized": False,
        }
    title = _fallback_issue_title(raw_title)
    source_hints = {
        "backlog": "这个问题来自待办梳理，建议补充触发条件、影响范围和验收标准后推进处理。",
        "plan": "这个问题来自开发计划梳理，建议进一步明确目标边界和验收条件。",
        "idle_scan": "这个问题来自空闲巡检，建议先确认触发原因、影响范围和修复边界。",
    }
    body = source_hints.get(source_name, "这个问题由 AutoCode 自动整理，建议先确认原因和影响范围。")
    labels = _infer_source_issue_labels(source_name=source_name, raw_title=raw_title, raw_body=raw_body)
    return {
        "action": "create",
        "title": _format_source_issue_title(title=title, labels=labels, fallback_title_source=raw_title),
        "body": _sanitize_source_issue_body(body),
        "labels": labels,
        "reason": "fallback",
        "optimized": False,
    }


def _draft_source_issue_payload(
    *,
    source_name: str,
    item_key: str,
    raw_title: str,
    raw_body: str,
    raw_labels: list[str],
) -> dict[str, object]:
    fallback = _fallback_source_issue_draft(
        source_name=source_name,
        raw_title=raw_title,
        raw_body=raw_body,
    )
    planner = getattr(controller, "planner", None) if controller is not None else None
    if planner is None:
        return fallback
    try:
        raw = planner.analyze(
            build_source_issue_prompt(
                source_name=source_name,
                raw_title=raw_title,
                raw_body=raw_body,
                raw_labels=raw_labels,
            ),
            system="你是保守的 AutoCode source issue 起草器，只输出 JSON。",
        )
        data = _extract_json_payload(raw)
        action = str(data.get("action", "") or "").strip().lower()
        if action not in {"create", "skip"}:
            action = str(fallback["action"])
        title = _normalize_issue_title(str(data.get("title", "") or "")) if action == "create" else ""
        body = _sanitize_source_issue_body(str(data.get("body", "") or "").strip()) if action == "create" else ""
        reason = str(data.get("reason", "") or "").strip() or str(fallback["reason"])
        if action == "create" and (not title or not body):
            return fallback
        labels = _normalize_source_issue_labels(
            [str(label) for label in data.get("labels", []) if str(label).strip()],
            source_name=source_name,
            raw_title=title or raw_title,
            raw_body=body or raw_body,
            for_skip=(action == "skip"),
        )
        return {
            "action": action,
            "title": (
                _format_source_issue_title(
                    title=title,
                    labels=labels,
                    fallback_title_source=raw_title,
                )
                if action == "create"
                else ""
            ),
            "body": body,
            "labels": labels,
            "reason": reason,
            "optimized": True,
        }
    except (LLMError, json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "source-issue-draft: fallback source=%s item=%s error=%s",
            source_name,
            item_key or "-",
            exc,
        )
        return fallback


def _build_background_scheduler() -> TaskScheduler:
    """按配置注册后台周期任务。

    这里注册的是“发现任务”和“消费队列”两类 cycle；不同 cycle 可以并发
    运行，避免长时间 issue 执行阻塞 source/pr/polling 等其他后台链路。
    """
    assert config and controller
    background_scheduler = TaskScheduler(logger=logger)
    scheduler_descriptions: list[str] = []
    if config.server.event_source == "polling":
        background_scheduler.register_task(
            name="polling-cycle",
            handler=_run_polling_cycle,
            interval_seconds=config.server.poll_interval_seconds,
            run_immediately=True,
        )
        scheduler_descriptions.append(f"polling-cycle={config.server.poll_interval_seconds}s(immediate)")
    background_scheduler.register_task(
        name="issue-task-cycle",
        handler=_run_issue_task_cycle,
        interval_seconds=getattr(config.autocode, "issue_task_interval_seconds", 5),
        run_immediately=True,
        priority="urgent",
    )
    scheduler_descriptions.append(
        f"issue-task-cycle={getattr(config.autocode, 'issue_task_interval_seconds', 5)}s(immediate)"
    )
    if (
        config.autocode.enable_plan_source
        or config.autocode.enable_backlog_source
        or config.autocode.enable_idle_scanner
    ):
        background_scheduler.register_task(
            name="source-task-cycle",
            handler=_run_source_task_cycle,
            interval_seconds=getattr(config.autocode, "source_task_interval_seconds", 5),
            run_immediately=True,
        )
        scheduler_descriptions.append(
            f"source-task-cycle={getattr(config.autocode, 'source_task_interval_seconds', 5)}s(immediate)"
        )
    if config.autocode.poll_review_feedback or config.autocode.enable_pr_queue:
        background_scheduler.register_task(
            name="pr-queue-cycle",
            handler=_run_pr_queue_cycle,
            interval_seconds=config.autocode.review_feedback_interval_seconds,
            run_immediately=True,
        )
        scheduler_descriptions.append(
            f"pr-queue-cycle={config.autocode.review_feedback_interval_seconds}s(immediate)"
        )
        background_scheduler.register_task(
            name="pr-task-cycle",
            handler=_run_pr_task_cycle,
            interval_seconds=getattr(config.autocode, "pr_task_interval_seconds", 5),
            run_immediately=True,
        )
        scheduler_descriptions.append(
            f"pr-task-cycle={getattr(config.autocode, 'pr_task_interval_seconds', 5)}s(immediate)"
        )
    if config.autocode.enable_idle_scanner:
        background_scheduler.register_task(
            name="idle-scan-cycle",
            handler=_run_idle_scan_cycle,
            interval_seconds=config.autocode.idle_scanner_interval_seconds,
            run_immediately=False,
            priority="background",
        )
        scheduler_descriptions.append(f"idle-scan-cycle={config.autocode.idle_scanner_interval_seconds}s(delayed)")
    if scheduler_descriptions:
        logger.info("scheduler: registered cycles %s", ", ".join(scheduler_descriptions))
    return background_scheduler


async def _run_polling_cycle():
    """执行一轮 polling 总流程，并把结果写进可观测状态。"""
    _mark_cycle_started("polling-cycle")
    try:
        stats = await _poll_once()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _record_cycle_result("polling-cycle", status="failed", error=str(exc))
        _record_state_event(
            category="scheduler_cycle",
            level="error",
            message="polling cycle failed",
            details={"cycle": "polling-cycle", "error": str(exc)},
        )
        logger.exception("AutoCode polling 周期执行失败")
        raise
    else:
        _record_cycle_result("polling-cycle", status="succeeded", stats=stats)


async def _run_issue_task_cycle():
    """执行一轮 issue 队列消费。"""
    assert issue_task_queue and state_store and gh and controller
    _mark_cycle_started("issue-task-cycle")
    try:
        stats = await _drain_issue_task_queue_once()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _record_cycle_result("issue-task-cycle", status="failed", error=str(exc))
        _record_state_event(
            category="scheduler_cycle",
            level="error",
            message="issue task cycle failed",
            details={"cycle": "issue-task-cycle", "error": str(exc)},
        )
        logger.exception("AutoCode issue 任务周期执行失败")
        raise
    else:
        _record_cycle_result("issue-task-cycle", status="succeeded", stats=stats)


async def _run_source_task_cycle():
    """执行一轮 source 同步与 source 队列消费。"""
    assert source_task_queue and gh_writer and state_store
    _mark_cycle_started("source-task-cycle")
    try:
        stats: dict[str, dict[str, int]] = {}
        if _should_sync_sources_in_background():
            stats["plans"] = await _safe_source_sync_once("plan", _sync_plan_files_once)
            stats["backlog"] = await _safe_source_sync_once("backlog", _sync_backlog_once)
        stats["queue"] = await _drain_source_task_queue_once()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _record_cycle_result("source-task-cycle", status="failed", error=str(exc))
        _record_state_event(
            category="scheduler_cycle",
            level="error",
            message="source task cycle failed",
            details={"cycle": "source-task-cycle", "error": str(exc)},
        )
        logger.exception("AutoCode source 任务周期执行失败")
        raise
    else:
        _record_cycle_result("source-task-cycle", status="succeeded", stats=stats)


async def _run_pr_queue_cycle():
    """执行一轮 tracked PR 扫描，把新的 review feedback 转成修复任务。"""
    assert controller and pr_task_queue
    _mark_cycle_started("pr-queue-cycle")
    try:
        stats = await _scan_pull_request_repairs_once()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _record_cycle_result("pr-queue-cycle", status="failed", error=str(exc))
        _record_state_event(
            category="scheduler_cycle",
            level="error",
            message="pr queue cycle failed",
            details={"cycle": "pr-queue-cycle", "error": str(exc)},
        )
        logger.exception("AutoCode PR 队列周期执行失败")
        raise
    else:
        _record_cycle_result("pr-queue-cycle", status="succeeded", stats=stats)


async def _run_pr_task_cycle():
    """执行一轮 PR 修复队列消费。"""
    assert controller and pr_task_queue and state_store and gh
    _mark_cycle_started("pr-task-cycle")
    try:
        stats = await _drain_pr_task_queue_once()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _record_cycle_result("pr-task-cycle", status="failed", error=str(exc))
        _record_state_event(
            category="scheduler_cycle",
            level="error",
            message="pr task cycle failed",
            details={"cycle": "pr-task-cycle", "error": str(exc)},
        )
        logger.exception("AutoCode PR 修复任务周期执行失败")
        raise
    else:
        _record_cycle_result("pr-task-cycle", status="succeeded", stats=stats)


async def _run_idle_scan_cycle():
    """系统空闲时扫描 TODO/FIXME，并把结果放入 source 队列。"""
    assert state_store
    _mark_cycle_started("idle-scan-cycle")
    if not _should_run_idle_scan():
        logger.info("idle-scan: skipped because active work is still pending")
        _record_cycle_result(
            "idle-scan-cycle",
            status="skipped",
            stats={"reason": "active_work_pending"},
        )
        _record_state_event(
            category="scheduler_cycle",
            message="idle scan skipped because active work is pending",
            details={"cycle": "idle-scan-cycle"},
        )
        return
    stats = await _enqueue_plan_source_items_once(
        plan_source=idle_scanner,
        source_name="idle_scan",
        priority="background",
    )
    logger.info("idle-scan: fetched=%d queued=%d", stats["fetched"], stats["queued"])
    _record_cycle_result("idle-scan-cycle", status="succeeded", stats=stats)


async def _drain_issue_task_queue_once() -> dict[str, int]:
    """消费一批 issue 执行任务。

    队列只保存“待处理的 issue 编号和触发方式”，真正执行前会再次拉取最新
    issue 状态，确保不会基于已关闭或已暂停的旧快照继续跑。
    """
    assert issue_task_queue and gh and controller and state_store
    stats = {"claimed": 0, "processed": 0, "failed": 0}
    max_batch = max(int(getattr(config.autocode, "issue_task_max_batch", 1) or 1), 1) if config else 1
    for _ in range(max_batch):
        task = issue_task_queue.claim_next()
        if task is None:
            break
        stats["claimed"] += 1
        issue_number = task.target_number
        logger.info(
            "issue-task-queue: 开始处理任务 task=%s issue=%s attempt=%s/%s source=%s explicit=%s",
            task.task_id,
            issue_number,
            task.attempt_count,
            task.max_attempts,
            task.source,
            bool(task.payload.get("explicit_command", False)),
        )
        try:
            if state_store.is_issue_paused(issue_number):
                issue_task_queue.mark_succeeded(task.task_id, result_summary="skipped_paused")
                logger.info("issue-task-queue: issue #%s skipped because it is paused", issue_number)
                _record_state_event(
                    category="issue_task_queue",
                    message=f"Skipped queued issue task for paused issue #{issue_number}",
                    target_kind="issue",
                    target_number=issue_number,
                    details={"task_id": task.task_id, "run_status": "skipped_paused"},
                )
                stats["processed"] += 1
                continue
            # 消费时再读取一次最新 issue，避免仅凭入队时的旧状态执行。
            issue = await asyncio.to_thread(gh.get_issue, issue_number)
            logger.info(
                "issue-task-queue: 已获取最新 issue #%s title=%s state=%s updated_at=%s",
                issue_number,
                _preview_text(str(issue.get("title", "") or "")),
                str(issue.get("state", "") or "-"),
                str(issue.get("updated_at", "") or "-"),
            )
            if str(issue.get("state", "") or "").lower() != "open":
                issue_task_queue.mark_succeeded(task.task_id, result_summary="skipped_closed")
                logger.info("issue-task-queue: issue #%s skipped because it is closed", issue_number)
                _record_state_event(
                    category="issue_task_queue",
                    message=f"Skipped queued issue task for closed issue #{issue_number}",
                    target_kind="issue",
                    target_number=issue_number,
                    details={"task_id": task.task_id, "run_status": "skipped_closed"},
                )
                stats["processed"] += 1
                continue
            result = await asyncio.to_thread(
                controller.execute_issue,
                issue,
                explicit_command=bool(task.payload.get("explicit_command", False)),
            )
            # 队列层只把 failed 视作需要重试；planned/blocked 也算“已处理完”。
            if str(result.run_record.status or "") == "failed":
                error = str(result.run_record.failure_reason or result.comment or "Issue execution failed")
                permanent = _is_permanent_task_failure(error)
                failed_task = issue_task_queue.mark_failed(
                    task.task_id,
                    error=error,
                    retry_delay_seconds=_issue_task_retry_delay_seconds(task.attempt_count),
                    permanent=permanent,
                )
                _record_state_event(
                    category="issue_task_queue",
                    level="error",
                    message=f"Queued issue task failed for issue #{issue_number}",
                    target_kind="issue",
                    target_number=issue_number,
                    details={
                        "task_id": task.task_id,
                        "error": error,
                        "run_status": result.run_record.status,
                        "run_id": result.run_record.run_id,
                        "queue_status": failed_task.status if failed_task is not None else "",
                    },
                )
                logger.error("issue-task-queue: task %s returned failed status: %s", task.task_id, error)
                stats["failed"] += 1
                continue
            issue_task_queue.mark_succeeded(task.task_id, result_summary=result.run_record.status)
            logger.info(
                "issue-task-queue: 任务完成 task=%s issue=%s run_status=%s run_id=%s",
                task.task_id,
                issue_number,
                result.run_record.status,
                result.run_record.run_id,
            )
            _record_state_event(
                category="issue_task_queue",
                message=f"Processed queued issue task for issue #{issue_number}",
                target_kind="issue",
                target_number=issue_number,
                details={
                    "task_id": task.task_id,
                    "run_status": result.run_record.status,
                    "run_id": result.run_record.run_id,
                },
            )
            stats["processed"] += 1
        except Exception as exc:
            failed_task = issue_task_queue.mark_failed(
                task.task_id,
                error=str(exc),
                retry_delay_seconds=_issue_task_retry_delay_seconds(task.attempt_count),
            )
            _record_state_event(
                category="issue_task_queue",
                level="error",
                message=f"Queued issue task failed for issue #{issue_number}",
                target_kind="issue",
                target_number=issue_number,
                details={
                    "task_id": task.task_id,
                    "error": str(exc),
                    "queue_status": failed_task.status if failed_task is not None else "",
                },
            )
            logger.exception("issue-task-queue: failed to process task %s", task.task_id)
            stats["failed"] += 1
    if stats["claimed"] == 0:
        logger.debug("issue-task-queue: no queued issue task this cycle")
    return stats


async def _scan_pull_request_repairs_once() -> dict[str, int]:
    """扫描 tracked PR，把最新且仍有效的 review feedback 放入修复队列。"""
    assert controller and pr_task_queue
    tracked_prs = await asyncio.to_thread(controller.collect_tracked_pull_requests)
    stats = {"scanned": len(tracked_prs), "queued": 0, "skipped": 0}
    if not tracked_prs:
        logger.debug("pr-queue: no tracked pull requests this cycle")
    else:
        logger.info("pr-queue: 开始扫描 %d 个待跟踪 PR", len(tracked_prs))
    for pr in tracked_prs:
        pr_number = int(pr.get("number", 0) or 0)
        previous = state_store.get_tracked_pr(pr_number) if state_store is not None and pr_number > 0 else None
        summary, candidate = await asyncio.to_thread(controller.plan_pull_request_repair, pr)
        current = state_store.get_tracked_pr(pr_number) if state_store is not None and pr_number > 0 else None
        summary_changed = (
            candidate is not None
            or not isinstance(previous, dict)
            or not isinstance(current, dict)
            or str(previous.get("status", "") or "") != str(current.get("status", "") or "")
            or str(previous.get("last_summary", "") or "") != str(current.get("last_summary", "") or "")
            or str(previous.get("review_state", "") or "") != str(current.get("review_state", "") or "")
        )
        if summary and summary_changed:
            logger.info("pr-queue: %s", summary)
        if candidate is None:
            stats["skipped"] += 1
            continue
        task, created = _enqueue_pull_request_repair_candidate(
            candidate,
            source=candidate.queue_source,
            priority="normal",
        )
        if created:
            stats["queued"] += 1
            _record_state_event(
                category="pr_task_queue",
                message=f"Queued PR repair task for PR #{candidate.pr_number}",
                target_kind="pull_request",
                target_number=candidate.pr_number,
                details={"task_id": task.task_id, "status": task.status},
            )
    if stats["skipped"] > 0:
        logger.info(
            "pr-queue: 扫描完成 total=%d queued=%d skipped_unchanged=%d",
            stats["scanned"], stats["queued"], stats["skipped"],
        )

    # Rebase pass: for PRs with no new feedback, check if they need rebasing
    rebased_count = 0
    for pr in tracked_prs:
        pr_number = int(pr.get("number", 0) or 0)
        if pr_number <= 0:
            continue
        try:
            result = await asyncio.to_thread(controller.rebase_tracked_pr_if_needed, pr)
            if result and "成功" in result:
                rebased_count += 1
                logger.info("pr-rebase: %s", result)
            elif result:
                logger.info("pr-rebase: %s", result)
        except Exception as exc:
            logger.warning("pr-rebase: PR #%s rebase error: %s", pr_number, exc)
    if rebased_count > 0:
        logger.info("pr-rebase: 本轮 rebase 完成 %d 个 PR", rebased_count)
    stats["rebased"] = rebased_count
    return stats


def _enqueue_pull_request_repair_candidate(candidate, *, source: str, priority: str):
    assert pr_task_queue
    try:
        task, created = pr_task_queue.enqueue(
            task_id=candidate.task_id,
            target_number=candidate.pr_number,
            source=source or candidate.queue_source,
            priority=priority,
            command="review_feedback",
            payload={
                "feedback_key": candidate.feedback_key,
                "feedback_updated_at": candidate.feedback_updated_at,
                "feedback_item_ids": candidate.feedback_item_ids,
                "feedback": getattr(candidate, "feedback", []),
                "legacy_feedback_key": getattr(candidate, "legacy_feedback_key", ""),
                "review_state": getattr(candidate, "review_state", ""),
            },
            max_attempts=getattr(config.autocode, "pr_task_max_attempts", 3) if config else 3,
            replace_existing=True,
        )
    except Exception:
        if state_store is not None and str(candidate.feedback_key or "").strip():
            state_store.release_feedback(candidate.feedback_key)
        raise
    if state_store is not None and str(candidate.feedback_key or "").strip():
        state_store.set_feedback_claim_owner(candidate.feedback_key, owner=task.task_id)
    return task, created


async def _enqueue_plan_source_items_once(
    *,
    plan_source,
    source_name: str,
    priority: str,
) -> dict[str, int]:
    """把 plan/backlog/idle 发现的条目转换成 source 队列任务。

    这里先做一次幂等过滤，真正的 GitHub issue 创建由 source-task-cycle
    里的 `_drain_source_task_queue_once()` 负责。
    """
    assert state_store and source_task_queue
    if plan_source is None:
        return {"fetched": 0, "queued": 0}
    items = await asyncio.to_thread(plan_source.collect_items)
    stats = {"fetched": len(items), "queued": 0}
    already_synced = 0
    requeued = 0
    pending_existing = 0
    require_real_issue = _source_sync_requires_real_issue()
    synced_keys = _synced_source_item_keys(items, source_name=source_name, require_real_issue=require_real_issue)
    for item in items:
        if item.key in synced_keys:
            already_synced += 1
            issue_enqueue = await _requeue_source_issue_execution_if_needed(
                item_key=item.key,
                source_name=source_name,
            )
            if str(issue_enqueue.get("reason", "") or "") == "closed_issue":
                task, created = _enqueue_source_issue_create_task(
                    item,
                    source_name=source_name,
                    priority=priority,
                )
                if created:
                    stats["queued"] += 1
                else:
                    pending_existing += 1
                logger.info(
                    "source-sync[%s]: linked issue closed for item=%s queued_recreate=%s task=%s",
                    source_name,
                    item.key,
                    created,
                    task.task_id,
                )
                continue
            if bool(issue_enqueue.get("queued", False)):
                requeued += 1
            enqueue_reason = str(issue_enqueue.get("reason", "") or "-")
            log_fn = logger.info if bool(issue_enqueue.get("queued", False)) else logger.debug
            log_fn(
                "source-sync[%s]: reused synced item=%s requeue_issue_task=%s reason=%s task=%s",
                source_name,
                item.key,
                bool(issue_enqueue.get("queued", False)),
                enqueue_reason,
                str(issue_enqueue.get("task_id", "") or "-"),
            )
            continue
        if source_name == "backlog":
            issue_reference = await asyncio.to_thread(_read_backlog_issue_reference, item.key)
            if issue_reference is not None:
                state_store.mark_backlog_item_synced(
                    item_key=item.key,
                    source_name=source_name,
                    source_path=item.source_path,
                    line_number=item.line_number,
                    title=item.title,
                    issue_title=item.title,
                    issue_number=int(issue_reference.get("issue_number", 0) or 0),
                    issue_url=str(issue_reference.get("issue_url", "") or ""),
                    sync_status="created",
                )
                already_synced += 1
                _record_state_event(
                    category="source_task_queue",
                    message=f"Recovered backlog issue link for {item.key}",
                    details={
                        "source": source_name,
                        "issue_number": int(issue_reference.get("issue_number", 0) or 0),
                        "source_path": item.source_path,
                    },
                )
                issue_enqueue = await _requeue_source_issue_execution_if_needed(
                    item_key=item.key,
                    source_name=source_name,
                )
                if bool(issue_enqueue.get("queued", False)):
                    requeued += 1
                logger.info(
                    "source-sync[%s]: recovered linked item=%s requeue_issue_task=%s reason=%s task=%s",
                    source_name,
                    item.key,
                    bool(issue_enqueue.get("queued", False)),
                    str(issue_enqueue.get("reason", "") or "-"),
                    str(issue_enqueue.get("task_id", "") or "-"),
                )
                continue
        task_id = _build_source_task_id(item.key)
        existing_task = source_task_queue.get_task(task_id)
        task, created = _enqueue_source_issue_create_task(
            item,
            source_name=source_name,
            priority=priority,
        )
        if created:
            stats["queued"] += 1
            _record_state_event(
                category="source_task_queue",
                message=f"Queued source issue creation for {item.key}",
                details={"task_id": task.task_id, "source": source_name, "source_path": item.source_path},
            )
            continue
        previous_status = str(existing_task.status if existing_task is not None else "")
        if previous_status in {"done", "dead_letter", "retry_waiting"} and task.status == "queued":
            stats["queued"] += 1
            requeued += 1
            _record_state_event(
                category="source_task_queue",
                message=f"Requeued source issue creation for {item.key}",
                details={
                    "task_id": task.task_id,
                    "source": source_name,
                    "source_path": item.source_path,
                    "previous_status": previous_status,
                },
            )
        else:
            pending_existing += 1
    logger.info(
        "source-sync[%s]: fetched=%d already_synced=%d queued=%d requeued=%d pending_existing=%d require_real_issue=%s",
        source_name,
        stats["fetched"],
        already_synced,
        stats["queued"],
        requeued,
        pending_existing,
        require_real_issue,
    )
    return stats


def _resolve_source_plan_source(source_name: str):
    if source_name == "backlog":
        return backlog_source
    if source_name == "plan":
        return plan_source
    if source_name == "idle_scan":
        return idle_scanner
    return None


async def _source_item_exists(source_name: str, item_key: str) -> bool:
    if not item_key:
        return True
    collector = _resolve_source_plan_source(source_name)
    if collector is None:
        return source_name not in {"backlog", "plan", "idle_scan"}
    checker = getattr(collector, "has_item_key", None)
    if callable(checker):
        return bool(await asyncio.to_thread(checker, item_key))
    items = await asyncio.to_thread(collector.collect_items)
    return any(item.key == item_key for item in items)


async def _source_item_by_key(
    source_name: str,
    item_key: str,
    *,
    expected_source_path: str = "",
    expected_title: str = "",
    expected_line_number: int = 0,
):
    if not item_key:
        return None
    collector = _resolve_source_plan_source(source_name)
    if collector is None:
        return None
    resolved = None
    getter = getattr(collector, "get_item_by_key", None)
    if callable(getter):
        resolved = await asyncio.to_thread(getter, item_key)
        if resolved is not None and _source_item_matches(
            resolved,
            source_path=expected_source_path,
            title=expected_title,
            line_number=expected_line_number,
        ):
            return resolved
    items = await asyncio.to_thread(collector.collect_items)
    if not items:
        return resolved
    exact = next((item for item in items if item.key == item_key), None)
    if exact is not None and _source_item_matches(
        exact,
        source_path=expected_source_path,
        title=expected_title,
        line_number=expected_line_number,
    ):
        return exact
    candidates = [
        item
        for item in items
        if _source_item_matches(
            item,
            source_path=expected_source_path,
            title=expected_title,
            line_number=0,
        )
    ]
    if candidates:
        if expected_line_number > 0:
            closest = sorted(
                candidates,
                key=lambda item: (
                    abs(int(item.line_number or 0) - int(expected_line_number)),
                    int(item.line_number or 0),
                    str(item.key or ""),
                ),
            )[0]
            return closest
        return candidates[0]
    return exact or resolved


def _source_item_matches(item, *, source_path: str, title: str, line_number: int) -> bool:
    if source_path and str(item.source_path or "") != source_path:
        return False
    if title and str(item.title or "") != title:
        return False
    if line_number > 0 and int(item.line_number or 0) != int(line_number):
        return False
    return True


def _synced_source_item_keys(items: list, *, source_name: str, require_real_issue: bool) -> set[str]:
    if state_store is None or not items:
        return set()
    records = state_store.list_backlog_items(require_real_issue=require_real_issue)
    direct_records: dict[str, dict] = {}
    filtered_records: list[dict] = []
    for record in records:
        record_source = str(record.get("source_name", "") or "")
        if record_source and record_source != source_name:
            continue
        direct_records[str(record.get("item_key", "") or "")] = record
        filtered_records.append(record)

    grouped_records: dict[tuple[str, str], list[dict]] = {}
    for record in filtered_records:
        group_key = (
            str(record.get("source_path", "") or ""),
            str(record.get("source_title", "") or record.get("title", "") or ""),
        )
        grouped_records.setdefault(group_key, []).append(record)

    grouped_items: dict[tuple[str, str], list] = {}
    for item in items:
        group_key = (str(item.source_path or ""), str(item.title or ""))
        grouped_items.setdefault(group_key, []).append(item)
    matched: set[str] = set()
    for group_key, current_items in grouped_items.items():
        synced_records = grouped_records.get(group_key, [])
        if not synced_records:
            continue
        current_sorted = sorted(
            current_items,
            key=lambda item: (
                int(item.line_number or 0),
                str(item.key or ""),
            ),
        )
        match_count = min(len(current_sorted), len(synced_records))
        if len(current_sorted) < len(synced_records):
            matched_items = list(reversed(current_sorted))[:match_count]
        else:
            matched_items = current_sorted[:match_count]
        matched.update(str(item.key or "") for item in matched_items if str(item.key or "").strip())
    for item in items:
        item_key = str(item.key or "")
        if not item_key or item_key in matched:
            continue
        group_key = (str(item.source_path or ""), str(item.title or ""))
        if group_key in grouped_records:
            continue
        record = direct_records.get(item_key)
        if record is None:
            continue
        record_source_path = str(record.get("source_path", "") or "")
        if record_source_path and str(item.source_path or "") and record_source_path != str(item.source_path or ""):
            continue
        matched.add(item_key)
    return matched


def _source_task_already_synced(task, *, require_real_issue: bool) -> bool:
    if state_store is None:
        return False
    item_key = str(task.payload.get("item_key", "") or "")
    if not item_key:
        return False
    record = state_store.get_backlog_item(item_key)
    if not isinstance(record, dict):
        return False
    if require_real_issue and bool(record.get("dry_run", False)):
        return False
    return True


def _write_backlog_issue_link(item_key: str, issue_number: int, issue_url: str) -> bool:
    if backlog_source is None:
        return False
    writer = getattr(backlog_source, "annotate_item_issue", None)
    if not callable(writer):
        return False
    return bool(writer(item_key, issue_number=issue_number, issue_url=issue_url))


def _read_backlog_issue_reference(item_key: str) -> dict[str, str | int] | None:
    if backlog_source is None:
        return None
    reader = getattr(backlog_source, "get_item_issue_reference", None)
    if not callable(reader):
        return None
    reference = reader(item_key)
    if not isinstance(reference, dict):
        return None
    issue_number = int(reference.get("issue_number", 0) or 0)
    if issue_number <= 0:
        return None
    return {
        "issue_number": issue_number,
        "issue_url": str(reference.get("issue_url", "") or ""),
    }


async def _adopt_existing_source_issue(
    *,
    task,
    item_key: str,
    latest_item,
    issue: dict,
    issue_title: str,
    result_summary: str,
    log_reason: str,
    queue_trigger: str,
    queue_existing_issue: bool = True,
) -> dict[str, object]:
    assert state_store and source_task_queue
    issue_number = int(issue.get("number", 0) or 0)
    issue_url = str(issue.get("html_url", "") or "")
    raw_source_title = latest_item.title if latest_item is not None else str(task.payload.get("title", "") or "")
    source_path = latest_item.source_path if latest_item is not None else str(task.payload.get("source_path", "") or "")
    line_number = int(latest_item.line_number if latest_item is not None else task.payload.get("line_number", 0) or 0)
    state_store.mark_backlog_item_synced(
        item_key=item_key,
        source_name=task.source,
        source_path=source_path,
        line_number=line_number,
        title=raw_source_title,
        issue_title=issue_title,
        issue_number=issue_number,
        issue_url=issue_url,
        sync_status="created",
    )
    if task.source == "backlog":
        writeback_ok = await asyncio.to_thread(_write_backlog_issue_link, item_key, issue_number, issue_url)
        logger.info(
            "source-task-queue: backlog writeback item=%s issue=%s updated=%s",
            item_key or "-",
            issue_number,
            writeback_ok,
        )
    if queue_existing_issue:
        issue_enqueue = _queue_issue_execution_task(
            issue,
            source=f"source:{task.source}",
            trigger=queue_trigger,
            explicit_command=False,
            allow_bot_user=True,
        )
    else:
        issue_enqueue = {"queued": False, "created": False, "reason": "linked_issue_not_open", "task_id": ""}
    source_task_queue.mark_succeeded(task.task_id, result_summary=result_summary)
    logger.info(
        "source-task-queue: reused existing issue #%s for %s source=%s title=%s reason=%s enqueue_issue_task=%s task=%s",
        issue_number,
        item_key or "-",
        task.source,
        issue_title,
        log_reason,
        bool(issue_enqueue.get("queued", False)),
        str(issue_enqueue.get("task_id", "") or "-"),
    )
    return issue_enqueue


async def _requeue_source_issue_execution_if_needed(
    *,
    item_key: str,
    source_name: str,
) -> dict[str, object]:
    if state_store is None or gh is None:
        return {"queued": False, "created": False, "reason": "issue_lookup_unavailable", "task_id": ""}
    if not item_key:
        return {"queued": False, "created": False, "reason": "missing_item_key", "task_id": ""}
    record = state_store.get_backlog_item(item_key)
    if not isinstance(record, dict):
        return {"queued": False, "created": False, "reason": "missing_backlog_record", "task_id": ""}
    issue_number = int(record.get("issue_number", 0) or 0)
    sync_status = str(record.get("sync_status", "") or "").strip().lower()
    if issue_number <= 0:
        if sync_status == "skipped":
            logger.debug(
                "source-sync: backlog item=%s 已标记为 skipped，跳过 issue 重入队检查",
                item_key,
            )
            return {"queued": False, "created": False, "reason": "skipped_non_actionable", "task_id": ""}
        if sync_status == "dry_run" or bool(record.get("dry_run", False)):
            return {"queued": False, "created": False, "reason": "dry_run", "task_id": ""}
        logger.warning("source-sync: backlog item=%s 缺少 issue_number，可能同步链路断裂", item_key)
        return {"queued": False, "created": False, "reason": "missing_issue_number", "task_id": ""}
    # 持久化冷却：优先用 backlog 记录中的 last_issue_check_at（跨重启），
    # 回退到内存缓存（同进程内去重）。
    last_check_iso = str(record.get("last_issue_check_at", "") or "")
    if last_check_iso:
        try:
            from datetime import datetime, timezone
            last_check_dt = datetime.fromisoformat(last_check_iso.replace("Z", "+00:00"))
            elapsed = (datetime.now(timezone.utc) - last_check_dt).total_seconds()
            if elapsed < _SOURCE_ISSUE_CHECK_COOLDOWN_SECONDS:
                return {"queued": False, "created": False, "reason": "cooldown", "task_id": ""}
        except (ValueError, TypeError):
            pass
    now = time.monotonic()
    last_check_mem = _source_issue_check_cache.get(issue_number, 0.0)
    if last_check_mem > 0 and now - last_check_mem < _SOURCE_ISSUE_CHECK_COOLDOWN_SECONDS:
        return {"queued": False, "created": False, "reason": "cooldown", "task_id": ""}
    issue = await asyncio.to_thread(gh.get_issue, issue_number)
    _source_issue_check_cache[issue_number] = time.monotonic()
    state_store.touch_backlog_issue_check(item_key)
    if "pull_request" in issue:
        return {"queued": False, "created": False, "reason": "pull_request", "task_id": ""}
    if str(issue.get("state", "") or "").lower() != "open":
        return {"queued": False, "created": False, "reason": "closed_issue", "task_id": ""}
    return _queue_issue_execution_task(
        issue,
        source=f"source:{source_name}",
        trigger="source_issue_existing",
        explicit_command=False,
        allow_bot_user=True,
    )


async def _requeue_synced_source_issue_if_needed(task) -> dict[str, object]:
    return await _requeue_source_issue_execution_if_needed(
        item_key=str(task.payload.get("item_key", "") or ""),
        source_name=str(task.source or ""),
    )


def _enqueue_source_issue_create_task(item, *, source_name: str, priority: str):
    assert source_task_queue
    task_id = _build_source_task_id(item.key)
    return source_task_queue.enqueue(
        task_id=task_id,
        target_number=0,
        source=source_name,
        priority=priority,
        command="create_issue",
        payload={
            "item_key": item.key,
            "title": item.title,
            "body": item.body,
            "source_path": item.source_path,
            "line_number": item.line_number,
            "labels": list(item.labels),
        },
        max_attempts=getattr(config.autocode, "source_task_max_attempts", 3) if config else 3,
        replace_existing=True,
    )


async def _drain_source_task_queue_once() -> dict[str, int]:
    """消费 source 队列，把本地条目同步成 GitHub issue。"""
    assert source_task_queue and gh_writer and state_store
    stats = {"claimed": 0, "processed": 0, "failed": 0}
    require_real_issue = _source_sync_requires_real_issue()
    open_autocode_issues: list[dict] | None = None
    max_batch = max(
        int(getattr(config.autocode, "backlog_max_items_per_cycle", 10) or 10),
        int(getattr(config.autocode, "plan_max_items_per_cycle", 10) or 10),
        int(getattr(config.autocode, "idle_scanner_max_items_per_cycle", 5) or 5),
    ) if config else 10
    for _ in range(max_batch):
        task = source_task_queue.claim_next()
        if task is None:
            break
        stats["claimed"] += 1
        item_key = str(task.payload.get("item_key", "") or "")
        try:
            if _source_task_already_synced(task, require_real_issue=require_real_issue):
                issue_enqueue = await _requeue_synced_source_issue_if_needed(task)
                if str(issue_enqueue.get("reason", "") or "") == "closed_issue":
                    logger.info(
                        "source-task-queue: synced item=%s source=%s points to closed issue; continue replacement creation",
                        item_key or "-",
                        task.source,
                    )
                else:
                    source_task_queue.mark_succeeded(task.task_id, result_summary="skipped_existing")
                    logger.info(
                        "source-task-queue: skipped existing item=%s source=%s requeue_issue_task=%s reason=%s task=%s",
                        item_key or "-",
                        task.source,
                        bool(issue_enqueue.get("queued", False)),
                        str(issue_enqueue.get("reason", "") or "-"),
                        str(issue_enqueue.get("task_id", "") or "-"),
                    )
                    stats["processed"] += 1
                    continue
            # source 文件可能在入队后被编辑，所以消费时按 item_key 再解析一次。
            latest_item = await _source_item_by_key(
                task.source,
                item_key,
                expected_source_path=str(task.payload.get("source_path", "") or ""),
                expected_title=str(task.payload.get("title", "") or ""),
                expected_line_number=int(task.payload.get("line_number", 0) or 0),
            ) if item_key else None
            if item_key and latest_item is None and _resolve_source_plan_source(task.source) is not None:
                source_task_queue.mark_succeeded(task.task_id, result_summary="skipped_missing_source")
                _record_state_event(
                    category="source_task_queue",
                    message=f"Skipped source issue creation for missing item {item_key}",
                    details={"task_id": task.task_id, "source": task.source},
                )
                stats["processed"] += 1
                continue
            issue_title = latest_item.title if latest_item is not None else str(task.payload.get("title", "") or "")
            issue_body = latest_item.body if latest_item is not None else str(task.payload.get("body", "") or "")
            issue_labels = (
                list(latest_item.labels)
                if latest_item is not None
                else [str(item) for item in task.payload.get("labels", []) if str(item).strip()]
            )
            source_path = latest_item.source_path if latest_item is not None else str(task.payload.get("source_path", "") or "")
            issue_draft = _draft_source_issue_payload(
                source_name=task.source,
                item_key=item_key,
                raw_title=issue_title,
                raw_body=issue_body,
                raw_labels=issue_labels,
            )
            logger.info(
                "source-task-queue: issue draft source=%s item=%s action=%s optimized=%s title=%s labels=%s body_preview=%s",
                task.source,
                item_key or "-",
                str(issue_draft.get("action", "") or "-"),
                bool(issue_draft.get("optimized", False)),
                str(issue_draft.get("title", "") or "-"),
                ",".join(str(label) for label in issue_draft.get("labels", []) if str(label).strip()) or "-",
                _preview_text(str(issue_draft.get("body", "") or "")),
            )
            if str(issue_draft.get("action", "") or "") == "skip":
                state_store.mark_backlog_item_synced(
                    item_key=item_key,
                    source_name=task.source,
                    source_path=source_path,
                    line_number=int(latest_item.line_number if latest_item is not None else task.payload.get("line_number", 0) or 0),
                    title=issue_title,
                    issue_number=0,
                    issue_url="",
                    sync_status="skipped",
                    sync_reason=str(issue_draft.get("reason", "") or ""),
                )
                source_task_queue.mark_succeeded(task.task_id, result_summary="skipped_non_actionable")
                logger.info(
                    "source-task-queue: skipped source=%s item=%s reason=%s",
                    task.source,
                    item_key or "-",
                    str(issue_draft.get("reason", "") or "-"),
                )
                stats["processed"] += 1
                continue
            issue_title = str(issue_draft.get("title", "") or issue_title)
            issue_body = str(issue_draft.get("body", "") or issue_body)
            issue_labels = [str(label) for label in issue_draft.get("labels", []) if str(label).strip()]
            if task.source == "backlog" and item_key:
                linked_reference = await asyncio.to_thread(_read_backlog_issue_reference, item_key)
                if linked_reference is not None:
                    linked_issue_number = int(linked_reference.get("issue_number", 0) or 0)
                    linked_issue_url = str(linked_reference.get("issue_url", "") or "")
                    linked_issue: dict = {
                        "number": linked_issue_number,
                        "html_url": linked_issue_url,
                        "title": issue_title,
                        "state": "open",
                        "updated_at": "",
                        "created_at": "",
                        "user": {"login": "autocode", "type": "Bot"},
                    }
                    if gh is not None and linked_issue_number > 0:
                        try:
                            linked_issue = await asyncio.to_thread(gh.get_issue, linked_issue_number)
                        except Exception as exc:
                            logger.warning(
                                "source-task-queue: failed to fetch linked backlog issue #%s for %s: %s",
                                linked_issue_number,
                                item_key or "-",
                                exc,
                            )
                    linked_issue.setdefault("number", linked_issue_number)
                    linked_issue.setdefault("html_url", linked_issue_url)
                    linked_issue.setdefault("title", issue_title)
                    is_open_issue = str(linked_issue.get("state", "") or "open").lower() == "open"
                    if is_open_issue:
                        await _adopt_existing_source_issue(
                            task=task,
                            item_key=item_key,
                            latest_item=latest_item,
                            issue=linked_issue,
                            issue_title=str(linked_issue.get("title", "") or issue_title),
                            result_summary=f"linked_issue:{linked_issue_number}",
                            log_reason="backlog_issue_link",
                            queue_trigger="source_issue_existing",
                            queue_existing_issue=True,
                        )
                        stats["processed"] += 1
                        continue
                    logger.info(
                        "source-task-queue: linked backlog issue #%s for %s is closed; creating replacement issue",
                        linked_issue_number,
                        item_key or "-",
                    )
            if open_autocode_issues is None:
                open_autocode_issues = await asyncio.to_thread(_list_open_autocode_issues)
            existing_open_issue = _find_matching_open_autocode_issue(
                open_autocode_issues,
                title=issue_title,
                labels=issue_labels,
            )
            if existing_open_issue is not None:
                await _adopt_existing_source_issue(
                    task=task,
                    item_key=item_key,
                    latest_item=latest_item,
                    issue=existing_open_issue,
                    issue_title=str(existing_open_issue.get("title", "") or issue_title),
                    result_summary=f"existing_issue:{int(existing_open_issue.get('number', 0) or 0)}",
                    log_reason="matching_open_autocode_issue",
                    queue_trigger="source_issue_existing",
                )
                stats["processed"] += 1
                continue
            capacity_block_reason = _source_issue_capacity_block_reason(
                open_autocode_issues,
                title=issue_title,
                body=issue_body,
                labels=issue_labels,
                raw_body=str(latest_item.body if latest_item is not None else task.payload.get("body", "") or ""),
            )
            if capacity_block_reason:
                deferred_task = source_task_queue.defer(
                    task.task_id,
                    reason=capacity_block_reason,
                    delay_seconds=max(float(getattr(config.server, "poll_interval_seconds", 300) or 300), 300.0) if config else 300.0,
                )
                logger.info(
                    "source-task-queue: deferred source=%s item=%s reason=%s available_at=%s",
                    task.source,
                    item_key or "-",
                    capacity_block_reason,
                    str(deferred_task.available_at if deferred_task is not None else "") or "-",
                )
                _record_state_event(
                    category="source_task_queue",
                    message=f"Deferred source issue creation for {item_key or task.task_id}",
                    details={
                        "task_id": task.task_id,
                        "source": task.source,
                        "reason": capacity_block_reason,
                    },
                )
                stats["processed"] += 1
                continue
            logger.info(
                "source-task-queue: creating issue source=%s item=%s title=%s labels=%s body_preview=%s",
                task.source,
                item_key or "-",
                issue_title,
                ",".join(issue_labels) or "-",
                _preview_text(issue_body),
            )
            created_issue = await asyncio.to_thread(
                gh_writer.create_issue,
                title=issue_title,
                body=issue_body,
                labels=issue_labels,
            )
            if not created_issue:
                if _is_dry_run_enabled():
                    state_store.mark_backlog_item_synced(
                        item_key=item_key,
                        source_name=task.source,
                        source_path=source_path,
                        line_number=int(latest_item.line_number if latest_item is not None else task.payload.get("line_number", 0) or 0),
                        title=issue_title,
                        issue_number=0,
                        issue_url="",
                        dry_run=True,
                    )
                    source_task_queue.mark_succeeded(task.task_id, result_summary="dry_run")
                    stats["processed"] += 1
                    continue
                raise RuntimeError("source issue creation returned empty result")
            issue_number = int(created_issue.get("number", 0) or 0)
            issue_url = str(created_issue.get("html_url", "") or "")
            raw_source_title = latest_item.title if latest_item is not None else str(task.payload.get("title", "") or "")
            state_store.mark_backlog_item_synced(
                item_key=item_key,
                source_name=task.source,
                source_path=source_path,
                line_number=int(latest_item.line_number if latest_item is not None else task.payload.get("line_number", 0) or 0),
                title=raw_source_title,
                issue_title=issue_title,
                issue_number=issue_number,
                issue_url=issue_url,
            )
            if task.source == "backlog":
                writeback_ok = await asyncio.to_thread(_write_backlog_issue_link, item_key, issue_number, issue_url)
                logger.info(
                    "source-task-queue: backlog writeback item=%s issue=%s updated=%s",
                    item_key or "-",
                    issue_number,
                    writeback_ok,
                )
            issue_enqueue = _queue_issue_execution_task(
                created_issue,
                source=f"source:{task.source}",
                trigger="source_issue_create",
                explicit_command=False,
                allow_bot_user=True,
            )
            source_task_queue.mark_succeeded(task.task_id, result_summary=f"issue:{issue_number}")
            if open_autocode_issues is not None:
                open_autocode_issues.append(created_issue)
            logger.info(
                "source-task-queue: created issue #%s for %s source=%s title=%s labels=%s url=%s enqueue_issue_task=%s reason=%s task=%s",
                issue_number,
                item_key or "-",
                task.source,
                issue_title,
                ",".join(issue_labels) or "-",
                issue_url or "-",
                bool(issue_enqueue.get("queued", False)),
                str(issue_enqueue.get("reason", "") or "-"),
                str(issue_enqueue.get("task_id", "") or "-"),
            )
            _record_state_event(
                category="source_task_queue",
                message=f"Created issue #{issue_number} for source item {item_key}",
                details={
                    "task_id": task.task_id,
                    "issue_number": issue_number,
                    "source": task.source,
                    "issue_task_queued": bool(issue_enqueue.get("queued", False)),
                    "issue_task_id": str(issue_enqueue.get("task_id", "") or ""),
                    "issue_task_reason": str(issue_enqueue.get("reason", "") or ""),
                },
            )
            stats["processed"] += 1
        except Exception as exc:
            failed_task = source_task_queue.mark_failed(
                task.task_id,
                error=str(exc),
                retry_delay_seconds=_source_task_retry_delay_seconds(task.attempt_count),
            )
            _record_state_event(
                category="source_task_queue",
                level="error",
                message=f"Source issue creation failed for {item_key or task.task_id}",
                details={
                    "task_id": task.task_id,
                    "error": str(exc),
                    "queue_status": failed_task.status if failed_task is not None else "",
                },
            )
            logger.exception("source-task-queue: failed to process task %s", task.task_id)
            stats["failed"] += 1
    if stats["claimed"] == 0:
        logger.debug("source-task-queue: no queued source task this cycle")
    return stats


async def _drain_pr_task_queue_once() -> dict[str, int]:
    """消费 PR 修复队列，在原分支上处理最新 review feedback。"""
    assert pr_task_queue and gh and controller and state_store
    stats = {"claimed": 0, "processed": 0, "failed": 0}
    max_batch = max(int(getattr(config.autocode, "pr_queue_max_batch", 1) or 1), 1) if config else 1
    for _ in range(max_batch):
        task = pr_task_queue.claim_next()
        if task is None:
            break
        stats["claimed"] += 1
        pr_number = task.target_number
        logger.info(
            "pr-task-queue: 开始处理 PR 修复任务 task=%s pr=%s attempt=%s/%s feedback_key=%s source=%s",
            task.task_id,
            pr_number,
            task.attempt_count,
            task.max_attempts,
            str(task.payload.get("feedback_key", "") or ""),
            task.source,
        )
        try:
            if state_store.is_tracked_pr_paused(pr_number):
                pause = state_store.get_tracked_pr_pause(pr_number) or {}
                state_store.mark_tracked_pr_status(
                    pr_number,
                    status="paused",
                    summary=f"PR #{pr_number} 已暂停自动修复: {pause.get('reason', '无说明')}",
                )
                pr_task_queue.mark_succeeded(task.task_id, result_summary="skipped_paused")
                stats["processed"] += 1
                continue
            pr = await asyncio.to_thread(gh.get_pr, pr_number)
            logger.info(
                "pr-task-queue: 已获取最新 PR #%s title=%s state=%s head=%s base=%s",
                pr_number,
                _preview_text(str(pr.get("title", "") or "")),
                str(pr.get("state", "") or "-"),
                str(pr.get("head", {}).get("ref", "") or "-"),
                str(pr.get("base", {}).get("ref", "") or "-"),
            )
            if str(pr.get("state", "") or "").lower() == "closed":
                state_store.mark_tracked_pr_status(pr_number, status="closed", summary="PR 已关闭。")
                pr_task_queue.mark_succeeded(task.task_id, result_summary="skipped_closed")
                stats["processed"] += 1
                continue
            cached_feedback = task.payload.get("feedback") if isinstance(task.payload.get("feedback"), list) else None
            summary = await asyncio.to_thread(
                controller.process_pull_request,
                pr,
                allow_repair=True,
                current_task_id=task.task_id,
                current_feedback_key=str(task.payload.get("feedback_key", "") or ""),
                cached_feedback=cached_feedback,
                cached_review_state=str(task.payload.get("review_state", "") or ""),
                cached_legacy_feedback_key=str(task.payload.get("legacy_feedback_key", "") or ""),
            )
            pr_task_queue.mark_succeeded(task.task_id, result_summary=summary)
            logger.info(
                "pr-task-queue: 任务完成 task=%s pr=%s summary=%s",
                task.task_id,
                pr_number,
                _preview_text(summary),
            )
            _record_state_event(
                category="pr_task_queue",
                message=f"Processed queued PR repair task for PR #{pr_number}",
                target_kind="pull_request",
                target_number=pr_number,
                details={"task_id": task.task_id, "summary": summary},
            )
            stats["processed"] += 1
        except Exception as exc:
            failed_task = pr_task_queue.mark_failed(
                task.task_id,
                error=str(exc),
                retry_delay_seconds=_pr_task_retry_delay_seconds(task.attempt_count),
            )
            _record_state_event(
                category="pr_task_queue",
                level="error",
                message=f"Queued PR repair task failed for PR #{pr_number}",
                target_kind="pull_request",
                target_number=pr_number,
                details={
                    "task_id": task.task_id,
                    "error": str(exc),
                    "queue_status": failed_task.status if failed_task is not None else "",
                },
            )
            logger.exception("pr-task-queue: failed to process task %s", task.task_id)
            logger.error(
                "pr-task-queue: task=%s pr=%s attempt=%s/%s transitioned_to=%s next_available_at=%s feedback_key=%s last_error=%s",
                task.task_id,
                pr_number,
                task.attempt_count,
                task.max_attempts,
                failed_task.status if failed_task is not None else "",
                failed_task.available_at if failed_task is not None else "",
                str(task.payload.get("feedback_key", "") or ""),
                str(exc),
            )
            stats["failed"] += 1
        finally:
            state_store.release_feedback(
                str(task.payload.get("feedback_key", "") or ""),
                owner=task.task_id,
            )
    return stats


def _should_run_idle_scan() -> bool:
    assert state_store
    state = state_store.summary()
    if int(state.get("pending_tracked_prs", 0) or 0) > 0:
        return False
    # Only block on issue/source tasks — PR repair tasks are routine
    # maintenance and should not prevent idle-scan from generating new work.
    non_pr_active = state_store.count_active_tasks(
        exclude_types={"pr_feedback_repair"},
    )
    if non_pr_active > 0:
        return False
    if scheduler is None:
        return True
    scheduled_tasks = scheduler.summary().get("tasks", [])
    for item in scheduled_tasks:
        if str(item.get("name", "") or "") in _IDLE_SCAN_EXCLUDED_CYCLES:
            continue
        if str(item.get("status", "")) in {"running", "retry_waiting"}:
            return False
    return True


async def _safe_source_sync_once(source_name: str, handler) -> dict[str, int]:
    try:
        return await handler()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("AutoCode %s source sync failed", source_name)
        _record_state_event(
            category="source_sync",
            level="error",
            message=f"{source_name} source sync failed",
            source=source_name,
            details={"error": str(exc)},
        )
        return {"fetched": 0, "queued": 0, "failed": 1}


_POLL_STEP_DEFAULTS: dict[str, int] = {"fetched": 0, "eligible": 0, "processed": 0}


async def _safe_poll_step(step_name: str, handler) -> dict[str, int]:
    """Run a polling sub-step with error isolation.

    Prevents transient failures (API timeouts, rate limits) from propagating
    to the scheduler and eventually dead-lettering the entire polling cycle.
    """
    try:
        return await handler()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("AutoCode %s poll step failed", step_name)
        _record_state_event(
            category="poll_step",
            level="error",
            message=f"{step_name} poll step failed",
            source=step_name,
            details={"error": str(exc)},
        )
        return dict(_POLL_STEP_DEFAULTS)


async def _poll_once() -> dict[str, dict[str, int]]:
    """执行一轮 polling 主流程。

    当前顺序是：先同步 plan/backlog 来源，再轮询 issue events / issue comments，
    最后补偿扫描仍处于 open 状态但可能错过事件的 issue。这样从
    本地文件派生、GitHub 原生创建、评论批准，以及历史遗留未入队的问题都能
    共用同一套后续 issue 管线。
    """
    assert config and gh and gh_writer and state_store
    if state_store.ensure_polling_baseline():
        logger.info(
            "AutoCode polling 基线已初始化: issue_event_after=%s",
            state_store.issue_baseline(),
        )

    plan_stats = await _safe_source_sync_once("plan", _sync_plan_files_once)
    backlog_stats = await _safe_source_sync_once("backlog", _sync_backlog_once)
    issue_stats = await _safe_poll_step("issue_poll", _poll_issues_once)
    comment_stats = await _safe_poll_step("issue_comment_poll", _poll_issue_comments_once)
    reconcile_stats = await _safe_poll_step("reconcile", _reconcile_open_issues_once)

    # Periodic housekeeping: prune old completed tasks to prevent unbounded
    # state file growth.  Runs every poll cycle but is very cheap (no I/O
    # unless tasks are actually removed).
    try:
        pruned = state_store.cleanup_completed_tasks()
        if pruned:
            logger.info("task-queue-cleanup: pruned=%d old completed tasks", pruned)
    except Exception:
        logger.debug("task-queue-cleanup: failed (non-critical)", exc_info=True)

    logger.info(
        "AutoCode polling 完成: plans fetched=%d queued=%d; backlog fetched=%d queued=%d; issues fetched=%d eligible=%d processed=%d; comments fetched=%d eligible=%d processed=%d; reconcile fetched=%d eligible=%d processed=%d",
        plan_stats["fetched"],
        plan_stats["queued"],
        backlog_stats["fetched"],
        backlog_stats["queued"],
        issue_stats["fetched"],
        issue_stats["eligible"],
        issue_stats["processed"],
        comment_stats["fetched"],
        comment_stats["eligible"],
        comment_stats["processed"],
        reconcile_stats["fetched"],
        reconcile_stats["eligible"],
        reconcile_stats["processed"],
    )
    return {
        "plans": plan_stats,
        "backlog": backlog_stats,
        "issues": issue_stats,
        "comments": comment_stats,
        "reconcile": reconcile_stats,
    }


async def _poll_issues_once() -> dict[str, int]:
    """基于仓库级 issue events 轮询新 issue。

    这里只认 `opened/reopened` 事件，而不是盯着 issue `updated_at`，目的是
    避免普通编辑、同步或评论把 issue 误判成新的执行任务。
    """
    assert config and gh and state_store
    baseline, baseline_id = state_store.issue_event_cursor()
    newest_created_at = baseline
    newest_event_id = baseline_id
    page = 1
    stats = {"fetched": 0, "eligible": 0, "processed": 0}
    new_event_count = 0
    action_counts: Counter[str] = Counter()
    action_samples: list[str] = []

    while True:
        events = await asyncio.to_thread(
            gh.list_repo_issue_events,
            per_page=config.server.poll_items_per_page,
            page=page,
        )
        if not events:
            break
        stats["fetched"] += len(events)

        new_events = [
            item
            for item in events
            if _is_after_cursor(
                str(item.get("created_at", "") or ""),
                int(item.get("id", 0) or 0),
                baseline,
                baseline_id,
            )
        ]
        new_events = sorted(
            new_events,
            key=lambda item: (str(item.get("created_at", "") or ""), int(item.get("id", 0) or 0)),
        )
        new_event_count += len(new_events)
        for event in new_events:
            event_id = event.get("id")
            action = str(event.get("event", "") or "")
            created_at = str(event.get("created_at", "") or "")
            action_counts[action or "unknown"] += 1
            if len(action_samples) < 8:
                action_samples.append(
                    f"{action or 'unknown'}#{_issue_number_from_issue_event(event) or '?'}"
                )
            if not isinstance(event_id, int):
                continue
            newest_created_at, newest_event_id = _max_cursor(
                newest_created_at,
                newest_event_id,
                created_at,
                event_id,
            )
            if action not in {"opened", "reopened"}:
                continue
            # 先按 event_id 去重，再决定是否把它交给 issue 入队逻辑。
            if state_store.has_processed_issue_event(event_id):
                continue
            issue_number = _issue_number_from_issue_event(event)
            if issue_number <= 0:
                continue
            claim = state_store.try_claim_issue_event(event_id)
            if claim != "claimed":
                continue
            processed = False
            try:
                issue = await asyncio.to_thread(gh.get_issue, issue_number)
                if "pull_request" in issue:
                    processed = True
                    continue
                if issue.get("state") != "open":
                    processed = True
                    continue
                stats["eligible"] += 1
                # 轮询只负责发现事件；是否入队由统一的 issue 事件处理器决定。
                await _handle_issue_event(issue, source="polling")
                stats["processed"] += 1
                processed = True
            finally:
                if processed:
                    state_store.mark_issue_event_processed(
                        event_id=event_id,
                        issue_number=issue_number,
                        action=action,
                        created_at=created_at,
                        source="polling",
                    )
                state_store.release_issue_event(event_id)
        if len(events) < config.server.poll_items_per_page or not new_events:
            break
        page += 1
    if newest_created_at and (newest_created_at != baseline or newest_event_id != baseline_id):
        state_store.advance_issue_baseline(newest_created_at, newest_event_id)
    action_summary = ", ".join(
        f"{name}:{count}" for name, count in action_counts.most_common(6)
    ) or "-"
    sample_summary = ", ".join(action_samples) or "-"
    logger.info(
        "issue-poll: fetched=%d new=%d eligible=%d processed=%d actions=[%s] sample=[%s] baseline=%s#%s newest=%s#%s",
        stats["fetched"],
        new_event_count,
        stats["eligible"],
        stats["processed"],
        action_summary,
        sample_summary,
        baseline or "-",
        baseline_id,
        newest_created_at or baseline or "-",
        newest_event_id,
    )
    return stats


async def _reconcile_open_issues_once() -> dict[str, int]:
    """补偿扫描仍处于 open 的 issue，避免漏掉历史存量、人工 issue 或事件缺失。"""
    assert config and gh and state_store
    if not bool(getattr(config.autocode, "auto_plan_on_issue_open", False)):
        logger.info("issue-reconcile: auto plan on issue open disabled, skipping")
        return {"fetched": 0, "eligible": 0, "processed": 0}
    open_issues = await asyncio.to_thread(_list_open_reconcile_issues)
    cluster_summary = _summarize_open_issue_clusters(open_issues)
    stats = {"fetched": len(open_issues), "eligible": 0, "processed": 0}
    remaining_budget = max(int(getattr(config.server, "poll_items_per_page", 30) or 0), 1)
    for issue in open_issues:
        issue_number = int(issue.get("number", 0) or 0)
        if issue_number <= 0:
            continue
        if str(issue.get("state", "") or "").lower() != "open":
            continue
        issue_updated_at = str(issue.get("updated_at") or issue.get("created_at") or "")
        if issue_updated_at and state_store.has_processed_issue_version(issue_number, issue_updated_at):
            last_outcome = state_store.get_issue_last_outcome(issue_number)
            if not last_outcome or last_outcome in _ISSUE_TERMINAL_SUCCESS_OUTCOMES:
                continue
            # Outcomes that require explicit human action should not be
            # auto-retried by reconcile (use POST /issues/{N}/implement).
            if last_outcome in _ISSUE_RECONCILE_SKIP_OUTCOMES:
                continue
            # Issue was processed but not successfully executed — eligible for retry.
        else:
            # New version not yet recorded. Skip if previous outcome was terminal —
            # the updated_at change is almost certainly from an autocode bot comment,
            # which would create an infinite self-loop if re-triggered.
            last_outcome = state_store.get_issue_last_outcome(issue_number)
            if last_outcome in _ISSUE_TERMINAL_SUCCESS_OUTCOMES:
                continue
        stats["eligible"] += 1
        if remaining_budget <= 0:
            continue
        labels = {label.lower() for label in _github_issue_label_names(issue)}
        enqueue_result = _queue_issue_execution_task(
            issue,
            source="reconcile",
            trigger="issue_reconcile",
            explicit_command=False,
            allow_bot_user="autocode" in labels,
        )
        if bool(enqueue_result.get("queued", False)):
            stats["processed"] += 1
            remaining_budget -= 1
    logger.info(
        "issue-reconcile: fetched=%d eligible=%d processed=%d clusters=[%s]",
        stats["fetched"],
        stats["eligible"],
        stats["processed"],
        _format_issue_cluster_summary(cluster_summary),
    )
    return stats


async def _reconcile_open_autocode_issues_once() -> dict[str, int]:
    """Backward-compatible alias for legacy callers/tests."""
    return await _reconcile_open_issues_once()


def _summarize_open_issue_clusters(open_issues: list[dict]) -> dict[str, int]:
    clusters: dict[str, int] = {}
    for issue in open_issues:
        combined = " ".join(
            str(part or "")
            for part in (
                issue.get("title", ""),
                issue.get("body", ""),
                " ".join(_github_issue_label_names(issue)),
            )
        ).lower()
        if not combined.strip():
            continue
        for cluster_name, keywords in _ISSUE_CLUSTER_RULES:
            if any(str(keyword).lower() in combined for keyword in keywords):
                clusters[cluster_name] = clusters.get(cluster_name, 0) + 1
                break
    return dict(sorted(clusters.items(), key=lambda item: (-item[1], item[0])))


def _format_issue_cluster_summary(clusters: dict[str, int], *, limit: int = 5) -> str:
    if not clusters:
        return "-"
    return ", ".join(f"{name}:{count}" for name, count in list(clusters.items())[:limit])


async def _poll_issue_comments_once() -> dict[str, int]:
    """轮询 issue comments，补偿 webhook 漏单或 edited comment 场景。"""
    assert config and gh and state_store
    baseline, baseline_id = state_store.issue_comment_cursor()
    newest_updated_at = baseline
    newest_comment_id = baseline_id
    page = 1
    stats = {"fetched": 0, "eligible": 0, "processed": 0}
    new_comment_count = 0
    sample_comments: list[str] = []

    while True:
        comments = await asyncio.to_thread(
            gh.list_repo_issue_comments,
            since=baseline or None,
            per_page=config.server.poll_items_per_page,
            page=page,
        )
        if not comments:
            break
        stats["fetched"] += len(comments)
        new_comments = [
            item
            for item in comments
            if _is_after_cursor(
                str(item.get("updated_at") or item.get("created_at") or ""),
                int(item.get("id", 0) or 0),
                baseline,
                baseline_id,
            )
        ]
        new_comments = sorted(
            new_comments,
            key=lambda item: (
                str(item.get("updated_at") or item.get("created_at") or ""),
                int(item.get("id", 0) or 0),
            ),
        )
        new_comment_count += len(new_comments)
        for comment in new_comments:
            comment_id = comment.get("id")
            updated_at = str(comment.get("updated_at") or comment.get("created_at") or "")
            if not isinstance(comment_id, int):
                continue
            issue_number = _issue_number_from_issue_url(str(comment.get("issue_url", "") or ""))
            if issue_number > 0 and len(sample_comments) < 8:
                sample_comments.append(f"{issue_number}#{comment_id}")
            claim = state_store.try_claim_issue_comment(comment_id, updated_at)
            if claim != "claimed":
                newest_updated_at, newest_comment_id = _max_cursor(
                    newest_updated_at,
                    newest_comment_id,
                    updated_at,
                    comment_id,
                )
                continue
            try:
                if issue_number <= 0:
                    _mark_issue_comment_processed(
                        comment_id=comment_id,
                        issue_number=0,
                        updated_at=updated_at,
                        command="",
                        source="polling",
                    )
                    newest_updated_at, newest_comment_id = _max_cursor(
                        newest_updated_at,
                        newest_comment_id,
                        updated_at,
                        comment_id,
                    )
                    continue
                try:
                    issue = await asyncio.to_thread(gh.get_issue, issue_number)
                except Exception as exc:
                    logger.warning(
                        "issue-comment-poll: 获取 issue #%s 失败，跳过该评论 comment_id=%s error=%s",
                        issue_number,
                        comment_id,
                        exc,
                    )
                    _record_state_event(
                        category="issue_comment",
                        level="error",
                        message=f"Failed to load issue #{issue_number} for comment poll",
                        source="polling",
                        target_kind="issue",
                        target_number=issue_number,
                        details={"comment_id": comment_id, "error": str(exc), "reason": "issue_fetch_failed"},
                    )
                    _mark_issue_comment_processed(
                        comment_id=comment_id,
                        issue_number=issue_number,
                        updated_at=updated_at,
                        command="",
                        source="polling",
                    )
                    newest_updated_at, newest_comment_id = _max_cursor(
                        newest_updated_at,
                        newest_comment_id,
                        updated_at,
                        comment_id,
                    )
                    continue
                if "pull_request" in issue:
                    _mark_issue_comment_processed(
                        comment_id=comment_id,
                        issue_number=issue_number,
                        updated_at=updated_at,
                        command="",
                        source="polling",
                    )
                    newest_updated_at, newest_comment_id = _max_cursor(
                        newest_updated_at,
                        newest_comment_id,
                        updated_at,
                        comment_id,
                    )
                    continue
                if str(issue.get("state", "") or "").lower() != "open":
                    _mark_issue_comment_processed(
                        comment_id=comment_id,
                        issue_number=issue_number,
                        updated_at=updated_at,
                        command="",
                        source="polling",
                    )
                    newest_updated_at, newest_comment_id = _max_cursor(
                        newest_updated_at,
                        newest_comment_id,
                        updated_at,
                        comment_id,
                    )
                    continue
                stats["eligible"] += 1
                await _handle_issue_comment_event({"issue": issue, "comment": comment}, source="polling")
                stats["processed"] += 1
                newest_updated_at, newest_comment_id = _max_cursor(
                    newest_updated_at,
                    newest_comment_id,
                    updated_at,
                    comment_id,
                )
            finally:
                state_store.release_issue_comment(comment_id)
        if len(comments) < config.server.poll_items_per_page or not new_comments:
            break
        page += 1
    if newest_updated_at and (newest_updated_at != baseline or newest_comment_id != baseline_id):
        state_store.advance_issue_comment_baseline(newest_updated_at, newest_comment_id)
    logger.info(
        "issue-comment-poll: fetched=%d new=%d eligible=%d processed=%d baseline=%s#%s newest=%s#%s sample=[%s]",
        stats["fetched"],
        new_comment_count,
        stats["eligible"],
        stats["processed"],
        baseline or "-",
        baseline_id,
        newest_updated_at or baseline or "-",
        newest_comment_id,
        ", ".join(sample_comments) or "-",
    )
    return stats


async def _sync_backlog_once() -> dict[str, int]:
    assert config and state_store
    if not config.autocode.enable_backlog_source:
        return {"fetched": 0, "queued": 0}
    return await _enqueue_plan_source_items_once(
        plan_source=backlog_source,
        source_name="backlog",
        priority="background",
    )


async def _sync_plan_files_once() -> dict[str, int]:
    assert config and state_store
    if not config.autocode.enable_plan_source:
        return {"fetched": 0, "queued": 0}
    return await _enqueue_plan_source_items_once(
        plan_source=plan_source,
        source_name="plan",
        priority="background",
    )


async def _handle_pull_request_event(pr: dict, *, source: str, action: str):
    assert controller and state_store
    pr_number = int(pr.get("number", 0) or 0)
    if pr_number <= 0:
        return
    if action == "closed":
        state_store.mark_tracked_pr_status(pr_number, status="closed", summary="PR 已关闭。")
        _record_state_event(
            category="pull_request",
            message=f"PR #{pr_number} marked closed",
            source=source,
            target_kind="pull_request",
            target_number=pr_number,
            details={"action": action, "status": "closed"},
        )
        return
    summary, task = await _schedule_pull_request_repair(pr, source=source, priority="normal")
    logger.info("pull-request-event: %s", summary)
    _record_state_event(
        category="pull_request",
        message=f"Processed PR #{pr_number} event {action}",
        source=source,
        target_kind="pull_request",
        target_number=pr_number,
        details={"action": action, "summary": summary, "task_id": task.task_id if task is not None else ""},
    )


def _mark_cycle_started(cycle_name: str):
    if state_store is None:
        return
    try:
        state_store.mark_cycle_started(cycle_name)
    except Exception:
        logger.exception("记录周期开始状态失败: %s", cycle_name)


def _record_cycle_result(cycle_name: str, *, status: str, stats: dict | None = None, error: str = ""):
    if state_store is None:
        return
    try:
        state_store.record_cycle_result(cycle_name, status=status, stats=stats, error=error)
    except Exception:
        logger.exception("记录周期结果失败: %s", cycle_name)


def _increment_state_counter(name: str, amount: int = 1):
    if state_store is None:
        return
    try:
        state_store.increment_counter(name, amount=amount)
    except Exception:
        logger.exception("记录可观测计数失败: %s", name)


def _record_state_event(
    *,
    category: str,
    message: str,
    level: str = "info",
    source: str = "",
    target_kind: str = "",
    target_number: int = 0,
    details: dict | None = None,
):
    if state_store is None:
        return
    try:
        state_store.record_event(
            category=category,
            message=message,
            level=level,
            source=source,
            target_kind=target_kind,
            target_number=target_number,
            details=details,
        )
    except Exception:
        logger.exception("记录可观测事件失败: %s", category)


def _build_issue_task_id(issue_number: int, issue_updated_at: str, source: str, *, trigger: str) -> str:
    digest = hashlib.sha256(
        f"{issue_number}\n{issue_updated_at}\n{source}\n{trigger}".encode("utf-8")
    ).hexdigest()[:16]
    return f"issue-task:{issue_number}:{digest}"


def _build_manual_issue_task_id(issue_number: int, issue_updated_at: str) -> str:
    digest = hashlib.sha256(
        f"{issue_number}\n{issue_updated_at}\nmanual_implement".encode("utf-8")
    ).hexdigest()[:16]
    return f"issue-task:{issue_number}:{digest}"

def _issue_task_retry_delay_seconds(attempt_count: int) -> float:
    exponent = max(int(attempt_count or 0) - 1, 0)
    return min(30.0 * (2 ** exponent), 900.0)


_PERMANENT_FAILURE_PATTERNS = (
    "删除行数超限",
    "新增行数超限",
    "改动文件数超限",
    "命中受限路径",
    "越界改动",
    "未包含测试改动",
)


def _is_permanent_task_failure(error: str) -> bool:
    text = str(error or "")
    return any(pattern in text for pattern in _PERMANENT_FAILURE_PATTERNS)


def _issue_task_queue_summary() -> dict:
    return _task_queue_summary(issue_task_queue, serializer=_issue_task_to_dict)


def _pr_task_retry_delay_seconds(attempt_count: int) -> float:
    exponent = max(int(attempt_count or 0) - 1, 0)
    return min(30.0 * (2 ** exponent), 900.0)


def _pr_task_queue_summary() -> dict:
    return _task_queue_summary(pr_task_queue, serializer=_pr_task_to_dict)


def _source_task_retry_delay_seconds(attempt_count: int) -> float:
    exponent = max(int(attempt_count or 0) - 1, 0)
    return min(30.0 * (2 ** exponent), 900.0)


def _source_task_queue_summary() -> dict:
    return _task_queue_summary(source_task_queue, serializer=_source_task_to_dict)


def _all_task_queue_summaries() -> dict[str, dict]:
    return {
        "issue": _issue_task_queue_summary(),
        "pr": _pr_task_queue_summary(),
        "source": _source_task_queue_summary(),
    }


def _aggregate_task_queue_summary(queue_summaries: dict[str, dict]) -> dict:
    summary = {
        "queues": 0,
        "total": 0,
        "queued": 0,
        "running": 0,
        "retry_waiting": 0,
        "dead_letter": 0,
        "dead_letter_action_required": 0,
        "dead_letter_superseded": 0,
        "problem": 0,
    }
    for item in queue_summaries.values():
        if not item:
            continue
        summary["queues"] += 1
        for key in (
            "total",
            "queued",
            "running",
            "retry_waiting",
            "dead_letter",
            "dead_letter_action_required",
            "dead_letter_superseded",
            "problem",
        ):
            summary[key] += int(item.get(key, 0) or 0)
    return summary


def _task_queue_bindings() -> dict[str, dict]:
    return {
        "issue": {
            "queue": issue_task_queue,
            "serializer": _issue_task_to_dict,
            "counter": "issue_task_queue.manual_retry",
            "category": "issue_task_queue_recovery",
            "display_name": "issue",
            "detail_name": "issue task",
            "not_initialized": "Issue task queue is not initialized",
            "target_kind": "issue",
        },
        "pr": {
            "queue": pr_task_queue,
            "serializer": _pr_task_to_dict,
            "counter": "pr_task_queue.manual_retry",
            "category": "pr_task_queue_recovery",
            "display_name": "PR",
            "detail_name": "PR task",
            "not_initialized": "PR task queue is not initialized",
            "target_kind": "pull_request",
        },
        "source": {
            "queue": source_task_queue,
            "serializer": _source_task_to_dict,
            "counter": "source_task_queue.manual_retry",
            "category": "source_task_queue_recovery",
            "display_name": "source",
            "detail_name": "source task",
            "not_initialized": "Source task queue is not initialized",
            "target_kind": "",
        },
    }


def _render_metrics() -> str:
    state = state_store.summary() if state_store else {}
    observability = state_store.observability_summary() if state_store else {}
    scheduler_summary = scheduler.summary() if scheduler else {}
    queue_summaries = _all_task_queue_summaries()
    lines = [
        _metric_line("autocode_up", 1),
        _metric_line("autocode_execution_enabled", _bool_metric(config.autocode.enabled if config else False)),
        _metric_line("autocode_webhook_enabled", _bool_metric(config.server.enable_webhook if config else False)),
        _metric_line("autocode_pr_queue_enabled", _bool_metric(config.autocode.enable_pr_queue if config else False)),
        _metric_line("autocode_plan_source_enabled", _bool_metric(config.autocode.enable_plan_source if config else False)),
        _metric_line(
            "autocode_backlog_source_enabled",
            _bool_metric(config.autocode.enable_backlog_source if config else False),
        ),
        _metric_line(
            "autocode_idle_scanner_enabled",
            _bool_metric(config.autocode.enable_idle_scanner if config else False),
        ),
        _metric_line("autocode_scheduler_registered_tasks", int(scheduler_summary.get("registered", 0) or 0)),
        _metric_line("autocode_scheduler_running_tasks", int(scheduler_summary.get("running", 0) or 0)),
        _metric_line("autocode_scheduler_retry_waiting_tasks", int(scheduler_summary.get("retry_waiting", 0) or 0)),
        _metric_line("autocode_scheduler_dead_letter_tasks", int(scheduler_summary.get("dead_letter", 0) or 0)),
    ]
    for queue_name, summary in queue_summaries.items():
        if not summary:
            continue
        for status_name in (
            "total",
            "queued",
            "running",
            "retry_waiting",
            "dead_letter",
            "dead_letter_action_required",
            "dead_letter_superseded",
            "problem",
        ):
            lines.append(
                _metric_line(
                    "autocode_task_queue_tasks",
                    int(summary.get(status_name, 0) or 0),
                    labels={"queue": queue_name, "status": status_name},
                )
            )
    for key in (
        "planned_issues",
        "frozen_issue_plans",
        "processed_issue_events",
        "processed_issue_comments",
        "processed_feedback_items",
        "paused_issues",
        "paused_tracked_prs",
        "backlog_items",
        "tracked_prs",
        "pending_tracked_prs",
        "runs",
        "queued_tasks",
        "running_tasks",
        "retry_waiting_tasks",
        "dead_letter_tasks",
        "observed_cycles",
        "recent_events",
    ):
        lines.append(
            _metric_line(
                "autocode_state_summary",
                int(state.get(key, 0) or 0),
                labels={"name": key},
            )
        )
    for name, value in dict(observability.get("counters", {}) or {}).items():
        lines.append(_metric_line("autocode_observability_counter_total", int(value or 0), labels={"name": str(name)}))
    for cycle in list(observability.get("cycles", []) or []):
        cycle_name = str(cycle.get("name", "") or "")
        if not cycle_name:
            continue
        lines.extend(
            [
                _metric_line("autocode_cycle_runs_total", int(cycle.get("run_count", 0) or 0), labels={"cycle": cycle_name}),
                _metric_line(
                    "autocode_cycle_success_total",
                    int(cycle.get("success_count", 0) or 0),
                    labels={"cycle": cycle_name},
                ),
                _metric_line(
                    "autocode_cycle_failure_total",
                    int(cycle.get("failure_count", 0) or 0),
                    labels={"cycle": cycle_name},
                ),
                _metric_line(
                    "autocode_cycle_skip_total",
                    int(cycle.get("skip_count", 0) or 0),
                    labels={"cycle": cycle_name},
                ),
                _metric_line(
                    "autocode_cycle_consecutive_failures",
                    int(cycle.get("consecutive_failures", 0) or 0),
                    labels={"cycle": cycle_name},
                ),
            ]
        )
        duration = cycle.get("last_duration_seconds")
        if duration not in (None, ""):
            lines.append(
                _metric_line(
                    "autocode_cycle_last_duration_seconds",
                    float(duration),
                    labels={"cycle": cycle_name},
                )
            )
    recent_events = list(observability.get("recent_events", []) or [])
    lines.extend(_event_count_metrics(recent_events))
    last_event_at = str(observability.get("last_event_at", "") or "")
    lines.append(_metric_line("autocode_last_event_timestamp_seconds", _iso_to_unix_seconds(last_event_at)))
    return "\n".join(lines) + "\n"


def _has_active_background_work() -> bool:
    scheduler_running = int((scheduler.summary() if scheduler else {}).get("running", 0) or 0)
    if scheduler_running > 0:
        return True
    queue_summaries = _all_task_queue_summaries()
    active_statuses = ("queued", "running", "retry_waiting")
    return any(
        any(int(summary.get(status, 0) or 0) > 0 for status in active_statuses)
        for summary in queue_summaries.values()
        if summary
    )


def _event_count_metrics(events: list[dict]) -> list[str]:
    by_level: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for event in events:
        level = str(event.get("level", "") or "")
        category = str(event.get("category", "") or "")
        if level:
            by_level[level] = by_level.get(level, 0) + 1
        if category:
            by_category[category] = by_category.get(category, 0) + 1
    lines: list[str] = []
    for level, count in sorted(by_level.items()):
        lines.append(_metric_line("autocode_recent_events_by_level", count, labels={"level": level}))
    for category, count in sorted(by_category.items()):
        lines.append(_metric_line("autocode_recent_events_by_category", count, labels={"category": category}))
    return lines


def _metric_line(name: str, value: int | float, *, labels: dict[str, str] | None = None) -> str:
    if labels:
        rendered_labels = ",".join(
            f'{key}="{_escape_metric_label(label_value)}"' for key, label_value in sorted(labels.items())
        )
        return f"{name}{{{rendered_labels}}} {value}"
    return f"{name} {value}"


def _escape_metric_label(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _bool_metric(value: bool) -> int:
    return 1 if value else 0


def _iso_to_unix_seconds(value: str) -> int:
    text = str(value or "").strip()
    if not text:
        return 0
    try:
        return int(datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return 0


def _parse_event_filter(value: str, *, allowed: set[str] | None = None) -> list[str] | None:
    raw = [item.strip().lower() for item in str(value or "").split(",") if item.strip()]
    if not raw:
        return None
    if allowed is not None:
        invalid = sorted(set(raw) - set(allowed))
        if invalid:
            raise HTTPException(status_code=400, detail=f"Unsupported event filter: {', '.join(invalid)}")
    ordered = []
    seen: set[str] = set()
    for item in raw:
        if item in seen:
            continue
        ordered.append(item)
        seen.add(item)
    return ordered


def _filter_recent_events(
    *,
    categories: list[str] | None = None,
    levels: list[str] | None = None,
    sources: list[str] | None = None,
    limit: int = 50,
) -> list[dict]:
    if state_store is None:
        return []
    events = list(state_store.observability_summary().get("recent_events", []) or [])
    category_set = set(categories or [])
    level_set = set(levels or [])
    source_set = set(sources or [])
    filtered = []
    for event in reversed(events):
        if category_set and str(event.get("category", "") or "").lower() not in category_set:
            continue
        if level_set and str(event.get("level", "") or "").lower() not in level_set:
            continue
        if source_set and str(event.get("source", "") or "").lower() not in source_set:
            continue
        filtered.append(dict(event))
        if len(filtered) >= limit:
            break
    filtered.reverse()
    return filtered


def _parse_task_queue_names(queue: str) -> list[str] | None:
    raw = [item.strip().lower() for item in str(queue or "").split(",") if item.strip()]
    if not raw:
        return None
    allowed = tuple(_task_queue_bindings().keys())
    invalid = sorted(set(raw) - set(allowed))
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unsupported task queue filter: {', '.join(invalid)}")
    return [name for name in allowed if name in set(raw)]


def _parse_task_statuses(
    status: str,
    *,
    default: set[str] | None = None,
    allowed: set[str] | None = None,
) -> set[str] | None:
    raw = [item.strip() for item in str(status or "").split(",") if item.strip()]
    if not raw:
        return set(default) if default is not None else None
    statuses = set(raw)
    supported = allowed or {"queued", "running", "retry_waiting", "dead_letter", "done"}
    invalid = sorted(statuses - supported)
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unsupported task status filter: {', '.join(invalid)}")
    return statuses


def _validate_task_limit(limit: int):
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=400, detail="Task limit must be between 1 and 200")


def _list_queue_tasks(queue_name: str, *, status: str = "") -> list[dict]:
    statuses = _parse_task_statuses(status)
    binding = _task_queue_bindings().get(queue_name)
    if binding is None:
        raise HTTPException(status_code=404, detail=f"Unknown task queue: {queue_name}")
    task_queue = binding["queue"]
    if task_queue is None:
        return []
    serializer = binding["serializer"]
    return [serializer(task) for task in task_queue.list_tasks(statuses=statuses)]


def _collect_task_items(
    *,
    statuses: set[str] | None,
    queue_names: list[str] | None = None,
    limit: int = 0,
    include_superseded: bool = True,
) -> list[dict]:
    items: list[dict] = []
    bindings = _task_queue_bindings()
    selected = queue_names or list(bindings.keys())
    for queue_name in selected:
        binding = bindings.get(queue_name)
        if not binding:
            continue
        task_queue = binding["queue"]
        if task_queue is None:
            continue
        serializer = binding["serializer"]
        tasks = task_queue.list_tasks()
        for task in tasks:
            if statuses and task.status not in statuses:
                continue
            payload = _serialize_task_with_triage(task, serializer=serializer, tasks=tasks)
            if not include_superseded and bool(payload.get("superseded", False)):
                continue
            payload["queue"] = queue_name
            items.append(payload)
    status_rank = {"running": 0, "queued": 1, "retry_waiting": 2, "dead_letter": 3}
    priority_rank = {"urgent": 0, "normal": 1, "background": 2}
    items.sort(
        key=lambda item: (
            status_rank.get(str(item.get("status", "") or ""), 9),
            priority_rank.get(str(item.get("priority", "") or ""), 9),
            str(item.get("available_at", "") or ""),
            str(item.get("created_at", "") or ""),
            str(item.get("task_id", "") or ""),
        )
    )
    return items[:limit] if limit > 0 else items


def _retry_queue_task(queue_name: str, task_id: str) -> dict:
    binding = _task_queue_bindings().get(queue_name)
    if binding is None:
        raise HTTPException(status_code=404, detail=f"Unknown task queue: {queue_name}")
    task_queue = binding["queue"]
    if task_queue is None:
        raise HTTPException(status_code=503, detail=str(binding["not_initialized"]))
    previous = task_queue.get_task(task_id)
    try:
        task = task_queue.retry_task(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown {binding['detail_name']}: {task_id}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    _increment_state_counter(str(binding["counter"]))
    _record_state_event(
        category=str(binding["category"]),
        message=f"Manually re-queued {binding['display_name']} task {task_id}",
        target_kind=str(binding["target_kind"] or ""),
        target_number=task.target_number,
        details={
            "queue": queue_name,
            "task_id": task.task_id,
            "previous_status": previous.status if previous is not None else "",
            "command": task.command,
        },
    )
    serializer = binding["serializer"]
    payload = serializer(task)
    payload["queue"] = queue_name
    return payload


def _task_queue_summary(task_queue: TaskQueue | None, *, serializer) -> dict:
    if task_queue is None:
        return {}
    tasks = task_queue.list_tasks()
    dead_letter_tasks = [task for task in tasks if task.status == "dead_letter"]
    dead_letter_superseded = sum(1 for task in dead_letter_tasks if _find_superseding_done_task(task, tasks) is not None)
    dead_letter_action_required = len(dead_letter_tasks) - dead_letter_superseded
    retry_waiting = sum(1 for task in tasks if task.status == "retry_waiting")
    return {
        "total": len(tasks),
        "queued": sum(1 for task in tasks if task.status == "queued"),
        "running": sum(1 for task in tasks if task.status == "running"),
        "retry_waiting": retry_waiting,
        "dead_letter": len(dead_letter_tasks),
        "dead_letter_action_required": dead_letter_action_required,
        "dead_letter_superseded": dead_letter_superseded,
        "problem": retry_waiting + dead_letter_action_required,
        "tasks": [_serialize_task_with_triage(task, serializer=serializer, tasks=tasks) for task in tasks[-10:]],
    }


def _serialize_task_with_triage(task, *, serializer, tasks: list) -> dict:
    payload = serializer(task)
    if task.status != "dead_letter":
        return payload
    superseding_task = _find_superseding_done_task(task, tasks)
    payload["superseded"] = superseding_task is not None
    payload["action_required"] = superseding_task is None
    if superseding_task is not None:
        payload["superseded_by_task_id"] = superseding_task.task_id
        payload["superseded_by_result_summary"] = superseding_task.result_summary
        payload["superseded_by_updated_at"] = superseding_task.updated_at
    return payload


def _find_superseding_done_task(task, tasks: list):
    if task.status != "dead_letter" or task.target_number <= 0:
        return None
    finished_at = _task_finished_sort_value(task)
    candidates = [
        item
        for item in tasks
        if item.task_id != task.task_id
        and item.task_type == task.task_type
        and item.target_number == task.target_number
        and item.status == "done"
        and _task_finished_sort_value(item) > finished_at
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (_task_finished_sort_value(item), item.task_id))
    return candidates[-1]


def _task_finished_sort_value(task) -> str:
    return task.last_finished_at or task.updated_at or task.available_at or task.created_at or ""


def _issue_task_to_dict(task) -> dict:
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "target_number": task.target_number,
        "source": task.source,
        "priority": task.priority,
        "command": task.command,
        "status": task.status,
        "attempt_count": task.attempt_count,
        "max_attempts": task.max_attempts,
        "available_at": task.available_at,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "last_error": task.last_error,
        "result_summary": task.result_summary,
    }


def _pr_task_to_dict(task) -> dict:
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "target_number": task.target_number,
        "source": task.source,
        "priority": task.priority,
        "command": task.command,
        "status": task.status,
        "attempt_count": task.attempt_count,
        "max_attempts": task.max_attempts,
        "available_at": task.available_at,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "last_error": task.last_error,
        "result_summary": task.result_summary,
    }


def _source_task_to_dict(task) -> dict:
    return {
        "task_id": task.task_id,
        "task_type": task.task_type,
        "source": task.source,
        "priority": task.priority,
        "command": task.command,
        "status": task.status,
        "attempt_count": task.attempt_count,
        "max_attempts": task.max_attempts,
        "available_at": task.available_at,
        "created_at": task.created_at,
        "updated_at": task.updated_at,
        "last_error": task.last_error,
        "result_summary": task.result_summary,
        "item_key": str(task.payload.get("item_key", "") or ""),
        "source_path": str(task.payload.get("source_path", "") or ""),
    }

def _build_source_task_id(item_key: str) -> str:
    digest = hashlib.sha256(str(item_key or "").encode("utf-8")).hexdigest()[:16]
    return f"source-task:{digest}"


async def _schedule_pull_request_repair(pr: dict, *, source: str, priority: str) -> tuple[str, object | None]:
    assert controller and pr_task_queue
    summary, candidate = await asyncio.to_thread(controller.plan_pull_request_repair, pr)
    if candidate is None:
        return summary, None
    task, created = _enqueue_pull_request_repair_candidate(candidate, source=source, priority=priority)
    if created:
        _record_state_event(
            category="pr_task_queue",
            message=f"Queued PR repair task for PR #{candidate.pr_number}",
            source=source,
            target_kind="pull_request",
            target_number=candidate.pr_number,
            details={"task_id": task.task_id, "status": task.status},
        )
    return summary, task



def _issue_number_from_issue_url(url: str) -> int:
    if not url:
        return 0
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    try:
        return int(tail)
    except ValueError:
        return 0


def _issue_number_from_issue_event(event: dict) -> int:
    issue = event.get("issue", {}) or {}
    number = issue.get("number")
    if isinstance(number, int):
        return number
    for candidate in (
        str(issue.get("url", "") or ""),
        str(event.get("issue_url", "") or ""),
    ):
        parsed = _issue_number_from_issue_url(candidate)
        if parsed > 0:
            return parsed
    return 0

def _is_after_cursor(timestamp: str, item_id: int, cursor_time: str, cursor_id: int) -> bool:
    if not timestamp:
        return False
    if not cursor_time:
        return True
    timestamp_dt = _parse_iso_timestamp(timestamp)
    cursor_dt = _parse_iso_timestamp(cursor_time)
    if timestamp_dt is not None and cursor_dt is not None:
        if timestamp_dt > cursor_dt:
            return True
        if timestamp_dt < cursor_dt:
            return False
        return int(item_id or 0) > int(cursor_id or 0)
    if timestamp > cursor_time:
        return True
    if timestamp < cursor_time:
        return False
    return int(item_id or 0) > int(cursor_id or 0)


def _max_cursor(
    current_time: str,
    current_id: int,
    candidate_time: str,
    candidate_id: int,
) -> tuple[str, int]:
    if _is_after_cursor(candidate_time, candidate_id, current_time, current_id):
        return candidate_time, candidate_id
    return current_time, current_id

def _parse_iso_timestamp(value: str) -> datetime | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_bot_user(user: dict) -> bool:
    user_type = str(user.get("type", "")).lower()
    login = str(user.get("login", "")).lower()
    return user_type == "bot" or "[bot]" in login


def _is_dry_run_enabled() -> bool:
    return bool(getattr(getattr(config, "server", None), "dry_run", False))


def _source_sync_requires_real_issue() -> bool:
    return not _is_dry_run_enabled()


def _should_sync_sources_in_background() -> bool:
    if config is None:
        return False
    event_source = str(getattr(getattr(config, "server", None), "event_source", "") or "").strip().lower()
    autocode_config = getattr(config, "autocode", None)
    return event_source != "polling" and (
        bool(getattr(autocode_config, "enable_plan_source", False))
        or bool(getattr(autocode_config, "enable_backlog_source", False))
    )


if __name__ == "__main__":
    _cfg = load_autocode_config()
    uvicorn.run(
        "autocode:app",
        host="0.0.0.0",
        port=_cfg.server.port,
        log_level=_cfg.server.log_level.lower(),
    )
