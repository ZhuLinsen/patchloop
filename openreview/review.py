"""
OpenReview Agent — Webhook / Polling 入口

功能:
1. 接收 GitHub Webhook 事件或主动轮询 GitHub
2. 在 webhook 模式下验证签名确保请求来自 GitHub
3. 分发事件到对应的处理流程
4. 调用本地 CLI 驱动的 Agent 分析并通过 GitHub API 回复

安全:
- Webhook 签名验证 (HMAC-SHA256)
- 仅处理 Issue 和 PR 事件
- 所有 GitHub 写操作仅限评论
"""
import asyncio
import fcntl
import hashlib
import hmac
import json
import logging
import os
import re
import sys
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
import uvicorn

from config import load_config, AppConfig
from adapters import create_adapter
from agent.core import ReviewAgent
from agent.context_builder import LocalContextBuilder
from agent.discussion import (
    build_issue_reference_pr_query,
    build_issue_search_query,
    build_pr_search_query,
    collect_discussion_entries,
    detect_duplicate_issues,
    format_discussion_context,
    format_duplicate_issues_context,
    format_related_issues_context,
    format_related_prs_context,
    format_resolution_signals_context,
    is_ignored_activity,
    is_ignored_actor,
    pick_latest_external_activity,
)
from agent.git_guard import GitGuard
from agent.github_client import GitHubClient
from agent.poller import GitHubPoller
from agent.state_store import (
    LocalStateStore,
    build_activity_fingerprint,
    build_issue_processing_key,
    build_pr_review_key,
)

# ------------------------------------------------------------------
# 日志配置
# ------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("openreview")

# ------------------------------------------------------------------
# 全局实例 (在 lifespan 中初始化)
# ------------------------------------------------------------------
config: AppConfig | None = None
agent: ReviewAgent | None = None
gh: GitHubClient | None = None
git_guard: GitGuard | None = None
state_store: LocalStateStore | None = None
poller_task: asyncio.Task | None = None
self_login: str = ""
_issue_processing_locks: dict[int, asyncio.Lock] = {}
_pr_processing_locks: dict[int, asyncio.Lock] = {}
_issue_processing_lock_refs: dict[int, int] = {}
_pr_processing_lock_refs: dict[int, int] = {}
_single_instance_lock_handle = None
_single_instance_lock_path = ""
PR_BLOCKED_NOTIFY_COOLDOWN_SECONDS = 1800
_AUTOCODE_COMMAND_RE = re.compile(r"(?mi)^\s*/(plan|implement|retry|abort|status)\b")
_IDEMPOTENCY_MARKER_PREFIX = "openreview:idempotency"
_IDEMPOTENCY_LOOKBACK_LIMIT = 200


def _build_idempotency_marker(kind: str, key: str) -> str:
    normalized_kind = re.sub(r"[^a-z0-9_-]+", "-", str(kind or "").strip().lower()) or "unknown"
    normalized_key = str(key or "").strip()
    digest = hashlib.sha256(f"{normalized_kind}\n{normalized_key}".encode("utf-8")).hexdigest()[:20]
    return f"<!-- {_IDEMPOTENCY_MARKER_PREFIX}:{normalized_kind}:{digest} -->"


def _append_idempotency_marker(content: str, marker: str) -> str:
    if not marker:
        return content
    if marker in content:
        return content
    stripped = content.rstrip()
    if not stripped:
        return marker
    return f"{stripped}\n\n{marker}"


async def _issue_comment_marker_exists(issue_number: int, marker: str) -> bool:
    assert gh
    if not marker:
        return False
    try:
        comments = await asyncio.to_thread(
            gh.list_recent_issue_comments,
            issue_number,
            limit=_IDEMPOTENCY_LOOKBACK_LIMIT,
            per_page=100,
        )
    except Exception as exc:
        logger.warning("读取 Issue #%d 最近评论失败，保守跳过以避免重复评论: %s", issue_number, exc)
        return True
    return any(marker in str(comment.get("body", "") or "") for comment in comments)


async def _pr_review_marker_exists(pr_number: int, marker: str) -> bool:
    assert gh
    if not marker:
        return False
    try:
        reviews = await asyncio.to_thread(
            gh.list_recent_pr_reviews,
            pr_number,
            limit=_IDEMPOTENCY_LOOKBACK_LIMIT,
            per_page=100,
        )
    except Exception as exc:
        logger.warning("读取 PR #%d 最近 reviews 失败，保守跳过以避免重复评论: %s", pr_number, exc)
        return True
    return any(marker in str(review.get("body", "") or "") for review in reviews)


def _single_instance_lockfile_path(state_file_path: str) -> Path:
    return Path(f"{state_file_path}.lock")


def _acquire_single_instance_lock(state_file_path: str):
    global _single_instance_lock_handle, _single_instance_lock_path

    if _single_instance_lock_handle is not None:
        return

    lock_path = _single_instance_lockfile_path(state_file_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.seek(0)
        owner = handle.read().strip()
        handle.close()
        owner_suffix = f"，当前锁信息: {owner}" if owner else ""
        raise RuntimeError(f"检测到已有 OpenReview 实例正在运行并占用锁文件: {lock_path}{owner_suffix}") from exc

    handle.seek(0)
    handle.truncate()
    handle.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "state_file": state_file_path,
            },
            ensure_ascii=False,
        )
    )
    handle.flush()
    _single_instance_lock_handle = handle
    _single_instance_lock_path = str(lock_path)
    logger.info("已获取单实例运行锁: %s", lock_path)


def _release_single_instance_lock():
    global _single_instance_lock_handle, _single_instance_lock_path

    handle = _single_instance_lock_handle
    if handle is None:
        return

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        logger.warning("释放单实例运行锁失败: %s", _single_instance_lock_path or "unknown")
    finally:
        handle.close()
        if _single_instance_lock_path:
            logger.info("已释放单实例运行锁: %s", _single_instance_lock_path)
        _single_instance_lock_handle = None
        _single_instance_lock_path = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时初始化组件"""
    global config, agent, gh, git_guard, state_store, poller_task, self_login

    logger.info("=== OpenReview Agent 启动中 ===")
    startup_completed = False

    try:
        # 加载配置
        config = load_config()
        logging.getLogger().setLevel(config.server.log_level)
        logger.info(
            "配置加载完成: repo=%s, event_source=%s, primary_cli=%s, fallback_cli=%s, dry_run=%s, port=%s",
            config.github.repo,
            config.server.event_source,
            config.cli.primary_cli,
            config.cli.fallback_cli or "none",
            config.server.dry_run,
            config.server.port,
        )
        if config.server.enable_webhook:
            logger.info("Webhook 接收地址已启用: POST /webhook")
        else:
            logger.info(
                "Webhook 当前未启用，使用 polling: interval=%ss, per_page=%s",
                config.server.poll_interval_seconds,
                config.server.poll_items_per_page,
            )

        _acquire_single_instance_lock(config.server.state_file_path)

        # 创建 CLI 分析适配器
        analyzer = create_adapter(
            cli_timeout=config.cli.cli_timeout,
            working_dir=config.local_repo.path,
            primary_cli=config.cli.primary_cli,
            primary_model=config.cli.primary_model,
            primary_fallback_model=config.cli.primary_fallback_model,
            primary_model_flag=config.cli.primary_model_flag,
            primary_command=config.cli.primary_command or None,
            fallback_cli=config.cli.fallback_cli,
            fallback_model=config.cli.fallback_model,
            fallback_fallback_model=config.cli.fallback_fallback_model,
            fallback_model_flag=config.cli.fallback_model_flag,
            fallback_command=config.cli.fallback_command or None,
        )

        # 创建 Git 安全守卫
        git_guard = GitGuard(
            repo_path=config.local_repo.path,
            default_branch=config.local_repo.default_branch,
        )
        try:
            git_guard.clone_if_missing(config.github.clone_url)
            if git_guard.is_git_repo():
                logger.info(
                    "Git 仓库已就绪: branch=%s, clean=%s",
                    git_guard.current_branch(),
                    not git_guard.is_dirty(),
                )
            else:
                logger.warning("LOCAL_REPO_PATH (%s) 不是 git 仓库，分支切换功能不可用", config.local_repo.path)
                git_guard = None
        except FileNotFoundError:
            logger.error("无法准备本地仓库，请确认 git 已安装且 LOCAL_REPO_PATH 可访问: %s", config.local_repo.path)
            raise
        except RuntimeError as exc:
            logger.error("自动拉取本地仓库失败: %s", exc)
            raise

        # 创建 Agent
        context_builder = LocalContextBuilder(
            repo_path=config.local_repo.path,
            max_context_files=config.local_repo.max_context_files,
            max_file_chars=config.local_repo.max_file_chars,
            max_total_chars=config.local_repo.max_total_context_chars,
        )
        agent = ReviewAgent(
            analyzer=analyzer,
            repo_name=config.github.repo,
            max_comment_length=config.server.max_comment_length,
            context_builder=context_builder,
            git_guard=git_guard,
        )

        # 创建 GitHub 客户端
        gh = GitHubClient(
            token=config.github.token,
            owner=config.github.owner,
            repo=config.github.repo_name,
            dry_run=config.server.dry_run,
        )
        try:
            self_login = (await asyncio.to_thread(gh.get_authenticated_login)).strip().lower()
            if self_login:
                logger.info("GitHub 写入身份: @%s", self_login)
        except Exception as exc:
            self_login = ""
            logger.warning("获取 GitHub 当前登录身份失败，将仅按 bot 规则过滤自触发: %s", exc)
        state_store = LocalStateStore(config.server.state_file_path)
        logger.info("本地状态文件: %s", state_store.summary()["state_file_path"])

        if config.server.event_source == "polling":
            poller = GitHubPoller(
                gh=gh,
                state_store=state_store,
                issue_processor=_process_issue,
                pr_processor=_process_pr,
                interval_seconds=config.server.poll_interval_seconds,
                items_per_page=config.server.poll_items_per_page,
                ignored_logins=_ignored_logins(),
                pending_pr_batch_size=config.server.poll_pending_pr_batch_size,
                pending_pr_max_age_days=config.server.poll_pending_pr_max_age_days,
                review_submission_batch_size=config.server.poll_review_submission_batch_size,
                review_submission_max_age_days=config.server.poll_review_submission_max_age_days,
            )
            poller_task = asyncio.create_task(poller.run(), name="github-poller")
            logger.info("Polling 后台任务已启动")

        logger.info("=== OpenReview Agent 就绪 ===")
        startup_completed = True
        yield
    finally:
        if poller_task is not None:
            poller_task.cancel()
            with suppress(asyncio.CancelledError):
                await poller_task
            poller_task = None

        # 关闭时确保仓库干净
        if startup_completed and git_guard is not None:
            logger.info("正在进行仓库收尾检查...")
            git_guard.ensure_clean()

        # 关闭 HTTP 客户端
        if gh is not None:
            gh.close()
            gh = None
        state_store = None
        agent = None
        git_guard = None
        self_login = ""
        _release_single_instance_lock()

        logger.info("=== OpenReview Agent 关闭 ===")


app = FastAPI(title="OpenReview Agent", lifespan=lifespan)


# ------------------------------------------------------------------
# Webhook 签名验证
# ------------------------------------------------------------------

def verify_signature(payload_body: bytes, signature_header: str | None, secret: str) -> bool:
    """验证 GitHub Webhook 的 HMAC-SHA256 签名"""
    if not signature_header:
        return False
    expected = "sha256=" + hmac.new(
        secret.encode("utf-8"),
        payload_body,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header)


# ------------------------------------------------------------------
# Webhook 端点
# ------------------------------------------------------------------

@app.post("/webhook")
async def github_webhook(request: Request):
    """接收并处理 GitHub Webhook 事件"""
    assert config and agent and gh, "应用未初始化"

    if config.server.event_source != "webhook":
        raise HTTPException(status_code=503, detail="Webhook is disabled")

    # 1. 签名验证
    body = await request.body()
    signature = request.headers.get("X-Hub-Signature-256")
    if not verify_signature(body, signature, config.github.webhook_secret):
        logger.warning("Webhook 签名验证失败，拒绝请求")
        raise HTTPException(status_code=403, detail="Invalid signature")

    # 2. 解析事件
    event_type = request.headers.get("X-GitHub-Event", "")
    content_type = request.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    if content_type == "application/json":
        data = await request.json()
    elif content_type == "application/x-www-form-urlencoded":
        form = await request.form()
        payload = form.get("payload")
        if not isinstance(payload, str):
            raise HTTPException(status_code=400, detail="Missing payload form field")
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Invalid form payload") from exc
    else:
        raise HTTPException(status_code=415, detail=f"Unsupported content type: {content_type or 'unknown'}")
    action = data.get("action", "")

    logger.info("收到 Webhook: event=%s, action=%s", event_type, action)

    # 3. 分发处理
    try:
        if event_type == "issues" and action in ("opened", "reopened", "edited"):
            if action == "edited" and _is_ignored_resource_owner(data.get("sender", {}) or {}):
                logger.info("忽略来自自身/bot 的 issues edited on #%s", data.get("issue", {}).get("number", "unknown"))
            else:
                await _handle_issue_event(data)
        elif event_type == "issue_comment" and action in ("created", "edited"):
            await _handle_issue_comment_event(data)
        elif event_type == "pull_request" and action in ("opened", "synchronize", "edited"):
            if action == "edited" and _is_ignored_resource_owner(data.get("sender", {}) or {}):
                logger.info(
                    "忽略来自自身/bot 的 pull_request edited on #%s",
                    data.get("pull_request", {}).get("number", "unknown"),
                )
            else:
                await _handle_pr_event(data)
        elif event_type == "pull_request_review" and action in ("submitted", "edited"):
            await _handle_pr_review_event(data)
        elif event_type == "pull_request_review_comment" and action in ("created", "edited"):
            await _handle_pr_review_comment_event(data)
        elif event_type in ("check_suite", "check_run"):
            await _handle_ci_event(data, event_type)
        elif event_type == "ping":
            logger.info("收到 ping 事件，Webhook 配置正常")
        else:
            logger.debug("忽略事件: %s/%s", event_type, action)
    except Exception as e:
        logger.exception("处理 Webhook 事件时发生异常: %s", e)
        # 返回 200 以免 GitHub 认为 Webhook 失败进行重试
        return {"status": "error", "message": str(e)}

    return {"status": "ok"}


# ------------------------------------------------------------------
# 事件处理函数
# ------------------------------------------------------------------

async def _handle_issue_event(data: dict):
    """处理 Issue 事件"""
    issue = data["issue"]
    await _process_issue(issue, "webhook")


async def _handle_issue_comment_event(data: dict):
    """处理 Issue/PR 会话评论事件。"""
    issue = data["issue"]
    comment = data.get("comment", {}) or {}
    if _is_ignored_discussion_actor(comment.get("user", {}), str(comment.get("body", "") or "")):
        logger.info("忽略来自自身/bot 的 issue_comment on #%s", issue.get("number", "unknown"))
        return
    if _AUTOCODE_COMMAND_RE.search(str(comment.get("body", "") or "")):
        logger.info("忽略 slash command issue_comment on #%s", issue.get("number", "unknown"))
        return

    activity_fingerprint = _build_activity_fingerprint_from_comment("issue_comment", comment)
    if "pull_request" in issue:
        await _process_pr(issue, "webhook:issue_comment", activity_fingerprint=activity_fingerprint)
        return

    await _process_issue(
        issue,
        "webhook:issue_comment",
        activity_fingerprint=activity_fingerprint,
        allow_unanswerable_reply=False,
    )


async def _handle_pr_event(data: dict):
    """处理 PR 事件"""
    pr = data["pull_request"]
    await _process_pr(pr, "webhook")


async def _handle_pr_review_event(data: dict):
    """处理 PR review 事件。"""
    review = data.get("review", {}) or {}
    if _is_ignored_discussion_actor(review.get("user", {}), str(review.get("body", "") or "")):
        logger.info("忽略来自自身/bot 的 pull_request_review on #%s", data.get("pull_request", {}).get("number", "unknown"))
        return
    if not str(review.get("body", "") or "").strip():
        logger.info("忽略空正文的 pull_request_review on #%s", data.get("pull_request", {}).get("number", "unknown"))
        return

    await _process_pr(
        data["pull_request"],
        "webhook:pull_request_review",
        activity_fingerprint=_build_activity_fingerprint_from_review(review),
    )


async def _handle_pr_review_comment_event(data: dict):
    """处理 PR review comment 事件。"""
    comment = data.get("comment", {}) or {}
    if _is_ignored_discussion_actor(comment.get("user", {}), str(comment.get("body", "") or "")):
        logger.info(
            "忽略来自自身/bot 的 pull_request_review_comment on #%s",
            data.get("pull_request", {}).get("number", "unknown"),
        )
        return
    if not str(comment.get("body", "") or "").strip():
        logger.info(
            "忽略空正文的 pull_request_review_comment on #%s",
            data.get("pull_request", {}).get("number", "unknown"),
        )
        return

    await _process_pr(
        data["pull_request"],
        "webhook:pull_request_review_comment",
        activity_fingerprint=_build_activity_fingerprint_from_comment("review_comment", comment),
    )

async def _handle_ci_event(data: dict, event_type: str):
    """处理 CI 完成事件，交给 PR 主流程决定是否可评审。"""

    if data.get("action") != "completed":
        return

    payload = data.get(event_type, {})
    conclusion = (payload.get("conclusion") or "").lower()
    if not conclusion:
        logger.info("忽略尚无结论的 CI 事件: %s", event_type)
        return

    pr_refs = payload.get("pull_requests", []) or []
    if not pr_refs:
        logger.debug("%s 事件未携带关联 PR，跳过", event_type)
        return

    for pr_ref in pr_refs:
        pr_number = pr_ref.get("number")
        if not isinstance(pr_number, int):
            continue
        logger.info("收到 %s 完成事件，准备复核 PR #%d 的整体 CI 状态", event_type, pr_number)
        await _process_pr({"number": pr_number}, "webhook")


def _issue_processing_lock(issue_number: int) -> asyncio.Lock:
    lock = _issue_processing_locks.get(issue_number)
    if lock is None:
        lock = asyncio.Lock()
        _issue_processing_locks[issue_number] = lock
    return lock


def _pr_processing_lock(pr_number: int) -> asyncio.Lock:
    lock = _pr_processing_locks.get(pr_number)
    if lock is None:
        lock = asyncio.Lock()
        _pr_processing_locks[pr_number] = lock
    return lock


def _retain_processing_lock(ref_map: dict[int, int], number: int):
    ref_map[number] = ref_map.get(number, 0) + 1


def _release_processing_lock(
    lock_map: dict[int, asyncio.Lock],
    ref_map: dict[int, int],
    number: int,
    lock: asyncio.Lock,
):
    remaining = max(ref_map.get(number, 0) - 1, 0)
    if remaining:
        ref_map[number] = remaining
        return
    ref_map.pop(number, None)
    current = lock_map.get(number)
    if current is lock and not lock.locked():
        lock_map.pop(number, None)


def _ignored_logins() -> set[str]:
    return {self_login} if self_login else set()


def _is_ignored_resource_owner(user: dict) -> bool:
    return is_ignored_actor(user)


def _is_ignored_discussion_actor(user: dict, body: str = "") -> bool:
    return is_ignored_activity(user, body=body, ignored_logins=_ignored_logins())


def _build_activity_fingerprint_from_comment(activity_type: str, comment: dict) -> str:
    return build_activity_fingerprint(
        activity_type,
        comment.get("id", 0) or 0,
        body=str(comment.get("body", "") or ""),
        updated_at=str(comment.get("updated_at", "") or comment.get("created_at", "") or ""),
    )


def _build_activity_fingerprint_from_review(review: dict) -> str:
    return build_activity_fingerprint(
        "review",
        review.get("id", 0) or 0,
        body=str(review.get("body", "") or ""),
        updated_at=str(review.get("submitted_at", "") or review.get("updated_at", "") or review.get("created_at", "") or ""),
    )


def _merge_search_results(*groups: list[dict]) -> list[dict]:
    """按编号合并搜索结果，保留更靠前查询的顺序。"""
    merged: list[dict] = []
    seen_numbers: set[int] = set()
    for group in groups:
        for item in group:
            number = int(item.get("number", 0) or 0)
            if number <= 0 or number in seen_numbers:
                continue
            seen_numbers.add(number)
            merged.append(item)
    return merged


async def _enrich_pr_search_results(prs: list[dict], *, limit: int = 6) -> list[dict]:
    """把 search/issues 返回的 PR 条目补全为 pull request 详情。"""
    assert gh

    selected = _merge_search_results(prs)[: max(limit, 1)]
    detail_tasks: list[tuple[dict, asyncio.Task]] = []
    for pr in selected:
        pr_number = int(pr.get("number", 0) or 0)
        if pr_number <= 0:
            continue
        detail_tasks.append((pr, asyncio.create_task(asyncio.to_thread(gh.get_pr, pr_number))))

    enriched: list[dict] = []
    for pr, task in detail_tasks:
        pr_number = int(pr.get("number", 0) or 0)
        try:
            detailed = await task
        except Exception as exc:
            logger.warning("读取相关 PR #%d 详情失败，将退回搜索结果: %s", pr_number, exc)
            enriched.append(pr)
            continue

        merged = dict(pr)
        if isinstance(detailed, dict):
            merged.update(detailed)
        enriched.append(merged)

    return enriched


async def _load_issue_runtime_context(issue_number: int, title: str, body: str) -> tuple[str, str, str, str, str]:
    """读取 Issue 当前讨论、相关 issue / PR 与修复线索。"""
    assert gh and config

    async def _load_comments() -> list[dict]:
        return await asyncio.to_thread(gh.list_recent_issue_comments, issue_number, limit=60, per_page=100)

    async def _search_similar_issues() -> list[dict]:
        query = build_issue_search_query(config.github.repo, title, body, max_terms=6)
        return await asyncio.to_thread(gh.search_issues, query, per_page=10, page=1)

    async def _search_related_prs() -> list[dict]:
        query = build_pr_search_query(config.github.repo, title, body, max_terms=6)
        return await asyncio.to_thread(gh.search_issues, query, per_page=10, page=1)

    async def _search_issue_reference_prs() -> list[dict]:
        query = build_issue_reference_pr_query(config.github.repo, issue_number)
        if not query:
            return []
        return await asyncio.to_thread(gh.search_issues, query, per_page=10, page=1)

    comments_task = asyncio.create_task(_load_comments())
    search_task = asyncio.create_task(_search_similar_issues())
    related_prs_task = asyncio.create_task(_search_related_prs())
    reference_prs_task = asyncio.create_task(_search_issue_reference_prs())

    comments: list[dict] = []
    similar_issues: list[dict] = []
    related_prs: list[dict] = []
    reference_prs: list[dict] = []

    try:
        comments = await comments_task
    except Exception as exc:
        logger.warning("读取 Issue #%d 评论失败，将在缺少讨论上下文的情况下继续: %s", issue_number, exc)

    try:
        similar_issues = await search_task
    except Exception as exc:
        logger.warning("搜索 Issue #%d 的相似 issue 失败，将跳过去重线索: %s", issue_number, exc)

    try:
        related_prs = await related_prs_task
    except Exception as exc:
        logger.warning("搜索 Issue #%d 的相关 PR 失败，将跳过 PR 线索: %s", issue_number, exc)

    try:
        reference_prs = await reference_prs_task
    except Exception as exc:
        logger.warning("搜索引用 Issue #%d 的 PR 失败，将跳过修复状态线索: %s", issue_number, exc)

    latest_comment_updated_at = ""
    for comment in comments:
        comment_updated_at = str(comment.get("updated_at", "") or comment.get("created_at", "") or "")
        if comment_updated_at and comment_updated_at > latest_comment_updated_at:
            latest_comment_updated_at = comment_updated_at

    discussion_entries = collect_discussion_entries(issue_comments=comments)
    discussion_context = format_discussion_context(discussion_entries, limit=8)
    latest_activity = pick_latest_external_activity(discussion_entries, ignored_logins=_ignored_logins())
    latest_activity_fingerprint = ""
    latest_activity_updated_at = ""
    if latest_activity is not None:
        latest_activity_fingerprint = build_activity_fingerprint(
            latest_activity.kind,
            latest_activity.activity_id,
            latest_activity.body,
            latest_activity.updated_at,
        )
        latest_activity_updated_at = latest_activity.updated_at

    related_pr_candidates = _merge_search_results(reference_prs, related_prs)
    enriched_related_prs = await _enrich_pr_search_results(related_pr_candidates, limit=6)
    duplicate_candidates = detect_duplicate_issues(
        title,
        body,
        similar_issues,
        current_issue_number=issue_number,
        limit=3,
    )
    duplicate_issue_numbers = {
        int(issue.get("number", 0) or 0)
        for issue in duplicate_candidates
        if int(issue.get("number", 0) or 0) > 0
    }
    remaining_issue_candidates = [
        issue
        for issue in similar_issues
        if int(issue.get("number", 0) or 0) not in duplicate_issue_numbers
    ]

    triage_parts: list[str] = []
    duplicate_context = format_duplicate_issues_context(duplicate_candidates, limit=3)
    if duplicate_context:
        triage_parts.append("疑似重复 Issue（高置信）:\n" + duplicate_context)
    related_issues = format_related_issues_context(
        remaining_issue_candidates,
        current_issue_number=issue_number,
        limit=4,
    )
    if related_issues:
        triage_parts.append("相关 Issue / 使用讨论:\n" + related_issues)
    related_pr_context = format_related_prs_context(enriched_related_prs, limit=6)
    if related_pr_context:
        triage_parts.append("可能相关的 PR:\n" + related_pr_context)
    resolution_signals = format_resolution_signals_context(enriched_related_prs, limit=4)
    if resolution_signals:
        triage_parts.append("修复状态线索:\n" + resolution_signals)
    triage_context = "\n\n".join(part for part in triage_parts if part.strip())
    return (
        discussion_context,
        triage_context,
        latest_activity_fingerprint,
        latest_activity_updated_at,
        latest_comment_updated_at,
    )


async def _load_pr_discussion_context(pr_number: int) -> tuple[str, str, str]:
    """读取 PR 当前讨论，并返回最新外部活动指纹与更新时间。"""
    assert gh

    issue_comments_task = asyncio.create_task(
        asyncio.to_thread(gh.list_recent_issue_comments, pr_number, limit=60, per_page=100)
    )
    reviews_task = asyncio.create_task(
        asyncio.to_thread(gh.list_recent_pr_reviews, pr_number, limit=60, per_page=100)
    )
    review_comments_task = asyncio.create_task(
        asyncio.to_thread(gh.list_recent_pr_review_comments, pr_number, limit=60, per_page=100)
    )

    issue_comments: list[dict] = []
    reviews: list[dict] = []
    review_comments: list[dict] = []

    try:
        issue_comments = await issue_comments_task
    except Exception as exc:
        logger.warning("读取 PR #%d 会话评论失败，将在缺少 discussion context 的情况下继续: %s", pr_number, exc)
    try:
        reviews = await reviews_task
    except Exception as exc:
        logger.warning("读取 PR #%d reviews 失败，将在缺少 review history 的情况下继续: %s", pr_number, exc)
    try:
        review_comments = await review_comments_task
    except Exception as exc:
        logger.warning("读取 PR #%d review comments 失败，将在缺少 inline discussion 的情况下继续: %s", pr_number, exc)

    discussion_entries = collect_discussion_entries(
        issue_comments=issue_comments,
        reviews=reviews,
        review_comments=review_comments,
    )
    discussion_context = format_discussion_context(discussion_entries, limit=10)
    latest_activity = pick_latest_external_activity(discussion_entries, ignored_logins=_ignored_logins())
    latest_activity_fingerprint = ""
    latest_activity_updated_at = ""
    if latest_activity is not None:
        latest_activity_updated_at = latest_activity.updated_at
        latest_activity_fingerprint = build_activity_fingerprint(
            latest_activity.kind,
            latest_activity.activity_id,
            latest_activity.body,
            latest_activity.updated_at,
        )
    return discussion_context, latest_activity_fingerprint, latest_activity_updated_at


async def _resolve_notify_login() -> str:
    """获取同步失败时需要 @ 的 GitHub 登录名。"""
    assert config

    configured = (config.github.notify_login or "").strip().lstrip("@")
    if configured:
        return configured

    assert gh
    try:
        return await asyncio.to_thread(gh.get_authenticated_login)
    except Exception as exc:
        logger.warning("获取通知登录名失败，将跳过 @mention: %s", exc)
        return ""


def _build_pr_sync_blocked_comment(reason: str, mention_login: str = "") -> str:
    """构造“因本地准备失败而跳过 review”的说明评论。"""
    mention_prefix = f"@{mention_login} " if mention_login else ""
    lines = [
        f"{mention_prefix}自动评审已跳过。",
        "",
        "本次在评审前未能准备出可靠的本地代码对比上下文。为了避免基于旧代码或不完整上下文给出错误结论，本次不会继续执行 review。",
        "",
        f"原因: {reason}",
        "",
        "请检查本地仓库同步状态、基线分支可达性或 Git 凭证后，再重新触发评审。",
    ]
    return "\n".join(lines)


def _summarize_pr_mergeability(pr: dict) -> str:
    mergeable = pr.get("mergeable")
    mergeable_state = str(pr.get("mergeable_state", "") or "").strip().lower()
    if mergeable_state == "dirty":
        return "存在冲突，当前不能直接合并"
    if mergeable is True:
        return f"可合并（mergeable_state={mergeable_state or 'clean'}）"
    if mergeable_state == "blocked":
        return "GitHub 标记为 blocked（mergeable_state=blocked），不等同于存在冲突，不能单独据此判断不可直接合入"
    if mergeable is False:
        return f"当前不能直接自动合并（mergeable_state={mergeable_state or 'unknown'}）"
    if mergeable_state:
        return f"GitHub 尚未给出稳定 merge 结果（mergeable_state={mergeable_state}）"
    return "GitHub 暂未返回 mergeable 信息"


def _build_pr_runtime_review_facts(pr: dict, ci_status: dict) -> list[str]:
    ci_state = str(ci_status.get("state", "missing") or "missing")
    ci_details = GitHubPoller.summarize_ci_details(ci_status)
    facts = [
        f"当前 CI 状态：{ci_state}",
        f"当前 merge 状态：{_summarize_pr_mergeability(pr)}",
        f"当前 draft 状态：{'是' if pr.get('draft') else '否'}",
    ]
    if ci_status.get("has_checks", False):
        facts.append(f"CI 细节摘要：{ci_details}")
    else:
        facts.append("当前未检测到 checks/statuses")
    return facts


def _should_allow_pr_review_before_ci_ready(*, followup_mode: bool, activity_fingerprint: str) -> bool:
    """只有 follow-up 或明确讨论信号时，才允许在 CI 未就绪时继续评审。"""
    return followup_mode or bool(activity_fingerprint)


def _is_finalized_ci_failure(ci_status: dict) -> bool:
    return str(ci_status.get("state", "") or "").lower() == "failure" and bool(ci_status.get("finalized"))


def _build_ci_failure_fingerprint(head_sha: str, ci_status: dict) -> str:
    summary = GitHubPoller.summarize_ci_details(ci_status)
    return build_activity_fingerprint(
        "ci_failure",
        head_sha,
        body=summary,
        updated_at=head_sha,
    )


def _build_ci_failure_discussion_context(ci_status: dict) -> str:
    return (
        "CI 状态变化触发复核：当前 head 的阻断检查失败。\n"
        f"- CI 摘要：{GitHubPoller.summarize_ci_details(ci_status)}\n"
        "- 请基于当前代码和失败检查给出可执行定位，不要只写“等待 CI”。"
    )


async def _is_pr_still_open_on_same_head(pr_number: int, expected_head_sha: str, stage: str) -> bool:
    """确认 PR 仍处于 open，且 head 在处理期间没有变化。"""
    assert gh

    try:
        latest_pr = await asyncio.to_thread(gh.get_pr, pr_number)
    except Exception as exc:
        logger.error("复核 PR #%d 最新 head 失败 (%s): %s", pr_number, stage, exc)
        return False

    if latest_pr.get("state") != "open":
        logger.info("PR #%d 在%s期间已不是 open 状态，本次结果作废", pr_number, stage)
        return False

    latest_head = latest_pr.get("head") or {}
    latest_head_sha = str(latest_head.get("sha", "") or "")
    if latest_head_sha != expected_head_sha:
        logger.info(
            "PR #%d 在%s期间 head 已变化: expected=%s, latest=%s，本次结果作废",
            pr_number,
            stage,
            expected_head_sha[:12] or "unknown",
            latest_head_sha[:12] or "unknown",
        )
        return False

    return True


async def _process_issue(
    issue: dict,
    source: str,
    *,
    activity_fingerprint: str = "",
    allow_unanswerable_reply: bool = True,
) -> bool:
    issue_number = issue["number"]
    lock = _issue_processing_lock(issue_number)
    _retain_processing_lock(_issue_processing_lock_refs, issue_number)
    try:
        async with lock:
            return await _process_issue_serialized(
                issue,
                source,
                activity_fingerprint=activity_fingerprint,
                allow_unanswerable_reply=allow_unanswerable_reply,
            )
    finally:
        _release_processing_lock(_issue_processing_locks, _issue_processing_lock_refs, issue_number, lock)


async def _process_issue_serialized(
    issue: dict,
    source: str,
    *,
    activity_fingerprint: str = "",
    allow_unanswerable_reply: bool = True,
) -> bool:
    """处理单个 Issue 对象。"""
    assert agent and gh and state_store

    issue_number = issue["number"]
    title = issue.get("title", "")
    body = issue.get("body", "") or ""
    labels = [l["name"] for l in issue.get("labels", [])]
    created_at = issue.get("created_at", "")
    updated_at = issue.get("updated_at", "")
    author_association = str(issue.get("author_association", "") or issue.get("authorAssociation", "") or "")

    if _is_ignored_resource_owner(issue.get("user", {})):
        logger.info("跳过来自自身/bot 的 Issue #%d", issue_number)
        return False
    if state_store.should_skip_issue_number(issue_number):
        logger.info("Issue #%d 小于等于阈值 %d，跳过", issue_number, state_store.issue_min_number())
        return False

    discussion_context = ""
    triage_context = ""
    _issue_context_loaded = False
    initial_key = build_issue_processing_key(title, body)
    initial_processed = state_store.is_issue_processed(
        issue_number,
        initial_key,
        updated_at=updated_at,
    )
    processing_key = initial_key
    event_processing_key = ""
    extra_processing_keys: list[str] = []
    latest_runtime_activity_fingerprint = ""
    should_allow_unanswerable_reply = allow_unanswerable_reply

    if activity_fingerprint:
        event_processing_key = build_issue_processing_key(title, body, activity_fingerprint)
        processing_key = event_processing_key
        should_allow_unanswerable_reply = False
        if processing_key != initial_key and not initial_processed:
            extra_processing_keys.append(initial_key)
    legacy_processed_without_keys = (
        source == "polling"
        and not initial_processed
        and state_store.has_processed_issue(issue_number)
        and not state_store.has_issue_processing_keys(issue_number)
    )

    if source == "polling" and (initial_processed or legacy_processed_without_keys):
        (
            discussion_context,
            triage_context,
            latest_activity_fingerprint,
            latest_activity_updated_at,
            latest_comment_updated_at,
        ) = await _load_issue_runtime_context(
            issue_number,
            title,
            body,
        )
        _issue_context_loaded = True
        latest_runtime_activity_fingerprint = latest_activity_fingerprint
        if legacy_processed_without_keys:
            processed_updated_at = state_store.issue_processed_updated_at(issue_number)
            has_newer_external_activity = bool(
                processed_updated_at and latest_activity_updated_at and processed_updated_at < latest_activity_updated_at
            )
            has_newer_comment_activity = bool(
                processed_updated_at and latest_comment_updated_at and processed_updated_at < latest_comment_updated_at
            )
            has_issue_changed_since_processed = bool(
                processed_updated_at
                and updated_at
                and processed_updated_at < updated_at
                and (not latest_comment_updated_at or latest_comment_updated_at < updated_at)
            )
            if not has_newer_external_activity and not has_issue_changed_since_processed:
                if has_newer_comment_activity:
                    logger.info("Issue #%d 更新时间仅由已忽略评论触发，且无新的外部活动，跳过", issue_number)
                else:
                    logger.info("Issue #%d 自上次处理后无新的外部活动或正文更新，跳过", issue_number)
                return False
            if not has_newer_external_activity:
                latest_runtime_activity_fingerprint = ""
            else:
                candidate_key = build_issue_processing_key(title, body, latest_activity_fingerprint)
                if candidate_key == initial_key or state_store.is_issue_processed(
                    issue_number,
                    candidate_key,
                    updated_at=updated_at,
                ):
                    logger.info("Issue #%d 的最新评论已处理过，跳过", issue_number)
                    return False
                processing_key = candidate_key
                should_allow_unanswerable_reply = False
        else:
            if not latest_activity_fingerprint:
                logger.info("Issue #%d 无新的外部评论需要跟进", issue_number)
                return False
            candidate_key = build_issue_processing_key(title, body, latest_activity_fingerprint)
            if candidate_key == initial_key or state_store.is_issue_processed(
                issue_number,
                candidate_key,
                updated_at=updated_at,
            ):
                logger.info("Issue #%d 的最新评论已处理过，跳过", issue_number)
                return False
            processing_key = candidate_key
            should_allow_unanswerable_reply = False

    claim_status = state_store.try_claim_issue_key(
        issue_number,
        processing_key,
        updated_at=updated_at,
    )
    if claim_status == "processed":
        logger.info("Issue #%d 的相同内容（key=%s）已处理过，跳过", issue_number, processing_key[:24] or "unknown")
        return False
    if claim_status == "inflight":
        logger.info("Issue #%d 的相同内容（key=%s）正在处理中，跳过重复触发", issue_number, processing_key[:24] or "unknown")
        return False

    try:
        if not _issue_context_loaded and not discussion_context and not triage_context:
            (
                discussion_context,
                triage_context,
                latest_runtime_activity_fingerprint,
                _latest_runtime_activity_updated_at,
                _latest_comment_updated_at,
            ) = await _load_issue_runtime_context(
                issue_number,
                title,
                body,
            )

        runtime_followup_key = ""
        if latest_runtime_activity_fingerprint:
            runtime_followup_key = build_issue_processing_key(
                title,
                body,
                latest_runtime_activity_fingerprint,
            )

        if activity_fingerprint:
            if initial_processed and runtime_followup_key:
                if state_store.is_issue_processed(issue_number, runtime_followup_key, updated_at=updated_at):
                    logger.info("Issue #%d 的较新讨论已处理过，跳过旧事件", issue_number)
                    return False
                if runtime_followup_key != processing_key:
                    if processing_key and processing_key not in extra_processing_keys:
                        extra_processing_keys.append(processing_key)
                    processing_key = runtime_followup_key
            elif not initial_processed:
                processing_key = initial_key
                should_allow_unanswerable_reply = True
                if event_processing_key and event_processing_key != initial_key and event_processing_key not in extra_processing_keys:
                    extra_processing_keys.append(event_processing_key)
                if runtime_followup_key and runtime_followup_key != initial_key and runtime_followup_key not in extra_processing_keys:
                    extra_processing_keys.append(runtime_followup_key)
        elif runtime_followup_key and runtime_followup_key != processing_key and runtime_followup_key not in extra_processing_keys:
            extra_processing_keys.append(runtime_followup_key)

        issue_marker = _build_idempotency_marker("issue", processing_key)
        if await _issue_comment_marker_exists(issue_number, issue_marker):
            state_store.mark_issue_processed(
                issue_number,
                processing_key=processing_key,
                extra_processing_keys=extra_processing_keys,
                source=source,
                created_at=created_at,
                updated_at=updated_at,
            )
            logger.info("Issue #%d 检测到已存在相同回复标记，补记本地状态并跳过重复发送", issue_number)
            return False

        logger.info(
            "处理 Issue #%d: %s (source=%s, key=%s)",
            issue_number,
            title,
            source,
            processing_key[:24] or "unknown",
        )
        reply, classification = await asyncio.to_thread(
            agent.handle_issue,
            title,
            body,
            labels,
            triage_context=triage_context,
            discussion_context=discussion_context,
            allow_unanswerable_reply=should_allow_unanswerable_reply,
            author_association=author_association,
        )

        if not reply:
            logger.warning("Issue #%d 未生成回复", issue_number)
            return False

        await asyncio.to_thread(gh.post_issue_comment, issue_number, _append_idempotency_marker(reply, issue_marker))
        state_store.mark_issue_processed(
            issue_number,
            processing_key=processing_key,
            extra_processing_keys=extra_processing_keys,
            source=source,
            created_at=created_at,
            updated_at=updated_at,
        )
        logger.info(
            "已回复 Issue #%d (classification: %s, source=%s)",
            issue_number,
            classification.classification,
            source,
        )
        return True
    finally:
        state_store.release_issue(issue_number, processing_key)


async def _process_pr(
    pr: dict,
    source: str,
    ci_status: dict | None = None,
    *,
    activity_fingerprint: str = "",
) -> bool:
    pr_number = pr["number"]
    lock = _pr_processing_lock(pr_number)
    _retain_processing_lock(_pr_processing_lock_refs, pr_number)
    try:
        async with lock:
            return await _process_pr_serialized(
                pr,
                source,
                ci_status,
                activity_fingerprint=activity_fingerprint,
            )
    finally:
        _release_processing_lock(_pr_processing_locks, _pr_processing_lock_refs, pr_number, lock)


async def _process_pr_serialized(
    pr: dict,
    source: str,
    ci_status: dict | None = None,
    *,
    activity_fingerprint: str = "",
) -> bool:
    """处理单个 PR 对象。"""
    assert agent and gh and state_store

    pr_number = pr["number"]
    original_head = pr.get("head") or {}
    original_head_sha = str(original_head.get("sha", "") or "")

    try:
        pr = await asyncio.to_thread(gh.get_pr, pr_number)
    except Exception as exc:
        logger.error("获取 PR #%d 最新详情失败: %s", pr_number, exc)
        return False

    title = pr.get("title", "")
    body = pr.get("body", "") or ""
    created_at = pr.get("created_at", "")
    updated_at = pr.get("updated_at", "")
    head = pr.get("head") or {}
    head_repo = head.get("repo") or {}
    head_ref = str(head.get("ref", "") or "")
    head_sha = str(head.get("sha", "") or "")
    head_repo_clone_url = str(head_repo.get("clone_url", "") or "")
    base_ref = pr.get("base", {}).get("ref", "") or ""

    if original_head_sha and original_head_sha != head_sha:
        logger.info(
            "PR #%d 收到的事件 head=%s 已过时，改用当前最新 head=%s",
            pr_number,
            original_head_sha[:12],
            head_sha[:12] or "unknown",
        )
        ci_status = None

    if _is_ignored_resource_owner(pr.get("user", {})):
        logger.info("跳过来自自身/bot 的 PR #%d", pr_number)
        return False
    if pr.get("state") != "open":
        logger.info("PR #%d 当前不是 open 状态，跳过", pr_number)
        return False
    if state_store.should_skip_pr_number(pr_number):
        logger.info("PR #%d 小于等于阈值 %d，跳过", pr_number, state_store.pr_min_number())
        return False
    if not head_sha:
        logger.warning("PR #%d 缺少 head sha，跳过", pr_number)
        return False

    initial_review_key = build_pr_review_key(head_sha, title, body, base_ref)
    review_key = initial_review_key
    event_review_key = ""
    followup_mode = False
    discussion_context = ""
    extra_review_keys: list[str] = []
    latest_runtime_activity_fingerprint = ""
    latest_runtime_activity_updated_at = ""
    compat_processed = False
    initial_processed = state_store.is_pr_review_processed(
        pr_number,
        initial_review_key,
        head_sha=head_sha,
        updated_at=updated_at,
    )
    if not initial_processed and state_store.has_processed_pr(pr_number):
        if (
            not state_store.has_pr_review_processing_keys(pr_number)
            and state_store.pr_processed_head_sha(pr_number) == head_sha
        ):
            initial_processed = True
            compat_processed = True
            logger.info(
                "PR #%d 命中 legacy 判重：旧状态仅记录 head，相同 head 按已完成首轮评审处理",
                pr_number,
            )
        elif state_store.is_pr_same_version_processed(
            pr_number,
            head_sha=head_sha,
            updated_at=updated_at,
        ):
            initial_processed = True
            compat_processed = True
            logger.info(
                "PR #%d 命中兼容判重：head 与 updated_at 未变化，按已完成首轮评审处理",
                pr_number,
            )

    if pr.get("draft") and not initial_processed:
        logger.info("PR #%d 当前是 draft，跳过首轮自动评审", pr_number)
        return False

    # Head SHA 稳定检查：已评审过的 PR 如果 head 刚变（无外部讨论），等待稳定后再评审
    if not initial_processed and not activity_fingerprint and state_store.has_processed_pr(pr_number):
        if not state_store.is_pr_head_stable(pr_number, head_sha, stabilize_seconds=config.server.pr_head_stabilize_seconds):
            logger.info(
                "PR #%d 的 head SHA 尚未稳定（首次观测不足 %d 秒），等待下轮",
                pr_number,
                config.server.pr_head_stabilize_seconds,
            )
            return False

    if ci_status is None:
        try:
            ci_status = await asyncio.to_thread(gh.get_commit_ci_status, head_sha)
        except Exception as exc:
            logger.error("获取 PR #%d CI 状态失败: %s", pr_number, exc)
            return False

    ci_state = ci_status.get("state", "missing")
    state_store.mark_pr_seen(
        pr_number,
        head_sha=head_sha,
        ci_state=ci_state,
        created_at=created_at,
        updated_at=updated_at,
    )
    logger.info(
        "PR #%d 当前 CI 状态: %s, details=%s",
        pr_number,
        ci_state,
        GitHubPoller.summarize_ci_details(ci_status),
    )
    ci_failure_fingerprint = (
        _build_ci_failure_fingerprint(head_sha, ci_status)
        if _is_finalized_ci_failure(ci_status)
        else ""
    )

    if activity_fingerprint:
        event_review_key = build_pr_review_key(
            head_sha,
            title,
            body,
            base_ref,
            discussion_fingerprint=activity_fingerprint,
        )
        review_key = event_review_key
        followup_mode = initial_processed and review_key != initial_review_key
        if review_key != initial_review_key and not initial_processed:
            extra_review_keys.append(initial_review_key)
    elif source == "polling" and initial_processed:
        discussion_context, latest_runtime_activity_fingerprint, latest_runtime_activity_updated_at = await _load_pr_discussion_context(pr_number)
        if not latest_runtime_activity_fingerprint:
            if ci_failure_fingerprint:
                candidate_key = build_pr_review_key(
                    head_sha,
                    title,
                    body,
                    base_ref,
                    discussion_fingerprint=ci_failure_fingerprint,
                )
                if not state_store.is_pr_review_processed(
                    pr_number,
                    candidate_key,
                    head_sha=head_sha,
                    updated_at=updated_at,
                ):
                    review_key = candidate_key
                    followup_mode = True
                    discussion_context = _build_ci_failure_discussion_context(ci_status)
                    logger.info("PR #%d 无新外部讨论，但 CI 失败，触发一次跟进评审", pr_number)
                else:
                    logger.info("PR #%d 的 CI 失败复核已处理过，跳过", pr_number)
                    return False
            else:
                logger.info("PR #%d 无新的外部讨论需要跟进", pr_number)
                return False
        processed_updated_at = state_store.pr_processed_updated_at(pr_number)
        if (
            processed_updated_at
            and latest_runtime_activity_updated_at
            and processed_updated_at >= latest_runtime_activity_updated_at
        ):
            if ci_failure_fingerprint:
                candidate_key = build_pr_review_key(
                    head_sha,
                    title,
                    body,
                    base_ref,
                    discussion_fingerprint=ci_failure_fingerprint,
                )
                if not state_store.is_pr_review_processed(
                    pr_number,
                    candidate_key,
                    head_sha=head_sha,
                    updated_at=updated_at,
                ):
                    review_key = candidate_key
                    followup_mode = True
                    discussion_context = _build_ci_failure_discussion_context(ci_status)
                    logger.info("PR #%d 外部讨论已处理，但 CI 失败，触发一次跟进评审", pr_number)
                else:
                    logger.info("PR #%d 自上次处理后无新的外部讨论，且 CI 失败复核已处理过，跳过", pr_number)
                    return False
            else:
                logger.info("PR #%d 自上次处理后无新的外部讨论，跳过", pr_number)
                return False
        if latest_runtime_activity_fingerprint:
            candidate_key = build_pr_review_key(
                head_sha,
                title,
                body,
                base_ref,
                discussion_fingerprint=latest_runtime_activity_fingerprint,
            )
            if candidate_key == initial_review_key or state_store.is_pr_review_processed(
                pr_number,
                candidate_key,
                head_sha=head_sha,
                updated_at=updated_at,
            ):
                logger.info("PR #%d 的最新讨论已处理过，跳过", pr_number)
                return False
            review_key = candidate_key
            followup_mode = True
    elif compat_processed:
        logger.info("PR #%d 当前版本已处理过，且没有新的讨论信号，跳过", pr_number)
        return False
    if not ci_status.get("ready", False):
        if _is_finalized_ci_failure(ci_status):
            logger.info("PR #%d 当前 CI 已失败，继续评审并输出可执行定位", pr_number)
        elif not _should_allow_pr_review_before_ci_ready(
            followup_mode=followup_mode,
            activity_fingerprint=activity_fingerprint,
        ):
            logger.info("PR #%d 当前 CI 未成功，且没有明确讨论信号，跳过首轮评审", pr_number)
            return False
        else:
            logger.info("PR #%d 当前 CI 未成功，但因存在明确讨论信号，继续评审", pr_number)
    runtime_review_facts = _build_pr_runtime_review_facts(pr, ci_status)

    if followup_mode or not initial_processed:
        if not discussion_context:
            discussion_context, latest_runtime_activity_fingerprint, latest_runtime_activity_updated_at = await _load_pr_discussion_context(pr_number)
        latest_runtime_review_key = ""
        if latest_runtime_activity_fingerprint:
            latest_runtime_review_key = build_pr_review_key(
                head_sha,
                title,
                body,
                base_ref,
                discussion_fingerprint=latest_runtime_activity_fingerprint,
            )
        if initial_processed:
            if latest_runtime_review_key and state_store.is_pr_review_processed(
                pr_number,
                latest_runtime_review_key,
                head_sha=head_sha,
                updated_at=updated_at,
            ):
                logger.info("PR #%d 的较新讨论已处理过，跳过旧事件", pr_number)
                return False
            if latest_runtime_review_key and latest_runtime_review_key != review_key:
                if review_key and review_key not in extra_review_keys:
                    extra_review_keys.append(review_key)
                review_key = latest_runtime_review_key
        else:
            review_key = initial_review_key
            if event_review_key and event_review_key != initial_review_key and event_review_key not in extra_review_keys:
                extra_review_keys.append(event_review_key)
            if (
                latest_runtime_review_key
                and latest_runtime_review_key != initial_review_key
                and latest_runtime_review_key not in extra_review_keys
            ):
                extra_review_keys.append(latest_runtime_review_key)

    claim_status = state_store.try_claim_pr_review(
        pr_number,
        review_key,
        head_sha=head_sha,
        updated_at=updated_at,
    )
    if claim_status == "processed":
        logger.info("PR #%d 的相同版本（review_key=%s）已处理过，跳过", pr_number, review_key[:24] or "unknown")
        return False
    if claim_status == "inflight":
        logger.info(
            "PR #%d 的相同版本（review_key=%s）正在处理中，跳过重复触发",
            pr_number,
            review_key[:24] or "unknown",
        )
        return False

    try:
        review_marker = _build_idempotency_marker("pr-review", review_key)
        if await _pr_review_marker_exists(pr_number, review_marker):
            state_store.mark_pr_processed(
                pr_number,
                review_key=review_key,
                extra_review_keys=extra_review_keys,
                head_sha=head_sha,
                source=source,
                ci_state=ci_state,
                created_at=created_at,
                updated_at=updated_at,
            )
            logger.info("PR #%d 检测到已存在相同 review 标记，补记本地状态并跳过重复发送", pr_number)
            return False

        logger.info(
            "处理 PR #%d: %s (branch: %s, source=%s, ci=%s, followup=%s)",
            pr_number,
            title,
            head_ref or "unknown",
            source,
            ci_state,
            followup_mode,
        )

        if not await _is_pr_still_open_on_same_head(pr_number, head_sha, "开始评审前"):
            return False

        if followup_mode:
            review_result = await asyncio.to_thread(
                agent.handle_pr_followup,
                title,
                body,
                base_ref,
                discussion_context,
                head_ref,
                head_repo_clone_url,
                head_sha,
                runtime_review_facts,
            )
        else:
            review_result = await asyncio.to_thread(
                agent.handle_pr,
                title,
                body,
                base_ref,
                head_ref,
                head_repo_clone_url,
                head_sha,
                discussion_context,
                runtime_review_facts,
            )
        if review_result.blocked_reason:
            if not await _is_pr_still_open_on_same_head(pr_number, head_sha, "发送同步失败通知前"):
                return False
            should_notify = state_store.should_notify_pr_blocked(
                pr_number,
                head_sha=head_sha,
                reason=review_result.blocked_reason,
                cooldown_seconds=PR_BLOCKED_NOTIFY_COOLDOWN_SECONDS,
            )
            if should_notify:
                blocked_marker = _build_idempotency_marker(
                    "pr-blocked",
                    f"{head_sha}\n{review_result.blocked_reason.strip()}",
                )
                if await _issue_comment_marker_exists(pr_number, blocked_marker):
                    state_store.mark_pr_blocked(
                        pr_number,
                        head_sha=head_sha,
                        source=source,
                        reason=review_result.blocked_reason,
                        notified=True,
                        ci_state=ci_state,
                        created_at=created_at,
                        updated_at=updated_at,
                    )
                    logger.warning("PR #%d 检测到已存在同步失败通知标记，补记本地状态并跳过重复通知", pr_number)
                    return False
                notify_login = await _resolve_notify_login()
                blocked_comment = _build_pr_sync_blocked_comment(review_result.blocked_reason, notify_login)
                await asyncio.to_thread(
                    gh.post_issue_comment,
                    pr_number,
                    _append_idempotency_marker(blocked_comment, blocked_marker),
                )
            state_store.mark_pr_blocked(
                pr_number,
                head_sha=head_sha,
                source=source,
                reason=review_result.blocked_reason,
                notified=should_notify,
                ci_state=ci_state,
                created_at=created_at,
                updated_at=updated_at,
            )
            if should_notify:
                logger.warning("PR #%d 因同步失败已跳过 review，并已通知: %s", pr_number, review_result.blocked_reason)
            else:
                logger.warning("PR #%d 因同步失败已跳过 review，冷却期内不重复通知: %s", pr_number, review_result.blocked_reason)
            return True

        reply = review_result.reply
        if not reply:
            logger.warning("PR #%d 未生成%s", pr_number, "继续评审评论" if followup_mode else "review")
            return False

        if not await _is_pr_still_open_on_same_head(pr_number, head_sha, "发表 review 前"):
            return False

        await asyncio.to_thread(gh.post_pr_review_comment, pr_number, _append_idempotency_marker(reply, review_marker))
        state_store.mark_pr_processed(
            pr_number,
            review_key=review_key,
            extra_review_keys=extra_review_keys,
            head_sha=head_sha,
            source=source,
            ci_state=ci_state,
            created_at=created_at,
            updated_at=updated_at,
        )
        logger.info("已%s PR #%d (source=%s)", "跟进评审" if followup_mode else "review", pr_number, source)
        return True
    finally:
        state_store.release_pr_review(pr_number, review_key, head_sha=head_sha)


# ------------------------------------------------------------------
# 健康检查
# ------------------------------------------------------------------

@app.get("/health")
async def health():
    state_summary = state_store.summary() if state_store else {}
    return {
        "status": "ok",
        "repo": config.github.repo if config else "not initialized",
        "event_source": config.server.event_source if config else "",
        "webhook_enabled": config.server.enable_webhook if config else False,
        "poll_interval_seconds": config.server.poll_interval_seconds if config else 0,
        "poll_items_per_page": config.server.poll_items_per_page if config else 0,
        "poll_pending_pr_batch_size": config.server.poll_pending_pr_batch_size if config else 0,
        "poll_pending_pr_max_age_days": config.server.poll_pending_pr_max_age_days if config else 0,
        "poll_review_submission_batch_size": config.server.poll_review_submission_batch_size if config else 0,
        "poll_review_submission_max_age_days": config.server.poll_review_submission_max_age_days if config else 0,
        "primary_cli": config.cli.primary_cli if config else "",
        "primary_model": config.cli.primary_model if config else "",
        "primary_fallback_model": config.cli.primary_fallback_model if config else "",
        "fallback_cli": config.cli.fallback_cli if config else "",
        "fallback_model": config.cli.fallback_model if config else "",
        "fallback_fallback_model": config.cli.fallback_fallback_model if config else "",
        "local_repo_path": config.local_repo.path if config else "",
        "default_branch": config.local_repo.default_branch if config else "",
        "state": state_summary,
    }


# ------------------------------------------------------------------
# 入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    # 先加载配置获取端口
    _cfg = load_config()
    uvicorn.run(
        "review:app",
        host="0.0.0.0",
        port=_cfg.server.port,
        log_level=_cfg.server.log_level.lower(),
    )
