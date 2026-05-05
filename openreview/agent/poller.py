"""
GitHub polling 后台任务

职责:
1. 周期性扫描 GitHub 上的新对象和新讨论活动
2. 对启动前已存在、但启动后重新活跃的 Issue / PR 也能继续跟进
3. 为 open PR 持续补充评审触发，并把首轮评审是否放宽交给主流程按策略判断
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
from collections.abc import Awaitable, Callable
import re
from datetime import datetime

import httpx

from agent.discussion import is_ignored_activity
from agent.github_client import GitHubClient
from agent.state_store import LocalStateStore, build_activity_fingerprint

logger = logging.getLogger(__name__)

IssueProcessor = Callable[..., Awaitable[bool]]
PRProcessor = Callable[..., Awaitable[bool]]
ActivityTarget = tuple[str, str, str]
_AUTOCODE_COMMAND_RE = re.compile(r"(?mi)^\s*/(plan|implement|retry|abort|status)\b")


@dataclass
class PollStats:
    fetched: int = 0
    eligible: int = 0
    processed: int = 0
    skipped_threshold: int = 0
    skipped_prebaseline: int = 0
    skipped_already_processed: int = 0
    skipped_not_open: int = 0
    skipped_non_issue: int = 0
    skipped_missing_identity: int = 0
    latest_updated_at: str = ""
    earliest_failed_updated_at: str = ""


@dataclass
class DiscussionPollStats:
    issue_comments_fetched: int = 0
    review_comments_fetched: int = 0
    issue_targets: int = 0
    pr_targets: int = 0
    processed: int = 0
    skipped_ignored: int = 0
    skipped_missing_identity: int = 0
    latest_updated_at: str = ""
    earliest_failed_updated_at: str = ""


class GitHubPoller:
    """GitHub 轮询器。"""

    def __init__(
        self,
        gh: GitHubClient,
        state_store: LocalStateStore,
        issue_processor: IssueProcessor,
        pr_processor: PRProcessor,
        *,
        interval_seconds: int = 60,
        items_per_page: int = 30,
        ignored_logins: set[str] | None = None,
        pending_pr_batch_size: int = 20,
        pending_pr_max_age_days: int = 30,
        review_submission_batch_size: int = 20,
        review_submission_max_age_days: int = 21,
    ):
        self.gh = gh
        self.state_store = state_store
        self.issue_processor = issue_processor
        self.pr_processor = pr_processor
        self.interval_seconds = interval_seconds
        self.items_per_page = items_per_page
        self.ignored_logins = {login.lower() for login in (ignored_logins or set()) if login}
        self.pending_pr_batch_size = max(int(pending_pr_batch_size or 0), 1)
        self.pending_pr_max_age_days = max(int(pending_pr_max_age_days or 0), 1)
        self.review_submission_batch_size = max(int(review_submission_batch_size or 0), 1)
        self.review_submission_max_age_days = max(int(review_submission_max_age_days or 0), 1)

    async def run(self):
        """持续执行轮询。"""
        while True:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Polling 周期执行失败")
            await asyncio.sleep(self.interval_seconds)

    async def poll_once(self):
        """执行一次完整轮询。"""
        if self.state_store.ensure_polling_baseline():
            logger.info(
                "Polling 基线已初始化: issue_after=%s, pr_after=%s, issue_cursor=%s, pr_cursor=%s",
                self.state_store.issue_baseline(),
                self.state_store.pr_baseline(),
                self.state_store.issue_cursor(),
                self.state_store.pr_cursor(),
            )
        else:
            logger.info(
                "Polling 开始: issue_after=%s, pr_after=%s, issue_cursor=%s, pr_cursor=%s, issue_comment_after=%s, review_comment_after=%s, issue_min=%s, pr_min=%s",
                self.state_store.issue_baseline(),
                self.state_store.pr_baseline(),
                self.state_store.issue_cursor(),
                self.state_store.pr_cursor(),
                self.state_store.issue_comment_cursor(),
                self.state_store.review_comment_cursor(),
                self.state_store.issue_min_number(),
                self.state_store.pr_min_number(),
            )

        discussion_stats = await self._poll_discussions()
        issue_stats = await self._poll_issues()
        pr_stats = await self._poll_prs()
        self._log_poll_summary(discussion_stats, issue_stats, pr_stats)

    async def _poll_discussions(self) -> DiscussionPollStats:
        issue_targets, issue_cursor, issue_stats = await self._collect_issue_comment_targets()
        review_targets, review_cursor, review_stats = await self._collect_review_comment_targets()
        review_submission_targets, review_submission_stats = await self._collect_review_submission_targets()
        issue_comment_failed_at = issue_stats.earliest_failed_updated_at
        review_comment_failed_at = review_stats.earliest_failed_updated_at

        stats = DiscussionPollStats(
            issue_comments_fetched=issue_stats.issue_comments_fetched,
            review_comments_fetched=review_stats.review_comments_fetched,
            skipped_ignored=issue_stats.skipped_ignored + review_stats.skipped_ignored,
            skipped_missing_identity=issue_stats.skipped_missing_identity + review_stats.skipped_missing_identity,
            latest_updated_at=self._max_timestamp(issue_cursor, review_cursor),
            earliest_failed_updated_at=self._min_timestamp(
                issue_stats.earliest_failed_updated_at,
                review_stats.earliest_failed_updated_at,
            ),
        )

        issue_detail_cache: dict[int, dict] = {}
        issue_activity: dict[int, ActivityTarget] = {}
        pr_activity: dict[int, ActivityTarget] = dict(review_targets)
        for pr_number, target in review_submission_targets.items():
            self._merge_activity_target(pr_activity, pr_number, target[0], target[1], target[2])

        for issue_number, (fingerprint, updated_at, source) in issue_targets.items():
            try:
                issue = issue_detail_cache.get(issue_number)
                if issue is None:
                    issue = await asyncio.to_thread(self.gh.get_issue, issue_number)
                    issue_detail_cache[issue_number] = issue
                if "pull_request" in issue:
                    self._merge_activity_target(pr_activity, issue_number, fingerprint, updated_at, source)
                else:
                    self._merge_activity_target(issue_activity, issue_number, fingerprint, updated_at, source)
            except Exception:
                stats.earliest_failed_updated_at = self._min_timestamp(stats.earliest_failed_updated_at, updated_at)
                if source == "issue_comment":
                    issue_comment_failed_at = self._min_timestamp(issue_comment_failed_at, updated_at)
                logger.exception("Polling 解析评论目标 #%d 失败，跳过该活动", issue_number)

        stats.issue_targets = len(issue_activity)
        stats.pr_targets = len(pr_activity)
        stats.skipped_ignored += review_submission_stats.skipped_ignored
        stats.skipped_missing_identity += review_submission_stats.skipped_missing_identity
        stats.latest_updated_at = self._max_timestamp(stats.latest_updated_at, review_submission_stats.latest_updated_at)
        stats.earliest_failed_updated_at = self._min_timestamp(
            stats.earliest_failed_updated_at,
            review_submission_stats.earliest_failed_updated_at,
        )

        for issue_number, (fingerprint, _updated_at, source) in issue_activity.items():
            try:
                issue = issue_detail_cache.get(issue_number)
                if issue is None:
                    issue = await asyncio.to_thread(self.gh.get_issue, issue_number)
                    issue_detail_cache[issue_number] = issue
                if await self.issue_processor(
                    issue,
                    "polling",
                    activity_fingerprint=fingerprint,
                    allow_unanswerable_reply=False,
                ):
                    stats.processed += 1
            except Exception:
                stats.earliest_failed_updated_at = self._min_timestamp(stats.earliest_failed_updated_at, _updated_at)
                if source == "issue_comment":
                    issue_comment_failed_at = self._min_timestamp(issue_comment_failed_at, _updated_at)
                logger.exception("Polling 处理 Issue 评论目标 #%d 失败", issue_number)

        for pr_number, (fingerprint, _updated_at, source) in pr_activity.items():
            try:
                if await self.pr_processor(
                    {"number": pr_number},
                    "polling",
                    None,
                    activity_fingerprint=fingerprint,
                ):
                    stats.processed += 1
            except Exception:
                stats.earliest_failed_updated_at = self._min_timestamp(stats.earliest_failed_updated_at, _updated_at)
                if source == "issue_comment":
                    issue_comment_failed_at = self._min_timestamp(issue_comment_failed_at, _updated_at)
                elif source == "review_comment":
                    review_comment_failed_at = self._min_timestamp(review_comment_failed_at, _updated_at)
                logger.exception("Polling 处理 PR 评论目标 #%d 失败", pr_number)

        issue_cursor_target = issue_comment_failed_at or issue_cursor
        if issue_cursor_target:
            self.state_store.advance_issue_comment_cursor(
                issue_cursor_target,
                overlap_seconds=5 if not issue_comment_failed_at else 30,
            )
        review_cursor_target = review_comment_failed_at or review_cursor
        if review_cursor_target:
            self.state_store.advance_review_comment_cursor(
                review_cursor_target,
                overlap_seconds=5 if not review_comment_failed_at else 30,
            )
        return stats

    async def _poll_issues(self) -> PollStats:
        cursor = self.state_store.issue_cursor()
        page = 1
        stats = PollStats()
        while True:
            issues = await asyncio.to_thread(
                self.gh.list_issues,
                state="all",
                sort="updated",
                direction="desc",
                since=cursor or None,
                per_page=self.items_per_page,
                page=page,
            )
            if not issues:
                break
            stats.fetched += len(issues)
            for issue in issues:
                if "pull_request" in issue:
                    stats.skipped_non_issue += 1
                    continue
                if issue.get("state") != "open":
                    stats.skipped_not_open += 1
                    continue
                created_at = issue.get("created_at", "")
                updated_at = issue.get("updated_at", "") or created_at
                stats.latest_updated_at = self._max_timestamp(stats.latest_updated_at, updated_at)
                issue_number = issue.get("number")
                if not isinstance(issue_number, int):
                    stats.skipped_missing_identity += 1
                    continue
                if self.state_store.should_skip_issue_number(issue_number):
                    stats.skipped_threshold += 1
                    continue
                is_after_cursor = bool(cursor and updated_at and updated_at >= cursor)
                is_tracked = self.state_store.has_tracked_issue(issue_number)
                if not is_after_cursor and not is_tracked:
                    stats.skipped_prebaseline += 1
                    continue
                stats.eligible += 1
                try:
                    if await self.issue_processor(issue, "polling"):
                        stats.processed += 1
                except Exception:
                    stats.earliest_failed_updated_at = self._min_timestamp(
                        stats.earliest_failed_updated_at,
                        updated_at,
                    )
                    logger.exception("Polling 处理 Issue #%d 失败", issue_number)
            if len(issues) < self.items_per_page:
                break
            page += 1

        cursor_target = stats.earliest_failed_updated_at or stats.latest_updated_at
        if cursor_target:
            self.state_store.advance_issue_cursor(
                cursor_target,
                overlap_seconds=5 if not stats.earliest_failed_updated_at else 30,
            )
        return stats

    async def _poll_prs(self) -> PollStats:
        cursor = self.state_store.pr_cursor()
        page = 1
        stats = PollStats()
        seen_pr_numbers: set[int] = set()
        while True:
            prs = await asyncio.to_thread(
                self.gh.list_pull_requests,
                state="open",
                sort="updated",
                direction="desc",
                per_page=self.items_per_page,
                page=page,
            )
            if not prs:
                break
            stats.fetched += len(prs)
            should_stop = False
            for pr in prs:
                pr_number = pr.get("number")
                if isinstance(pr_number, int):
                    seen_pr_numbers.add(pr_number)
                created_at = pr.get("created_at", "")
                updated_at = pr.get("updated_at", "")
                stats.latest_updated_at = self._max_timestamp(stats.latest_updated_at, updated_at)
                body = pr.get("body", "") or ""
                head = pr.get("head") or {}
                head_sha = str(head.get("sha", "") or "")
                if not isinstance(pr_number, int) or not head_sha:
                    stats.skipped_missing_identity += 1
                    continue
                if self.state_store.should_skip_pr_number(pr_number):
                    stats.skipped_threshold += 1
                    continue

                is_after_cursor = bool(cursor and updated_at and updated_at >= cursor)
                is_tracked = self.state_store.has_tracked_pr(pr_number)
                if not is_after_cursor and not is_tracked:
                    stats.skipped_prebaseline += 1
                    should_stop = True
                    continue

                stats.eligible += 1
                await self._process_polled_pr(pr, stats)
            if len(prs) < self.items_per_page:
                break
            if should_stop:
                break
            page += 1

        for pr_number in self.state_store.pending_pr_numbers_for_polling(
            limit=max(self.items_per_page, self.pending_pr_batch_size),
            max_age_days=self.pending_pr_max_age_days,
        ):
            if pr_number in seen_pr_numbers:
                continue
            try:
                pr = await asyncio.to_thread(self.gh.get_pr, pr_number)
            except Exception:
                logger.exception("Polling 补拉已跟踪 PR #%d 失败", pr_number)
                continue
            if pr.get("state") != "open":
                continue
            stats.eligible += 1
            updated_at = str(pr.get("updated_at", "") or pr.get("created_at", "") or "")
            stats.latest_updated_at = self._max_timestamp(stats.latest_updated_at, updated_at)
            await self._process_polled_pr(pr, stats)

        cursor_target = stats.earliest_failed_updated_at or stats.latest_updated_at
        if cursor_target:
            self.state_store.advance_pr_cursor(
                cursor_target,
                overlap_seconds=5 if not stats.earliest_failed_updated_at else 30,
            )
        return stats

    def _log_poll_summary(self, discussion_stats: DiscussionPollStats, issue_stats: PollStats, pr_stats: PollStats):
        if (
            discussion_stats.issue_targets == 0 and
            discussion_stats.pr_targets == 0 and
            discussion_stats.issue_comments_fetched == 0 and
            discussion_stats.review_comments_fetched == 0
        ):
            logger.info("本轮无新的讨论活动需要跟进")
        else:
            logger.info(
                "本轮讨论扫描完成: issue_comments=%d, review_comments=%d, issue_targets=%d, pr_targets=%d, processed=%d, ignored=%d",
                discussion_stats.issue_comments_fetched,
                discussion_stats.review_comments_fetched,
                discussion_stats.issue_targets,
                discussion_stats.pr_targets,
                discussion_stats.processed,
                discussion_stats.skipped_ignored,
            )

        if issue_stats.eligible == 0:
            logger.info(
                "本轮无新的 Issue 需要处理: fetched=%d, threshold=%d, prebaseline=%d, processed=%d",
                issue_stats.fetched,
                issue_stats.skipped_threshold,
                issue_stats.skipped_prebaseline,
                issue_stats.skipped_already_processed,
            )
        else:
            logger.info(
                "本轮 Issue 扫描完成: fetched=%d, eligible=%d, processed=%d, processed_before=%d",
                issue_stats.fetched,
                issue_stats.eligible,
                issue_stats.processed,
                issue_stats.skipped_already_processed,
            )

        if pr_stats.eligible == 0:
            logger.info(
                "本轮无新的 PR 需要处理: fetched=%d, threshold=%d, prebaseline=%d, processed=%d",
                pr_stats.fetched,
                pr_stats.skipped_threshold,
                pr_stats.skipped_prebaseline,
                pr_stats.skipped_already_processed,
            )
        else:
            logger.info(
                "本轮 PR 扫描完成: fetched=%d, eligible=%d, processed=%d, processed_before=%d",
                pr_stats.fetched,
                pr_stats.eligible,
                pr_stats.processed,
                pr_stats.skipped_already_processed,
            )

    async def _process_polled_pr(self, pr: dict, stats: PollStats):
        pr_number = pr.get("number")
        created_at = str(pr.get("created_at", "") or "")
        updated_at = str(pr.get("updated_at", "") or created_at)
        head = pr.get("head") or {}
        head_sha = str(head.get("sha", "") or "")
        if not isinstance(pr_number, int) or not head_sha:
            stats.skipped_missing_identity += 1
            return

        try:
            ci_status = await asyncio.to_thread(self.gh.get_commit_ci_status, head_sha)
            ci_state = ci_status.get("state", "missing")
            self.state_store.mark_pr_seen(
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
                self.summarize_ci_details(ci_status),
            )

            if await self.pr_processor(pr, "polling", ci_status):
                stats.processed += 1
        except Exception:
            stats.earliest_failed_updated_at = self._min_timestamp(
                stats.earliest_failed_updated_at,
                updated_at,
            )
            logger.exception("Polling 处理 PR #%d 失败", pr_number)

    async def _collect_issue_comment_targets(self) -> tuple[dict[int, ActivityTarget], str, DiscussionPollStats]:
        cursor = self.state_store.issue_comment_cursor()
        page = 1
        max_updated_at = ""
        targets: dict[int, ActivityTarget] = {}
        stats = DiscussionPollStats()

        while True:
            comments = await asyncio.to_thread(
                self.gh.list_repository_issue_comments,
                sort="updated",
                direction="desc",
                since=cursor or None,
                per_page=self.items_per_page,
                page=page,
            )
            if not comments:
                break

            stats.issue_comments_fetched += len(comments)
            for comment in comments:
                updated_at = str(comment.get("updated_at", "") or comment.get("created_at", "") or "")
                stats.latest_updated_at = self._max_timestamp(stats.latest_updated_at, updated_at)
                if updated_at and (not max_updated_at or self._is_newer_timestamp(updated_at, max_updated_at)):
                    max_updated_at = updated_at
                if is_ignored_activity(
                    comment.get("user", {}),
                    body=str(comment.get("body", "") or ""),
                    ignored_logins=self.ignored_logins,
                ):
                    stats.skipped_ignored += 1
                    continue
                if _AUTOCODE_COMMAND_RE.search(str(comment.get("body", "") or "")):
                    logger.info("Polling 忽略 slash command issue_comment")
                    continue
                issue_number = self._extract_trailing_number(comment.get("issue_url", ""), "issues")
                if issue_number <= 0:
                    stats.skipped_missing_identity += 1
                    continue
                fingerprint = build_activity_fingerprint(
                    "issue_comment",
                    comment.get("id", 0) or 0,
                    body=str(comment.get("body", "") or ""),
                    updated_at=updated_at,
                )
                self._merge_activity_target(targets, issue_number, fingerprint, updated_at, "issue_comment")

            if len(comments) < self.items_per_page:
                break
            page += 1

        return targets, max_updated_at, stats

    async def _collect_review_comment_targets(self) -> tuple[dict[int, ActivityTarget], str, DiscussionPollStats]:
        cursor = self.state_store.review_comment_cursor()
        page = 1
        max_updated_at = ""
        targets: dict[int, ActivityTarget] = {}
        stats = DiscussionPollStats()

        while True:
            comments = await asyncio.to_thread(
                self.gh.list_repository_pr_review_comments,
                sort="updated",
                direction="desc",
                since=cursor or None,
                per_page=self.items_per_page,
                page=page,
            )
            if not comments:
                break

            stats.review_comments_fetched += len(comments)
            for comment in comments:
                updated_at = str(comment.get("updated_at", "") or comment.get("created_at", "") or "")
                stats.latest_updated_at = self._max_timestamp(stats.latest_updated_at, updated_at)
                if updated_at and (not max_updated_at or self._is_newer_timestamp(updated_at, max_updated_at)):
                    max_updated_at = updated_at
                if is_ignored_activity(
                    comment.get("user", {}),
                    body=str(comment.get("body", "") or ""),
                    ignored_logins=self.ignored_logins,
                ):
                    stats.skipped_ignored += 1
                    continue
                pr_number = self._extract_trailing_number(comment.get("pull_request_url", ""), "pulls")
                if pr_number <= 0:
                    stats.skipped_missing_identity += 1
                    continue
                fingerprint = build_activity_fingerprint(
                    "review_comment",
                    comment.get("id", 0) or 0,
                    body=str(comment.get("body", "") or ""),
                    updated_at=updated_at,
                )
                self._merge_activity_target(targets, pr_number, fingerprint, updated_at, "review_comment")

            if len(comments) < self.items_per_page:
                break
            page += 1

        return targets, max_updated_at, stats

    async def _collect_review_submission_targets(self) -> tuple[dict[int, ActivityTarget], DiscussionPollStats]:
        targets: dict[int, ActivityTarget] = {}
        stats = DiscussionPollStats()

        for pr_number in self.state_store.pr_numbers_for_review_polling(
            limit=max(self.items_per_page, self.review_submission_batch_size),
            max_age_days=self.review_submission_max_age_days,
        ):
            cutoff = self.state_store.pr_review_polling_cutoff(pr_number)
            latest_review_updated_at = ""
            page = 1
            scan_completed = False
            while True:
                try:
                    reviews = await asyncio.to_thread(
                        self.gh.list_pr_reviews,
                        pr_number,
                        per_page=self.items_per_page,
                        page=page,
                    )
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code == 404:
                        self.state_store.mark_pr_unreachable(pr_number)
                        logger.warning("PR #%d reviews 返回 404，标记为不可达", pr_number)
                    else:
                        stats.earliest_failed_updated_at = self._min_timestamp(
                            stats.earliest_failed_updated_at,
                            stats.latest_updated_at,
                        )
                        logger.exception("Polling 拉取 PR #%d 的 reviews 失败", pr_number)
                    break
                except Exception:
                    stats.earliest_failed_updated_at = self._min_timestamp(
                        stats.earliest_failed_updated_at,
                        stats.latest_updated_at,
                    )
                    logger.exception("Polling 拉取 PR #%d 的 reviews 失败", pr_number)
                    break

                # 成功拿到响应（含空列表），清除不可达标记
                self.state_store.clear_pr_unreachable(pr_number)

                if not reviews:
                    scan_completed = True
                    break

                for review in reviews:
                    review_updated_at = str(
                        review.get("updated_at", "") or review.get("submitted_at", "") or review.get("created_at", "") or ""
                    )
                    stats.latest_updated_at = self._max_timestamp(stats.latest_updated_at, review_updated_at)
                    latest_review_updated_at = self._max_timestamp(latest_review_updated_at, review_updated_at)
                    if cutoff and review_updated_at and not self._is_newer_timestamp(review_updated_at, cutoff):
                        continue
                    body = str(review.get("body", "") or "").strip()
                    if not body:
                        continue
                    if is_ignored_activity(
                        review.get("user", {}),
                        body=body,
                        ignored_logins=self.ignored_logins,
                    ):
                        stats.skipped_ignored += 1
                        continue

                    fingerprint = build_activity_fingerprint(
                        "review",
                        review.get("id", 0) or 0,
                        body=body,
                        updated_at=review_updated_at,
                    )
                    self._merge_activity_target(targets, pr_number, fingerprint, review_updated_at, "review")

                if len(reviews) < self.items_per_page:
                    scan_completed = True
                    break
                page += 1

            if scan_completed:
                self.state_store.mark_pr_review_polling_scanned(
                    pr_number,
                    latest_review_updated_at=latest_review_updated_at,
                )

        return targets, stats

    @staticmethod
    def _merge_activity_target(
        targets: dict[int, ActivityTarget],
        number: int,
        fingerprint: str,
        updated_at: str,
        source: str,
    ):
        existing = targets.get(number)
        if existing is None or GitHubPoller._is_newer_timestamp(updated_at, existing[1]):
            targets[number] = (fingerprint, updated_at, source)

    @staticmethod
    def _extract_trailing_number(url: str, resource_name: str) -> int:
        match = re.search(rf"/{re.escape(resource_name)}/(\d+)$", str(url or ""))
        return int(match.group(1)) if match else 0

    @staticmethod
    def _is_newer_timestamp(left: str, right: str) -> bool:
        try:
            return datetime.fromisoformat(left.replace("Z", "+00:00")) > datetime.fromisoformat(
                right.replace("Z", "+00:00")
            )
        except ValueError:
            return left > right

    @staticmethod
    def _max_timestamp(left: str, right: str) -> str:
        if not left:
            return right
        if not right:
            return left
        return left if GitHubPoller._is_newer_timestamp(left, right) else right

    @staticmethod
    def _min_timestamp(left: str, right: str) -> str:
        if not left:
            return right
        if not right:
            return left
        return right if GitHubPoller._is_newer_timestamp(left, right) else left

    @staticmethod
    def summarize_ci_details(ci_status: dict) -> str:
        """提炼 CI 明细，便于日志排查。"""
        prioritized: list[tuple[int, str]] = []
        for item in ci_status.get("details", []):
            name = item.get("name") or item.get("type") or "unknown"
            status = item.get("status") or ""
            conclusion = item.get("conclusion") or ""
            if status and status != "completed":
                prioritized.append((0, f"{name}:{status}"))
            elif conclusion:
                priority = 1 if conclusion in {"failure", "error", "cancelled", "timed_out"} else 2
                prioritized.append((priority, f"{name}:{conclusion}"))
        if not prioritized:
            return "no-check-details"
        prioritized.sort(key=lambda item: (item[0], item[1]))
        return ", ".join(text for _, text in prioritized[:6])
