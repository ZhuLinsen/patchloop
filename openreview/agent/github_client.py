"""
GitHub API 客户端 — 仅限只读 + 评论操作

安全设计:
- 只实现 GET (读取) 和 POST comment (评论) 接口
- **绝不实现** push / commit / merge / close / delete 等写操作
- 所有评论操作在 dry_run 模式下只打印不实际发送
"""
import logging
import re
import time
from collections.abc import Callable
import httpx

logger = logging.getLogger(__name__)

_GITHUB_API = "https://api.github.com"
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0
_MAX_RETRY_DELAY = 60.0  # 重试延迟上限，避免 X-RateLimit-Reset 导致超长等待
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_RATE_LIMITED_STATUS_CODES = {429, 403}  # 仅这些状态码才参考 X-RateLimit-Reset


class GitHubClient:
    """GitHub REST API 封装 — 安全只读 + 评论"""

    def __init__(self, token: str, owner: str, repo: str, dry_run: bool = False):
        self.token = token
        self.owner = owner
        self.repo = repo
        self.dry_run = dry_run
        self._viewer_login: str | None = None
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        self._client = httpx.Client(timeout=30, headers=self._headers)

    def close(self):
        """关闭底层 HTTP 客户端"""
        self._client.close()

    # ------------------------------------------------------------------
    # 读取接口
    # ------------------------------------------------------------------

    def get_issue(self, issue_number: int) -> dict:
        """获取 Issue 详情"""
        url = f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/issues/{issue_number}"
        return self._get(url)

    def get_pr(self, pr_number: int) -> dict:
        """获取 PR 详情"""
        url = f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/pulls/{pr_number}"
        return self._get(url)

    def get_pr_diff(self, pr_number: int) -> str:
        """获取 PR 的 diff 文本"""
        url = f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/pulls/{pr_number}"
        resp = self._client.get(url, headers={"Accept": "application/vnd.github.diff"})
        resp.raise_for_status()
        return resp.text

    def get_pr_files(self, pr_number: int) -> list[str]:
        """获取 PR 变更的文件列表"""
        url = f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/pulls/{pr_number}/files"
        files: list[str] = []
        page = 1
        while True:
            data = self._get(url, params={"per_page": 100, "page": page})
            if not isinstance(data, list) or not data:
                break
            files.extend(str(f.get("filename", "")) for f in data if f.get("filename"))
            if len(data) < 100:
                break
            page += 1
        return files

    def list_issues(
        self,
        *,
        state: str = "all",
        sort: str = "updated",
        direction: str = "desc",
        since: str | None = None,
        per_page: int = 30,
        page: int = 1,
    ) -> list[dict]:
        """列出 Issue（GitHub 会一并返回 PR issue 条目，调用方自行过滤）。"""
        url = f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/issues"
        params = {
            "state": state,
            "sort": sort,
            "direction": direction,
            "per_page": per_page,
            "page": page,
        }
        if since:
            params["since"] = since
        data = self._get(url, params=params)
        return data if isinstance(data, list) else []

    def list_pull_requests(
        self,
        *,
        state: str = "open",
        sort: str = "updated",
        direction: str = "desc",
        per_page: int = 30,
        page: int = 1,
    ) -> list[dict]:
        """列出 Pull Request。"""
        url = f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/pulls"
        params = {
            "state": state,
            "sort": sort,
            "direction": direction,
            "per_page": per_page,
            "page": page,
        }
        data = self._get(url, params=params)
        return data if isinstance(data, list) else []

    def list_issue_comments(
        self,
        issue_number: int,
        *,
        per_page: int = 30,
        page: int = 1,
    ) -> list[dict]:
        """列出 Issue/PR 会话评论。"""
        url = f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/issues/{issue_number}/comments"
        params = {
            "per_page": per_page,
            "page": page,
        }
        data = self._get(url, params=params)
        return data if isinstance(data, list) else []

    def list_repository_issue_comments(
        self,
        *,
        sort: str = "updated",
        direction: str = "desc",
        since: str | None = None,
        per_page: int = 30,
        page: int = 1,
    ) -> list[dict]:
        """列出仓库范围内的 Issue/PR 会话评论。"""
        url = f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/issues/comments"
        params = {
            "sort": sort,
            "direction": direction,
            "per_page": per_page,
            "page": page,
        }
        if since:
            params["since"] = since
        data = self._get(url, params=params)
        return data if isinstance(data, list) else []

    def list_pr_reviews(
        self,
        pr_number: int,
        *,
        per_page: int = 30,
        page: int = 1,
    ) -> list[dict]:
        """列出 PR reviews。"""
        url = f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/pulls/{pr_number}/reviews"
        params = {
            "per_page": per_page,
            "page": page,
        }
        data = self._get(url, params=params)
        return data if isinstance(data, list) else []

    def list_pr_review_comments(
        self,
        pr_number: int,
        *,
        per_page: int = 30,
        page: int = 1,
    ) -> list[dict]:
        """列出 PR diff review comments。"""
        url = f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/pulls/{pr_number}/comments"
        params = {
            "per_page": per_page,
            "page": page,
        }
        data = self._get(url, params=params)
        return data if isinstance(data, list) else []

    def list_repository_pr_review_comments(
        self,
        *,
        sort: str = "updated",
        direction: str = "desc",
        since: str | None = None,
        per_page: int = 30,
        page: int = 1,
    ) -> list[dict]:
        """列出仓库范围内的 PR inline review comments。"""
        url = f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/pulls/comments"
        params = {
            "sort": sort,
            "direction": direction,
            "per_page": per_page,
            "page": page,
        }
        if since:
            params["since"] = since
        data = self._get(url, params=params)
        return data if isinstance(data, list) else []

    def list_recent_issue_comments(
        self,
        issue_number: int,
        *,
        limit: int = 60,
        per_page: int = 100,
    ) -> list[dict]:
        """读取最新评论尾部，避免只拿到第一页的旧评论。"""
        return self._collect_recent_paginated(
            lambda page, page_size: self._get_list_page(
                f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/issues/{issue_number}/comments",
                params={"per_page": page_size, "page": page},
            ),
            limit=limit,
            per_page=per_page,
        )

    def list_recent_pr_reviews(
        self,
        pr_number: int,
        *,
        limit: int = 60,
        per_page: int = 100,
    ) -> list[dict]:
        """读取最近的 review submissions。"""
        return self._collect_recent_paginated(
            lambda page, page_size: self._get_list_page(
                f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/pulls/{pr_number}/reviews",
                params={"per_page": page_size, "page": page},
            ),
            limit=limit,
            per_page=per_page,
        )

    def list_recent_pr_review_comments(
        self,
        pr_number: int,
        *,
        limit: int = 60,
        per_page: int = 100,
    ) -> list[dict]:
        """读取最近的 inline review comments。"""
        return self._collect_recent_paginated(
            lambda page, page_size: self._get_list_page(
                f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/pulls/{pr_number}/comments",
                params={"per_page": page_size, "page": page},
            ),
            limit=limit,
            per_page=per_page,
        )

    def search_issues(
        self,
        query: str,
        *,
        per_page: int = 10,
        page: int = 1,
    ) -> list[dict]:
        """搜索仓库内相关 Issue。"""
        url = f"{_GITHUB_API}/search/issues"
        params = {
            "q": query,
            "per_page": per_page,
            "page": page,
        }
        data = self._get(url, params=params)
        items = data.get("items", []) if isinstance(data, dict) else []
        return items if isinstance(items, list) else []

    def get_authenticated_login(self) -> str:
        """获取当前 token 对应的 GitHub 登录名。"""
        if self._viewer_login is not None:
            return self._viewer_login

        data = self._get(f"{_GITHUB_API}/user")
        login = str(data.get("login", "")).strip()
        if not login:
            raise RuntimeError("GitHub /user 响应中缺少 login")
        self._viewer_login = login
        return login

    def _collect_recent_paginated(
        self,
        fetch_page: Callable[[int, int], tuple[list[dict], str]],
        *,
        limit: int,
        per_page: int,
    ) -> list[dict]:
        max_items = max(limit, 1)
        page_size = max(per_page, 1)
        items, link_header = fetch_page(1, page_size)
        if not items:
            return []

        last_page = self._parse_last_page(link_header)
        if last_page and last_page > 1:
            pages_needed = max(1, (max_items + page_size - 1) // page_size)
            start_page = max(1, last_page - pages_needed + 1)
            recent = list(items) if start_page == 1 else []
            for page in range(max(2, start_page), last_page + 1):
                page_items, _ = fetch_page(page, page_size)
                if not page_items:
                    break
                recent.extend(page_items)
            return recent[-max_items:]

        recent = list(items)
        page = 2
        while len(items) == page_size:
            items, _ = fetch_page(page, page_size)
            if not items:
                break
            recent.extend(items)
            if len(recent) > max_items:
                recent = recent[-max_items:]
            page += 1

        return recent[-max_items:]

    @staticmethod
    def _parse_last_page(link_header: str) -> int | None:
        for part in link_header.split(","):
            if 'rel="last"' not in part:
                continue
            match = re.search(r"[?&]page=(\d+)", part)
            if match:
                return int(match.group(1))
        return None

    def get_commit_ci_status(self, ref: str) -> dict:
        """
        获取提交的 CI 汇总状态。

        返回:
            {
              "state": "success|pending|failure|missing",
              "ready": bool,
              "finalized": bool,
              "has_checks": bool,
              "details": [...]
            }
        """
        check_runs_url = f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/commits/{ref}/check-runs"
        commit_status_url = f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/commits/{ref}/status"

        check_runs_data = self._get(check_runs_url)
        commit_status_data = self._get(commit_status_url)

        details: list[dict] = []
        explicit_success = False
        explicit_pending = False
        explicit_failure = False

        for check_run in check_runs_data.get("check_runs", []):
            status = (check_run.get("status") or "").lower()
            conclusion = (check_run.get("conclusion") or "").lower()
            details.append(
                {
                    "type": "check_run",
                    "name": check_run.get("name", ""),
                    "status": status,
                    "conclusion": conclusion,
                }
            )
            if status and status != "completed":
                explicit_pending = True
            elif conclusion in {"success", "neutral", "skipped"}:
                explicit_success = True
            elif conclusion in {
                "action_required",
                "cancelled",
                "failure",
                "stale",
                "startup_failure",
                "timed_out",
            }:
                explicit_failure = True

        for status_item in commit_status_data.get("statuses", []):
            state = (status_item.get("state") or "").lower()
            details.append(
                {
                    "type": "commit_status",
                    "name": status_item.get("context", ""),
                    "status": state,
                    "conclusion": state,
                }
            )
            if state == "pending":
                explicit_pending = True
            elif state == "success":
                explicit_success = True
            elif state in {"error", "failure"}:
                explicit_failure = True

        combined_state = (commit_status_data.get("state") or "").lower()
        has_checks = bool(details)

        # 优先信任显式的 check-runs / commit statuses 明细。
        # 只有没有任何明细时，才回退使用 combined state。
        if explicit_pending:
            state = "pending"
        elif explicit_failure:
            state = "failure"
        elif explicit_success:
            state = "success"
        else:
            if combined_state == "pending":
                state = "pending"
            elif combined_state in {"error", "failure"}:
                state = "failure"
            elif combined_state == "success":
                state = "success"
            else:
                state = "missing"

        return {
            "state": state,
            "ready": state == "success" and has_checks,
            "finalized": state in {"success", "failure"} and has_checks,
            "has_checks": has_checks,
            "details": details,
        }

    # ------------------------------------------------------------------
    # 评论接口 (唯一的写操作)
    # ------------------------------------------------------------------

    def post_issue_comment(self, issue_number: int, body: str) -> dict | None:
        """
        在 Issue/PR 下发表评论。
        
        注意: GitHub 的 Issue 和 PR 共用评论 API。
        """
        if self.dry_run:
            logger.info("[DRY RUN] 将在 #%d 发表评论:\n%s", issue_number, body[:200])
            return None

        url = f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/issues/{issue_number}/comments"
        payload = {"body": body}
        return self._post(url, payload)

    def post_pr_review_comment(self, pr_number: int, body: str) -> dict | None:
        """
        在 PR 上创建一个 review (general comment)。
        使用 COMMENT 事件类型 — 不会 approve 也不会 request changes。
        """
        if self.dry_run:
            logger.info("[DRY RUN] 将在 PR #%d 发表 review:\n%s", pr_number, body[:200])
            return None

        url = f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/pulls/{pr_number}/reviews"
        payload = {
            "body": body,
            "event": "COMMENT",  # 安全: 只评论，不 approve/request changes
        }
        return self._post(url, payload)

    # ------------------------------------------------------------------
    # HTTP 工具方法
    # ------------------------------------------------------------------

    def _get(self, url: str, params: dict | None = None) -> dict | list:
        return self._request_with_retry("GET", url, params=params)

    def _get_list_page(self, url: str, params: dict | None = None) -> tuple[list[dict], str]:
        resp = self._raw_request_with_retry("GET", url, params=params)
        data = resp.json()
        items = data if isinstance(data, list) else []
        return items, resp.headers.get("Link", "")

    def _post(self, url: str, payload: dict) -> dict:
        return self._request_with_retry("POST", url, json=payload)

    def _request_with_retry(self, method: str, url: str, **kwargs) -> dict | list:
        resp = self._raw_request_with_retry(method, url, **kwargs)
        return resp.json()

    def _raw_request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._client.request(method, url, **kwargs)
                if resp.status_code in _RETRYABLE_STATUS_CODES:
                    retry_after = _parse_retry_after(resp, resp.status_code)
                    delay = min(max(retry_after, _RETRY_BASE_DELAY * (2 ** attempt)), _MAX_RETRY_DELAY)
                    if attempt < _MAX_RETRIES - 1:
                        logger.warning(
                            "GitHub API %s %s 返回 %d，%0.1fs 后重试 (%d/%d)",
                            method, url.split("?")[0], resp.status_code, delay, attempt + 1, _MAX_RETRIES,
                        )
                        time.sleep(delay)
                        continue
                resp.raise_for_status()
                return resp
            except httpx.HTTPStatusError:
                raise
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    delay = _RETRY_BASE_DELAY * (2 ** attempt)
                    logger.warning(
                        "GitHub API %s %s 网络异常，%0.1fs 后重试 (%d/%d): %s",
                        method, url.split("?")[0], delay, attempt + 1, _MAX_RETRIES, exc,
                    )
                    time.sleep(delay)
                    continue
                raise
        raise last_exc  # type: ignore[misc]


def _parse_retry_after(resp: httpx.Response, status_code: int = 0) -> float:
    """从 Retry-After 或 X-RateLimit-Reset 头解析等待秒数。

    仅在 429/403 (rate limit) 时才参考 X-RateLimit-Reset；
    502/503/504 等服务端瞬时故障只用 Retry-After 头或回退到指数退避。
    """
    retry_after = resp.headers.get("Retry-After", "")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    if status_code in _RATE_LIMITED_STATUS_CODES:
        reset_at = resp.headers.get("X-RateLimit-Reset", "")
        if reset_at:
            try:
                return max(float(reset_at) - time.time(), 0)
            except ValueError:
                pass
    return 0
