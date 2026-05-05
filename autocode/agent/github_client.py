"""
GitHub API 客户端 — 仅限只读 + 评论操作

安全设计:
- 只实现 GET (读取) 和 POST comment (评论) 接口
- **绝不实现** push / commit / merge / close / delete 等写操作
- 所有评论操作在 dry_run 模式下只打印不实际发送
"""
import logging
import httpx

logger = logging.getLogger(__name__)

_GITHUB_API = "https://api.github.com"


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

    def list_issue_comments(
        self,
        issue_number: int,
        *,
        per_page: int = 100,
        page: int = 1,
    ) -> list[dict]:
        """列出 Issue/PR 下的普通评论。"""
        url = f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/issues/{issue_number}/comments"
        data = self._get(url, params={"per_page": per_page, "page": page})
        return data if isinstance(data, list) else []

    def list_repo_issue_comments(
        self,
        *,
        since: str | None = None,
        per_page: int = 100,
        page: int = 1,
    ) -> list[dict]:
        """列出仓库范围内的 Issue comments，供 polling 模式扫描命令评论。"""
        url = f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/issues/comments"
        params = {"per_page": per_page, "page": page}
        if since:
            params["since"] = since
        data = self._get(url, params=params)
        return data if isinstance(data, list) else []

    def list_repo_issue_events(
        self,
        *,
        per_page: int = 100,
        page: int = 1,
    ) -> list[dict]:
        """列出仓库范围内的 Issue events，供 polling 模式识别 opened/reopened。"""
        url = f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/issues/events"
        data = self._get(url, params={"per_page": per_page, "page": page})
        return data if isinstance(data, list) else []

    def list_pr_review_comments(
        self,
        pr_number: int,
        *,
        per_page: int = 100,
        page: int = 1,
    ) -> list[dict]:
        """列出 PR review line comments。"""
        url = f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/pulls/{pr_number}/comments"
        data = self._get(url, params={"per_page": per_page, "page": page})
        return data if isinstance(data, list) else []

    def list_pr_reviews(
        self,
        pr_number: int,
        *,
        per_page: int = 100,
        page: int = 1,
    ) -> list[dict]:
        """列出 PR reviews（含 review body）。"""
        url = f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/pulls/{pr_number}/reviews"
        data = self._get(url, params={"per_page": per_page, "page": page})
        return data if isinstance(data, list) else []

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
        head: str | None = None,
        base: str | None = None,
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
        if head:
            params["head"] = head
        if base:
            params["base"] = base
        data = self._get(url, params=params)
        return data if isinstance(data, list) else []

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
                    "completed_at": check_run.get("completed_at", ""),
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

    def get_branch_ci_check_summary(self, base_ref: str) -> tuple[set[str], set[str]]:
        """
        返回 (pre_existing_failures, recently_passing) 两个 check 名称集合。

        pre_existing_failures: 在多数最近 CI run 中都失败的 check（预存失败）
        recently_passing: 在最近 CI run 中至少有一次成功的 check（说明该 check 本身没问题）

        因为 CI 通常只在 PR 上跑（push to main 不触发），所以无法直接用 base branch
        的 commit SHA 查 check-runs。替代策略：采样最近若干条已完成的 CI workflow run
        （任意 PR/branch），统计每个 job 的失败/成功情况。
        """
        runs_url = f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/actions/runs"
        try:
            runs_data = self._get(runs_url, params={"status": "completed", "per_page": 10})
        except Exception:
            return set(), set()
        runs = runs_data.get("workflow_runs", [])
        if not runs:
            return set(), set()

        # 只看 CI 相关 workflow（跳过 stale/review/dependabot 等）
        ci_run_ids = []
        for r in runs:
            name = (r.get("name") or "").lower()
            event = (r.get("event") or "").lower()
            if event in {"pull_request", "push"} and any(kw in name for kw in ("ci", "test", "build", "gate", "backend")):
                ci_run_ids.append(r["id"])
            if len(ci_run_ids) >= 5:
                break

        if not ci_run_ids:
            ci_run_ids = [r["id"] for r in runs[:5] if r.get("id")]

        fail_counts: dict[str, int] = {}
        pass_counts: dict[str, int] = {}
        total_runs = 0
        for run_id in ci_run_ids:
            jobs_url = f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/actions/runs/{run_id}/jobs"
            try:
                jobs_data = self._get(jobs_url, params={"per_page": 100})
            except Exception:
                continue
            jobs = jobs_data.get("jobs", [])
            if not jobs:
                continue
            total_runs += 1
            for job in jobs:
                conclusion = (job.get("conclusion") or "").lower()
                name = job.get("name", "")
                if not name:
                    continue
                if conclusion == "failure":
                    fail_counts[name] = fail_counts.get(name, 0) + 1
                elif conclusion in {"success", "neutral", "skipped"}:
                    pass_counts[name] = pass_counts.get(name, 0) + 1

        if total_runs == 0:
            return set(), set()

        threshold = max(1, total_runs // 2)
        pre_existing = {name for name, count in fail_counts.items() if count > threshold}
        recently_passing = {name for name, count in pass_counts.items() if count >= 1}
        return pre_existing, recently_passing

    def get_branch_ci_failing_checks(self, base_ref: str) -> set[str]:
        """Convenience wrapper returning only the pre-existing failures set."""
        pre_existing, _ = self.get_branch_ci_check_summary(base_ref)
        return pre_existing

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
        resp = self._client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()

    def _post(self, url: str, payload: dict) -> dict:
        resp = self._client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()
