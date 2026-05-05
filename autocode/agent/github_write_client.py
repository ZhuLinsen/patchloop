"""GitHub 写客户端 — 仅用于创建/更新 PR，严禁 merge。"""
from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

_GITHUB_API = "https://api.github.com"


class GitHubWriteClient:
    """最小写权限客户端。"""

    def __init__(
        self,
        *,
        token: str,
        owner: str,
        repo: str,
        dry_run: bool = False,
        force_draft_pr: bool = True,
        forbid_main_base_branch: bool = False,
        default_branch: str = "main",
    ):
        self.token = token
        self.owner = owner
        self.repo = repo
        self.dry_run = dry_run
        self.force_draft_pr = force_draft_pr
        self.forbid_main_base_branch = forbid_main_base_branch
        self.default_branch = str(default_branch or "main").strip() or "main"
        self._viewer_login: str | None = None
        self._client = httpx.Client(
            timeout=30,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    def close(self):
        self._client.close()

    def get_authenticated_login(self) -> str:
        if self._viewer_login is not None:
            return self._viewer_login
        response = self._client.get(f"{_GITHUB_API}/user")
        response.raise_for_status()
        data = response.json()
        login = str(data.get("login", "")).strip()
        if not login:
            raise RuntimeError("GitHub /user 响应中缺少 login")
        self._viewer_login = login
        return login

    def create_issue(self, *, title: str, body: str, labels: list[str] | None = None) -> dict | None:
        payload = {
            "title": title,
            "body": body,
        }
        normalized_labels = [item.strip() for item in (labels or []) if item.strip()]
        if normalized_labels:
            payload["labels"] = normalized_labels
        if self.dry_run:
            logger.info("[DRY RUN] 将创建 Issue: title=%s labels=%s", title, ",".join(normalized_labels))
            return None
        response = self._client.post(f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/issues", json=payload)
        response.raise_for_status()
        return response.json()

    def create_pull_request(
        self,
        *,
        title: str,
        body: str,
        head: str,
        base: str,
        issue_number: int | None = None,
    ) -> dict | None:
        self._validate_base_branch(base)
        payload = {
            "title": title,
            "body": body,
            "head": head,
            "base": base,
            "draft": self.force_draft_pr,
        }
        if self.dry_run:
            logger.info(
                "[DRY RUN] 将创建 PR: head=%s, base=%s, draft=%s, title=%s, issue=%s",
                head,
                base,
                self.force_draft_pr,
                title,
                issue_number or "",
            )
            return None
        response = self._client.post(f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/pulls", json=payload)
        response.raise_for_status()
        return response.json()

    def update_pull_request(self, pr_number: int, *, title: str | None = None, body: str | None = None) -> dict | None:
        payload = {}
        if title is not None:
            payload["title"] = title
        if body is not None:
            payload["body"] = body
        if not payload:
            return None
        if self.dry_run:
            logger.info("[DRY RUN] 将更新 PR #%d: fields=%s", pr_number, ",".join(payload.keys()))
            return None
        response = self._client.patch(f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/pulls/{pr_number}", json=payload)
        response.raise_for_status()
        return response.json()

    def mark_ready_for_review(self, pr_number: int) -> dict | None:
        if self.dry_run:
            logger.info("[DRY RUN] 将把 PR #%d 转为 ready for review", pr_number)
            return None
        pr_response = self._client.get(f"{_GITHUB_API}/repos/{self.owner}/{self.repo}/pulls/{pr_number}")
        pr_response.raise_for_status()
        pr_data = pr_response.json()
        pr_node_id = str(pr_data.get("node_id", "") or "").strip()
        if not pr_node_id:
            raise RuntimeError(f"PR #{pr_number} 缺少 node_id，无法切换为 ready for review")
        response = self._client.post(
            f"{_GITHUB_API}/graphql",
            json={
                "query": (
                    "mutation MarkReadyForReview($pullRequestId: ID!) {"
                    "  markPullRequestReadyForReview(input: {pullRequestId: $pullRequestId}) {"
                    "    pullRequest { number url isDraft }"
                    "  }"
                    "}"
                ),
                "variables": {"pullRequestId": pr_node_id},
            },
        )
        response.raise_for_status()
        payload = response.json()
        errors = payload.get("errors", [])
        if errors:
            raise RuntimeError(f"GitHub GraphQL mark_ready_for_review 失败: {errors}")
        pull_request = (
            payload.get("data", {})
            .get("markPullRequestReadyForReview", {})
            .get("pullRequest", {})
        )
        if not isinstance(pull_request, dict) or not pull_request:
            raise RuntimeError(f"PR #{pr_number} ready_for_review 响应缺少 pullRequest")
        updated_pr = dict(pr_data)
        updated_pr["number"] = int(pull_request.get("number", pr_number) or pr_number)
        updated_pr["html_url"] = str(pull_request.get("url", "") or pr_data.get("html_url", "") or "")
        updated_pr["draft"] = bool(pull_request.get("isDraft", pr_data.get("draft", False)))
        return updated_pr

    def _validate_base_branch(self, base: str):
        normalized = base.strip()
        if not normalized:
            raise RuntimeError("PR base branch 不能为空")
        if self.forbid_main_base_branch and normalized == self.default_branch:
            raise RuntimeError(f"策略禁止向默认分支 {self.default_branch} 发起 AutoCode PR")
