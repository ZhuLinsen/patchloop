#!/usr/bin/env python3
"""一次性脚本：修复已有 autocode PR 的标题质量。

使用存储在 .autocode-state.json 中的 plan goal 重新生成标题。
默认 dry-run，加 --apply 实际更新。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
from agent.pr_title import build_pr_title, should_keep_existing_pr_title

STATE_FILE = Path(__file__).resolve().parent.parent / ".autocode-state.json"
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

_PR_TITLE_ISSUE_RE = re.compile(r"\(#(\d+)\)\s*$")


def _load_env() -> dict[str, str]:
    env = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def main():
    apply = "--apply" in sys.argv
    env = _load_env()
    token = env.get("GITHUB_TOKEN", "")
    repo_full = env.get("GITHUB_REPO", "")
    if not token or not repo_full:
        print("ERROR: GITHUB_TOKEN or GITHUB_REPO not found in .env")
        return 1

    owner, repo = repo_full.split("/", 1)

    # Load state for plan goals
    state = json.loads(STATE_FILE.read_text())
    plans = state.get("issue_plans", {})

    # Build issue_number -> (task_type, goal) map
    issue_goals: dict[int, tuple[str, str]] = {}
    for key, val in plans.items():
        try:
            issue_num = int(str(key).split(":")[0]) if ":" in str(key) else int(key)
        except (ValueError, TypeError):
            continue
        task_type = str(val.get("triage", {}).get("task_type", ""))
        goal = str(val.get("plan", {}).get("goal", ""))
        if goal:
            issue_goals[issue_num] = (task_type, goal)

    # Fetch open autocode PRs
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    client = httpx.Client(timeout=30, headers=headers)

    prs = []
    page = 1
    while True:
        resp = client.get(
            f"https://api.github.com/repos/{owner}/{repo}/pulls",
            params={"state": "open", "per_page": 100, "page": page},
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        prs.extend(batch)
        page += 1

    autocode_prs = [p for p in prs if str(p.get("head", {}).get("ref", "")).startswith("autocode/")]
    print(f"Found {len(autocode_prs)} open autocode PRs\n")

    updated = 0
    skipped = 0
    for pr in sorted(autocode_prs, key=lambda x: x["number"]):
        pr_num = pr["number"]
        old_title = pr["title"]

        # Extract linked issue number from title
        m = _PR_TITLE_ISSUE_RE.search(old_title)
        if not m:
            print(f"  PR #{pr_num}: SKIP (no issue ref in title)")
            skipped += 1
            continue

        issue_num = int(m.group(1))
        if issue_num not in issue_goals:
            print(f"  PR #{pr_num}: SKIP (no plan data for issue #{issue_num})")
            skipped += 1
            continue

        task_type, goal = issue_goals[issue_num]

        if should_keep_existing_pr_title(old_title, task_type, issue_num):
            print(f"  PR #{pr_num}: OK (title unchanged)")
            skipped += 1
            continue

        # Fetch original issue title for context
        issue_resp = client.get(f"https://api.github.com/repos/{owner}/{repo}/issues/{issue_num}")
        issue_title = issue_resp.json().get("title", "") if issue_resp.status_code == 200 else ""

        new_title = build_pr_title(task_type, issue_title, issue_num, plan_goal=goal)

        if new_title == old_title:
            print(f"  PR #{pr_num}: OK (title unchanged)")
            skipped += 1
            continue

        print(f"  PR #{pr_num}:")
        print(f"    OLD: {old_title}")
        print(f"    NEW: {new_title}")

        if apply:
            resp = client.patch(
                f"https://api.github.com/repos/{owner}/{repo}/pulls/{pr_num}",
                json={"title": new_title},
            )
            if resp.status_code == 200:
                print(f"    -> UPDATED")
                updated += 1
            else:
                print(f"    -> FAILED: {resp.status_code} {resp.text[:200]}")
        else:
            print(f"    -> DRY RUN (use --apply to update)")
            updated += 1

    print(f"\nDone: {updated} to update, {skipped} skipped")
    if not apply and updated > 0:
        print("Run with --apply to actually update PR titles on GitHub.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
