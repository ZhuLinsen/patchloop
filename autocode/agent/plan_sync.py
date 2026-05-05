"""计划源同步。"""
from __future__ import annotations

from typing import Protocol

from agent.autocode_state_store import AutoCodeStateStore
from agent.plan_source import PlanItem, PlanSource


class IssueWriter(Protocol):
    def create_issue(self, *, title: str, body: str, labels: list[str] | None = None) -> dict | None:
        ...


async def sync_plan_source_once(
    *,
    plan_source: PlanSource | None,
    state_store: AutoCodeStateStore,
    gh_writer: IssueWriter,
    to_thread,
) -> dict[str, int]:
    if plan_source is None:
        return {"fetched": 0, "created": 0}

    items = plan_source.collect_items()
    stats = {"fetched": len(items), "created": 0}
    for item in items:
        if state_store.has_backlog_item(item.key):
            continue
        created_issue = await to_thread(
            gh_writer.create_issue,
            title=item.title,
            body=item.body,
            labels=item.labels,
        )
        if not created_issue:
            continue
        issue_number = int(created_issue.get("number", 0) or 0)
        issue_url = str(created_issue.get("html_url", "") or "")
        state_store.mark_backlog_item_synced(
            item_key=item.key,
            source_name="plan_sync",
            source_path=item.source_path,
            line_number=item.line_number,
            title=item.title,
            issue_number=issue_number,
            issue_url=issue_url,
        )
        stats["created"] += 1
    return stats
