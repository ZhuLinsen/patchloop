import asyncio
import tempfile
import unittest
from pathlib import Path

from agent.autocode_state_store import AutoCodeStateStore
from agent.plan_source import PlanItem
from agent.plan_sync import sync_plan_source_once


class _FakePlanSource:
    def __init__(self, items: list[PlanItem]):
        self.items = items

    def collect_items(self) -> list[PlanItem]:
        return list(self.items)


class _FakeWriter:
    def __init__(self):
        self.created: list[dict] = []

    def create_issue(self, *, title: str, body: str, labels: list[str] | None = None) -> dict:
        payload = {"title": title, "body": body, "labels": list(labels or [])}
        self.created.append(payload)
        return {
            "number": 500 + len(self.created),
            "html_url": f"https://example.com/issues/{500 + len(self.created)}",
        }


async def _run_inline(func, *args, **kwargs):
    return func(*args, **kwargs)


class PlanSyncTests(unittest.TestCase):
    def test_sync_plan_source_creates_issue_and_persists_mapping(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / 'state.json'))
            writer = _FakeWriter()
            item = PlanItem(
                key='plan:1',
                title='Add tests',
                body='body',
                source_path='PLAN.md',
                line_number=4,
                labels=['autocode'],
            )

            stats = asyncio.run(
                sync_plan_source_once(
                    plan_source=_FakePlanSource([item]),
                    state_store=state_store,
                    gh_writer=writer,
                    to_thread=_run_inline,
                )
            )

            self.assertEqual({'fetched': 1, 'created': 1}, stats)
            self.assertEqual(1, len(writer.created))
            self.assertTrue(state_store.has_backlog_item('plan:1'))

    def test_sync_plan_source_skips_existing_item(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state_store = AutoCodeStateStore(str(Path(tmpdir) / 'state.json'))
            state_store.mark_backlog_item_synced(
                item_key='plan:1',
                source_name='plan_sync',
                source_path='PLAN.md',
                line_number=3,
                title='Add tests',
                issue_number=501,
                issue_url='https://example.com/issues/501',
            )
            writer = _FakeWriter()
            item = PlanItem(
                key='plan:1',
                title='Add tests',
                body='body',
                source_path='PLAN.md',
                line_number=4,
                labels=['autocode'],
            )

            stats = asyncio.run(
                sync_plan_source_once(
                    plan_source=_FakePlanSource([item]),
                    state_store=state_store,
                    gh_writer=writer,
                    to_thread=_run_inline,
                )
            )

            self.assertEqual({'fetched': 1, 'created': 0}, stats)
            self.assertEqual([], writer.created)


if __name__ == '__main__':
    unittest.main()
