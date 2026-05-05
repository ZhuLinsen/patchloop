import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent.autocode_state_store import AutoCodeStateStore
from agent.task_scheduler import TaskScheduler
import autocode as main


class SchedulerFlowTests(unittest.TestCase):
    def test_dead_letter_task_can_be_recovered_via_main_endpoint(self):
        attempts = {"count": 0}

        async def fake_poll_once():
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("boom")
            return {
                "plans": {"fetched": 0, "queued": 0},
                "backlog": {"fetched": 0, "queued": 0},
                "issues": {"fetched": 1, "eligible": 1, "processed": 1},
            }

        async def run_test():
            with tempfile.TemporaryDirectory() as tmpdir:
                state_store = AutoCodeStateStore(str(Path(tmpdir) / "autocode-state.json"))
                scheduler = TaskScheduler()
                scheduler.register_task(
                    name="polling-cycle",
                    handler=main._run_polling_cycle,
                    interval_seconds=60,
                    max_retries=0,
                )

                with mock.patch.object(main, "state_store", state_store), mock.patch.object(
                    main, "_poll_once", new=fake_poll_once
                ), mock.patch.object(main, "scheduler", scheduler):
                    await scheduler.run_pending()
                    self.assertEqual("dead_letter", scheduler.get_task_record("polling-cycle").status)

                    retry_response = await main.retry_scheduler_task("polling-cycle")
                    self.assertEqual("ok", retry_response["status"])
                    self.assertEqual("queued", retry_response["task"]["status"])

                    await scheduler.run_pending()

                record = scheduler.get_task_record("polling-cycle")
                assert record is not None
                self.assertEqual("succeeded", record.status)
                observability = state_store.observability_summary()
                cycles = {item["name"]: item for item in observability["cycles"]}
                self.assertEqual("succeeded", cycles["polling-cycle"]["last_status"])
                self.assertEqual(1, observability["counters"]["scheduler.manual_retry"])

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
