import asyncio
from datetime import datetime, timedelta, timezone
import unittest

from agent.task_scheduler import TaskLease, TaskScheduler, utc_now_iso


class TaskSchedulerTests(unittest.TestCase):
    def test_run_pending_updates_success_summary(self):
        calls: list[str] = []

        async def handler():
            calls.append("ok")

        async def run_test():
            scheduler = TaskScheduler()
            scheduler.register_task(name="polling-cycle", handler=handler, interval_seconds=60)
            await scheduler.run_pending()
            summary = scheduler.summary()
            record = scheduler.get_task_record("polling-cycle")
            self.assertEqual(["ok"], calls)
            self.assertEqual(1, summary["registered"])
            self.assertEqual("succeeded", summary["tasks"][0]["status"])
            self.assertEqual(1, summary["tasks"][0]["run_count"])
            self.assertEqual(0, summary["dead_letter"])
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual("succeeded", record.status)
            self.assertTrue(record.last_success_at)

        asyncio.run(run_test())

    def test_failure_enters_retry_waiting_with_backoff(self):
        async def handler():
            raise RuntimeError("boom")

        async def run_test():
            scheduler = TaskScheduler()
            scheduler.register_task(
                name="pr-queue-cycle",
                handler=handler,
                interval_seconds=60,
                max_retries=2,
                backoff_base_seconds=5,
            )
            await scheduler.run_pending()
            summary = scheduler.summary()
            record = scheduler.get_task_record("pr-queue-cycle")
            self.assertEqual(1, summary["retry_waiting"])
            self.assertEqual("retry_waiting", summary["tasks"][0]["status"])
            self.assertEqual("boom", summary["tasks"][0]["last_error"])
            self.assertEqual(1, summary["tasks"][0]["failure_count"])
            self.assertEqual(1, summary["tasks"][0]["consecutive_failures"])
            self.assertTrue(summary["tasks"][0]["next_run_at"])
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual("retry_waiting", record.status)

        asyncio.run(run_test())

    def test_task_becomes_dead_letter_after_retry_budget_exhausted(self):
        async def handler():
            raise RuntimeError("boom")

        async def run_test():
            scheduler = TaskScheduler()
            scheduler.register_task(
                name="pr-queue-cycle",
                handler=handler,
                interval_seconds=60,
                max_retries=1,
                backoff_base_seconds=0.1,
            )
            await scheduler.run_pending()
            task = scheduler._tasks["pr-queue-cycle"]
            task.next_run_at = datetime.now(timezone.utc) - timedelta(seconds=1)
            await scheduler.run_pending()
            summary = scheduler.summary()
            self.assertEqual(1, summary["dead_letter"])
            self.assertEqual("dead_letter", summary["tasks"][0]["status"])
            self.assertEqual("", summary["tasks"][0]["next_run_at"])
            self.assertEqual(2, summary["tasks"][0]["failure_count"])

        asyncio.run(run_test())

    def test_due_tasks_respect_priority_order(self):
        calls: list[str] = []

        async def urgent_handler():
            calls.append("urgent")

        async def normal_handler():
            calls.append("normal")

        async def run_test():
            scheduler = TaskScheduler()
            scheduler.register_task(name="normal-task", handler=normal_handler, interval_seconds=60, priority="normal")
            scheduler.register_task(name="urgent-task", handler=urgent_handler, interval_seconds=60, priority="urgent")
            await scheduler.run_pending()
            self.assertEqual(["urgent", "normal"], calls)

        asyncio.run(run_test())

    def test_retry_task_revives_dead_letter_and_runs_again(self):
        attempts = {"count": 0}

        async def handler():
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("boom")

        async def run_test():
            scheduler = TaskScheduler()
            scheduler.register_task(
                name="polling-cycle",
                handler=handler,
                interval_seconds=60,
                max_retries=0,
            )
            await scheduler.run_pending()
            self.assertEqual("dead_letter", scheduler.get_task_record("polling-cycle").status)

            revived = scheduler.retry_task("polling-cycle")
            self.assertEqual("queued", revived.status)
            self.assertEqual(0, revived.consecutive_failures)
            await scheduler.run_pending()

            record = scheduler.get_task_record("polling-cycle")
            assert record is not None
            self.assertEqual("succeeded", record.status)
            self.assertEqual(2, record.run_count)

        asyncio.run(run_test())

    def test_retry_task_rejects_running_task(self):
        scheduler = TaskScheduler()

        async def handler():
            return None

        scheduler.register_task(name="polling-cycle", handler=handler, interval_seconds=60)
        scheduler._tasks["polling-cycle"].lease = TaskLease(owner="test", acquired_at=utc_now_iso())

        with self.assertRaisesRegex(RuntimeError, "无法恢复"):
            scheduler.retry_task("polling-cycle")

    def test_long_running_task_does_not_block_other_due_tasks(self):
        calls: list[str] = []
        blocker = asyncio.Event()
        slow_started = asyncio.Event()

        async def slow_handler():
            calls.append("slow-start")
            slow_started.set()
            await blocker.wait()
            calls.append("slow-finish")

        async def fast_handler():
            calls.append("fast")

        async def run_test():
            scheduler = TaskScheduler()
            scheduler.register_task(name="issue-task-cycle", handler=slow_handler, interval_seconds=60, priority="urgent")
            scheduler.register_task(name="source-task-cycle", handler=fast_handler, interval_seconds=60)
            await scheduler.run_pending()
            await slow_started.wait()
            await asyncio.sleep(0)

            issue_record = scheduler.get_task_record("issue-task-cycle")
            source_record = scheduler.get_task_record("source-task-cycle")
            assert issue_record is not None
            assert source_record is not None
            self.assertEqual("running", issue_record.status)
            self.assertEqual("succeeded", source_record.status)
            self.assertEqual(["slow-start", "fast"], calls[:2])

            blocker.set()
            await asyncio.sleep(0)
            await asyncio.sleep(0)

            issue_record = scheduler.get_task_record("issue-task-cycle")
            assert issue_record is not None
            self.assertEqual("succeeded", issue_record.status)
            self.assertEqual(["slow-start", "fast", "slow-finish"], calls)
            await scheduler.stop()

        asyncio.run(run_test())

    def test_stop_waits_for_active_run_to_finish(self):
        calls: list[str] = []
        blocker = asyncio.Event()
        slow_started = asyncio.Event()

        async def handler():
            calls.append("slow-start")
            slow_started.set()
            await blocker.wait()
            calls.append("slow-finish")

        async def run_test():
            scheduler = TaskScheduler()
            scheduler.register_task(name="polling-cycle", handler=handler, interval_seconds=60)
            await scheduler.run_pending()
            await slow_started.wait()

            stop_task = asyncio.create_task(scheduler.stop())
            await asyncio.sleep(0)
            self.assertFalse(stop_task.done())
            record = scheduler.get_task_record("polling-cycle")
            assert record is not None
            self.assertEqual("running", record.status)

            blocker.set()
            await stop_task

            record = scheduler.get_task_record("polling-cycle")
            assert record is not None
            self.assertEqual("succeeded", record.status)
            self.assertEqual(["slow-start", "slow-finish"], calls)

        asyncio.run(run_test())

    def test_stop_cancels_active_run_after_timeout(self):
        calls: list[str] = []
        slow_started = asyncio.Event()
        blocker = asyncio.Event()

        async def handler():
            calls.append("slow-start")
            slow_started.set()
            await blocker.wait()
            calls.append("slow-finish")

        async def run_test():
            scheduler = TaskScheduler(stop_wait_seconds=0.01)
            scheduler.register_task(name="polling-cycle", handler=handler, interval_seconds=60)
            await scheduler.run_pending()
            await slow_started.wait()

            await asyncio.wait_for(scheduler.stop(), timeout=0.5)

            record = scheduler.get_task_record("polling-cycle")
            assert record is not None
            self.assertEqual("cancelled", record.status)
            self.assertEqual(["slow-start"], calls)

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
