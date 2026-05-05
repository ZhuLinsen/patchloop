import os
import unittest
from unittest import mock

import config


class ConfigTests(unittest.TestCase):
    def _base_env(self) -> dict[str, str]:
        return {
            "GITHUB_TOKEN": "test-token",
            "GITHUB_REPO": "example/repo",
            "ENABLE_WEBHOOK": "false",
            "EVENT_SOURCE": "polling",
        }

    def test_polling_catchup_limits_read_from_env(self):
        env = self._base_env()
        env["POLL_PENDING_PR_BATCH_SIZE"] = "7"
        env["POLL_PENDING_PR_MAX_AGE_DAYS"] = "14"
        env["POLL_REVIEW_SUBMISSION_BATCH_SIZE"] = "9"
        env["POLL_REVIEW_SUBMISSION_MAX_AGE_DAYS"] = "10"

        with mock.patch.dict(os.environ, env, clear=True):
            loaded = config.load_config()

        self.assertEqual(7, loaded.server.poll_pending_pr_batch_size)
        self.assertEqual(14, loaded.server.poll_pending_pr_max_age_days)
        self.assertEqual(9, loaded.server.poll_review_submission_batch_size)
        self.assertEqual(10, loaded.server.poll_review_submission_max_age_days)


if __name__ == "__main__":
    unittest.main()
