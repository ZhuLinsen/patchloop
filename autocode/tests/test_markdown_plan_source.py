import tempfile
import unittest
from pathlib import Path

from agent.markdown_plan_source import MarkdownPlanSource


class MarkdownPlanSourceTests(unittest.TestCase):
    def test_collect_items_reads_open_checkboxes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "PLAN.md").write_text(
                "# Q2\n\n- [ ] add regression tests\n- [x] done item\n\n## Cleanup\n\n* [ ] trim logs\n",
                encoding="utf-8",
            )
            source = MarkdownPlanSource(
                repo_path=str(repo),
                file_paths=["PLAN.md"],
                default_labels=["autocode", "plan"],
                title_prefix="[Plan]",
                max_items_per_cycle=10,
            )

            items = source.collect_items()

            self.assertEqual(2, len(items))
            self.assertTrue(items[0].title.startswith("[Plan]"))
            self.assertIn("PLAN.md:3", items[0].body)
            self.assertEqual(["autocode", "plan"], items[0].labels)

    def test_identical_items_get_distinct_keys_and_later_copy_stays_stable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            file_path = repo / "PLAN.md"
            file_path.write_text(
                "# Sprint\n\n"
                "- [ ] add regression tests\n"
                "- [ ] add regression tests\n",
                encoding="utf-8",
            )
            source = MarkdownPlanSource(
                repo_path=str(repo),
                file_paths=["PLAN.md"],
                default_labels=["autocode", "plan"],
                title_prefix="[Plan]",
                max_items_per_cycle=10,
            )

            initial = source.collect_items()

            self.assertEqual(2, len(initial))
            self.assertNotEqual(initial[0].key, initial[1].key)

            file_path.write_text(
                "# Sprint\n\n"
                "- [ ] add regression tests\n",
                encoding="utf-8",
            )
            shifted = source.collect_items()

            self.assertEqual(1, len(shifted))
            self.assertEqual(initial[1].key, shifted[0].key)

    def test_collect_items_use_plain_issue_style_body_when_prefix_is_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "PLAN.md").write_text(
                "# Q2\n\n- [ ] add retry coverage\n",
                encoding="utf-8",
            )
            source = MarkdownPlanSource(
                repo_path=str(repo),
                file_paths=["PLAN.md"],
                default_labels=["autocode"],
                title_prefix="",
                max_items_per_cycle=10,
            )

            items = source.collect_items()

            self.assertEqual(1, len(items))
            self.assertEqual("add retry coverage", items[0].title)
            self.assertEqual(["autocode"], items[0].labels)
            self.assertNotIn("AutoCode Development Plan Item", items[0].body)
            self.assertNotIn("Item key", items[0].body)
            self.assertIn("来源：`PLAN.md:3`", items[0].body)


if __name__ == "__main__":
    unittest.main()
