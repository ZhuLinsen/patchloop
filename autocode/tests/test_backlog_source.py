import tempfile
import unittest
from pathlib import Path

from agent.backlog_source import BacklogSource


class BacklogSourceTests(unittest.TestCase):
    def test_collect_items_reads_open_checkboxes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "TODO.md").write_text(
                "# Sprint\n\n- [ ] add API test\n- [x] done item\n\n## Cleanup\n\n* [ ] trim logs\n",
                encoding="utf-8",
            )
            source = BacklogSource(
                repo_path=str(repo),
                file_paths=["TODO.md"],
                default_labels=["autocode", "backlog"],
                title_prefix="[Backlog]",
                max_items_per_cycle=10,
            )

            items = source.collect_items()

            self.assertEqual(2, len(items))
            self.assertTrue(items[0].title.startswith("[Backlog]"))
            self.assertIn("TODO.md:3", items[0].body)
            self.assertEqual(["autocode", "backlog"], items[0].labels)

    def test_identical_items_get_distinct_keys_and_later_copy_stays_stable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            file_path = repo / "TODO.md"
            file_path.write_text(
                "# Sprint\n\n"
                "- [ ] add API test\n"
                "- [ ] add API test\n",
                encoding="utf-8",
            )
            source = BacklogSource(
                repo_path=str(repo),
                file_paths=["TODO.md"],
                default_labels=["autocode", "backlog"],
                title_prefix="[Backlog]",
                max_items_per_cycle=10,
            )

            initial = source.collect_items()

            self.assertEqual(2, len(initial))
            self.assertNotEqual(initial[0].key, initial[1].key)

            file_path.write_text(
                "# Sprint\n\n"
                "- [ ] add API test\n",
                encoding="utf-8",
            )
            shifted = source.collect_items()

            self.assertEqual(1, len(shifted))
            self.assertEqual(initial[1].key, shifted[0].key)

    def test_collect_items_use_plain_issue_style_body_when_prefix_is_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "TODO.md").write_text(
                "# Audit\n\n- [ ] fix flaky test\n",
                encoding="utf-8",
            )
            source = BacklogSource(
                repo_path=str(repo),
                file_paths=["TODO.md"],
                default_labels=["autocode"],
                title_prefix="",
                max_items_per_cycle=10,
            )

            items = source.collect_items()

            self.assertEqual(1, len(items))
            self.assertEqual("fix flaky test", items[0].title)
            self.assertEqual(["autocode"], items[0].labels)
            self.assertNotIn("AutoCode Backlog Item", items[0].body)
            self.assertNotIn("Item key", items[0].body)
            self.assertIn("来源：`TODO.md:3`", items[0].body)

    def test_annotate_item_issue_appends_visible_issue_link_without_changing_item_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            todo_path = repo / "TODO.md"
            todo_path.write_text(
                "- [ ] add API test\n- [ ] trim logs\n",
                encoding="utf-8",
            )
            source = BacklogSource(
                repo_path=str(repo),
                file_paths=["TODO.md"],
                default_labels=["autocode"],
                title_prefix="",
                max_items_per_cycle=10,
            )

            items = source.collect_items()
            updated = source.annotate_item_issue(
                items[0].key,
                issue_number=875,
                issue_url="https://github.com/octo-org/example-repo/issues/875",
            )

            self.assertTrue(updated)
            contents = todo_path.read_text(encoding="utf-8")
            self.assertIn("issue: [#875](https://github.com/octo-org/example-repo/issues/875)", contents)

            recollected = source.collect_items()
            self.assertEqual(items[0].key, recollected[0].key)
            self.assertEqual("add API test", recollected[0].title)
            self.assertEqual("trim logs", recollected[1].title)

    def test_annotate_item_issue_replaces_existing_link_instead_of_duplication(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            todo_path = repo / "TODO.md"
            todo_path.write_text(
                "- [ ] add API test\n  issue: [#875](https://example.com/issues/875)\n",
                encoding="utf-8",
            )
            source = BacklogSource(
                repo_path=str(repo),
                file_paths=["TODO.md"],
                default_labels=["autocode"],
                title_prefix="",
                max_items_per_cycle=10,
            )

            item = source.collect_items()[0]
            updated = source.annotate_item_issue(
                item.key,
                issue_number=883,
                issue_url="https://github.com/octo-org/example-repo/issues/883",
            )

            self.assertTrue(updated)
            contents = todo_path.read_text(encoding="utf-8")
            self.assertEqual(1, contents.count("issue: [#"))
            self.assertIn("issue: [#883](https://github.com/octo-org/example-repo/issues/883)", contents)

    def test_get_item_issue_reference_reads_existing_issue_link(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            todo_path = repo / "TODO.md"
            todo_path.write_text(
                "- [ ] add API test\n  issue: [#883](https://github.com/octo-org/example-repo/issues/883)\n",
                encoding="utf-8",
            )
            source = BacklogSource(
                repo_path=str(repo),
                file_paths=["TODO.md"],
                default_labels=["autocode"],
                title_prefix="",
                max_items_per_cycle=10,
            )

            item = source.collect_items()[0]
            reference = source.get_item_issue_reference(item.key)

            self.assertEqual(
                {
                    "issue_number": 883,
                    "issue_url": "https://github.com/octo-org/example-repo/issues/883",
                },
                reference,
            )


if __name__ == "__main__":
    unittest.main()
