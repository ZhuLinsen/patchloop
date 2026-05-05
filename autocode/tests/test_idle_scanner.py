import tempfile
import unittest
from pathlib import Path

from agent.idle_scanner import IdleScanner


class IdleScannerTests(unittest.TestCase):
    def test_collect_items_reads_todo_and_fixme_findings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "app.py").write_text(
                "def run():\n"
                "    pass  # TODO add validation\n"
                "    return True  # FIXME handle false branch\n",
                encoding="utf-8",
            )
            scanner = IdleScanner(
                repo_path=str(repo),
                default_labels=["autocode", "idle-scan"],
                title_prefix="[Idle]",
                max_items_per_cycle=10,
            )

            items = scanner.collect_items()

            self.assertEqual(2, len(items))
            self.assertTrue(items[0].title.startswith("[Idle]"))
            self.assertEqual(["autocode", "idle-scan"], items[0].labels)
            self.assertIn("app.py:2", items[0].body)
            self.assertIn("TODO", items[0].body)
            self.assertNotIn("AutoCode Idle Scanner Finding", items[0].body)

    def test_item_key_is_stable_when_unrelated_lines_are_inserted_above(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            file_path = repo / "app.py"
            file_path.write_text(
                "def helper():\n"
                "    return True\n"
                "\n"
                "def run():\n"
                "    pass  # TODO add validation\n"
                "    return True\n",
                encoding="utf-8",
            )
            scanner = IdleScanner(
                repo_path=str(repo),
                default_labels=["autocode", "idle-scan"],
                title_prefix="[Idle]",
                max_items_per_cycle=10,
            )

            initial = scanner.collect_items()
            self.assertEqual(1, len(initial))

            file_path.write_text(
                "# inserted header\n"
                "def helper():\n"
                "    return True\n"
                "\n"
                "def run():\n"
                "    pass  # TODO add validation\n"
                "    return True\n",
                encoding="utf-8",
            )
            shifted = scanner.collect_items()

            self.assertEqual(1, len(shifted))
            self.assertEqual(initial[0].key, shifted[0].key)
            self.assertNotEqual(initial[0].line_number, shifted[0].line_number)

    def test_identical_findings_in_same_file_get_distinct_keys_and_later_copy_stays_stable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            file_path = repo / "app.py"
            file_path.write_text(
                "def run():\n"
                "    pass  # TODO add validation\n"
                "    print('ok')\n"
                "    return True  # TODO add validation\n",
                encoding="utf-8",
            )
            scanner = IdleScanner(
                repo_path=str(repo),
                default_labels=["autocode", "idle-scan"],
                title_prefix="[Idle]",
                max_items_per_cycle=10,
            )

            initial = scanner.collect_items()

            self.assertEqual(2, len(initial))
            self.assertNotEqual(initial[0].key, initial[1].key)

            file_path.write_text(
                "def run():\n"
                "    print('ok')\n"
                "    return True  # TODO add validation\n",
                encoding="utf-8",
            )
            shifted = scanner.collect_items()

            self.assertEqual(1, len(shifted))
            self.assertEqual(initial[1].key, shifted[0].key)

    def test_identical_comment_findings_keep_later_key_when_earlier_duplicate_is_removed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            file_path = repo / "app.py"
            file_path.write_text(
                "def run():\n"
                "    # TODO add tests\n"
                "    # TODO add tests\n",
                encoding="utf-8",
            )
            scanner = IdleScanner(
                repo_path=str(repo),
                default_labels=["autocode", "idle-scan"],
                title_prefix="[Idle]",
                max_items_per_cycle=10,
            )

            initial = scanner.collect_items()

            self.assertEqual(2, len(initial))
            self.assertNotEqual(initial[0].key, initial[1].key)

            file_path.write_text(
                "def run():\n"
                "    # TODO add tests\n",
                encoding="utf-8",
            )
            shifted = scanner.collect_items()

            self.assertEqual(1, len(shifted))
            self.assertEqual(initial[1].key, shifted[0].key)

    def test_collect_items_skips_ignored_directories(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "src").mkdir()
            (repo / "node_modules").mkdir()
            (repo / "src" / "app.py").write_text(
                "def run():\n"
                "    pass  # TODO add validation\n",
                encoding="utf-8",
            )
            (repo / "node_modules" / "vendor.js").write_text(
                "const value = 1; // TODO should be ignored\n",
                encoding="utf-8",
            )
            scanner = IdleScanner(
                repo_path=str(repo),
                default_labels=["autocode", "idle-scan"],
                title_prefix="[Idle]",
                max_items_per_cycle=10,
            )

            items = scanner.collect_items()
            scanned_paths = {path.relative_to(repo).as_posix() for path in scanner._iter_candidate_files()}

            self.assertEqual(1, len(items))
            self.assertEqual("src/app.py", items[0].source_path)
            self.assertEqual({"src/app.py"}, scanned_paths)

    def test_collect_items_falls_back_to_heuristic_candidates_when_todo_is_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "app.py").write_text(
                "import requests\n\n"
                "def run():\n"
                "    return requests.get('https://example.com')\n",
                encoding="utf-8",
            )
            scanner = IdleScanner(
                repo_path=str(repo),
                default_labels=["autocode", "idle-scan"],
                title_prefix="[Idle]",
                max_items_per_cycle=10,
                enable_llm_enrichment=False,
            )

            items = scanner.collect_items()

            self.assertEqual(1, len(items))
            self.assertEqual("app.py", items[0].source_path)
            self.assertEqual(["autocode", "idle-scan"], items[0].labels)
            self.assertIn("timeout", items[0].title)
            self.assertIn("requests.get", items[0].body)
            self.assertNotIn("AutoCode Idle Analysis", items[0].body)

    def test_get_item_by_key_can_resolve_heuristic_candidate_even_if_todo_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            file_path = repo / "app.py"
            file_path.write_text(
                "import requests\n\n"
                "def run():\n"
                "    # TODO add retries\n"
                "    return requests.get('https://example.com')\n",
                encoding="utf-8",
            )
            scanner = IdleScanner(
                repo_path=str(repo),
                default_labels=["autocode", "idle-scan"],
                title_prefix="[Idle]",
                max_items_per_cycle=10,
                enable_llm_enrichment=False,
            )

            heuristic_items = scanner._collect_advanced_items()
            resolved = scanner.get_item_by_key(heuristic_items[0].key)

            self.assertTrue(heuristic_items)
            self.assertIsNotNone(resolved)
            self.assertEqual(heuristic_items[0].key, resolved.key)
            self.assertEqual(["autocode", "idle-scan"], resolved.labels)

    def test_collect_items_ignores_requests_examples_in_comments_and_docstrings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "app.py").write_text(
                'def run():\n'
                '    """\n'
                '    requests.get("https://example.com")\n'
                '    """\n'
                '    # requests.get("https://example.com")\n'
                "    return True\n",
                encoding="utf-8",
            )
            scanner = IdleScanner(
                repo_path=str(repo),
                default_labels=["autocode", "idle-scan"],
                title_prefix="[Idle]",
                max_items_per_cycle=10,
                enable_llm_enrichment=False,
            )

            items = scanner.collect_items()

            self.assertEqual([], items)

    def test_collect_items_ignores_multiline_requests_with_timeout_on_late_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "app.py").write_text(
                "import requests\n\n"
                "def run():\n"
                "    return requests.get(\n"
                '        "https://example.com",\n'
                '        headers={"A": "B"},\n'
                '        params={"q": "x"},\n'
                '        auth=("u", "p"),\n'
                '        cookies={"c": "d"},\n'
                '        json={"k": "v"},\n'
                "        timeout=5,\n"
                "    )\n",
                encoding="utf-8",
            )
            scanner = IdleScanner(
                repo_path=str(repo),
                default_labels=["autocode", "idle-scan"],
                title_prefix="[Idle]",
                max_items_per_cycle=10,
                enable_llm_enrichment=False,
            )

            items = scanner.collect_items()

            self.assertEqual([], items)

    def test_collect_items_ignores_multiline_add_argument_with_help_on_late_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "app.py").write_text(
                "import argparse\n\n"
                "def build_parser():\n"
                "    parser = argparse.ArgumentParser()\n"
                "    parser.add_argument(\n"
                '        "--symbol",\n'
                '        dest="symbol",\n'
                '        metavar="SYMBOL",\n'
                '        default="AAPL",\n'
                '        choices=["AAPL", "MSFT"],\n'
                '        help="stock symbol",\n'
                "    )\n"
                "    return parser\n",
                encoding="utf-8",
            )
            scanner = IdleScanner(
                repo_path=str(repo),
                default_labels=["autocode", "idle-scan"],
                title_prefix="[Idle]",
                max_items_per_cycle=10,
                enable_llm_enrichment=False,
            )

            items = scanner.collect_items()

            self.assertEqual([], items)

    def test_heuristic_item_key_and_title_stay_stable_when_unrelated_lines_are_inserted_above(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            file_path = repo / "app.py"
            file_path.write_text(
                "import requests\n\n"
                "def run():\n"
                "    return requests.get('https://example.com')\n",
                encoding="utf-8",
            )
            scanner = IdleScanner(
                repo_path=str(repo),
                default_labels=["autocode", "idle-scan"],
                title_prefix="[Idle]",
                max_items_per_cycle=10,
                enable_llm_enrichment=False,
            )

            initial = scanner._collect_advanced_items()

            file_path.write_text(
                "# inserted header\n"
                "import requests\n\n"
                "def run():\n"
                "    return requests.get('https://example.com')\n",
                encoding="utf-8",
            )
            shifted = scanner._collect_advanced_items()

            self.assertEqual(1, len(initial))
            self.assertEqual(1, len(shifted))
            self.assertEqual(initial[0].key, shifted[0].key)
            self.assertEqual(initial[0].title, shifted[0].title)
            self.assertNotEqual(initial[0].line_number, shifted[0].line_number)

    def test_collect_items_use_plain_issue_style_and_autocode_label_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = Path(tmpdir)
            (repo / "app.py").write_text(
                "import requests\n\n"
                "def run():\n"
                "    return requests.get('https://example.com')\n",
                encoding="utf-8",
            )
            scanner = IdleScanner(
                repo_path=str(repo),
                default_labels=["autocode"],
                title_prefix="",
                max_items_per_cycle=10,
                enable_llm_enrichment=False,
            )

            items = scanner.collect_items()

            self.assertEqual(1, len(items))
            self.assertEqual(["autocode"], items[0].labels)
            self.assertEqual("app.py: 网络请求缺少 timeout", items[0].title)
            self.assertNotIn("AutoCode Idle Analysis", items[0].body)
            self.assertIn("在 `app.py:4` 发现了一个值得确认的问题。", items[0].body)


if __name__ == "__main__":
    unittest.main()
