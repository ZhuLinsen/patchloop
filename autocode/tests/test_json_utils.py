import unittest

from agent.json_utils import extract_json_object


class JsonUtilsTests(unittest.TestCase):
    def test_extract_json_object_parses_fenced_json(self):
        raw = '```json\n{"task_type":"bug_fix","confidence":0.9}\n```'

        parsed = extract_json_object(raw, context="triage 回复")

        self.assertEqual("bug_fix", parsed["task_type"])
        self.assertEqual(0.9, parsed["confidence"])

    def test_extract_json_object_finds_first_valid_object_without_greedy_regex(self):
        raw = '分析如下:\n{"first": true}\n补充说明\n{"second": true}'

        parsed = extract_json_object(raw, context="execution plan 回复")

        self.assertEqual({"first": True}, parsed)

    def test_extract_json_object_raises_value_error_for_invalid_json(self):
        with self.assertRaisesRegex(ValueError, "idle narrative"):
            extract_json_object('prefix {"broken": } suffix', context="idle narrative")


if __name__ == "__main__":
    unittest.main()
