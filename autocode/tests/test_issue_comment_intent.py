import unittest

from agent.issue_comment_intent import (
    classify_issue_comment_implementation_intent,
    issue_comment_requests_implementation,
)


class FakeIntentAnalyzer:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def analyze(self, prompt: str, system: str = "") -> str:
        del system
        self.prompts.append(prompt)
        return self.response


class IssueCommentIntentTests(unittest.TestCase):
    def test_accepts_direct_natural_language_commands(self):
        for body in (
            "你直接修一下",
            "进行修复",
            "请修复",
            "开始修复",
            "按这个方案修复",
            "按你说的方案处理\n需要的话补测试",
            "开个 PR 吧",
            "please fix it",
        ):
            with self.subTest(body=body):
                self.assertTrue(issue_comment_requests_implementation(body))

    def test_accepts_not_resolved_followups(self):
        for body in (
            "还是不行，继续修",
            "没有解决，再试一下",
            "不是这个根因，重新处理",
            "still failing, try again",
        ):
            with self.subTest(body=body):
                self.assertTrue(issue_comment_requests_implementation(body))

    def test_rejects_ambiguous_or_negative_comments(self):
        for body in (
            "实现\n但记得补测试",
            "这个功能还未实现",
            "先别实现",
            "看起来不错",
        ):
            with self.subTest(body=body):
                self.assertFalse(issue_comment_requests_implementation(body))

    def test_llm_classifier_is_primary_when_available(self):
        analyzer = FakeIntentAnalyzer('{"intent":"IGNORE","reason":"测试覆盖","confidence":0.9}')

        result = classify_issue_comment_implementation_intent("实现", analyzer=analyzer)

        self.assertFalse(result.requests_implementation)
        self.assertEqual("llm", result.source)
        self.assertEqual(1, len(analyzer.prompts))

    def test_llm_classifier_falls_back_to_regex_on_invalid_response(self):
        analyzer = FakeIntentAnalyzer("not json")

        result = classify_issue_comment_implementation_intent("进行修复", analyzer=analyzer)

        self.assertTrue(result.requests_implementation)
        self.assertEqual("regex_fallback", result.source)
        self.assertTrue(result.degraded)


if __name__ == "__main__":
    unittest.main()
