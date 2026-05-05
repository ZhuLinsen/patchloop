import unittest

from agent.issue_comment_intent import issue_comment_requests_implementation


class IssueCommentIntentTests(unittest.TestCase):
    def test_accepts_direct_natural_language_commands(self):
        for body in (
            "你直接修一下",
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


if __name__ == "__main__":
    unittest.main()
