import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from adapters.base import BaseLLMAdapter
from agent.autocode_classifier import AutoCodeTriageResult
from agent.autocode_state_store import (
    AutoCodeStateStore,
    build_feedback_item_fingerprint,
    build_feedback_key,
)
from agent.execution_controller import ExecutionController, PlanningResult
from config import CLIProfile
from agent.execution_planner import ExecutionPlan
from agent.execution_policy import PolicyDecision
from agent.patch_inspector import PatchInspectionResult
from agent.validation_runner import ValidationResult, ValidationStepResult
from prompts import format_execution_plan_reply


class RecordingAdapter(BaseLLMAdapter):
    def __init__(self, *, working_dir: str | None = None, calls: list[str | None] | None = None):
        self.working_dir = working_dir
        self.calls = calls if calls is not None else []

    def analyze(self, prompt: str, system: str = "") -> str:
        del prompt, system
        self.calls.append(self.working_dir)
        return "done"

    def with_working_dir(self, working_dir: str | None) -> "RecordingAdapter":
        return RecordingAdapter(working_dir=working_dir, calls=self.calls)


class ExplodingAdapter(BaseLLMAdapter):
    def __init__(self, exc: Exception):
        self.exc = exc

    def analyze(self, prompt: str, system: str = "") -> str:
        del prompt, system
        raise self.exc

    def with_working_dir(self, working_dir: str | None) -> "ExplodingAdapter":
        del working_dir
        return self


class SequenceAdapter(BaseLLMAdapter):
    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def analyze(self, prompt: str, system: str = "") -> str:
        del system
        self.prompts.append(prompt)
        if not self.responses:
            raise AssertionError("No response left for SequenceAdapter")
        return self.responses.pop(0)

    def with_working_dir(self, working_dir: str | None) -> "SequenceAdapter":
        del working_dir
        return self


class PreviousFailureStateStore:
    def __init__(self, runs: list[dict]):
        self.runs = runs

    def list_failed_runs_for_target(self, kind: str, target_number: int, limit: int = 3) -> list[dict]:
        del kind, target_number, limit
        return list(self.runs)


class FakeGitHubClient:
    def __init__(
        self,
        *,
        pull_request_pages=None,
        review_pages=None,
        review_comment_pages=None,
        existing_prs=None,
        pr_details=None,
        pr_files=None,
        issue_comment_pages=None,
    ):
        self.pull_request_pages = pull_request_pages or {}
        self.review_pages = review_pages or {}
        self.review_comment_pages = review_comment_pages or {}
        self.existing_prs = existing_prs or {}
        self.pr_details = pr_details or {}
        self.pr_files = pr_files or {}
        self.issue_comment_pages = issue_comment_pages or {}
        self.posted_comments: list[tuple[int, str]] = []

    def list_pull_requests(self, *, page: int = 1, **kwargs) -> list[dict]:
        head = kwargs.get("head")
        base = kwargs.get("base")
        if head or base:
            return list(self.existing_prs.get((head or "", base or ""), []))
        return list(self.pull_request_pages.get(page, []))

    def list_pr_reviews(self, pr_number: int, *, page: int = 1, **kwargs) -> list[dict]:
        del pr_number, kwargs
        return list(self.review_pages.get(page, []))

    def list_pr_review_comments(self, pr_number: int, *, page: int = 1, **kwargs) -> list[dict]:
        del pr_number, kwargs
        return list(self.review_comment_pages.get(page, []))

    def get_pr(self, pr_number: int) -> dict:
        return dict(self.pr_details.get(pr_number, _build_pr(pr_number)))

    def get_pr_files(self, pr_number: int) -> list[str]:
        return list(self.pr_files.get(pr_number, []))

    def list_issue_comments(self, issue_number: int, *, page: int = 1, **kwargs) -> list[dict]:
        del issue_number, kwargs
        return list(self.issue_comment_pages.get(page, []))

    def post_issue_comment(self, issue_number: int, body: str) -> dict:
        self.posted_comments.append((issue_number, body))
        return {"id": len(self.posted_comments)}


class FakeGitHubWriteClient:
    def __init__(self, *, dry_run: bool = False):
        self.created_payloads: list[dict] = []
        self.updated_payloads: list[tuple[int, dict]] = []
        self.ready_calls: list[int] = []
        self.dry_run = dry_run

    def create_pull_request(self, **kwargs) -> dict:
        self.created_payloads.append(dict(kwargs))
        if self.dry_run:
            return None
        return {"number": 77, "html_url": "https://example.com/pr/77", "payload": kwargs}

    def update_pull_request(self, pr_number: int, *, title: str | None = None, body: str | None = None) -> dict:
        payload = {}
        if title is not None:
            payload["title"] = title
        if body is not None:
            payload["body"] = body
        self.updated_payloads.append((pr_number, payload))
        if self.dry_run:
            return None
        return {"number": pr_number, "html_url": f"https://example.com/pr/{pr_number}", "payload": payload}

    def mark_ready_for_review(self, pr_number: int) -> dict:
        self.ready_calls.append(pr_number)
        if self.dry_run:
            return None
        return {"number": pr_number, "html_url": f"https://example.com/pr/{pr_number}", "draft": False}


class FakeWorkspaceManager:
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.sync_calls = 0
        self.commit_calls: list[str] = []
        self.push_calls: list[str] = []
        self.branch_locks: dict[str, dict[str, object]] = {}

    def sync_default_branch(self):
        self.sync_calls += 1

    def is_protected_branch(self, branch_name: str) -> bool:
        return branch_name == "main"

    @contextmanager
    def prepare_issue_workspace(self, *, issue_number: int, title: str, base_ref: str):
        del issue_number, title, base_ref
        yield SimpleNamespace(
            path=self.workspace_path,
            branch_name="autocode/issue-1-demo",
            base_ref="origin/release",
        )

    @contextmanager
    def prepare_existing_branch_workspace(self, *, branch_name: str, remote: str | None = None):
        del remote
        yield SimpleNamespace(
            path=self.workspace_path,
            branch_name=branch_name,
            base_ref=f"origin/{branch_name}",
        )

    def active_branch_lock(self, branch_name: str) -> dict[str, object] | None:
        return self.branch_locks.get(branch_name)

    def has_changes(self, workspace_path: Path) -> bool:
        del workspace_path
        return True

    def commit_all(self, workspace_path: Path, message: str):
        del workspace_path
        self.commit_calls.append(message)

    def push_branch(self, workspace_path: Path, branch_name: str, remote: str | None = None, *, force: bool = False):
        del workspace_path, remote, force
        self.push_calls.append(branch_name)

    def rebase_onto_default_branch(self, workspace_path: Path) -> bool:
        del workspace_path
        return False


class FakeValidationRunner:
    def run_all(self, workspace_path: Path, *, test_timeout_override: int = 0) -> ValidationResult:
        del workspace_path, test_timeout_override
        return ValidationResult(success=True, steps=[])


class FakePatchInspector:
    def __init__(self, result: PatchInspectionResult | None = None):
        self.calls: list[dict] = []
        self.result = result

    def inspect(
        self,
        workspace_path: Path,
        *,
        require_test_changes: bool,
        allowed_paths: list[str] | None = None,
        documentation_only: bool = False,
    ) -> PatchInspectionResult:
        del workspace_path, require_test_changes
        self.calls.append({
            "allowed_paths": list(allowed_paths or []),
            "documentation_only": documentation_only,
        })
        if self.result is not None:
            return self.result
        return PatchInspectionResult(
            blocked=False,
            reasons=[],
            changed_files=["app.py"],
            added_lines=1,
            deleted_lines=0,
            has_test_changes=True,
        )


class FakePolicyEngine:
    def evaluate_issue(self, **kwargs) -> PolicyDecision:
        del kwargs
        return PolicyDecision(allowed=True, requires_approval=False, reasons=[])

    def validate_publish_target(self, base_branch: str) -> PolicyDecision:
        del base_branch
        return PolicyDecision(allowed=True, requires_approval=False, reasons=[])


def _build_pr(number: int) -> dict:
    return {
        "number": number,
        "title": f"PR #{number}",
        "body": "body",
        "html_url": f"https://example.com/pr/{number}",
        "user": {"login": "reviewer"},
        "head": {
            "ref": f"autocode/issue-{number}-demo",
            "sha": f"sha-{number}",
            "repo": {"full_name": "demo/repo"},
        },
        "base": {"ref": "release"},
    }


def _build_controller(
    *,
    workspace_path: Path,
    gh: FakeGitHubClient | None = None,
    planner: BaseLLMAdapter | None = None,
    executor: RecordingAdapter | None = None,
    state_store: AutoCodeStateStore | None = None,
    gh_writer: FakeGitHubWriteClient | None = None,
    validation_runner: FakeValidationRunner | None = None,
    patch_inspector: FakePatchInspector | None = None,
    dry_run: bool = False,
) -> ExecutionController:
    config = SimpleNamespace(
        github=SimpleNamespace(repo="demo/repo", owner="repoowner"),
        cli=SimpleNamespace(
            primary_cli="codex",
            primary_model="gpt-5.4",
            primary_fallback_model="gpt-5.3-codex",
            primary_model_flag="",
            primary_command=[],
            fallback_cli="copilot",
            fallback_clis=["copilot"],
            fallback_model="gpt-4.1",
            fallback_fallback_model="",
            fallback_model_flag="",
            fallback_command=[],
            executor_cli_timeout=1800,
            todo_refactor_executor_cli_timeout=900,
            todo_refactor_executor_model="gpt-5.3-codex",
            todo_refactor_executor_max_idle_seconds=180,
            simple_executor_model="",
            simple_executor_max_feedback_items=1,
            cli_profiles={
                "codex": CLIProfile(
                    name="codex",
                    primary_model="gpt-5.4",
                    fallback_models=["gpt-5.3-codex"],
                    model_flag="",
                    billing="token",
                ),
                "copilot": CLIProfile(
                    name="copilot",
                    primary_model="gpt-4.1",
                    fallback_models=[],
                    model_flag="",
                    billing="call",
                ),
            },
        ),
        local_repo=SimpleNamespace(
            path=str(workspace_path),
            default_branch="main",
            max_context_files=3,
            max_file_chars=400,
            max_total_context_chars=1200,
        ),
        autocode=SimpleNamespace(
            pr_base_branch="release",
            mode="manual",
            auto_implement_on_issue_open=False,
            allow_high_risk_autorun=False,
            plan_comment_header="## AutoCode 计划",
            format_commands=[],
            lint_commands=[],
            test_commands=[],
            enable_pr_queue=True,
            poll_review_feedback=True,
            auto_ready_for_review=False,
            pr_queue_max_batch=1,
            branch_prefix="autocode",
            tracked_pr_authors=["reviewer"],
            forbid_main_base_branch=True,
            max_changed_files=20,
            max_added_lines=400,
            max_deleted_lines=200,
            pr_max_repair_rounds=5,
            max_open_autocode_prs=20,
            worktree_root="/tmp/autocode-runs",
            repair_test_timeout_seconds=600,
            issue_repair_max_attempts=2,
            review_feedback_repair_max_attempts=2,
            bot_feedback_max_repair_rounds=4,
            validation_fail_fast=False,
        ),
        server=SimpleNamespace(poll_items_per_page=5, dry_run=dry_run),
    )
    return ExecutionController(
        planner=planner or RecordingAdapter(),
        executor=executor or RecordingAdapter(),
        gh=gh or FakeGitHubClient(),
        gh_writer=gh_writer or FakeGitHubWriteClient(dry_run=dry_run),
        state_store=state_store or AutoCodeStateStore(str(workspace_path / ".autocode-state.json")),
        workspace_manager=FakeWorkspaceManager(workspace_path),
        validation_runner=validation_runner or FakeValidationRunner(),
        patch_inspector=patch_inspector or FakePatchInspector(),
        policy_engine=FakePolicyEngine(),
        config=config,
    )


class FakeFailedValidationRunner:
    def __init__(self, result: ValidationResult):
        self.result = result

    def run_all(self, workspace_path: Path, *, test_timeout_override: int = 0) -> ValidationResult:
        del workspace_path, test_timeout_override
        return self.result


class SequenceValidationRunner:
    def __init__(self, results: list[ValidationResult]):
        self.results = list(results)
        self.calls = 0

    def run_all(self, workspace_path: Path, *, test_timeout_override: int = 0) -> ValidationResult:
        del workspace_path, test_timeout_override
        self.calls += 1
        if not self.results:
            raise AssertionError("No validation result left")
        return self.results.pop(0)


class ExecutionControllerTests(unittest.TestCase):
    def test_executor_settings_use_todo_refactor_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = _build_controller(workspace_path=Path(tmpdir))

            settings = controller._execution_adapter_settings("todo_refactor")

            self.assertEqual("todo_refactor", settings["profile"])
            self.assertEqual(900, settings["cli_timeout"])
            self.assertEqual(180, settings["max_idle_seconds"])
            self.assertEqual("gpt-5.3-codex", settings["primary_model"])
            self.assertEqual("gpt-5.4", settings["primary_fallback_model"])

    def test_should_auto_execute_blocks_non_owner_feature_types(self):
        """非 owner 创建的 feature 仍不在 issue opened 时自动执行。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = _build_controller(workspace_path=Path(tmpdir))
            controller.config.autocode.mode = "auto"
            controller.config.autocode.auto_implement_on_issue_open = True
            controller.config.autocode.allow_high_risk_autorun = True

            for task_type in ("small_feature", "high_risk_feature"):
                should_auto = controller._should_auto_execute(
                    AutoCodeTriageResult(
                        task_type=task_type,
                        action="implement",
                        risk_level="high",
                        reason="critical migration",
                        confidence=0.9,
                    )
                )
                self.assertFalse(should_auto, f"{task_type} should NOT auto-execute")

    def test_should_auto_execute_allows_owner_feature_types(self):
        """Repo owner 自己创建的 feature issue 视为已授权执行。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = _build_controller(workspace_path=Path(tmpdir))
            controller.config.autocode.mode = "auto"
            controller.config.autocode.auto_implement_on_issue_open = True
            controller.config.autocode.allow_high_risk_autorun = True

            for task_type in ("small_feature", "high_risk_feature"):
                should_auto = controller._should_auto_execute(
                    AutoCodeTriageResult(
                        task_type=task_type,
                        action="plan_only",
                        risk_level="high",
                        reason="owner requested feature",
                        confidence=0.9,
                    ),
                    owner_authored_issue=True,
                )
                self.assertTrue(should_auto, f"{task_type} should auto-execute for owner-authored issues")

    def test_should_auto_execute_blocks_high_risk_feature_when_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = _build_controller(workspace_path=Path(tmpdir))
            controller.config.autocode.mode = "auto"
            controller.config.autocode.auto_implement_on_issue_open = True
            controller.config.autocode.allow_high_risk_autorun = False

            should_auto = controller._should_auto_execute(
                AutoCodeTriageResult(
                    task_type="high_risk_feature",
                    action="implement",
                    risk_level="high",
                    reason="critical migration",
                    confidence=0.9,
                )
            )

            self.assertFalse(should_auto)

    def test_execute_issue_runs_executor_in_workspace_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            executor = RecordingAdapter(working_dir="/tmp/not-the-workspace")
            controller = _build_controller(workspace_path=workspace_path, executor=executor)
            planning = PlanningResult(
                issue_number=1,
                title="Fix app",
                body="Adjust workspace execution",
                labels=["bug"],
                updated_at="2026-03-24T00:00:00Z",
                comment="plan",
                triage=AutoCodeTriageResult(
                    task_type="bug_fix",
                    action="implement",
                    risk_level="low",
                    reason="bug",
                    confidence=0.9,
                ),
                plan=ExecutionPlan(
                    goal="Fix app",
                    assumptions=[],
                    acceptance_criteria=["workspace run succeeds"],
                    risk_level="low",
                    estimated_files=["app.py"],
                    suggested_tests=[],
                    needs_human_approval=False,
                    blocked_reasons=[],
                ),
                issue_key="issue:1:test",
            )

            result = controller.execute_issue(
                {"number": 1, "updated_at": "2026-03-24T00:00:00Z"},
                explicit_command=True,
                planning=planning,
            )

            self.assertTrue(result.success)
            self.assertEqual([str(workspace_path)], executor.calls)
            self.assertEqual(1, controller.workspace_manager.sync_calls)

    def test_execute_issue_repairs_after_validation_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            executor = SequenceAdapter(["first attempt", "repair attempt"])
            validation_runner = SequenceValidationRunner(
                [
                    ValidationResult(
                        success=False,
                        steps=[
                            ValidationStepResult(
                                name="test",
                                command=["pytest", "tests/test_app.py"],
                                success=False,
                                output="AssertionError: expected timeout guard",
                            )
                        ],
                    ),
                    ValidationResult(success=True, steps=[]),
                ]
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                executor=executor,
                validation_runner=validation_runner,
            )
            planning = PlanningResult(
                issue_number=1,
                title="Fix app",
                body="Adjust workspace execution",
                labels=["bug"],
                updated_at="2026-03-24T00:00:00Z",
                comment="plan",
                triage=AutoCodeTriageResult(
                    task_type="bug_fix",
                    action="implement",
                    risk_level="low",
                    reason="bug",
                    confidence=0.9,
                ),
                plan=ExecutionPlan(
                    goal="Fix app",
                    assumptions=[],
                    acceptance_criteria=["workspace run succeeds"],
                    risk_level="low",
                    estimated_files=["app.py"],
                    suggested_tests=[],
                    needs_human_approval=False,
                    blocked_reasons=[],
                ),
                issue_key="issue:1:test",
            )

            result = controller.execute_issue(
                {"number": 1, "updated_at": "2026-03-24T00:00:00Z"},
                explicit_command=True,
                planning=planning,
            )

            self.assertTrue(result.success)
            self.assertEqual(2, validation_runner.calls)
            self.assertEqual(2, len(executor.prompts))
            self.assertIn("issue 执行第 1 轮在 validation 阶段失败", executor.prompts[1])
            self.assertIn("AssertionError: expected timeout guard", executor.prompts[1])

    def test_executor_for_workspace_builds_special_adapter_for_todo_refactor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            controller = _build_controller(workspace_path=workspace_path)
            replacement = RecordingAdapter()
            with mock.patch("agent.execution_controller.create_adapter_from_profiles", return_value=replacement) as factory:
                adapter = controller._executor_for_workspace(
                    workspace_path,
                    task_type="todo_refactor",
                    log_context="issue-execution#883",
                )

            self.assertIs(replacement, adapter)
            factory.assert_called_once_with(
                cli_profiles=controller.config.cli.cli_profiles,
                primary_cli="codex",
                fallback_clis=["copilot"],
                cli_timeout=900,
                max_idle_seconds=180,
                working_dir=str(workspace_path),
                execution_mode=True,
                primary_model_override="gpt-5.3-codex",
                primary_fallback_model_override="gpt-5.4",
            )

    def test_review_feedback_executor_uses_simple_model_for_single_feedback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            controller = _build_controller(workspace_path=workspace_path)
            controller.config.cli.simple_executor_model = "gpt-5.3-codex-spark"
            replacement = RecordingAdapter()
            with mock.patch("agent.execution_controller.create_adapter_from_profiles", return_value=replacement) as factory:
                adapter = controller._review_feedback_executor_for_workspace(
                    workspace_path,
                    feedback=[{"id": 101, "body": "Please refresh the changed files list"}],
                    log_context="pr-feedback#11",
                )

            self.assertIs(replacement, adapter)
            factory.assert_called_once_with(
                cli_profiles=controller.config.cli.cli_profiles,
                primary_cli="codex",
                fallback_clis=["copilot"],
                cli_timeout=1800,
                max_idle_seconds=300,
                working_dir=str(workspace_path),
                execution_mode=True,
                primary_model_override="gpt-5.3-codex-spark",
                primary_fallback_model_override="gpt-5.4",
            )

    def test_review_feedback_executor_keeps_primary_model_for_code_feedback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            executor = RecordingAdapter()
            controller = _build_controller(workspace_path=workspace_path, executor=executor)
            controller.config.cli.simple_executor_model = "gpt-5.3-codex-spark"
            with mock.patch("agent.execution_controller.create_adapter_from_profiles") as factory:
                adapter = controller._review_feedback_executor_for_workspace(
                    workspace_path,
                    feedback=[{"id": 101, "path": "src/analyzer.py", "body": "Redact quoted values"}],
                    log_context="pr-feedback#897",
                )

            self.assertIsInstance(adapter, RecordingAdapter)
            factory.assert_not_called()

    def test_execute_issue_reuses_existing_pull_request(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            gh = FakeGitHubClient(
                existing_prs={
                    ("demo:autocode/issue-1-demo", "release"): [
                        {
                            "number": 91,
                            "html_url": "https://example.com/pr/91",
                            "title": "fix: 修复工作区执行问题 (#1)",
                        }
                    ]
                }
            )
            gh_writer = FakeGitHubWriteClient()
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                gh_writer=gh_writer,
            )
            planning = PlanningResult(
                issue_number=1,
                title="Fix app",
                body="Adjust workspace execution",
                labels=["bug"],
                updated_at="2026-03-24T00:00:00Z",
                comment="plan",
                triage=AutoCodeTriageResult(
                    task_type="bug_fix",
                    action="implement",
                    risk_level="low",
                    reason="bug",
                    confidence=0.9,
                ),
                plan=ExecutionPlan(
                    goal="Fix app",
                    assumptions=[],
                    acceptance_criteria=["workspace run succeeds"],
                    risk_level="low",
                    estimated_files=["app.py"],
                    suggested_tests=[],
                    needs_human_approval=False,
                    blocked_reasons=[],
                ),
                issue_key="issue:1:test",
            )

            result = controller.execute_issue(
                {"number": 1, "updated_at": "2026-03-24T00:00:00Z"},
                explicit_command=True,
                planning=planning,
            )

            self.assertTrue(result.success)
            self.assertEqual([], gh_writer.created_payloads)
            self.assertEqual(1, len(gh_writer.updated_payloads))
            self.assertEqual(91, gh_writer.updated_payloads[0][0])
            self.assertEqual("fix: 修复工作区执行问题 (#1)", gh_writer.updated_payloads[0][1]["title"])
            self.assertEqual(91, result.run_record.pr_number)

    def test_execute_issue_pr_body_highlights_key_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            gh_writer = FakeGitHubWriteClient()
            controller = _build_controller(
                workspace_path=workspace_path,
                gh_writer=gh_writer,
            )
            planning = PlanningResult(
                issue_number=1,
                title="[Bug] Fix auth timeout",
                body="Adjust request handling",
                labels=["bug", "security"],
                updated_at="2026-03-24T00:00:00Z",
                comment="plan",
                triage=AutoCodeTriageResult(
                    task_type="bug_fix",
                    action="implement",
                    risk_level="high",
                    reason="security bug",
                    confidence=0.9,
                ),
                plan=ExecutionPlan(
                    goal="Fix auth timeout handling",
                    assumptions=[],
                    acceptance_criteria=["requests have timeout"],
                    risk_level="high",
                    estimated_files=["app.py"],
                    suggested_tests=["pytest tests/test_auth.py"],
                    needs_human_approval=True,
                    blocked_reasons=[],
                ),
                issue_key="issue:1:test",
            )

            result = controller.execute_issue(
                {"number": 1, "updated_at": "2026-03-24T00:00:00Z"},
                explicit_command=True,
                planning=planning,
            )

            self.assertTrue(result.success)
            self.assertEqual(1, len(gh_writer.created_payloads))
            payload = gh_writer.created_payloads[0]
            self.assertEqual("fix: Fix auth timeout (#1)", payload["title"])
            body = payload["body"]
            self.assertIn("## PR Type", body)
            self.assertIn("- [x] fix", body)
            self.assertIn("## Background And Problem", body)
            self.assertIn("当前问题：Fix auth timeout handling", body)
            self.assertIn("触发来源：显式 implement/API 执行（Issue #1）。", body)
            self.assertIn("## Scope Of Change", body)
            self.assertIn("`app.py`", body)
            self.assertIn("## Documentation And Changelog", body)
            self.assertIn("当前补丁未包含 `README.md`、`docs/` 或 `docs/CHANGELOG.md` 更新", body)
            self.assertIn("## Issue Link", body)
            self.assertIn("Closes #1", body)
            self.assertIn("## Verification Commands And Results", body)
            self.assertIn("# no validation commands recorded", body)
            self.assertIn("## Compatibility And Risk", body)
            self.assertIn("**High**", body)
            self.assertIn("请重点审查变更范围", body)
            self.assertIn("## Rollback Plan", body)
            self.assertIn("## Acceptance Criteria", body)
            self.assertIn("requests have timeout", body)
            self.assertNotIn("## Implementation", body)
            self.assertNotIn("done", body)
            self.assertIn("## Notes", body)
            self.assertIn("## Checklist", body)
            self.assertIn("- [ ] 文档与 `docs/CHANGELOG.md` 同步仍需确认", body)
            self.assertNotIn("### Safety", body)
            self.assertNotIn("manual_implement", body)

    def test_build_pr_title_prefers_plan_goal_for_low_signal_issue_title(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = _build_controller(workspace_path=Path(tmpdir))

            pr_title = controller._build_pr_title(
                "bug_fix",
                "今天突然就跑不动了",
                839,
                plan_goal="修复启动早期失败时日志缺失与根因不透明问题",
            )

            self.assertEqual("fix: 修复启动早期失败时日志缺失与根因不透明问题 (#839)", pr_title)

    def test_build_pr_title_uses_plan_goal_for_bare_prefix_title(self):
        """Issue titles like '[Bug]' or '[Feature]' should fall back to plan.goal."""
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = _build_controller(workspace_path=Path(tmpdir))

            pr_title = controller._build_pr_title(
                "bug_fix",
                "[Bug]",
                819,
                plan_goal="修复 WebUI 修改 SCHEDULE_TIME 后运行中的调度器不生效",
            )
            self.assertNotIn("[Bug]", pr_title.split("(#")[0])
            self.assertIn("修复 WebUI", pr_title)

            pr_title2 = controller._build_pr_title(
                "small_feature",
                "[Feature]",
                827,
                plan_goal="为 Windows 安装包增加可选目标目录",
            )
            self.assertNotIn("[Feature]", pr_title2.split("(#")[0])
            self.assertIn("Windows 安装包", pr_title2)

    def test_execute_issue_pr_body_marks_doc_sync_complete_when_docs_are_updated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "main.py").write_text("print('hello')\n", encoding="utf-8")
            (workspace_path / "README.md").write_text("readme\n", encoding="utf-8")
            docs_dir = workspace_path / "docs"
            docs_dir.mkdir()
            (docs_dir / "CHANGELOG.md").write_text("changelog\n", encoding="utf-8")
            gh_writer = FakeGitHubWriteClient()
            controller = _build_controller(
                workspace_path=workspace_path,
                gh_writer=gh_writer,
            )
            controller.patch_inspector.inspect = mock.Mock(
                return_value=PatchInspectionResult(
                    blocked=False,
                    reasons=[],
                    changed_files=["main.py", "README.md", "docs/CHANGELOG.md"],
                    added_lines=12,
                    deleted_lines=3,
                    has_test_changes=True,
                )
            )
            planning = PlanningResult(
                issue_number=1,
                title="[Bug] Fix startup logging",
                body="Improve early failure logging",
                labels=["bug"],
                updated_at="2026-03-24T00:00:00Z",
                comment="plan",
                triage=AutoCodeTriageResult(
                    task_type="bug_fix",
                    action="implement",
                    risk_level="medium",
                    reason="bug",
                    confidence=0.9,
                ),
                plan=ExecutionPlan(
                    goal="Fix startup logging fallback",
                    assumptions=[],
                    acceptance_criteria=["startup failures are logged"],
                    risk_level="medium",
                    estimated_files=["main.py", "README.md", "docs/CHANGELOG.md"],
                    suggested_tests=[],
                    needs_human_approval=False,
                    blocked_reasons=[],
                ),
                issue_key="issue:1:test",
            )

            result = controller.execute_issue(
                {"number": 1, "updated_at": "2026-03-24T00:00:00Z"},
                explicit_command=True,
                planning=planning,
            )

            self.assertTrue(result.success)
            body = gh_writer.created_payloads[0]["body"]
            self.assertIn("已同步更新文档/变更记录：`README.md`, `docs/CHANGELOG.md`。", body)
            self.assertIn("- [x] 已同步更新相关文档与 `docs/CHANGELOG.md`", body)

    def test_extract_pr_changed_files_from_body_reads_scope_section(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = _build_controller(workspace_path=Path(tmpdir))

            changed_files = controller._extract_pr_changed_files_from_body(
                "## Scope Of Change\n- `main.py`\n- `tests/test_main.py`\n\n## Issue Link\nCloses #1\n"
            )

            self.assertEqual(["main.py", "tests/test_main.py"], changed_files)

    def test_execute_issue_dry_run_skips_commit_push_and_marks_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            controller = _build_controller(workspace_path=workspace_path, dry_run=True)
            planning = PlanningResult(
                issue_number=1,
                title="Fix app",
                body="Adjust workspace execution",
                labels=["bug"],
                updated_at="2026-03-24T00:00:00Z",
                comment="plan",
                triage=AutoCodeTriageResult(
                    task_type="bug_fix",
                    action="implement",
                    risk_level="low",
                    reason="bug",
                    confidence=0.9,
                ),
                plan=ExecutionPlan(
                    goal="Fix app",
                    assumptions=[],
                    acceptance_criteria=["workspace run succeeds"],
                    risk_level="low",
                    estimated_files=["app.py"],
                    suggested_tests=[],
                    needs_human_approval=False,
                    blocked_reasons=[],
                ),
                issue_key="issue:1:test",
            )

            result = controller.execute_issue(
                {"number": 1, "updated_at": "2026-03-24T00:00:00Z"},
                explicit_command=True,
                planning=planning,
            )

            self.assertTrue(result.success)
            self.assertEqual("dry_run", result.run_record.status)
            self.assertEqual(0, result.run_record.pr_number)
            self.assertEqual([], controller.workspace_manager.commit_calls)
            self.assertEqual([], controller.workspace_manager.push_calls)
            self.assertIn("dry-run", result.comment)

    def test_execute_issue_marks_failed_on_unexpected_executor_exception(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            controller = _build_controller(
                workspace_path=workspace_path,
                executor=ExplodingAdapter(ValueError("boom")),
                state_store=state_store,
            )
            planning = PlanningResult(
                issue_number=1,
                title="Fix app",
                body="Adjust workspace execution",
                labels=["bug"],
                updated_at="2026-03-24T00:00:00Z",
                comment="plan",
                triage=AutoCodeTriageResult(
                    task_type="bug_fix",
                    action="implement",
                    risk_level="low",
                    reason="bug",
                    confidence=0.9,
                ),
                plan=ExecutionPlan(
                    goal="Fix app",
                    assumptions=[],
                    acceptance_criteria=["workspace run succeeds"],
                    risk_level="low",
                    estimated_files=["app.py"],
                    suggested_tests=[],
                    needs_human_approval=False,
                    blocked_reasons=[],
                ),
                issue_key="issue:1:test",
            )

            result = controller.execute_issue(
                {"number": 1, "updated_at": "2026-03-24T00:00:00Z"},
                explicit_command=True,
                planning=planning,
            )

            self.assertFalse(result.success)
            self.assertEqual("failed", result.run_record.status)
            self.assertIn("未预期异常", result.comment)
            self.assertIn("ValueError: boom", result.run_record.failure_reason or "")
            latest_run = state_store.latest_run_for_target("issue", 1)
            self.assertIsNotNone(latest_run)
            self.assertEqual("failed", latest_run["status"])

    def test_execute_issue_marks_validation_environment_blocker_as_blocked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            validation_runner = FakeFailedValidationRunner(
                ValidationResult(
                    success=False,
                    steps=[
                        ValidationStepResult(
                            name="lint",
                            command=["./scripts/ci_gate.sh", "flake8"],
                            success=False,
                            output="./scripts/ci_gate.sh: line 15: flake8: command not found",
                        )
                    ],
                )
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                state_store=state_store,
                validation_runner=validation_runner,
            )
            planning = PlanningResult(
                issue_number=1,
                title="Fix app",
                body="Adjust workspace execution",
                labels=["bug"],
                updated_at="2026-03-24T00:00:00Z",
                comment="plan",
                triage=AutoCodeTriageResult(
                    task_type="bug_fix",
                    action="implement",
                    risk_level="low",
                    reason="bug",
                    confidence=0.9,
                ),
                plan=ExecutionPlan(
                    goal="Fix app",
                    assumptions=[],
                    acceptance_criteria=["workspace run succeeds"],
                    risk_level="low",
                    estimated_files=["app.py"],
                    suggested_tests=[],
                    needs_human_approval=False,
                    blocked_reasons=[],
                ),
                issue_key="issue:1:test",
            )

            result = controller.execute_issue(
                {"number": 1, "updated_at": "2026-03-24T00:00:00Z"},
                explicit_command=True,
                planning=planning,
            )

            self.assertFalse(result.success)
            self.assertEqual("blocked", result.run_record.status)
            self.assertIn("本地验证环境未就绪", result.run_record.failure_reason)
            self.assertTrue(state_store.has_processed_issue_version(1, "2026-03-24T00:00:00Z"))
            latest_run = state_store.latest_run_for_target("issue", 1)
            self.assertIsNotNone(latest_run)
            self.assertEqual("blocked", latest_run["status"])

    def test_execute_issue_surfaces_blocked_reason_and_updates_project_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            project_dir = workspace_path / "projects"
            project_dir.mkdir(parents=True, exist_ok=True)
            project_log = project_dir / "example_repo.md"
            project_log.write_text(
                (
                    "# example_repo Execution Log\n\n"
                    "## 已处理记录\n\n"
                    "| 日期 | 类型 | 编号 | 主题 | 动作 | 结果 | 备注 |\n"
                    "| --- | --- | --- | --- | --- | --- | --- |\n"
                    "| 待补 | Issue/PR | - | - | - | - | 这里补 AutoCode 实际处理过的条目 |\n"
                ),
                encoding="utf-8",
            )
            controller = _build_controller(workspace_path=workspace_path)
            controller.config.github.repo = "demo/example_repo"
            controller.config.autocode.project_log_root = str(project_dir)
            controller.policy_engine = SimpleNamespace(
                evaluate_issue=lambda **kwargs: PolicyDecision(
                    allowed=False,
                    requires_approval=True,
                    reasons=["README 信息优先级需要维护者确认"],
                ),
                validate_publish_target=lambda base_branch: PolicyDecision(
                    allowed=True,
                    requires_approval=False,
                    reasons=[],
                ),
            )
            planning = PlanningResult(
                issue_number=883,
                title="[Docs] 精简 README",
                body="需要把细节下沉到文档",
                labels=["documentation"],
                updated_at="2026-03-24T00:00:00Z",
                comment=format_execution_plan_reply(
                    header="## AutoCode Plan",
                    task_type="todo_refactor",
                    action="auto_fix",
                    plan_summary="精简 README 并下沉细节",
                    risk_level="medium",
                    estimated_files=["README.md", "docs/full-guide.md"],
                    acceptance_criteria=["README 更聚焦"],
                    needs_human_approval=False,
                    blocked_reasons=["README 信息优先级需要维护者确认"],
                ),
                triage=AutoCodeTriageResult(
                    task_type="todo_refactor",
                    action="auto_fix",
                    risk_level="medium",
                    reason="docs",
                    confidence=0.9,
                ),
                plan=ExecutionPlan(
                    goal="精简 README 并下沉细节",
                    assumptions=["保留高频入口"],
                    acceptance_criteria=["README 更聚焦"],
                    risk_level="medium",
                    estimated_files=["README.md", "docs/full-guide.md"],
                    suggested_tests=["manual review"],
                    needs_human_approval=False,
                    blocked_reasons=["README 信息优先级需要维护者确认"],
                ),
                issue_key="issue:883:test",
            )

            result = controller.execute_issue(
                {"number": 883, "updated_at": "2026-03-24T00:00:00Z"},
                explicit_command=True,
                planning=planning,
            )

            self.assertFalse(result.success)
            self.assertEqual("blocked", result.run_record.status)
            self.assertIn("README 信息优先级需要维护者确认", result.comment)
            log_text = project_log.read_text(encoding="utf-8")
            self.assertIn("| Issue | 883 |", log_text)
            self.assertIn("README 信息优先级需要维护者确认", log_text)

    def test_apply_review_feedback_updates_project_log(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            project_dir = workspace_path / "projects"
            controller = _build_controller(workspace_path=workspace_path)
            controller.config.github.repo = "demo/example_repo"
            controller.config.autocode.project_log_root = str(project_dir)

            summary = controller._apply_review_feedback(
                _build_pr(11),
                [{"id": 101, "body": "Please fix the logging path", "user": {"login": "reviewer"}}],
            )

            self.assertIn("已根据 review 反馈更新分支", summary)
            project_log = project_dir / "example_repo.md"
            log_text = project_log.read_text(encoding="utf-8")
            self.assertIn("| PR | 11 | PR #11 | review_feedback | updated |", log_text)

    def test_apply_review_feedback_refreshes_pr_body_from_issue_plan_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            gh_writer = FakeGitHubWriteClient()
            controller = _build_controller(
                workspace_path=workspace_path,
                state_store=state_store,
                gh_writer=gh_writer,
            )
            planning = PlanningResult(
                issue_number=877,
                title="修复 INFO 日志泄露 LLM 预览",
                body="修复日志泄露",
                labels=["bug", "autocode"],
                updated_at="2026-03-24T00:00:00Z",
                comment="plan",
                triage=AutoCodeTriageResult(
                    task_type="bug_fix",
                    action="implement",
                    risk_level="low",
                    reason="bug",
                    confidence=0.9,
                ),
                plan=ExecutionPlan(
                    goal="修复日志预览泄露问题",
                    assumptions=[],
                    acceptance_criteria=["日志预览不再泄露敏感字段"],
                    risk_level="low",
                    estimated_files=["src/analyzer.py", "tests/test_analyzer_logging.py"],
                    suggested_tests=["pytest tests/test_analyzer_logging.py"],
                    needs_human_approval=False,
                    blocked_reasons=[],
                ),
                issue_key="issue:877:test",
            )
            state_store.save_issue_plan_snapshot(877, planning.to_snapshot_record())
            pr = _build_pr(11)
            pr["title"] = "fix: 修复 INFO 日志默认输出 LLM prompt/response 预览的问题 (#877)"
            pr["body"] = "## Issue Link\n- `Fixes #877`\n"

            summary = controller._apply_review_feedback(
                pr,
                [{"id": 101, "body": "Please refresh the changed files list", "user": {"login": "reviewer"}}],
            )

            self.assertIn("已根据 review 反馈更新分支", summary)
            self.assertTrue(gh_writer.updated_payloads)
            _, payload = gh_writer.updated_payloads[-1]
            self.assertIn("## Scope Of Change", payload.get("body", ""))
            self.assertIn("Closes #877", payload.get("body", ""))

    def test_apply_review_feedback_refreshes_pr_body_without_code_changes_for_metadata_only_feedback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            existing_pr_body = """## PR Type
- [x] fix

## Scope Of Change
- `src/analyzer.py`
- `tests/test_analyzer_logging.py`

## Verification Commands And Results
```bash
pytest tests/test_analyzer_logging.py
```

关键输出/结论 / Key output & conclusion:
- pytest passed

## Implementation
已完成旧版本修复摘要。
"""
            gh = FakeGitHubClient(
                pr_details={
                    11: {
                        **_build_pr(11),
                        "title": "fix: 修复 INFO 日志默认输出 LLM prompt/response 预览的问题 (#877)",
                        "body": existing_pr_body,
                        "additions": 684,
                        "deletions": 15,
                    }
                },
                pr_files={
                    11: [
                        "docs/CHANGELOG.md",
                        "src/analyzer.py",
                        "src/logging_config.py",
                        "tests/test_analyzer_logging.py",
                    ]
                },
            )
            gh_writer = FakeGitHubWriteClient()
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                state_store=state_store,
                gh_writer=gh_writer,
            )
            controller.workspace_manager.has_changes = mock.Mock(return_value=False)
            planning = PlanningResult(
                issue_number=877,
                title="修复 INFO 日志泄露 LLM 预览",
                body="修复日志泄露",
                labels=["bug", "autocode"],
                updated_at="2026-03-24T00:00:00Z",
                comment="plan",
                triage=AutoCodeTriageResult(
                    task_type="bug_fix",
                    action="implement",
                    risk_level="low",
                    reason="bug",
                    confidence=0.9,
                ),
                plan=ExecutionPlan(
                    goal="修复日志预览泄露问题",
                    assumptions=[],
                    acceptance_criteria=["日志预览不再泄露敏感字段"],
                    risk_level="low",
                    estimated_files=["src/analyzer.py", "tests/test_analyzer_logging.py"],
                    suggested_tests=["pytest tests/test_analyzer_logging.py"],
                    needs_human_approval=False,
                    blocked_reasons=[],
                ),
                issue_key="issue:877:test",
            )
            state_store.save_issue_plan_snapshot(877, planning.to_snapshot_record())
            pr = _build_pr(11)
            pr["title"] = "fix: 修复 INFO 日志默认输出 LLM prompt/response 预览的问题 (#877)"
            pr["body"] = "## Issue Link\n- `Fixes #877`\n"

            summary = controller._apply_review_feedback(
                pr,
                [{"id": 101, "body": "Please refresh the PR description changed files list", "user": {"login": "reviewer"}}],
            )

            self.assertIn("刷新 PR 描述", summary)
            self.assertEqual([], controller.workspace_manager.commit_calls)
            self.assertEqual([], controller.workspace_manager.push_calls)
            self.assertTrue(gh_writer.updated_payloads)
            _, payload = gh_writer.updated_payloads[-1]
            body = payload.get("body", "")
            self.assertIn("## Scope Of Change", body)
            self.assertIn("本次改动涉及 4 个文件", body)
            self.assertIn("Diff 为 `+684 / -15`", body)
            self.assertIn("- `src/logging_config.py`", body)
            self.assertIn("pytest tests/test_analyzer_logging.py", body)
            self.assertNotIn("## Implementation", body)
            self.assertNotIn("已完成旧版本修复摘要", body)
            self.assertNotIn("# no validation commands recorded", body)
            self.assertNotIn("本次改动涉及 0 个文件", body)
            latest_run = state_store.latest_run_for_target("review_feedback", 11)
            self.assertIsNotNone(latest_run)
            assert latest_run is not None
            self.assertEqual("done", latest_run["status"])

    def test_resolve_planning_replans_when_explicit_snapshot_is_stale(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            controller = _build_controller(workspace_path=workspace_path, state_store=state_store)
            stale_planning = PlanningResult(
                issue_number=5,
                title="Old title",
                body="Old body",
                labels=["bug"],
                updated_at="2026-03-24T00:00:00Z",
                comment="old plan",
                triage=AutoCodeTriageResult(
                    task_type="bug_fix",
                    action="implement",
                    risk_level="low",
                    reason="bug",
                    confidence=0.9,
                ),
                plan=ExecutionPlan(
                    goal="Old goal",
                    assumptions=[],
                    acceptance_criteria=["old"],
                    risk_level="low",
                    estimated_files=["app.py"],
                    suggested_tests=[],
                    needs_human_approval=False,
                    blocked_reasons=[],
                ),
                issue_key="issue:5:stale",
            )
            fresh_planning = PlanningResult(
                issue_number=5,
                title="Fresh title",
                body="Fresh body",
                labels=["bug"],
                updated_at="2026-03-25T00:00:00Z",
                comment="fresh plan",
                triage=stale_planning.triage,
                plan=stale_planning.plan,
                issue_key="issue:5:fresh",
            )
            state_store.save_issue_plan_snapshot(5, stale_planning.to_snapshot_record())
            issue = {"number": 5, "updated_at": "2026-03-25T00:00:00Z"}

            with mock.patch.object(controller, "plan_issue", return_value=fresh_planning) as plan_issue:
                planning = controller._resolve_planning(issue, explicit_command=True, provided_planning=None)

            self.assertEqual(fresh_planning, planning)
            plan_issue.assert_called_once_with(issue)

    def test_resolve_planning_reuses_cache_in_auto_flow(self):
        """Auto flow (explicit_command=False) should also reuse cached plan when updated_at matches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            controller = _build_controller(workspace_path=workspace_path, state_store=state_store)
            cached_planning = PlanningResult(
                issue_number=7,
                title="Cached title",
                body="Cached body",
                labels=["enhancement"],
                updated_at="2026-03-25T10:00:00Z",
                comment="cached plan",
                triage=AutoCodeTriageResult(
                    task_type="small_feature",
                    action="implement",
                    risk_level="low",
                    reason="small feature",
                    confidence=0.85,
                ),
                plan=ExecutionPlan(
                    goal="Cached goal",
                    assumptions=[],
                    acceptance_criteria=["cached"],
                    risk_level="low",
                    estimated_files=["app.py"],
                    suggested_tests=[],
                    needs_human_approval=False,
                    blocked_reasons=[],
                ),
                issue_key="issue:7:cached",
            )
            state_store.save_issue_plan_snapshot(7, cached_planning.to_snapshot_record())
            issue = {"number": 7, "updated_at": "2026-03-25T10:00:00Z"}

            with mock.patch.object(controller, "plan_issue") as plan_issue:
                planning = controller._resolve_planning(issue, explicit_command=False, provided_planning=None)

            plan_issue.assert_not_called()
            self.assertEqual(cached_planning.issue_key, planning.issue_key)

    def test_apply_review_feedback_marks_failed_on_unexpected_executor_exception(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            controller = _build_controller(
                workspace_path=workspace_path,
                executor=ExplodingAdapter(ValueError("boom")),
                state_store=state_store,
            )

            with self.assertRaises(RuntimeError) as exc:
                controller._apply_review_feedback(
                    _build_pr(1),
                    [{"id": 11, "body": "Please fix the failing path"}],
                )

            self.assertIn("未预期异常", str(exc.exception))
            latest_run = state_store.latest_run_for_target("review_feedback", 1)
            self.assertIsNotNone(latest_run)
            self.assertEqual("failed", latest_run["status"])
            self.assertIn("ValueError: boom", latest_run["failure_reason"])

    def test_apply_review_feedback_repairs_after_validation_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            executor = SequenceAdapter(["first repair", "second repair"])
            validation_runner = SequenceValidationRunner(
                [
                    ValidationResult(
                        success=False,
                        steps=[
                            ValidationStepResult(
                                name="test",
                                command=["pytest", "tests/test_app.py"],
                                success=False,
                                output="AssertionError: missing review fix",
                            )
                        ],
                    ),
                    ValidationResult(success=True, steps=[]),
                ]
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                executor=executor,
                validation_runner=validation_runner,
            )

            summary = controller._apply_review_feedback(
                _build_pr(11),
                [{"id": 101, "path": "app.py", "body": "Please update app.py to address the review"}],
            )

            self.assertIn("已根据 review 反馈更新分支", summary)
            self.assertEqual(2, validation_runner.calls)
            self.assertEqual(2, len(executor.prompts))
            self.assertIn("review feedback 第 1 轮在 validation 阶段失败", executor.prompts[1])
            self.assertIn("AssertionError: missing review fix", executor.prompts[1])

    def test_documentation_pr_feedback_uses_scope_guard_and_light_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "README.md").write_text("# hello\n", encoding="utf-8")
            executor = SequenceAdapter(["docs fixed"])
            validation_runner = SequenceValidationRunner([])
            patch_inspector = FakePatchInspector()
            gh = FakeGitHubClient(pr_files={1120: ["docs/README_EN.md", "docs/full-guide.md"]})
            controller = _build_controller(
                workspace_path=workspace_path,
                executor=executor,
                gh=gh,
                validation_runner=validation_runner,
                patch_inspector=patch_inspector,
            )
            pr = _build_pr(1120)
            pr["title"] = "feat: streamline README documentation"
            pr["body"] = "Docs only, tests not run."

            with mock.patch.object(
                controller,
                "_run_documentation_only_validation",
                return_value=ValidationResult(success=True, steps=[]),
            ) as doc_validation:
                summary = controller._apply_review_feedback(
                    pr,
                    [{"id": 101, "path": "docs/README_EN.md", "body": "Please fix the README anchor"}],
                )

            self.assertIn("已根据 review 反馈更新分支", summary)
            self.assertEqual(0, validation_runner.calls)
            doc_validation.assert_called_once()
            self.assertTrue(patch_inspector.calls)
            self.assertTrue(patch_inspector.calls[-1]["documentation_only"])
            self.assertEqual(["docs/README_EN.md", "docs/full-guide.md"], patch_inspector.calls[-1]["allowed_paths"])
            self.assertIn("本轮范围硬约束", executor.prompts[0])
            self.assertIn("禁止修改运行时代码", executor.prompts[0])

    def test_documentation_pr_with_existing_code_files_blocks_auto_repair(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            executor = SequenceAdapter(["should not run"])
            gh = FakeGitHubClient(pr_files={1120: ["docs/README_EN.md", "bot/dispatcher.py"]})
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            controller = _build_controller(
                workspace_path=workspace_path,
                executor=executor,
                gh=gh,
                state_store=state_store,
            )
            pr = _build_pr(1120)
            pr["title"] = "feat: streamline README documentation"
            pr["body"] = "Docs only, tests not run."

            with self.assertRaises(RuntimeError) as exc:
                controller._apply_review_feedback(
                    pr,
                    [{"id": 101, "path": "docs/README_EN.md", "body": "Please fix the README anchor"}],
                )

            self.assertIn("文档/治理 PR 已包含非文档改动", str(exc.exception))
            self.assertEqual([], executor.prompts)
            latest_run = state_store.latest_run_for_target("review_feedback", 1120)
            self.assertIsNotNone(latest_run)
            self.assertEqual("failed", latest_run["status"])

    def test_execute_issue_marks_planning_runtime_error_as_failed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            controller = _build_controller(workspace_path=workspace_path)

            with mock.patch.object(controller, "plan_issue", side_effect=RuntimeError("planner offline")):
                result = controller.execute_issue(
                    {"number": 5, "updated_at": "2026-03-25T00:00:00Z"},
                    explicit_command=True,
                )

            self.assertFalse(result.success)
            self.assertEqual("failed", result.run_record.status)
            self.assertEqual("planner offline", result.run_record.failure_reason)

    def test_execute_issue_answers_configuration_issue_without_creating_pr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            planner = SequenceAdapter([
                json.dumps({
                    "classification": "ANSWERABLE",
                    "reason": "配置排查",
                    "confidence": 0.92,
                }),
            ])
            gh = FakeGitHubClient()
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                planner=planner,
            )

            result = controller.execute_issue(
                {
                    "number": 21,
                    "title": "[Bug] The job has exceeded the maximum execution time of 30m0s",
                    "body": "GitHub Actions 提示 The job has exceeded the maximum execution time of 30m0s",
                    "labels": [{"name": "bug"}],
                    "updated_at": "2026-04-01T05:00:00Z",
                },
                explicit_command=False,
            )

            self.assertFalse(result.success)
            self.assertEqual("answered", result.run_record.status)
            self.assertEqual(0, len(gh.posted_comments))
            self.assertEqual(1, len(planner.prompts))

    def test_execute_issue_continues_planning_for_answerable_actionable_bug(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            planner = SequenceAdapter([
                json.dumps({
                    "classification": "ANSWERABLE",
                    "reason": "配置排查",
                    "confidence": 0.93,
                }),
                json.dumps({
                    "triage": {
                        "task_type": "bug_fix",
                        "action": "auto_fix",
                        "risk_level": "low",
                        "reason": "保存缺陷",
                        "confidence": 0.9,
                    },
                    "plan": {
                        "goal": "定位并修复模型配置保存失败的问题",
                        "assumptions": [],
                        "acceptance_criteria": ["保存后配置可正常持久化"],
                        "risk_level": "low",
                        "estimated_files": ["src/config.py", "tests/"],
                        "suggested_tests": ["pytest tests/test_config.py"],
                        "needs_human_approval": False,
                        "blocked_reasons": [],
                    },
                }),
            ])
            gh = FakeGitHubClient()
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                planner=planner,
            )

            result = controller.execute_issue(
                {
                    "number": 2109,
                    "title": "[Bug] Docker 端可检测 Deepseek v4 模型，却无法使用（保存）",
                    "body": "模型列表能检测到，但保存失败，之后无法使用。",
                    "labels": [{"name": "bug"}],
                    "updated_at": "2026-04-24T05:00:00Z",
                },
                explicit_command=False,
            )

            self.assertFalse(result.success)
            self.assertEqual("planned", result.run_record.status)
            self.assertEqual(0, len(gh.posted_comments))
            self.assertEqual(2, len(planner.prompts))

    def test_execute_issue_marks_resolved_confirmation_without_planning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            planner = SequenceAdapter([])
            gh = FakeGitHubClient(
                issue_comment_pages={
                    1: [
                        {
                            "id": 1,
                            "body": "感谢，已经成功了",
                            "user": {"login": "reporter"},
                            "author_association": "NONE",
                        },
                    ],
                }
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                planner=planner,
            )

            result = controller.execute_issue(
                {
                    "number": 2110,
                    "title": "[Bug] 不显示个股分析",
                    "body": "只显示大盘复盘",
                    "labels": [{"name": "bug"}],
                    "updated_at": "2026-04-24T05:10:00Z",
                },
                explicit_command=False,
            )

            self.assertFalse(result.success)
            self.assertEqual("resolved_by_user", result.run_record.status)
            self.assertEqual(0, len(planner.prompts))

    def test_execute_issue_respects_owner_non_pr_guidance_even_when_explicit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            planner = SequenceAdapter([])
            gh = FakeGitHubClient(
                issue_comment_pages={
                    1: [
                        {
                            "id": 1,
                            "body": "伪需求，这个改github配置即可",
                            "user": {"login": "repoowner"},
                            "author_association": "OWNER",
                        },
                        {
                            "id": 2,
                            "body": "收到",
                            "user": {"login": "repoowner"},
                            "author_association": "OWNER",
                        },
                    ],
                }
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                planner=planner,
            )

            result = controller.execute_issue(
                {
                    "number": 22,
                    "title": "[Bug] job timeout",
                    "body": "运行超时",
                    "labels": [{"name": "bug"}],
                    "updated_at": "2026-04-01T05:10:00Z",
                },
                explicit_command=True,
            )

            self.assertFalse(result.success)
            self.assertEqual("manual_action", result.run_record.status)
            self.assertEqual(0, len(gh.posted_comments))
            self.assertEqual(0, len(planner.prompts))

    def test_execute_issue_ignores_stale_owner_non_pr_guidance_after_newer_owner_comment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            planner = SequenceAdapter([
                json.dumps({
                    "triage": {
                        "task_type": "bug_fix",
                        "action": "auto_fix",
                        "risk_level": "low",
                        "reason": "明确缺陷",
                        "confidence": 0.9,
                    },
                    "plan": {
                        "goal": "修复问题",
                        "assumptions": [],
                        "acceptance_criteria": ["问题消失"],
                        "risk_level": "low",
                        "estimated_files": ["app.py"],
                        "suggested_tests": [],
                        "needs_human_approval": False,
                        "blocked_reasons": [],
                    },
                }),
            ])
            gh = FakeGitHubClient(
                issue_comment_pages={
                    1: [
                        {
                            "id": 1,
                            "body": "伪需求，这个改github配置即可",
                            "user": {"login": "repoowner"},
                            "author_association": "OWNER",
                        },
                        {
                            "id": 2,
                            "body": "实现",
                            "user": {"login": "repoowner"},
                            "author_association": "OWNER",
                        },
                        {
                            "id": 3,
                            "body": "收到，记得补测试",
                            "user": {"login": "repoowner"},
                            "author_association": "OWNER",
                        },
                    ],
                }
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                planner=planner,
            )

            result = controller.execute_issue(
                {
                    "number": 222,
                    "title": "[Bug] job timeout",
                    "body": "运行超时",
                    "labels": [{"name": "bug"}],
                    "updated_at": "2026-04-01T05:12:00Z",
                },
                explicit_command=True,
            )

            self.assertTrue(result.success)
            self.assertEqual("pr_opened", result.run_record.status)
            self.assertEqual([], gh.posted_comments)

    def test_execute_issue_promotes_latest_owner_implement_comment_to_explicit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient(
                issue_comment_pages={
                    1: [
                        {
                            "id": 3,
                            "body": "实现",
                            "user": {"login": "repoowner"},
                            "author_association": "OWNER",
                        },
                        {
                            "id": 4,
                            "body": "收到，记得补测试",
                            "user": {"login": "repoowner"},
                            "author_association": "OWNER",
                        },
                    ],
                }
            )
            gh_writer = FakeGitHubWriteClient()
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                gh_writer=gh_writer,
            )
            planning = PlanningResult(
                issue_number=223,
                title="[Feature] add export",
                body="please add export",
                labels=["enhancement"],
                updated_at="2026-04-01T05:20:00Z",
                comment="plan",
                triage=AutoCodeTriageResult(
                    task_type="small_feature",
                    action="implement",
                    risk_level="low",
                    reason="feature approved by owner",
                    confidence=0.9,
                ),
                plan=ExecutionPlan(
                    goal="Add export",
                    assumptions=[],
                    acceptance_criteria=["export action is available"],
                    risk_level="low",
                    estimated_files=["app.py"],
                    suggested_tests=[],
                    needs_human_approval=False,
                    blocked_reasons=[],
                ),
                issue_key="issue:223:test",
            )

            result = controller.execute_issue(
                {
                    "number": 223,
                    "title": "[Feature] add export",
                    "body": "please add export",
                    "labels": [{"name": "enhancement"}],
                    "updated_at": "2026-04-01T05:20:00Z",
                },
                explicit_command=False,
                planning=planning,
            )

            self.assertTrue(result.success)
            self.assertEqual("pr_opened", result.run_record.status)
            self.assertEqual(1, len(gh_writer.created_payloads))

    def test_execute_issue_promotes_semantic_owner_implement_comment_to_explicit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient(
                issue_comment_pages={
                    1: [
                        {
                            "id": 6,
                            "body": "进行实现",
                            "user": {"login": "repoowner", "type": "User"},
                            "author_association": "OWNER",
                        },
                        {
                            "id": 7,
                            "body": (
                                "这个方向可以直接实现，现有代码里已经有合适插入点。\n\n"
                                "<!-- openreview:idempotency:issue:abc123 -->"
                            ),
                            "user": {"login": "repoowner", "type": "User"},
                            "author_association": "OWNER",
                        },
                    ],
                }
            )
            gh_writer = FakeGitHubWriteClient()
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                gh_writer=gh_writer,
            )
            planning = PlanningResult(
                issue_number=223,
                title="[Feature] add export",
                body="please add export",
                labels=["enhancement"],
                updated_at="2026-04-01T05:20:00Z",
                comment="plan",
                triage=AutoCodeTriageResult(
                    task_type="small_feature",
                    action="implement",
                    risk_level="low",
                    reason="feature approved by owner",
                    confidence=0.9,
                ),
                plan=ExecutionPlan(
                    goal="Add export",
                    assumptions=[],
                    acceptance_criteria=["export action is available"],
                    risk_level="low",
                    estimated_files=["app.py"],
                    suggested_tests=[],
                    needs_human_approval=False,
                    blocked_reasons=[],
                ),
                issue_key="issue:223:test",
            )

            result = controller.execute_issue(
                {
                    "number": 223,
                    "title": "[Feature] add export",
                    "body": "please add export",
                    "labels": [{"name": "enhancement"}],
                    "updated_at": "2026-04-01T05:20:00Z",
                },
                explicit_command=False,
                planning=planning,
            )

            self.assertTrue(result.success)
            self.assertEqual("pr_opened", result.run_record.status)
            self.assertEqual(1, len(gh_writer.created_payloads))

    def test_execute_issue_does_not_promote_multiline_conditional_implement_comment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient(
                issue_comment_pages={
                    1: [
                        {
                            "id": 5,
                            "body": "实现\n但记得补测试",
                            "user": {"login": "repoowner"},
                            "author_association": "OWNER",
                        },
                    ],
                }
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
            )
            planning = PlanningResult(
                issue_number=224,
                title="[Feature] add export",
                body="please add export",
                labels=["enhancement"],
                updated_at="2026-04-01T05:21:00Z",
                comment="plan",
                triage=AutoCodeTriageResult(
                    task_type="small_feature",
                    action="implement",
                    risk_level="low",
                    reason="feature pending approval",
                    confidence=0.9,
                ),
                plan=ExecutionPlan(
                    goal="Add export",
                    assumptions=[],
                    acceptance_criteria=["export action is available"],
                    risk_level="low",
                    estimated_files=["app.py"],
                    suggested_tests=[],
                    needs_human_approval=False,
                    blocked_reasons=[],
                ),
                issue_key="issue:224:test",
            )

            result = controller.execute_issue(
                {
                    "number": 224,
                    "title": "[Feature] add export",
                    "body": "please add export",
                    "labels": [{"name": "enhancement"}],
                    "updated_at": "2026-04-01T05:21:00Z",
                },
                explicit_command=False,
                planning=planning,
            )

            self.assertFalse(result.success)
            self.assertEqual("planned", result.run_record.status)

    def test_plan_issue_includes_issue_discussion_in_planner_prompt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            planner = SequenceAdapter([
                json.dumps({
                    "triage": {
                        "task_type": "bug_fix",
                        "action": "auto_fix",
                        "risk_level": "low",
                        "reason": "明确缺陷",
                        "confidence": 0.91,
                    },
                    "plan": {
                        "goal": "修复问题",
                        "assumptions": [],
                        "acceptance_criteria": ["问题消失"],
                        "risk_level": "low",
                        "estimated_files": ["app.py"],
                        "suggested_tests": [],
                        "needs_human_approval": False,
                        "blocked_reasons": [],
                    },
                }),
            ])
            gh = FakeGitHubClient(
                issue_comment_pages={
                    1: [
                        {
                            "id": 2,
                            "body": "修改 `.github/workflows/daily_analysis.yml` 并调整 `timeout-minutes` 即可",
                            "user": {"login": "repoowner"},
                            "author_association": "OWNER",
                        },
                    ],
                }
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                planner=planner,
            )

            controller.plan_issue(
                {
                    "number": 23,
                    "title": "[Bug] job timeout",
                    "body": "运行超时",
                    "labels": [{"name": "bug"}],
                    "updated_at": "2026-04-01T05:20:00Z",
                }
            )

            self.assertTrue(planner.prompts)
            self.assertIn("修改 `.github/workflows/daily_analysis.yml` 并调整 `timeout-minutes` 即可", planner.prompts[0])

    def test_collect_tracked_pull_requests_reads_all_pages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            gh = FakeGitHubClient(
                pull_request_pages={
                    1: [_build_pr(number) for number in range(1, 31)],
                    2: [_build_pr(31)],
                }
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                state_store=state_store,
            )

            prs = controller._collect_tracked_pull_requests()

            self.assertEqual(31, len(prs))
            self.assertEqual(31, prs[-1]["number"])
            tracked = {item["pr_number"]: item for item in state_store.list_tracked_prs()}
            self.assertNotEqual("closed", tracked[31].get("status"))

    def test_bot_review_comments_are_included_in_feedback(self):
        """Bot accounts (copilot, chatgpt-codex-connector) should trigger repairs like human reviewers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 100,
                            "user": {"login": "copilot-pull-request-reviewer[bot]"},
                            "state": "COMMENTED",
                            "body": "## Pull request overview\nSome suggestions...",
                            "submitted_at": "2026-03-29T01:00:00Z",
                        },
                    ],
                },
                review_comment_pages={
                    1: [
                        {
                            "id": 200,
                            "user": {"login": "chatgpt-codex-connector[bot]"},
                            "pull_request_review_id": 100,
                            "body": "Guard non-string link",
                            "path": "src/app.py",
                            "line": 10,
                            "updated_at": "2026-03-29T01:00:00Z",
                        },
                    ],
                },
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                state_store=state_store,
            )
            feedback = controller._collect_pr_feedback(1, pr_author_login="autocode[bot]")
            self.assertGreater(len(feedback), 0)

    def test_latest_openreview_feedback_drops_stale_bot_comments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 100,
                            "user": {"login": "chatgpt-codex-connector[bot]"},
                            "state": "COMMENTED",
                            "body": "### 💡 Codex Review",
                            "submitted_at": "2026-03-29T01:00:00Z",
                        },
                        {
                            "id": 101,
                            "user": {"login": "repoowner"},
                            "state": "COMMENTED",
                            "body": (
                                "**评审结论**\n"
                                "- **是否可直接合入**：不可\n"
                                "\n"
                                "**主要问题**\n"
                                "1. [Correctness blocker] 只处理最新 blocker。\n"
                                "\n"
                                "<!-- openreview:idempotency:pr-review:latest -->"
                            ),
                            "submitted_at": "2026-03-29T02:00:00Z",
                        },
                    ],
                },
                review_comment_pages={
                    1: [
                        {
                            "id": 200,
                            "user": {"login": "chatgpt-codex-connector[bot]"},
                            "pull_request_review_id": 100,
                            "body": "Older bot-only inline note",
                            "path": "src/app.py",
                            "line": 10,
                            "updated_at": "2026-03-29T01:00:00Z",
                        },
                    ],
                },
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                state_store=state_store,
            )

            feedback = controller._collect_pr_feedback(1, pr_author_login="reviewer")

            self.assertEqual([101], [item.get("id") for item in feedback])

    def test_repair_limit_blocks_further_repairs(self):
        """PR should stop auto-repairing after max repair rounds with bot-only feedback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 300,
                            "user": {"login": "copilot-pull-request-reviewer"},
                            "state": "COMMENTED",
                            "body": "请修复这个问题",
                            "submitted_at": "2026-03-29T02:00:00Z",
                        },
                    ],
                },
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                state_store=state_store,
            )
            # Override max repair rounds to 2
            controller.config.autocode.pr_max_repair_rounds = 2
            pr = _build_pr(10)
            # Simulate 2 completed repairs
            state_store.upsert_tracked_pr(
                pr_number=10, branch_name="autocode/issue-10", base_ref="main",
                head_sha="abc", author_login="autocode[bot]", pr_url="", queue_source="autocode",
            )
            state_store.increment_tracked_pr_repair_count(10)
            state_store.increment_tracked_pr_repair_count(10)

            summary, candidate = controller.plan_pull_request_repair(pr)

            self.assertIsNone(candidate)
            self.assertIn("最大修复轮数", summary)
            tracked = state_store.get_tracked_pr(10)
            self.assertEqual("repair_limit_reached", tracked["status"])

    def test_plan_pull_request_repair_returns_candidate_when_feedback_is_pending(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 100,
                            "body": "please update this",
                            "state": "CHANGES_REQUESTED",
                            "updated_at": "2026-03-24T01:00:00Z",
                            "user": {"login": "reviewer"},
                        }
                    ]
                }
            )
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            controller = _build_controller(workspace_path=workspace_path, gh=gh, state_store=state_store)

            summary, candidate = controller.plan_pull_request_repair(_build_pr(11))

            self.assertIn("已进入修复队列", summary)
            self.assertIsNotNone(candidate)
            assert candidate is not None
            self.assertEqual(11, candidate.pr_number)
            self.assertTrue(candidate.task_id.startswith("pr-task:pr:11:"))
            tracked = state_store.get_tracked_pr(11)
            self.assertIsNotNone(tracked)
            assert tracked is not None
            self.assertEqual("review_feedback_pending", tracked["status"])
            self.assertEqual("inflight", state_store.try_claim_feedback(candidate.feedback_key))

    def test_plan_pull_request_repair_blocks_when_owner_says_config_change_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 100,
                            "body": "伪需求，这个改github配置即可",
                            "state": "COMMENTED",
                            "updated_at": "2026-04-01T05:30:00Z",
                            "user": {"login": "repoowner"},
                        }
                    ]
                }
            )
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            controller = _build_controller(workspace_path=workspace_path, gh=gh, state_store=state_store)

            summary, candidate = controller.plan_pull_request_repair(_build_pr(24))

            self.assertIsNone(candidate)
            self.assertIn("停止继续自动改代码", summary)
            tracked = state_store.get_tracked_pr(24)
            self.assertIsNotNone(tracked)
            assert tracked is not None
            self.assertEqual("blocked", tracked["status"])
            self.assertEqual("2026-04-01T05:30:00Z", str(tracked.get("last_feedback_updated_at", "") or ""))
            self.assertEqual([100], tracked.get("last_feedback_item_ids", []))

    def test_plan_pull_request_repair_does_not_treat_openreview_marker_as_owner_stop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 100,
                            "body": (
                                "**评审结论**\n"
                                "- **是否可直接合入**：不可\n"
                                "1. [Correctness blocker] PR 描述仍缺少 Rollback Plan，请补充。\n"
                                "说明：无需修改 src 代码，只修 PR 描述。\n"
                                "---\n"
                                "<!-- openreview:idempotency:pr-review:abc123 -->"
                            ),
                            "state": "COMMENTED",
                            "updated_at": "2026-04-01T05:30:00Z",
                            "user": {"login": "repoowner"},
                        }
                    ]
                }
            )
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            controller = _build_controller(workspace_path=workspace_path, gh=gh, state_store=state_store)

            summary, candidate = controller.plan_pull_request_repair(_build_pr(24))

            self.assertIsNotNone(candidate)
            self.assertIn("已进入修复队列", summary)
            self.assertNotIn("停止继续自动改代码", summary)

    def test_plan_pull_request_repair_does_not_treat_structured_owner_review_as_stop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 100,
                            "body": (
                                "**评审结论**\n"
                                "- **是否可直接合入**：不可 + 需要先补齐兼容性边界。\n"
                                "\n"
                                "**主要问题**\n"
                                "1. [Compatibility blocker] 当前实现还需要处理 non-thinking 模式。\n"
                                "说明：这里不需要复杂预案，按最小代码路径修复即可。"
                            ),
                            "state": "COMMENTED",
                            "updated_at": "2026-04-01T05:30:00Z",
                            "user": {"login": "repoowner"},
                        }
                    ]
                }
            )
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            controller = _build_controller(workspace_path=workspace_path, gh=gh, state_store=state_store)

            summary, candidate = controller.plan_pull_request_repair(_build_pr(24))

            self.assertIsNotNone(candidate)
            self.assertIn("已进入修复队列", summary)
            self.assertNotIn("停止继续自动改代码", summary)

    def test_plan_pull_request_repair_ignores_stale_owner_redirect_after_newer_owner_feedback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 100,
                            "body": "伪需求，这个改github配置即可",
                            "state": "COMMENTED",
                            "updated_at": "2026-04-01T05:30:00Z",
                            "user": {"login": "repoowner"},
                        },
                        {
                            "id": 101,
                            "body": "请修复这个回归问题",
                            "state": "COMMENTED",
                            "updated_at": "2026-04-01T05:40:00Z",
                            "user": {"login": "repoowner"},
                        },
                    ]
                }
            )
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            controller = _build_controller(workspace_path=workspace_path, gh=gh, state_store=state_store)

            summary, candidate = controller.plan_pull_request_repair(_build_pr(25))

            self.assertIsNotNone(candidate)
            self.assertIn("已进入修复队列", summary)

    def test_plan_pull_request_repair_returns_none_when_no_new_feedback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient()
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            controller = _build_controller(workspace_path=workspace_path, gh=gh, state_store=state_store)

            summary, candidate = controller.plan_pull_request_repair(_build_pr(12))

            self.assertIn("暂无新的可执行检视意见", summary)
            self.assertIsNone(candidate)

    def test_plan_pull_request_repair_returns_none_when_pr_repair_is_disabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 100,
                            "body": "please update this",
                            "state": "CHANGES_REQUESTED",
                            "updated_at": "2026-03-24T01:00:00Z",
                            "user": {"login": "reviewer"},
                        }
                    ]
                }
            )
            controller = _build_controller(workspace_path=workspace_path, gh=gh)
            controller.config.autocode.enable_pr_queue = False
            controller.config.autocode.poll_review_feedback = False

            summary, candidate = controller.plan_pull_request_repair(_build_pr(11))

            self.assertIn("不在 AutoCode 跟踪范围", summary)
            self.assertIsNone(candidate)

    def test_plan_pull_request_repair_ignores_feedback_when_latest_review_is_approved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 100,
                            "body": "please update this",
                            "state": "CHANGES_REQUESTED",
                            "updated_at": "2026-03-24T01:00:00Z",
                            "user": {"login": "reviewer"},
                        },
                        {
                            "id": 101,
                            "body": "looks good now",
                            "state": "APPROVED",
                            "updated_at": "2026-03-24T02:00:00Z",
                            "user": {"login": "reviewer"},
                        },
                    ]
                }
            )
            controller = _build_controller(workspace_path=workspace_path, gh=gh)

            summary, candidate = controller.plan_pull_request_repair(_build_pr(11))

            self.assertIn("已通过最新 review", summary)
            self.assertIsNone(candidate)

    def test_plan_pull_request_repair_ignores_feedback_when_latest_review_is_dismissed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 100,
                            "body": "please update this",
                            "state": "CHANGES_REQUESTED",
                            "updated_at": "2026-03-24T01:00:00Z",
                            "user": {"login": "reviewer"},
                        },
                        {
                            "id": 101,
                            "body": "dismissed later",
                            "state": "DISMISSED",
                            "updated_at": "2026-03-24T02:00:00Z",
                            "user": {"login": "reviewer"},
                        },
                    ]
                }
            )
            controller = _build_controller(workspace_path=workspace_path, gh=gh)

            summary, candidate = controller.plan_pull_request_repair(_build_pr(11))

            self.assertIn("暂无新的可执行检视意见", summary)
            self.assertIsNone(candidate)

    def test_process_pull_request_returns_paused_summary_when_pr_is_paused(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            state_store.set_tracked_pr_paused(11, paused=True, reason="manual")
            controller = _build_controller(workspace_path=workspace_path, state_store=state_store)

            summary = controller.process_pull_request(_build_pr(11))

            self.assertIn("已暂停自动修复", summary)
            tracked = state_store.get_tracked_pr(11)
            self.assertIsNotNone(tracked)
            assert tracked is not None
            self.assertEqual("paused", tracked["status"])

    def test_plan_pull_request_repair_blocks_default_branch_not_named_main(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 100,
                            "body": "please update this",
                            "state": "CHANGES_REQUESTED",
                            "updated_at": "2026-03-24T01:00:00Z",
                            "user": {"login": "reviewer"},
                        }
                    ]
                }
            )
            controller = _build_controller(workspace_path=workspace_path, gh=gh)
            controller.config.local_repo.default_branch = "master"
            pr = _build_pr(11)
            pr["base"]["ref"] = "master"

            summary, candidate = controller.plan_pull_request_repair(pr)

            self.assertIn("base=master", summary)
            self.assertIsNone(candidate)

    def test_process_pull_request_advances_cursor_even_when_repair_fails(self):
        """Cursor should advance on failure to prevent infinite retry of the same feedback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 100,
                            "body": "please update this",
                            "state": "CHANGES_REQUESTED",
                            "updated_at": "2026-03-24T01:00:00Z",
                            "user": {"login": "reviewer"},
                        }
                    ]
                }
            )
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            controller = _build_controller(workspace_path=workspace_path, gh=gh, state_store=state_store)
            controller.workspace_manager.has_changes = lambda workspace_path: False

            with self.assertRaises(RuntimeError) as ctx:
                controller.process_pull_request(_build_pr(11))

            self.assertIn("review 反馈处理失败", str(ctx.exception))
            tracked = state_store.get_tracked_pr(11)
            self.assertIsNotNone(tracked)
            assert tracked is not None
            self.assertEqual("failed", tracked["status"])
            # Cursor should be advanced even on failure to prevent same feedback from retrying
            self.assertEqual("2026-03-24T01:00:00Z", str(tracked.get("last_feedback_updated_at", "") or ""))
            self.assertEqual([100], tracked.get("last_feedback_item_ids", []))

    def test_process_pull_request_ignores_current_running_feedback_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            review = {
                "id": 100,
                "body": "please update this",
                "state": "CHANGES_REQUESTED",
                "updated_at": "2026-03-24T01:00:00Z",
                "user": {"login": "reviewer"},
            }
            gh = FakeGitHubClient(review_pages={1: [review]})
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            controller = _build_controller(workspace_path=workspace_path, gh=gh, state_store=state_store)
            feedback_key = build_feedback_key(
                11,
                [100],
                ["2026-03-24T01:00:00Z"],
                item_fingerprints=[build_feedback_item_fingerprint(review)],
            )
            task_id = f"pr-task:{feedback_key}"
            state_store.enqueue_task(
                task_id=task_id,
                task_type="pr_feedback_repair",
                target_number=11,
                source="polling",
                command="review_feedback",
                payload={"feedback_key": feedback_key},
                max_attempts=3,
            )
            claimed = state_store.claim_next_task(task_type="pr_feedback_repair", owner="test")
            self.assertIsNotNone(claimed)
            state_store.try_claim_feedback(feedback_key, owner=task_id)

            summary = controller.process_pull_request(
                _build_pr(11),
                current_task_id=task_id,
                current_feedback_key=feedback_key,
            )

            self.assertIn("已根据 review 反馈更新分支", summary)
            self.assertEqual(["autocode/issue-11-demo"], controller.workspace_manager.push_calls)
            self.assertEqual("processed", state_store.try_claim_feedback(feedback_key))

    def test_feedback_pending_normalizes_iso_timestamps_before_comparing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            controller = _build_controller(workspace_path=workspace_path)

            item = {
                "id": 101,
                "updated_at": "2026-03-28T10:00:00Z",
            }

            self.assertFalse(
                controller._is_feedback_item_pending(
                    item,
                    after_updated_at="2026-03-28T10:00:00+00:00",
                    after_ids={101},
                )
            )
            self.assertTrue(
                controller._is_feedback_item_pending(
                    item,
                    after_updated_at="2026-03-28T10:00:00+00:00",
                    after_ids=set(),
                )
            )

    def test_plan_pull_request_repair_respects_legacy_processed_feedback_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 100,
                            "body": "please update this",
                            "state": "CHANGES_REQUESTED",
                            "updated_at": "2026-03-24T01:00:00Z",
                            "user": {"login": "reviewer"},
                        }
                    ]
                }
            )
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            legacy_key = build_feedback_key(11, [100], ["2026-03-24T01:00:00Z"])
            state_store.mark_feedback_processed(
                feedback_key=legacy_key,
                pr_number=11,
                summary="already handled",
            )
            controller = _build_controller(workspace_path=workspace_path, gh=gh, state_store=state_store)

            summary, candidate = controller.plan_pull_request_repair(_build_pr(11))

            self.assertIsNone(candidate)
            self.assertIn("最新检视意见已处理", summary)

    def test_plan_pull_request_repair_respects_legacy_key_for_submitted_at_only_review(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 100,
                            "body": "please update this",
                            "state": "CHANGES_REQUESTED",
                            "submitted_at": "2026-03-24T01:00:00Z",
                            "user": {"login": "reviewer"},
                        }
                    ]
                }
            )
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            legacy_key = build_feedback_key(11, [100], [""])
            state_store.mark_feedback_processed(
                feedback_key=legacy_key,
                pr_number=11,
                summary="already handled",
            )
            controller = _build_controller(workspace_path=workspace_path, gh=gh, state_store=state_store)

            summary, candidate = controller.plan_pull_request_repair(_build_pr(11))

            self.assertIsNone(candidate)
            self.assertIn("最新检视意见已处理", summary)

    def test_collect_pr_feedback_reads_all_review_pages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            review_page_one = [
                {
                    "id": number,
                    "body": f"review-{number}",
                    "state": "COMMENTED",
                    "updated_at": "2026-03-24T00:00:00Z",
                    "user": {"login": "reviewer-a"},
                }
                for number in range(1, 101)
            ]
            review_comment_page_one = [
                {
                    "id": number + 200,
                    "body": f"line-{number}",
                    "path": "app.py",
                    "updated_at": "2026-03-24T00:00:00Z",
                    "pull_request_review_id": 100,
                    "user": {"login": "reviewer-a"},
                }
                for number in range(1, 101)
            ]
            gh = FakeGitHubClient(
                review_pages={
                    1: review_page_one,
                    2: [
                        {
                            "id": 101,
                            "body": "review-page-2",
                            "state": "CHANGES_REQUESTED",
                            "updated_at": "2026-03-24T00:00:00Z",
                            "user": {"login": "reviewer-b"},
                        }
                    ],
                },
                review_comment_pages={
                    1: review_comment_page_one,
                    2: [
                        {
                            "id": 301,
                            "body": "line-page-2",
                            "path": "app.py",
                            "updated_at": "2026-03-24T00:00:00Z",
                            "pull_request_review_id": 101,
                            "user": {"login": "reviewer-b"},
                        }
                    ],
                },
            )
            controller = _build_controller(workspace_path=workspace_path, gh=gh)

            feedback = controller._collect_pr_feedback(88)

            bodies = {item["body"] for item in feedback}
            self.assertIn("review-page-2", bodies)
            self.assertIn("line-page-2", bodies)
            self.assertIn("line-100", bodies)
            self.assertNotIn("review-1", bodies)
            self.assertEqual(103, len(feedback))

    def test_collect_pr_feedback_only_returns_items_newer_than_cursor(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 100,
                            "body": "old review",
                            "state": "CHANGES_REQUESTED",
                            "updated_at": "2026-03-24T00:00:00Z",
                            "user": {"login": "reviewer"},
                        },
                        {
                            "id": 101,
                            "body": "new review",
                            "state": "CHANGES_REQUESTED",
                            "updated_at": "2026-03-24T01:00:00Z",
                            "user": {"login": "reviewer"},
                        },
                    ]
                },
                review_comment_pages={
                    1: [
                        {
                            "id": 201,
                            "body": "old line",
                            "path": "app.py",
                            "pull_request_review_id": 100,
                            "updated_at": "2026-03-24T00:00:00Z",
                            "user": {"login": "reviewer"},
                        },
                        {
                            "id": 202,
                            "body": "new line",
                            "path": "app.py",
                            "pull_request_review_id": 101,
                            "updated_at": "2026-03-24T01:00:00Z",
                            "user": {"login": "reviewer"},
                        },
                    ]
                },
            )
            controller = _build_controller(workspace_path=workspace_path, gh=gh)

            feedback = controller._collect_pr_feedback(
                88,
                after_updated_at="2026-03-24T00:00:00Z",
                after_item_ids=[100, 201],
            )

            bodies = {item["body"] for item in feedback}
            self.assertEqual({"new review", "new line"}, bodies)

    def test_collect_pr_feedback_keeps_multiple_standalone_comments_on_same_line(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient(
                review_comment_pages={
                    1: [
                        {
                            "id": 201,
                            "body": "rename this field",
                            "path": "app.py",
                            "line": 10,
                            "side": "RIGHT",
                            "updated_at": "2026-03-24T00:00:00Z",
                            "user": {"login": "reviewer"},
                        },
                        {
                            "id": 202,
                            "body": "also add a test",
                            "path": "app.py",
                            "line": 10,
                            "side": "RIGHT",
                            "updated_at": "2026-03-24T01:00:00Z",
                            "user": {"login": "reviewer"},
                        },
                    ]
                }
            )
            controller = _build_controller(workspace_path=workspace_path, gh=gh)

            feedback = controller._collect_pr_feedback(88)

            self.assertEqual(["rename this field", "also add a test"], [item["body"] for item in feedback])

    def test_collect_pr_feedback_ignores_non_actionable_self_authored_review_bodies_and_outdated_comments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 100,
                            "body": "ready to merge",
                            "state": "COMMENTED",
                            "updated_at": "2026-03-24T01:00:00Z",
                            "user": {"login": "author"},
                        },
                        {
                            "id": 101,
                            "body": "### 💡 Codex Review Here are some automated review suggestions for this pull request.",
                            "state": "COMMENTED",
                            "updated_at": "2026-03-24T01:10:00Z",
                            "user": {"login": "chatgpt-codex-connector[bot]"},
                        },
                        {
                            "id": 102,
                            "body": "please update this",
                            "state": "COMMENTED",
                            "updated_at": "2026-03-24T01:20:00Z",
                            "user": {"login": "reviewer"},
                        },
                    ]
                },
                review_comment_pages={
                    1: [
                        {
                            "id": 201,
                            "body": "outdated line comment",
                            "path": "app.py",
                            "line": None,
                            "original_line": 12,
                            "pull_request_review_id": 102,
                            "updated_at": "2026-03-24T01:21:00Z",
                            "user": {"login": "reviewer"},
                        },
                        {
                            "id": 202,
                            "body": "current line comment",
                            "path": "app.py",
                            "line": 13,
                            "pull_request_review_id": 102,
                            "updated_at": "2026-03-24T01:22:00Z",
                            "user": {"login": "reviewer"},
                        },
                        {
                            "id": 203,
                            "body": "please keep this self review line comment",
                            "path": "app.py",
                            "line": 15,
                            "pull_request_review_id": 100,
                            "updated_at": "2026-03-24T01:23:00Z",
                            "user": {"login": "author"},
                        },
                    ]
                },
            )
            controller = _build_controller(workspace_path=workspace_path, gh=gh)

            feedback = controller._collect_pr_feedback(88, pr_author_login="author")

            self.assertEqual(
                [
                    "please update this",
                    "current line comment",
                    "please keep this self review line comment",
                ],
                [item["body"] for item in feedback],
            )

    def test_collect_pr_feedback_keeps_actionable_self_authored_review_body(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 100,
                            "body": (
                                "**评审结论**\n"
                                "- **是否可直接合入**：不可 + 需要先补齐 `src/core/pipeline.py`"
                                " 对 `SH600519` / `000001.SZ` 的处理。\n\n"
                                "**主要问题**\n"
                                "1. `[Correctness blocker]` `src/core/pipeline.py` 当前仍遗漏已支持代码格式。"
                            ),
                            "state": "COMMENTED",
                            "updated_at": "2026-03-24T01:00:00Z",
                            "user": {"login": "author"},
                        }
                    ]
                }
            )
            controller = _build_controller(workspace_path=workspace_path, gh=gh)

            feedback = controller._collect_pr_feedback(88, pr_author_login="author")

            self.assertEqual(1, len(feedback))
            self.assertIn("不可", feedback[0]["body"])

    def test_plan_pull_request_repair_ignores_non_actionable_structured_review_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 100,
                            "body": (
                                "**评审结论**\n"
                                "- **必要性**：通过 + 修复目标成立。\n"
                                "- **是否有对应 issue**：通过。\n"
                                "- **PR 类型**：通过。\n"
                                "- **description 完整性**：通过。\n"
                                "- **是否可直接合入**：通过。"
                            ),
                            "state": "COMMENTED",
                            "updated_at": "2026-03-24T01:00:00Z",
                            "user": {"login": "reviewer"},
                        }
                    ]
                }
            )
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            controller = _build_controller(workspace_path=workspace_path, gh=gh, state_store=state_store)

            summary, candidate = controller.plan_pull_request_repair(_build_pr(11))

            self.assertIn("暂无新的可执行检视意见", summary)
            self.assertIsNone(candidate)

    def test_plan_pull_request_repair_returns_candidate_for_actionable_self_authored_review_body(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 100,
                            "body": (
                                "**评审结论**\n"
                                "- **是否可直接合入**：不可 + 需要先修复 `src/core/pipeline.py`。\n\n"
                                "**主要问题**\n"
                                "1. `[Correctness blocker]` `src/core/pipeline.py` 仍遗漏合法股票代码格式。"
                            ),
                            "state": "COMMENTED",
                            "updated_at": "2026-03-24T01:00:00Z",
                            "user": {"login": "author"},
                        }
                    ]
                }
            )
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            controller = _build_controller(workspace_path=workspace_path, gh=gh, state_store=state_store)
            pr = _build_pr(11)
            pr["user"] = {"login": "author"}

            summary, candidate = controller.plan_pull_request_repair(pr)

            self.assertIn("已进入修复队列", summary)
            self.assertIsNotNone(candidate)
            assert candidate is not None
            self.assertEqual([100], candidate.feedback_item_ids)

    def test_feedback_text_includes_line_location(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            controller = _build_controller(workspace_path=workspace_path)

            text = controller._feedback_text(
                {
                    "body": "rename variable",
                    "path": "app.py",
                    "original_line": 30,
                    "side": "RIGHT",
                }
            )

            self.assertEqual("app.py:30 (RIGHT): rename variable", text)

    def test_process_pull_request_dry_run_skips_commit_push_and_marks_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 100,
                            "body": "please update this",
                            "state": "CHANGES_REQUESTED",
                            "updated_at": "2026-03-24T01:00:00Z",
                            "user": {"login": "reviewer"},
                        }
                    ]
                }
            )
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            controller = _build_controller(workspace_path=workspace_path, gh=gh, state_store=state_store, dry_run=True)

            summary = controller.process_pull_request(_build_pr(11))

            self.assertIn("dry-run", summary)
            self.assertEqual([], controller.workspace_manager.commit_calls)
            self.assertEqual([], controller.workspace_manager.push_calls)
            tracked = state_store.get_tracked_pr(11)
            self.assertIsNotNone(tracked)
            assert tracked is not None
            self.assertEqual("dry_run", tracked["status"])

    def test_plan_pull_request_repair_returns_dead_letter_summary_for_existing_feedback_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            review = {
                "id": 100,
                "body": "please update this",
                "state": "CHANGES_REQUESTED",
                "updated_at": "2026-03-24T01:00:00Z",
                "user": {"login": "reviewer"},
            }
            gh = FakeGitHubClient(review_pages={1: [review]})
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            feedback_key = build_feedback_key(
                11,
                [100],
                ["2026-03-24T01:00:00Z"],
                item_fingerprints=[build_feedback_item_fingerprint(review)],
            )
            task_id = f"pr-task:{feedback_key}"
            state_store.enqueue_task(
                task_id=task_id,
                task_type="pr_feedback_repair",
                target_number=11,
                source="polling",
                command="review_feedback",
                payload={"feedback_key": feedback_key},
                max_attempts=3,
            )
            for _ in range(3):
                claimed = state_store.claim_next_task(task_type="pr_feedback_repair", owner="test")
                self.assertIsNotNone(claimed)
                state_store.mark_task_failed(task_id, error="boom", retry_delay_seconds=0)

            controller = _build_controller(workspace_path=workspace_path, gh=gh, state_store=state_store)

            summary, candidate = controller.plan_pull_request_repair(_build_pr(11))

            self.assertIsNone(candidate)
            self.assertIn("最大自动修复重试次数", summary)
            tracked = state_store.get_tracked_pr(11)
            self.assertIsNotNone(tracked)
            assert tracked is not None
            self.assertEqual("failed", tracked["status"])

    def test_plan_pull_request_repair_skips_when_same_pr_has_active_repair_task(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            review = {
                "id": 100,
                "body": "please update this",
                "state": "CHANGES_REQUESTED",
                "updated_at": "2026-03-24T01:00:00Z",
                "user": {"login": "reviewer"},
            }
            gh = FakeGitHubClient(review_pages={1: [review]})
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            state_store.enqueue_task(
                task_id="pr-task:existing",
                task_type="pr_feedback_repair",
                target_number=11,
                source="polling",
                command="review_feedback",
                payload={},
                max_attempts=3,
            )
            controller = _build_controller(workspace_path=workspace_path, gh=gh, state_store=state_store)

            summary, candidate = controller.plan_pull_request_repair(_build_pr(11))

            self.assertIsNone(candidate)
            self.assertIn("已有 review feedback 修复任务", summary)
            self.assertEqual(1, len(state_store.list_tasks(task_type="pr_feedback_repair")))

    def test_execute_issue_marks_draft_pr_ready_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print(\'hello\')\n", encoding="utf-8")
            gh_writer = FakeGitHubWriteClient()
            controller = _build_controller(
                workspace_path=workspace_path,
                gh_writer=gh_writer,
            )
            controller.config.autocode.auto_ready_for_review = True
            controller.config.autocode.force_draft_pr = True
            planning = PlanningResult(
                issue_number=1,
                title="Fix app",
                body="Adjust workspace execution",
                labels=["bug"],
                updated_at="2026-03-24T00:00:00Z",
                comment="plan",
                triage=AutoCodeTriageResult(
                    task_type="bug_fix",
                    action="implement",
                    risk_level="low",
                    reason="bug",
                    confidence=0.9,
                ),
                plan=ExecutionPlan(
                    goal="Fix app",
                    assumptions=[],
                    acceptance_criteria=["workspace run succeeds"],
                    risk_level="low",
                    estimated_files=["app.py"],
                    suggested_tests=[],
                    needs_human_approval=False,
                    blocked_reasons=[],
                ),
                issue_key="issue:1:test",
            )

            result = controller.execute_issue(
                {"number": 1, "updated_at": "2026-03-24T00:00:00Z"},
                explicit_command=True,
                planning=planning,
            )

            self.assertTrue(result.success)
            self.assertEqual([77], gh_writer.ready_calls)
            self.assertEqual("pr_ready", result.run_record.status)
            self.assertIn("ready for review", result.comment)


    def test_process_tracked_pr_blocks_when_repair_limit_reached(self):
        """Bot-only feedback should still be blocked by pr_max_repair_rounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 500,
                            "user": {"login": "copilot-pull-request-reviewer"},
                            "state": "COMMENTED",
                            "body": "请修复这个问题",
                            "submitted_at": "2026-03-29T03:00:00Z",
                        },
                    ],
                },
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                state_store=state_store,
            )
            controller.config.autocode.pr_max_repair_rounds = 2
            pr = _build_pr(20)
            state_store.upsert_tracked_pr(
                pr_number=20, branch_name="autocode/issue-20", base_ref="main",
                head_sha="abc", author_login="autocode[bot]", pr_url="", queue_source="autocode",
            )
            state_store.increment_tracked_pr_repair_count(20)
            state_store.increment_tracked_pr_repair_count(20)

            summary, consumed = controller._process_tracked_pull_request(pr, allow_repair=True)

            self.assertIn("最大修复轮数", summary)
            self.assertFalse(consumed)
            tracked = state_store.get_tracked_pr(20)
            self.assertEqual("repair_limit_reached", tracked["status"])

    def test_repair_limit_auto_resets_on_human_feedback(self):
        """Human feedback should auto-reset repair count when at limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 501,
                            "user": {"login": "human-reviewer"},
                            "state": "CHANGES_REQUESTED",
                            "body": "这里有个 bug 需要修复",
                            "submitted_at": "2026-03-29T03:00:00Z",
                        },
                    ],
                },
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                state_store=state_store,
            )
            controller.config.autocode.pr_max_repair_rounds = 2
            pr = _build_pr(20)
            state_store.upsert_tracked_pr(
                pr_number=20, branch_name="autocode/issue-20", base_ref="main",
                head_sha="abc", author_login="autocode[bot]", pr_url="", queue_source="autocode",
            )
            state_store.increment_tracked_pr_repair_count(20)
            state_store.increment_tracked_pr_repair_count(20)
            self.assertEqual(2, state_store.get_tracked_pr_repair_count(20))

            summary, consumed = controller._process_tracked_pull_request(pr, allow_repair=True)

            # Should NOT be blocked — human feedback triggers auto-reset
            self.assertNotIn("最大修复轮数", summary)
            # Count was reset from 2→0, then incremented by the repair execution
            self.assertLess(state_store.get_tracked_pr_repair_count(20), 2)

    def test_openreview_approved_skips_bot_feedback(self):
        """When OpenReview approved, bot-only feedback should be skipped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 601,
                            "user": {"login": "chatgpt-codex-connector[bot]"},
                            "state": "COMMENTED",
                            "body": "### Codex Review\nSome suggestion",
                            "submitted_at": "2026-03-31T02:00:00Z",
                        },
                        {
                            "id": 602,
                            "user": {"login": "maintainer"},
                            "state": "COMMENTED",
                            "body": (
                                "**评审结论**\n"
                                "- **是否可直接合入**：可\n"
                                "---\n"
                                "<!-- openreview:idempotency:pr-review:abc123 -->"
                            ),
                            "submitted_at": "2026-03-31T05:00:00Z",
                        },
                    ],
                },
                review_comment_pages={
                    1: [
                        {
                            "id": 701,
                            "user": {"login": "chatgpt-codex-connector[bot]"},
                            "body": "P1: Fix this",
                            "path": "docs/CHANGELOG.md",
                            "pull_request_review_id": 601,
                            "updated_at": "2026-03-31T02:00:00Z",
                        },
                    ],
                },
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                state_store=state_store,
            )
            pr = _build_pr(25)
            state_store.upsert_tracked_pr(
                pr_number=25, branch_name="autocode/issue-25", base_ref="main",
                head_sha="abc", author_login="autocode[bot]", pr_url="", queue_source="autocode",
            )

            summary, candidate = controller.plan_pull_request_repair(pr)

            self.assertIn("OpenReview 审批", summary)
            self.assertIsNone(candidate)
            tracked = state_store.get_tracked_pr(25) or {}
            self.assertEqual("approved", tracked.get("status"))

    def test_openreview_rejected_allows_repair(self):
        """When OpenReview says '不可', feedback should still trigger repair."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 611,
                            "user": {"login": "maintainer"},
                            "state": "COMMENTED",
                            "body": (
                                "**评审结论**\n"
                                "- **是否可直接合入**：不可\n"
                                "1. [Correctness blocker] Fix the bug\n"
                                "---\n"
                                "<!-- openreview:idempotency:pr-review:def456 -->"
                            ),
                            "submitted_at": "2026-03-31T05:00:00Z",
                        },
                    ],
                },
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                state_store=state_store,
            )
            pr = _build_pr(26)
            state_store.upsert_tracked_pr(
                pr_number=26, branch_name="autocode/issue-26", base_ref="main",
                head_sha="abc", author_login="autocode[bot]", pr_url="", queue_source="autocode",
            )

            summary, candidate = controller.plan_pull_request_repair(pr)

            # Should NOT be skipped — OpenReview rejected
            self.assertNotIn("OpenReview 审批", summary)

    def test_stale_openreview_review_is_not_repaired_against_new_head(self):
        """OpenReview feedback for an older head should wait for a fresh review."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 621,
                            "user": {"login": "maintainer"},
                            "state": "COMMENTED",
                            "body": (
                                "**评审结论**\n"
                                "- **是否可直接合入**：不可\n"
                                "1. [Correctness blocker] Fix the old-head bug\n"
                                "---\n"
                                "<!-- openreview:idempotency:pr-review:old -->"
                            ),
                            "submitted_at": "2026-03-31T05:00:00Z",
                            "commit_id": "old-sha",
                        },
                    ],
                },
                review_comment_pages={
                    1: [
                        {
                            "id": 721,
                            "user": {"login": "maintainer"},
                            "body": "Old-head inline note",
                            "path": "src/app.py",
                            "pull_request_review_id": 621,
                            "updated_at": "2026-03-31T05:00:00Z",
                        },
                    ],
                },
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                state_store=state_store,
            )
            pr = _build_pr(27)
            pr["head"]["sha"] = "new-sha"
            state_store.upsert_tracked_pr(
                pr_number=27, branch_name="autocode/issue-27", base_ref="main",
                head_sha="new-sha", author_login="autocode[bot]", pr_url="", queue_source="autocode",
            )

            summary, candidate = controller.plan_pull_request_repair(pr)

            self.assertIsNone(candidate)
            self.assertIn("暂无新的可执行检视意见", summary)

    def test_cached_stale_openreview_feedback_is_not_executed(self):
        """Recovered queued tasks should not execute old-head OpenReview feedback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            controller = _build_controller(
                workspace_path=workspace_path,
                state_store=state_store,
            )
            pr = _build_pr(28)
            pr["head"]["sha"] = "new-sha"
            cached_feedback = [
                {
                    "id": 631,
                    "user": {"login": "maintainer"},
                    "state": "COMMENTED",
                    "body": (
                        "**评审结论**\n"
                        "- **是否可直接合入**：不可\n"
                        "1. [Correctness blocker] Fix the old-head bug\n"
                        "---\n"
                        "<!-- openreview:idempotency:pr-review:old -->"
                    ),
                    "submitted_at": "2026-03-31T05:00:00Z",
                    "commit_id": "old-sha",
                },
                {
                    "id": 731,
                    "user": {"login": "maintainer"},
                    "body": "Old-head inline note",
                    "path": "src/app.py",
                    "pull_request_review_id": 631,
                    "updated_at": "2026-03-31T05:00:00Z",
                },
            ]
            state_store.upsert_tracked_pr(
                pr_number=28, branch_name="autocode/issue-28", base_ref="main",
                head_sha="new-sha", author_login="autocode[bot]", pr_url="", queue_source="autocode",
            )

            summary, consumed = controller._process_tracked_pull_request(
                pr,
                allow_repair=True,
                cached_feedback=cached_feedback,
                cached_review_state="commented",
            )

            self.assertFalse(consumed)
            self.assertIn("暂无新的可执行检视意见", summary)
            self.assertFalse(controller.workspace_manager.commit_calls)

    def test_resume_resets_repair_count(self):
        """Resuming a paused PR should reset repair_count to 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            state_store.upsert_tracked_pr(
                pr_number=30, branch_name="autocode/issue-30", base_ref="main",
                head_sha="abc", author_login="autocode[bot]", pr_url="", queue_source="autocode",
            )
            state_store.increment_tracked_pr_repair_count(30)
            state_store.increment_tracked_pr_repair_count(30)
            state_store.increment_tracked_pr_repair_count(30)
            self.assertEqual(3, state_store.get_tracked_pr_repair_count(30))

            state_store.set_tracked_pr_paused(30, paused=True, reason="manual pause")
            self.assertEqual(3, state_store.get_tracked_pr_repair_count(30))

            state_store.set_tracked_pr_paused(30, paused=False)
            self.assertEqual(0, state_store.get_tracked_pr_repair_count(30))

    def test_reset_tracked_pr_repair_count_explicit(self):
        """Explicit reset should zero out the counter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            state_store.upsert_tracked_pr(
                pr_number=40, branch_name="autocode/issue-40", base_ref="main",
                head_sha="abc", author_login="autocode[bot]", pr_url="", queue_source="autocode",
            )
            state_store.increment_tracked_pr_repair_count(40)
            state_store.increment_tracked_pr_repair_count(40)
            self.assertEqual(2, state_store.get_tracked_pr_repair_count(40))

            state_store.reset_tracked_pr_repair_count(40)
            self.assertEqual(0, state_store.get_tracked_pr_repair_count(40))

    def test_is_repair_diverging_detects_consecutive_increase(self):
        """Feedback counts [2, 3, 5] should be detected as diverging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            state_store.upsert_tracked_pr(
                pr_number=50, branch_name="autocode/issue-50", base_ref="main",
                head_sha="abc", author_login="autocode[bot]", pr_url="", queue_source="autocode",
            )
            state_store.increment_tracked_pr_repair_count(50, feedback_count=2)
            self.assertFalse(state_store.is_repair_diverging(50))
            state_store.increment_tracked_pr_repair_count(50, feedback_count=3)
            self.assertFalse(state_store.is_repair_diverging(50))
            state_store.increment_tracked_pr_repair_count(50, feedback_count=5)
            self.assertTrue(state_store.is_repair_diverging(50))

    def test_is_repair_diverging_false_when_feedback_decreases(self):
        """Feedback counts [5, 3, 1] should NOT be detected as diverging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            state_store.upsert_tracked_pr(
                pr_number=51, branch_name="autocode/issue-51", base_ref="main",
                head_sha="abc", author_login="autocode[bot]", pr_url="", queue_source="autocode",
            )
            state_store.increment_tracked_pr_repair_count(51, feedback_count=5)
            state_store.increment_tracked_pr_repair_count(51, feedback_count=3)
            state_store.increment_tracked_pr_repair_count(51, feedback_count=1)
            self.assertFalse(state_store.is_repair_diverging(51))


class BotFeedbackDetectionTests(unittest.TestCase):
    """Tests for _is_bot_only_feedback method."""

    def _make_controller(self):
        config = SimpleNamespace(
            github=SimpleNamespace(repo="demo/repo"),
            autocode=SimpleNamespace(
                tracked_pr_authors=["reviewer"],
                pr_max_repair_rounds=3,
                worktree_root="/tmp/autocode-runs",
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            state = AutoCodeStateStore(str(Path(tmpdir) / "state.json"))
            ctrl = ExecutionController.__new__(ExecutionController)
            ctrl.config = config
            ctrl.state_store = state
            return ctrl

    def test_bot_only_feedback_all_bots(self):
        ctrl = self._make_controller()
        feedback = [
            {"body": "some comment", "user": {"login": "copilot-pull-request-reviewer"}},
            {"body": "another comment", "user": {"login": "chatgpt-codex-connector"}},
        ]
        self.assertTrue(ctrl._is_bot_only_feedback(feedback))

    def test_bot_only_feedback_mixed(self):
        ctrl = self._make_controller()
        feedback = [
            {"body": "bot comment", "user": {"login": "copilot-pull-request-reviewer"}},
            {"body": "human comment", "user": {"login": "human-user"}},
        ]
        self.assertFalse(ctrl._is_bot_only_feedback(feedback))

    def test_bot_only_feedback_empty(self):
        ctrl = self._make_controller()
        self.assertFalse(ctrl._is_bot_only_feedback([]))

    def test_bot_only_feedback_no_user(self):
        ctrl = self._make_controller()
        feedback = [{"body": "no user field"}]
        self.assertFalse(ctrl._is_bot_only_feedback(feedback))

    def test_bot_only_feedback_openreview_marker(self):
        ctrl = self._make_controller()
        feedback = [
            {"body": "评审结论\n<!-- openreview:idempotency:abc123 -->", "user": {"login": "maintainer"}},
        ]
        self.assertTrue(ctrl._is_bot_only_feedback(feedback))

    def test_bot_review_item_by_marker(self):
        ctrl = self._make_controller()
        item = {"body": "some review <!-- openreview: marker -->", "user": {"login": "unknown-user"}}
        self.assertTrue(ctrl._is_bot_review_item(item))

    def test_bot_review_item_by_login(self):
        ctrl = self._make_controller()
        item = {"body": "human-like text", "user": {"login": "copilot-pull-request-reviewer"}}
        self.assertTrue(ctrl._is_bot_review_item(item))

    def test_bot_review_item_human(self):
        ctrl = self._make_controller()
        item = {"body": "please fix this bug", "user": {"login": "real-reviewer"}}
        self.assertFalse(ctrl._is_bot_review_item(item))

    def test_sanitize_execution_summary_strips_tmp_paths(self):
        ctrl = self._make_controller()
        text = "Modified [src/config.py](/tmp/autocode-runs/autocode-issue-832-bug-1774791906/src/config.py) and [tests/test_main.py](/tmp/autocode-runs/autocode-issue-832-bug-1774791906/tests/test_main.py)"
        result = ctrl._sanitize_execution_summary(text)
        self.assertNotIn("/tmp/autocode-runs/", result)
        self.assertIn("src/config.py", result)
        self.assertIn("tests/test_main.py", result)

    def test_sanitize_execution_summary_preserves_clean_text(self):
        ctrl = self._make_controller()
        text = "Modified src/config.py with no local paths"
        self.assertEqual(text, ctrl._sanitize_execution_summary(text))

    def test_sanitize_execution_summary_empty(self):
        ctrl = self._make_controller()
        self.assertEqual("", ctrl._sanitize_execution_summary(""))

    def test_check_open_pr_cap_blocks_at_limit(self):
        """_check_open_pr_cap returns a blocking reason when cap is reached."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            autocode_prs = [
                _build_pr(i) for i in range(1, 4)
            ]
            gh = FakeGitHubClient(pull_request_pages={1: autocode_prs})
            controller = _build_controller(workspace_path=workspace_path, gh=gh)
            controller.config.autocode.max_open_autocode_prs = 3
            reason = controller._check_open_pr_cap()
            self.assertIn("上限", reason)
            self.assertIn("3/3", reason)

    def test_check_open_pr_cap_allows_below_limit(self):
        """_check_open_pr_cap returns empty string when below cap."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient(pull_request_pages={1: [_build_pr(1)]})
            controller = _build_controller(workspace_path=workspace_path, gh=gh)
            controller.config.autocode.max_open_autocode_prs = 5
            self.assertEqual("", controller._check_open_pr_cap())

    def test_check_open_pr_cap_disabled_when_zero(self):
        """_check_open_pr_cap is disabled when cap is 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            gh = FakeGitHubClient(pull_request_pages={1: [_build_pr(i) for i in range(1, 50)]})
            controller = _build_controller(workspace_path=workspace_path, gh=gh)
            controller.config.autocode.max_open_autocode_prs = 0
            self.assertEqual("", controller._check_open_pr_cap())

    def test_check_open_pr_cap_ignores_non_autocode_branches(self):
        """_check_open_pr_cap only counts autocode/ prefixed branches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            non_autocode_pr = {
                "number": 99,
                "title": "Manual PR",
                "body": "body",
                "html_url": "https://example.com/pr/99",
                "user": {"login": "human"},
                "head": {"ref": "feature/manual", "sha": "abc", "repo": {"full_name": "demo/repo"}},
                "base": {"ref": "main"},
            }
            gh = FakeGitHubClient(pull_request_pages={1: [non_autocode_pr]})
            controller = _build_controller(workspace_path=workspace_path, gh=gh)
            controller.config.autocode.max_open_autocode_prs = 1
            self.assertEqual("", controller._check_open_pr_cap())

    def test_bot_feedback_throttle_blocks_at_limit(self):
        """Bot-only feedback should be throttled after configured bot repair rounds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "state.json"))
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 100,
                            "user": {"login": "copilot-pull-request-reviewer"},
                            "state": "COMMENTED",
                            "body": "bot suggestions",
                            "submitted_at": "2026-03-29T01:00:00Z",
                        },
                    ],
                },
                review_comment_pages={
                    1: [
                        {
                            "id": 200,
                            "user": {"login": "copilot-pull-request-reviewer"},
                            "pull_request_review_id": 100,
                            "body": "Guard non-string link",
                            "path": "src/app.py",
                            "line": 10,
                            "updated_at": "2026-03-29T01:00:00Z",
                        },
                    ],
                },
            )
            controller = _build_controller(
                workspace_path=workspace_path, gh=gh, state_store=state_store,
            )
            controller.config.autocode.bot_feedback_max_repair_rounds = 2
            pr = _build_pr(10)
            state_store.upsert_tracked_pr(
                pr_number=10, branch_name="autocode/issue-10", base_ref="main",
                head_sha="abc", author_login="autocode[bot]", pr_url="", queue_source="autocode",
            )
            # Simulate 2 completed repairs (reaches configured bot-only limit)
            state_store.increment_tracked_pr_repair_count(10)
            state_store.increment_tracked_pr_repair_count(10)

            summary, candidate = controller.plan_pull_request_repair(pr)
            self.assertIsNone(candidate)
            self.assertIn("bot", summary.lower())
            self.assertIn("修复上限", summary)

    def test_bot_feedback_throttle_allows_actionable_openreview_feedback(self):
        """OpenReview blockers should not be stopped by the pure-bot throttle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "state.json"))
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 100,
                            "user": {"login": "repoowner"},
                            "state": "COMMENTED",
                            "body": (
                                "**评审结论**\n"
                                "- **是否可直接合入**：不可 + 需要先修复兼容问题。\n"
                                "\n"
                                "**主要问题**\n"
                                "1. [Compatibility blocker] Kimi K2.6 non-thinking 模式应使用 0.6。\n"
                                "\n"
                                "<!-- openreview:idempotency:pr-review:abc123 -->"
                            ),
                            "submitted_at": "2026-03-29T01:00:00Z",
                        },
                    ],
                },
            )
            controller = _build_controller(
                workspace_path=workspace_path, gh=gh, state_store=state_store,
            )
            controller.config.autocode.bot_feedback_max_repair_rounds = 2
            pr = _build_pr(10)
            state_store.upsert_tracked_pr(
                pr_number=10, branch_name="autocode/issue-10", base_ref="main",
                head_sha="abc", author_login="autocode[bot]", pr_url="", queue_source="autocode",
            )
            state_store.increment_tracked_pr_repair_count(10)
            state_store.increment_tracked_pr_repair_count(10)
            state_store.increment_tracked_pr_repair_count(10)

            summary, candidate = controller.plan_pull_request_repair(pr)

            self.assertIsNotNone(candidate)
            self.assertIn("已进入修复队列", summary)
            self.assertNotIn("bot 修复上限", summary)

    def test_openreview_feedback_resets_total_repair_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "state.json"))
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 100,
                            "user": {"login": "repoowner"},
                            "state": "COMMENTED",
                            "body": (
                                "**评审结论**\n"
                                "- **是否可直接合入**：不可\n"
                                "\n"
                                "**主要问题**\n"
                                "1. [Correctness blocker] 仍需补一个边界条件。\n"
                                "\n"
                                "<!-- openreview:idempotency:pr-review:def456 -->"
                            ),
                            "submitted_at": "2026-03-29T01:00:00Z",
                        },
                    ],
                },
            )
            controller = _build_controller(
                workspace_path=workspace_path, gh=gh, state_store=state_store,
            )
            controller.config.autocode.pr_max_repair_rounds = 2
            pr = _build_pr(10)
            state_store.upsert_tracked_pr(
                pr_number=10, branch_name="autocode/issue-10", base_ref="main",
                head_sha="abc", author_login="autocode[bot]", pr_url="", queue_source="autocode",
            )
            state_store.increment_tracked_pr_repair_count(10)
            state_store.increment_tracked_pr_repair_count(10)

            summary, candidate = controller.plan_pull_request_repair(pr)

            self.assertIsNotNone(candidate)
            self.assertIn("已进入修复队列", summary)
            self.assertEqual(0, state_store.get_tracked_pr_repair_count(10))

    def test_bot_feedback_throttle_allows_human_feedback(self):
        """Mixed human+bot feedback should NOT be throttled by bot limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "state.json"))
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 100,
                            "user": {"login": "human-reviewer"},
                            "state": "CHANGES_REQUESTED",
                            "body": "please fix the null check",
                            "submitted_at": "2026-03-29T01:00:00Z",
                        },
                    ],
                },
                review_comment_pages={
                    1: [
                        {
                            "id": 200,
                            "user": {"login": "human-reviewer"},
                            "pull_request_review_id": 100,
                            "body": "This needs a null check",
                            "path": "src/app.py",
                            "line": 10,
                            "updated_at": "2026-03-29T01:00:00Z",
                        },
                    ],
                },
            )
            controller = _build_controller(
                workspace_path=workspace_path, gh=gh, state_store=state_store,
            )
            pr = _build_pr(10)
            state_store.upsert_tracked_pr(
                pr_number=10, branch_name="autocode/issue-10", base_ref="main",
                head_sha="abc", author_login="autocode[bot]", pr_url="", queue_source="autocode",
            )
            # Even with 1 repair, human feedback should pass through
            state_store.increment_tracked_pr_repair_count(10)

            summary, candidate = controller.plan_pull_request_repair(pr)
            # Should produce a repair candidate (not throttled)
            self.assertIsNotNone(candidate)

    def test_bot_feedback_throttle_uses_last_summary_for_log_downgrade(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "state.json"))
            controller = _build_controller(workspace_path=workspace_path, state_store=state_store)
            controller.config.autocode.bot_feedback_max_repair_rounds = 2
            state_store.upsert_tracked_pr(
                pr_number=10,
                branch_name="autocode/issue-10",
                base_ref="main",
                head_sha="abc",
                author_login="autocode[bot]",
                pr_url="",
                queue_source="autocode",
            )
            state_store.mark_tracked_pr_status(
                10,
                status="waiting_review",
                summary="PR #10 当前反馈均来自 bot reviewer，且已达到 bot 修复上限 (2/2)，跳过自动修复",
                review_state="commented",
            )
            state_store.increment_tracked_pr_repair_count(10)
            state_store.increment_tracked_pr_repair_count(10)

            feedback = [
                {
                    "id": 200,
                    "body": "Guard non-string link",
                    "user": {"login": "copilot-pull-request-reviewer"},
                    "path": "src/app.py",
                    "line": 10,
                    "updated_at": "2026-03-29T01:00:00Z",
                }
            ]

            with mock.patch("agent.execution_controller.logger") as logger_mock:
                summary = controller._check_bot_feedback_throttle(10, feedback, "commented")

            self.assertIn("bot 修复上限", summary)
            logger_mock.debug.assert_called_once()
            logger_mock.info.assert_not_called()

    def test_check_repair_limit_helper(self):
        """_check_repair_limit returns empty string when under limit, blocking reason when at limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "state.json"))
            controller = _build_controller(workspace_path=workspace_path, state_store=state_store)
            controller.config.autocode.pr_max_repair_rounds = 2
            state_store.upsert_tracked_pr(
                pr_number=5, branch_name="autocode/issue-5", base_ref="main",
                head_sha="abc", author_login="autocode[bot]", pr_url="", queue_source="autocode",
            )
            # Under limit
            self.assertEqual("", controller._check_repair_limit(5, "changes_requested"))
            # At limit
            state_store.increment_tracked_pr_repair_count(5)
            state_store.increment_tracked_pr_repair_count(5)
            reason = controller._check_repair_limit(5, "changes_requested")
            self.assertIn("最大修复轮数", reason)
            self.assertEqual("repair_limit_reached", state_store.get_tracked_pr(5)["status"])

    def test_check_bot_feedback_throttle_helper(self):
        """_check_bot_feedback_throttle returns empty for human feedback, blocking for bot at limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            state_store = AutoCodeStateStore(str(Path(tmpdir) / "state.json"))
            controller = _build_controller(workspace_path=workspace_path, state_store=state_store)
            controller.config.autocode.bot_feedback_max_repair_rounds = 2
            state_store.upsert_tracked_pr(
                pr_number=5, branch_name="autocode/issue-5", base_ref="main",
                head_sha="abc", author_login="autocode[bot]", pr_url="", queue_source="autocode",
            )
            human_fb = [{"body": "fix this", "user": {"login": "reviewer"}, "id": 1}]
            bot_fb = [{"body": "suggestion", "user": {"login": "copilot-pull-request-reviewer"}, "id": 2}]
            # Human feedback: always empty even at bot limit
            state_store.increment_tracked_pr_repair_count(5)
            state_store.increment_tracked_pr_repair_count(5)
            self.assertEqual("", controller._check_bot_feedback_throttle(5, human_fb, "changes_requested"))
            # Bot feedback at configured limit: blocking
            reason = controller._check_bot_feedback_throttle(5, bot_fb, "changes_requested")
            self.assertIn("bot", reason.lower())


    def test_process_tracked_pr_sets_approved_via_openreview(self):
        """_process_tracked_pull_request should set status='approved' when OpenReview
        says '可直接合入' even though review_state is COMMENTED (not APPROVED)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 801,
                            "user": {"login": "maintainer"},
                            "state": "COMMENTED",
                            "body": (
                                "**评审结论**\n"
                                "- **是否可直接合入**：可\n"
                                "---\n"
                                "<!-- openreview:idempotency:pr-review:xyz123 -->"
                            ),
                            "submitted_at": "2026-03-31T05:00:00Z",
                        },
                    ],
                },
                review_comment_pages={1: []},
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                state_store=state_store,
            )
            pr = _build_pr(40)
            state_store.upsert_tracked_pr(
                pr_number=40, branch_name="autocode/issue-40", base_ref="main",
                head_sha="abc", author_login="autocode[bot]", pr_url="", queue_source="autocode",
            )

            summary, consumed = controller._process_tracked_pull_request(pr, allow_repair=True)

            tracked = state_store.get_tracked_pr(40) or {}
            self.assertEqual("approved", tracked.get("status"))
            self.assertFalse(consumed)

    def test_process_tracked_pr_skips_bot_feedback_when_openreview_approved(self):
        """_process_tracked_pull_request should skip bot-only feedback when
        OpenReview approved, setting status to 'approved'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 901,
                            "user": {"login": "chatgpt-codex-connector[bot]"},
                            "state": "COMMENTED",
                            "body": "### Codex Review\nSome P1 suggestion",
                            "submitted_at": "2026-03-31T02:00:00Z",
                        },
                        {
                            "id": 902,
                            "user": {"login": "maintainer"},
                            "state": "COMMENTED",
                            "body": (
                                "**评审结论**\n"
                                "- **是否可直接合入**：可\n"
                                "---\n"
                                "<!-- openreview:idempotency:pr-review:abc456 -->"
                            ),
                            "submitted_at": "2026-03-31T05:00:00Z",
                        },
                    ],
                },
                review_comment_pages={
                    1: [
                        {
                            "id": 1001,
                            "user": {"login": "chatgpt-codex-connector[bot]"},
                            "body": "P1: Fix CHANGELOG format",
                            "path": "docs/CHANGELOG.md",
                            "pull_request_review_id": 901,
                            "updated_at": "2026-03-31T02:00:00Z",
                        },
                    ],
                },
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                state_store=state_store,
            )
            pr = _build_pr(41)
            state_store.upsert_tracked_pr(
                pr_number=41, branch_name="autocode/issue-41", base_ref="main",
                head_sha="abc", author_login="autocode[bot]", pr_url="", queue_source="autocode",
            )

            summary, consumed = controller._process_tracked_pull_request(pr, allow_repair=True)

            self.assertIn("OpenReview 审批", summary)
            self.assertFalse(consumed)
            tracked = state_store.get_tracked_pr(41) or {}
            self.assertEqual("approved", tracked.get("status"))

    def test_rebase_skipped_when_active_repair_task(self):
        """rebase_tracked_pr_if_needed should skip when there is an active
        repair task for this PR to avoid interfering with in-flight work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            gh = FakeGitHubClient(review_pages={1: []}, review_comment_pages={1: []})
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                state_store=state_store,
            )
            pr = _build_pr(42)
            state_store.upsert_tracked_pr(
                pr_number=42, branch_name="autocode/issue-42", base_ref="main",
                head_sha="abc", author_login="autocode[bot]", pr_url="", queue_source="autocode",
            )
            # Simulate one repair round having happened
            state_store.increment_tracked_pr_repair_count(42)
            # Add an active repair task for this PR
            state_store.enqueue_task(
                task_id="pr-task:pr:42:test",
                task_type="pr_feedback_repair",
                target_number=42,
                source="test",
                command="review_feedback",
                payload={},
            )

            result = controller.rebase_tracked_pr_if_needed(pr)

            self.assertIsNone(result)

    def test_rebase_allowed_when_repair_count_positive_no_active_task(self):
        """rebase should proceed when repair_count > 0 but no active task
        is running — the old blanket block caused CHANGELOG conflicts to
        accumulate indefinitely."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            dirty_pr = {**_build_pr(42), "mergeable_state": "dirty"}
            gh = FakeGitHubClient(
                review_pages={1: []}, review_comment_pages={1: []},
                pr_details={42: dirty_pr},
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                state_store=state_store,
            )
            pr = _build_pr(42)
            state_store.upsert_tracked_pr(
                pr_number=42, branch_name="autocode/issue-42", base_ref="main",
                head_sha="abc", author_login="autocode[bot]", pr_url="", queue_source="autocode",
            )
            # Simulate completed repair (count > 0 but no active task)
            state_store.increment_tracked_pr_repair_count(42)

            controller.rebase_tracked_pr_if_needed(pr)

            # sync_default_branch was called → method passed all guard checks
            self.assertGreater(controller.workspace_manager.sync_calls, 0)

    def test_rebase_allowed_for_approved_pr_with_conflicts(self):
        """Approved PRs with merge conflicts should be rebased."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            dirty_pr = {**_build_pr(42), "mergeable_state": "dirty"}
            gh = FakeGitHubClient(
                review_pages={1: []}, review_comment_pages={1: []},
                pr_details={42: dirty_pr},
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                state_store=state_store,
            )
            pr = _build_pr(42)
            state_store.upsert_tracked_pr(
                pr_number=42, branch_name="autocode/issue-42", base_ref="main",
                head_sha="abc", author_login="autocode[bot]", pr_url="", queue_source="autocode",
            )
            state_store.mark_tracked_pr_status(42, status="approved", summary="approved")

            controller.rebase_tracked_pr_if_needed(pr)

            # sync_default_branch was called → approved dirty PRs pass guard checks
            self.assertGreater(controller.workspace_manager.sync_calls, 0)

    def test_rebase_skipped_when_branch_workspace_locked(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            dirty_pr = {**_build_pr(42), "mergeable_state": "dirty"}
            gh = FakeGitHubClient(
                review_pages={1: []},
                review_comment_pages={1: []},
                pr_details={42: dirty_pr},
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                state_store=state_store,
            )
            pr = _build_pr(42)
            controller.workspace_manager.branch_locks[pr["head"]["ref"]] = {
                "branch": pr["head"]["ref"],
                "worktree_path": "/tmp/autocode-runs/autocode-issue-42-demo",
            }

            result = controller.rebase_tracked_pr_if_needed(pr)

            self.assertIsNone(result)
            self.assertEqual(0, controller.workspace_manager.sync_calls)

    def test_rebase_skipped_when_no_conflicts(self):
        """PRs without merge conflicts (clean/blocked) should NOT be rebased
        to avoid unnecessary force-pushes that re-run CI and disrupt reviews."""
        for state in ("clean", "blocked", "has_hooks"):
            with self.subTest(mergeable_state=state):
                with tempfile.TemporaryDirectory() as tmpdir:
                    workspace_path = Path(tmpdir)
                    state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
                    clean_pr = {**_build_pr(42), "mergeable_state": state}
                    gh = FakeGitHubClient(
                        review_pages={1: []}, review_comment_pages={1: []},
                        pr_details={42: clean_pr},
                    )
                    controller = _build_controller(
                        workspace_path=workspace_path,
                        gh=gh,
                        state_store=state_store,
                    )
                    pr = _build_pr(42)
                    state_store.upsert_tracked_pr(
                        pr_number=42, branch_name="autocode/issue-42", base_ref="main",
                        head_sha="abc", author_login="autocode[bot]", pr_url="",
                        queue_source="autocode",
                    )

                    result = controller.rebase_tracked_pr_if_needed(pr)

                    self.assertIsNone(result)
                    # sync_default_branch should NOT have been called
                    self.assertEqual(controller.workspace_manager.sync_calls, 0)

    def test_rebase_skips_repeat_failure_for_same_head_sha(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            dirty_pr = {**_build_pr(42), "mergeable_state": "dirty"}
            gh = FakeGitHubClient(
                review_pages={1: []},
                review_comment_pages={1: []},
                pr_details={42: dirty_pr},
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                state_store=state_store,
            )
            controller.workspace_manager.rebase_onto_default_branch = mock.Mock(side_effect=RuntimeError("conflict"))
            pr = _build_pr(42)
            state_store.upsert_tracked_pr(
                pr_number=42,
                branch_name="autocode/issue-42",
                base_ref="main",
                head_sha="sha-42",
                author_login="autocode[bot]",
                pr_url="",
                queue_source="autocode",
            )

            first = controller.rebase_tracked_pr_if_needed(pr)
            second = controller.rebase_tracked_pr_if_needed(pr)

            self.assertIn("rebase 失败", str(first))
            self.assertIsNone(second)
            self.assertEqual(1, controller.workspace_manager.sync_calls)
            tracked = state_store.get_tracked_pr(42) or {}
            self.assertEqual("sha-42", tracked.get("last_rebase_failed_head_sha"))

    def test_rebase_retries_after_head_sha_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            dirty_pr = {**_build_pr(42), "mergeable_state": "dirty"}
            gh = FakeGitHubClient(
                review_pages={1: []},
                review_comment_pages={1: []},
                pr_details={42: dirty_pr},
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                state_store=state_store,
            )
            controller.workspace_manager.rebase_onto_default_branch = mock.Mock(
                side_effect=[RuntimeError("conflict"), False]
            )
            pr = _build_pr(42)
            state_store.upsert_tracked_pr(
                pr_number=42,
                branch_name="autocode/issue-42",
                base_ref="main",
                head_sha="sha-42",
                author_login="autocode[bot]",
                pr_url="",
                queue_source="autocode",
            )

            first = controller.rebase_tracked_pr_if_needed(pr)
            self.assertIn("rebase 失败", str(first))

            updated_pr = _build_pr(42)
            updated_pr["head"]["sha"] = "sha-new"
            state_store.upsert_tracked_pr(
                pr_number=42,
                branch_name="autocode/issue-42",
                base_ref="main",
                head_sha="sha-new",
                author_login="autocode[bot]",
                pr_url="",
                queue_source="autocode",
            )

            second = controller.rebase_tracked_pr_if_needed(updated_pr)

            self.assertIsNone(second)
            self.assertEqual(2, controller.workspace_manager.sync_calls)


    def test_repair_limit_advances_cursor(self):
        """When repair limit blocks, feedback cursor should advance so same
        feedback isn't re-evaluated next poll cycle."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 501,
                            "user": {"login": "copilot-pull-request-reviewer"},
                            "state": "COMMENTED",
                            "body": "Bot review suggestion",
                            "submitted_at": "2026-03-31T06:00:00Z",
                        },
                    ],
                },
                review_comment_pages={
                    1: [
                        {
                            "id": 601,
                            "user": {"login": "copilot-pull-request-reviewer"},
                            "body": "Fix the import",
                            "path": "src/app.py",
                            "pull_request_review_id": 501,
                            "updated_at": "2026-03-31T06:00:00Z",
                        },
                    ],
                },
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                state_store=state_store,
            )
            pr = _build_pr(44)
            state_store.upsert_tracked_pr(
                pr_number=44, branch_name="autocode/issue-44", base_ref="main",
                head_sha="abc", author_login="autocode[bot]", pr_url="", queue_source="autocode",
            )
            # Push repair count past the limit
            for _ in range(10):
                state_store.increment_tracked_pr_repair_count(44)

            # First call: should block AND advance cursor
            summary1, consumed1 = controller._process_tracked_pull_request(pr, allow_repair=True)
            self.assertIn("最大修复轮数", summary1)
            self.assertFalse(consumed1)

            # Verify cursor was advanced - tracked PR should have cursor set
            tracked = state_store.get_tracked_pr(44) or {}
            self.assertTrue(tracked.get("last_feedback_updated_at", ""), "Cursor should have been set")

            # Second call: same feedback should now be filtered out by cursor,
            # resulting in "no new feedback" rather than hitting the limit again
            summary2, consumed2 = controller._process_tracked_pull_request(pr, allow_repair=True)
            self.assertNotIn("最大修复轮数", summary2)
            self.assertIn("暂无新的", summary2)

    def test_bot_throttle_advances_cursor(self):
        """When bot throttle blocks, feedback cursor should advance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_path = Path(tmpdir)
            (workspace_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
            state_store = AutoCodeStateStore(str(workspace_path / ".autocode-state.json"))
            gh = FakeGitHubClient(
                review_pages={
                    1: [
                        {
                            "id": 701,
                            "user": {"login": "chatgpt-codex-connector[bot]"},
                            "state": "COMMENTED",
                            "body": "Codex review suggestion",
                            "submitted_at": "2026-03-31T07:00:00Z",
                        },
                    ],
                },
                review_comment_pages={
                    1: [
                        {
                            "id": 801,
                            "user": {"login": "chatgpt-codex-connector[bot]"},
                            "body": "P1: Fix this issue",
                            "path": "docs/CHANGELOG.md",
                            "pull_request_review_id": 701,
                            "updated_at": "2026-03-31T07:00:00Z",
                        },
                    ],
                },
            )
            controller = _build_controller(
                workspace_path=workspace_path,
                gh=gh,
                state_store=state_store,
            )
            controller.config.autocode.bot_feedback_max_repair_rounds = 2
            pr = _build_pr(45)
            state_store.upsert_tracked_pr(
                pr_number=45, branch_name="autocode/issue-45", base_ref="main",
                head_sha="abc", author_login="autocode[bot]", pr_url="", queue_source="autocode",
            )
            # Push past bot repair limit
            for _ in range(3):
                state_store.increment_tracked_pr_repair_count(45)

            # First call: bot throttle blocks
            summary1, _ = controller._process_tracked_pull_request(pr, allow_repair=True)
            self.assertIn("bot", summary1.lower())

            # Second call: cursor advanced, no new feedback
            summary2, _ = controller._process_tracked_pull_request(pr, allow_repair=True)
            self.assertNotIn("bot 修复上限", summary2)

    def test_issue_previous_failure_reasons_drop_stale_scope_errors(self):
        controller = ExecutionController.__new__(ExecutionController)
        controller.state_store = PreviousFailureStateStore(
            [
                {"failure_reason": "越界改动: src/market_analyzer.py"},
                {"failure_reason": "验证失败: pytest tests/test_market.py"},
            ]
        )

        reasons = controller._get_previous_failure_reasons("issue", 1115)

        self.assertEqual(["验证失败: pytest tests/test_market.py"], reasons)

    def test_pr_previous_failure_reasons_keep_scope_errors(self):
        controller = ExecutionController.__new__(ExecutionController)
        controller.state_store = PreviousFailureStateStore(
            [{"failure_reason": "越界改动: bot/dispatcher.py"}]
        )

        reasons = controller._get_previous_failure_reasons("review_feedback", 1120)

        self.assertEqual(["越界改动: bot/dispatcher.py"], reasons)


if __name__ == "__main__":
    unittest.main()
