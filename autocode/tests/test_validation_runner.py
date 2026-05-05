import signal
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent.validation_runner import ValidationRunner, ValidationStepResult


class ValidationRunnerTests(unittest.TestCase):
    def test_run_all_returns_failed_result_when_command_times_out(self):
        runner = ValidationRunner(
            format_commands=[["python", "-m", "black", "."]],
            lint_commands=[],
            test_commands=[],
            timeout_seconds=30,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            with mock.patch(
                "agent.validation_runner.ValidationRunner._run_command",
                side_effect=subprocess.TimeoutExpired(cmd=["python", "-m", "black", "."], timeout=30),
            ):
                result = runner.run_all(workspace)

        self.assertFalse(result.success)
        self.assertEqual(1, len(result.steps))
        self.assertEqual("format", result.steps[0].name)
        self.assertIn("命令超时", result.steps[0].output)
        self.assertIn("python -m black .", result.steps[0].output)

    def test_only_test_timed_out_when_lint_passes_but_test_times_out(self):
        runner = ValidationRunner(
            format_commands=[],
            lint_commands=[["flake8"]],
            test_commands=[["pytest"]],
            timeout_seconds=30,
            test_timeout_seconds=60,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            call_count = 0

            def fake_run(*args, **kwargs):
                del kwargs
                nonlocal call_count
                call_count += 1
                if call_count == 1:  # lint
                    return subprocess.CompletedProcess(args=args[1], returncode=0, stdout="ok", stderr="")
                raise subprocess.TimeoutExpired(cmd=["pytest"], timeout=60)  # test

            with mock.patch("agent.validation_runner.ValidationRunner._run_command", side_effect=fake_run):
                result = runner.run_all(workspace)

        self.assertFalse(result.success)
        self.assertTrue(result.only_test_timed_out)
        self.assertEqual(2, len(result.steps))
        self.assertTrue(result.steps[1].is_timeout)
        self.assertEqual("test", result.steps[1].name)

    def test_only_test_timed_out_false_when_lint_also_fails(self):
        from agent.validation_runner import ValidationResult, ValidationStepResult
        result = ValidationResult(success=False, steps=[
            ValidationStepResult(name="lint", command=["flake8"], success=False, output="error"),
            ValidationStepResult(name="test", command=["pytest"], success=False, output="timeout", is_timeout=True),
        ])
        self.assertFalse(result.only_test_timed_out)

    def test_environment_blocker_detects_missing_command(self):
        result = ValidationStepResult(
            name="lint",
            command=["./scripts/ci_gate.sh", "flake8"],
            success=False,
            output="./scripts/ci_gate.sh: line 15: flake8: command not found",
        )

        blocker = result.environment_blocker()

        self.assertIn("./scripts/ci_gate.sh flake8", blocker)
        self.assertIn("缺少命令 `flake8`", blocker)

    def test_summarize_reports_timeout_instead_of_fail(self):
        from agent.validation_runner import ValidationResult
        result = ValidationResult(success=False, steps=[
            ValidationStepResult(name="format", command=["black", "."], success=True, output="ok"),
            ValidationStepResult(name="test", command=["pytest"], success=False, output="timeout", is_timeout=True),
        ])
        summary = result.summarize()
        self.assertIn("format:PASS", summary)
        self.assertIn("test:TIMEOUT", summary)
        self.assertNotIn("test:FAIL", summary)

    def test_summarize_reports_fail_for_non_timeout(self):
        from agent.validation_runner import ValidationResult
        result = ValidationResult(success=False, steps=[
            ValidationStepResult(name="lint", command=["flake8"], success=False, output="errors"),
        ])
        self.assertIn("lint:FAIL", result.summarize())

    def test_run_all_respects_test_timeout_override(self):
        runner = ValidationRunner(
            format_commands=[],
            lint_commands=[],
            test_commands=[["pytest"]],
            timeout_seconds=1200,
            test_timeout_seconds=1200,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            captured_timeouts = []

            def fake_run(path, command, timeout_seconds):
                del path
                captured_timeouts.append(timeout_seconds)
                return subprocess.CompletedProcess(args=command, returncode=0, stdout="ok", stderr="")

            with mock.patch("agent.validation_runner.ValidationRunner._run_command", side_effect=fake_run):
                runner.run_all(workspace, test_timeout_override=600)

        self.assertEqual([600], captured_timeouts)

    def test_run_all_collects_multiple_failures_by_default(self):
        runner = ValidationRunner(
            format_commands=[],
            lint_commands=[["ruff", "check", "."]],
            test_commands=[["pytest"]],
            timeout_seconds=30,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)

            def fake_run(path, command, timeout_seconds):
                del path, timeout_seconds
                if command[0] == "ruff":
                    return subprocess.CompletedProcess(args=command, returncode=1, stdout="lint failed", stderr="")
                return subprocess.CompletedProcess(args=command, returncode=1, stdout="test failed", stderr="")

            with mock.patch("agent.validation_runner.ValidationRunner._run_command", side_effect=fake_run):
                result = runner.run_all(workspace)

        self.assertFalse(result.success)
        self.assertEqual(["lint", "test"], [step.name for step in result.steps])
        summary = result.failed_steps_summary()
        self.assertIn("lint:FAIL ruff check .", summary)
        self.assertIn("test:FAIL pytest", summary)

    def test_run_all_fail_fast_stops_on_first_failure(self):
        runner = ValidationRunner(
            format_commands=[],
            lint_commands=[["ruff", "check", "."]],
            test_commands=[["pytest"]],
            timeout_seconds=30,
            fail_fast=True,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            calls = []

            def fake_run(path, command, timeout_seconds):
                del path, timeout_seconds
                calls.append(command)
                return subprocess.CompletedProcess(args=command, returncode=1, stdout="failed", stderr="")

            with mock.patch("agent.validation_runner.ValidationRunner._run_command", side_effect=fake_run):
                result = runner.run_all(workspace)

        self.assertFalse(result.success)
        self.assertEqual([["ruff", "check", "."]], calls)
        self.assertEqual(["lint"], [step.name for step in result.steps])

    def test_run_all_retries_after_cleaning_referenced_transient_artifacts(self):
        runner = ValidationRunner(
            format_commands=[],
            lint_commands=[["flake8"]],
            test_commands=[],
            timeout_seconds=30,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            artifact_dir = workspace / "apps" / "desktop" / "node_modules"
            artifact_dir.mkdir(parents=True)
            (artifact_dir / "bad.py").write_text("g\n", encoding="utf-8")
            calls = []

            def fake_run(path, command, timeout_seconds):
                del path, timeout_seconds
                calls.append(command)
                if artifact_dir.exists():
                    return subprocess.CompletedProcess(
                        args=command,
                        returncode=1,
                        stdout="apps/desktop/node_modules/pkg/bad.py:1: undefined name 'g'",
                        stderr="",
                    )
                return subprocess.CompletedProcess(args=command, returncode=0, stdout="ok", stderr="")

            with mock.patch("agent.validation_runner.ValidationRunner._run_command", side_effect=fake_run):
                result = runner.run_all(workspace)

        self.assertTrue(result.success)
        self.assertEqual([["flake8"], ["flake8"]], calls)
        self.assertFalse(artifact_dir.exists())

    def test_transient_artifact_dirs_are_configurable(self):
        runner = ValidationRunner(
            format_commands=[],
            lint_commands=[["flake8"]],
            test_commands=[],
            timeout_seconds=30,
            transient_artifact_dirs=["generated-vendor"],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            artifact_dir = workspace / "generated-vendor"
            artifact_dir.mkdir()
            (artifact_dir / "bad.py").write_text("g\n", encoding="utf-8")
            calls = []

            def fake_run(path, command, timeout_seconds):
                del path, timeout_seconds
                calls.append(command)
                if artifact_dir.exists():
                    return subprocess.CompletedProcess(
                        args=command,
                        returncode=1,
                        stdout="generated-vendor/bad.py:1: undefined name 'g'",
                        stderr="",
                    )
                return subprocess.CompletedProcess(args=command, returncode=0, stdout="ok", stderr="")

            with mock.patch("agent.validation_runner.ValidationRunner._run_command", side_effect=fake_run):
                result = runner.run_all(workspace)

        self.assertTrue(result.success)
        self.assertEqual([["flake8"], ["flake8"]], calls)
        self.assertFalse(artifact_dir.exists())

    def test_run_command_starts_new_session_and_kills_process_group_on_timeout(self):
        process = mock.Mock()
        process.pid = 1234
        process.poll.return_value = None
        process.returncode = None
        process.communicate.side_effect = [
            subprocess.TimeoutExpired(cmd=["pytest"], timeout=1, output="partial", stderr="err"),
            ("final stdout", "final stderr"),
        ]
        process.wait.side_effect = [
            subprocess.TimeoutExpired(cmd=["pytest"], timeout=3),
            None,
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            with mock.patch("agent.validation_runner.subprocess.Popen", return_value=process) as popen:
                with mock.patch("agent.validation_runner.os.getpgid", return_value=1234):
                    with mock.patch("agent.validation_runner.os.killpg") as killpg:
                        with self.assertRaises(subprocess.TimeoutExpired) as raised:
                            ValidationRunner._run_command(workspace, ["pytest"], 1)

        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.assertEqual(
            [mock.call(1234, signal.SIGTERM), mock.call(1234, signal.SIGKILL)],
            killpg.mock_calls,
        )
        self.assertEqual("final stdout", raised.exception.stdout)
        self.assertEqual("final stderr", raised.exception.stderr)


if __name__ == "__main__":
    unittest.main()
