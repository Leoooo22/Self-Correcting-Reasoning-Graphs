import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reasoning_graphs.problems.problem import TestCase


@dataclass
class TestResult:
    test: TestCase
    status: str  # "passed" | "failed" | "error" | "timeout"
    actual: Any = None
    error: str | None = None


@dataclass
class ExecutionResult:
    results: list[TestResult]

    @property
    def num_passed(self) -> int:
        return sum(t.status == "passed" for t in self.results)

    @property
    def all_passed(self) -> bool:
        if not self.results:
            return False
        return all(t.status == "passed" for t in self.results)


class CodeExecutor:
    def __init__(self, timeout: float = 5.0):
        self.timeout = timeout
        self.runner_path = Path(__file__).parent / "runner.py"

    def run(self, code: str, function_name: str, tests: list[TestCase]) -> ExecutionResult:
        payload = json.dumps({
            "code": code,
            "function_name": function_name,
            "tests": [{"args": t.args, "expected": t.expected} for t in tests],
        })

        # Run in a temporary directory, so files created by the model's code don't end up in the project
        with tempfile.TemporaryDirectory() as work_dir:
            try:
                completed = subprocess.run(
                    [sys.executable, str(self.runner_path)],
                    input=payload,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=work_dir,
                )
            except subprocess.TimeoutExpired:
                return _all_tests(tests, "timeout", f"timed out after {self.timeout} seconds")

        # The runner may die without answering (e.g. os._exit, out of memory)
        try:
            raw_results = json.loads(completed.stdout)["results"]
        except (json.JSONDecodeError, KeyError, TypeError):
            stderr_tail = completed.stderr.strip()[-300:]
            return _all_tests(tests, "error", f"runner crashed (exit code {completed.returncode}): {stderr_tail}")
        if len(raw_results) != len(tests):
            return _all_tests(tests, "error", f"runner returned {len(raw_results)} results for {len(tests)} tests")

        return ExecutionResult([
            TestResult(test, r["status"], r["actual"], r["error"])
            for test, r in zip(tests, raw_results)
        ])


def _all_tests(tests: list[TestCase], status: str, error: str) -> ExecutionResult:
    """Same outcome for every test, used when the code could not be run at all."""
    return ExecutionResult([TestResult(test, status, None, error) for test in tests])
