import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from problem import TestCase


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


if __name__ == "__main__":
    import time

    from problems import TWO_SUM
    from step_format import extract_code
    import main as demo

    executor = CodeExecutor(timeout=2.0)
    public = TWO_SUM.public_tests

    def check(description: str, code: str, tests: list[TestCase] = public) -> ExecutionResult:
        result = executor.run(code, "two_sum", tests)
        statuses = [r.status for r in result.results]
        print(f"{description:<38} {statuses}")
        return result

    correct = """
def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
"""
    result = check("correct solution", correct)
    assert result.all_passed and result.num_passed == 2

    values_instead_of_indices = """
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [nums[i], nums[j]]
"""
    result = check("returns values instead of indices", values_instead_of_indices)
    assert all(r.status == "failed" for r in result.results)
    assert result.results[0].actual == [2, 7]

    result = check("always [0, 1]: right on first test only", "def two_sum(nums, target):\n    return [0, 1]")
    assert result.num_passed == 1 and not result.all_passed

    result = check("raises IndexError", "def two_sum(nums, target):\n    return nums[100]")
    assert all(r.status == "error" and "IndexError" in r.error for r in result.results)

    result = check("function with the wrong name", "def solve(nums, target):\n    return [0, 1]")
    assert all(r.status == "error" and "not found" in r.error for r in result.results)

    start = time.perf_counter()
    result = check("infinite loop", "def two_sum(nums, target):\n    while True:\n        pass")
    elapsed = time.perf_counter() - start
    assert all(r.status == "timeout" for r in result.results)
    assert elapsed < executor.timeout + 1, f"timeout took too long: {elapsed:.1f}s"
    print(f"{'':<38} (stopped after {elapsed:.1f}s)")

    result = check("os._exit kills the runner", "import os\ndef two_sum(nums, target):\n    os._exit(1)")
    assert all(r.status == "error" and "crashed" in r.error for r in result.results)

    result = check("no tests", correct, tests=[])
    assert not result.all_passed

    # Integration: the reference solution passes every Two Sum test, public and hidden
    result = check("reference on all 7 tests", correct, TWO_SUM.public_tests + TWO_SUM.hidden_tests)
    assert result.all_passed

    # Integration: the demo's codes get the right verdict on all tests
    all_tests = TWO_SUM.public_tests + TWO_SUM.hidden_tests
    for name, block, should_pass in [
        ("demo HASH_MAP_CODE", demo.HASH_MAP_CODE, True),
        ("demo BRUTE_FORCE_CODE", demo.BRUTE_FORCE_CODE, True),
        ("demo WRONG_CODE", demo.WRONG_CODE, False),
    ]:
        result = check(name, extract_code(block), all_tests)
        assert result.all_passed == should_pass

    print("\nAll checks passed")
