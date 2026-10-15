import time

import pytest

import main as demo
from reasoning_graphs.execution.executor import CodeExecutor
from reasoning_graphs.problems.two_sum import TWO_SUM
from reasoning_graphs.step_format import extract_code

PUBLIC = TWO_SUM.public_tests
ALL_TESTS = TWO_SUM.public_tests + TWO_SUM.hidden_tests

CORRECT = """
def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
"""


@pytest.fixture
def executor():
    return CodeExecutor(timeout=2.0)


def statuses(result):
    return [r.status for r in result.results]


def test_correct_solution(executor):
    result = executor.run(CORRECT, "two_sum", PUBLIC)
    assert result.all_passed and result.num_passed == 2


def test_values_instead_of_indices(executor):
    code = """
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [nums[i], nums[j]]
"""
    result = executor.run(code, "two_sum", PUBLIC)
    assert statuses(result) == ["failed", "failed"]
    assert result.results[0].actual == [2, 7]


def test_right_on_the_first_test_only(executor):
    result = executor.run("def two_sum(nums, target):\n    return [0, 1]", "two_sum", PUBLIC)
    assert result.num_passed == 1 and not result.all_passed


def test_exception_is_an_error(executor):
    result = executor.run("def two_sum(nums, target):\n    return nums[100]", "two_sum", PUBLIC)
    assert all(r.status == "error" and "IndexError" in r.error for r in result.results)


def test_function_with_the_wrong_name(executor):
    result = executor.run("def solve(nums, target):\n    return [0, 1]", "two_sum", PUBLIC)
    assert all(r.status == "error" and "not found" in r.error for r in result.results)


def test_infinite_loop_times_out(executor):
    start = time.perf_counter()
    result = executor.run("def two_sum(nums, target):\n    while True:\n        pass", "two_sum", PUBLIC)
    elapsed = time.perf_counter() - start
    assert statuses(result) == ["timeout", "timeout"]
    assert elapsed < executor.timeout + 1, f"timeout took too long: {elapsed:.1f}s"


def test_runner_killed_by_os_exit(executor):
    result = executor.run("import os\ndef two_sum(nums, target):\n    os._exit(1)", "two_sum", PUBLIC)
    assert all(r.status == "error" and "crashed" in r.error for r in result.results)


def test_no_tests_is_not_a_pass(executor):
    assert not executor.run(CORRECT, "two_sum", []).all_passed


def test_reference_passes_all_two_sum_tests(executor):
    assert executor.run(CORRECT, "two_sum", ALL_TESTS).all_passed


@pytest.mark.parametrize("block, should_pass", [
    pytest.param(demo.HASH_MAP_CODE, True, id="hash map"),
    pytest.param(demo.BRUTE_FORCE_CODE, True, id="brute force"),
    pytest.param(demo.WRONG_CODE, False, id="wrong but valid"),
])
def test_demo_codes_get_the_right_verdict(executor, block, should_pass):
    assert executor.run(extract_code(block), "two_sum", ALL_TESTS).all_passed == should_pass
