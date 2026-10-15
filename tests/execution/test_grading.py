import pytest

from reasoning_graphs.execution.executor import CodeExecutor, ExecutionResult, TestResult as Result
from reasoning_graphs.execution.grading import grade_on_public, pick_solution
from reasoning_graphs.problems.two_sum import TWO_SUM
from reasoning_graphs.reasoning_node import ReasoningNode

CORRECT = """def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i"""
WRONG = "def two_sum(nums, target):\n    return [0, 1]"  # right on the first public test only


def make_leaves(codes: list[str]) -> list[ReasoningNode]:
    root = ReasoningNode("Problem: two sum.")
    leaves = []
    for code in codes:
        leaf = root.add_child("Code:\n```python\n" + code + "\n```")
        leaf.is_terminal = True
        leaf.code = code
        leaves.append(leaf)
    return leaves


def fake_result(passed: bool) -> ExecutionResult:
    """A result without running anything: one test, passed or failed."""
    return ExecutionResult([Result(test=None, status="passed" if passed else "failed")])


def test_grade_on_public_runs_each_leaf_on_the_public_tests():
    leaves = make_leaves([CORRECT, WRONG])
    results = grade_on_public(leaves, TWO_SUM, CodeExecutor(timeout=2.0))

    assert set(results) == {leaf.id for leaf in leaves}
    assert all(len(r.results) == len(TWO_SUM.public_tests) for r in results.values())
    assert results[leaves[0].id].all_passed
    assert not results[leaves[1].id].all_passed


@pytest.mark.parametrize("passes, expected_index", [
    pytest.param([False, True, True], 1, id="first passing leaf, not the first leaf"),
    pytest.param([True, True], 0, id="several pass: the first one"),
    pytest.param([False, False], None, id="none passes"),
    pytest.param([], None, id="no leaves"),
])
def test_pick_solution(passes, expected_index):
    leaves = make_leaves(["pass"] * len(passes))
    results = {leaf.id: fake_result(p) for leaf, p in zip(leaves, passes)}

    chosen = pick_solution(leaves, results)

    assert chosen is (None if expected_index is None else leaves[expected_index])
