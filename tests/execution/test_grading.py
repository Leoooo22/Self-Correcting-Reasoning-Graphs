import pytest

from reasoning_graphs.execution.executor import CodeExecutor, ExecutionResult, TestResult as Result
from reasoning_graphs.execution.grading import branch_value, grade_on_public, pick_by_branch, pick_solution
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


def make_branch_leaf(root: ReasoningNode, parent_score: float | None) -> ReasoningNode:
    """A code leaf under a reasoning step with the given score (None: code written directly under the root)."""
    parent = root
    if parent_score is not None:
        parent = root.add_child(f"Step 1: branch scored {parent_score}")
        parent.score = parent_score
    leaf = parent.add_child("Code:\n```python\npass\n```")
    leaf.is_terminal = True
    leaf.code = "pass"
    return leaf


@pytest.mark.parametrize("parent_score, expected", [
    pytest.param(0.75, 0.75, id="scored branch"),
    pytest.param(0.0, 0.0, id="zero is a real score"),
    pytest.param(None, -1.0, id="code written under the root"),
])
def test_branch_value(parent_score, expected):
    leaf = make_branch_leaf(ReasoningNode("Problem: p."), parent_score)
    assert branch_value(leaf) == expected


@pytest.mark.parametrize("parent_scores, passes, expected_index", [
    pytest.param([0.25, 0.75, 0.5], [True, True, True], 1, id="best branch, not the first leaf"),
    pytest.param([0.25, 1.0], [True, False], 0, id="a failing leaf never wins, whatever its branch"),
    pytest.param([None, 0.0], [True, True], 1, id="direct code loses even to a branch scored 0"),
    pytest.param([None], [True], 0, id="direct code is kept when it is the only passing leaf"),
    pytest.param([0.5, 0.5], [True, True], 0, id="ties: the first one"),
    pytest.param([0.5, 0.5], [False, False], None, id="none passes"),
    pytest.param([], [], None, id="no leaves"),
])
def test_pick_by_branch(parent_scores, passes, expected_index):
    root = ReasoningNode("Problem: p.")
    leaves = [make_branch_leaf(root, score) for score in parent_scores]
    results = {leaf.id: fake_result(p) for leaf, p in zip(leaves, passes)}

    chosen = pick_by_branch(leaves, results)

    assert chosen is (None if expected_index is None else leaves[expected_index])


def test_without_scores_pick_by_branch_is_pick_solution():
    # A tree grown without an evaluator has no scores: every branch value is -1, so the first passing leaf wins
    leaves = make_leaves(["pass"] * 3)
    results = {leaf.id: fake_result(p) for leaf, p in zip(leaves, [False, True, True])}
    assert pick_by_branch(leaves, results) is pick_solution(leaves, results) is leaves[1]



def make_rollout_leaf(origin: ReasoningNode) -> ReasoningNode:
    """A code leaf as a rollout leaves it: on an unscored copy of the path, with .origin = the real node."""
    copy = origin.detached_copy().add_child("Step 2: more.")
    leaf = copy.add_child("Code:\n```python\npass\n```")
    leaf.is_terminal = True
    leaf.code = "pass"
    leaf.origin = origin
    return leaf


@pytest.mark.parametrize("origin_score, expected", [
    pytest.param(0.75, 0.75, id="score of the node the rollout started from"),
    pytest.param(0.0, 0.0, id="zero is a real score"),
    pytest.param(None, -1.0, id="unscored origin, e.g. the root in Best-of-N"),
])
def test_branch_value_of_a_rollout_solution(origin_score, expected):
    root = ReasoningNode("Problem: p.")
    origin = root.add_child("Step 1: plan.") if origin_score is not None else root
    origin.score = origin_score
    leaf = make_rollout_leaf(origin)

    assert leaf.parent.score is None  # its parent is an unscored copy...
    assert branch_value(leaf) == expected  # ...so the origin decides


def test_pick_by_branch_mixes_tree_leaves_and_rollout_solutions():
    root = ReasoningNode("Problem: p.")
    tree_leaf = make_branch_leaf(root, 0.5)
    good_origin = root.add_child("Step 1: good plan.")
    good_origin.score = 1.0
    rollout_leaf = make_rollout_leaf(good_origin)
    leaves = [tree_leaf, rollout_leaf]
    results = {leaf.id: fake_result(True) for leaf in leaves}

    assert pick_by_branch(leaves, results) is rollout_leaf  # comes from the better branch
