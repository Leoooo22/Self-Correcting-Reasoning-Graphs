import pytest

from reasoning_graphs.config import SearchConfig
from reasoning_graphs.problems.two_sum import TWO_SUM
from reasoning_graphs.reasoning_node import ReasoningNode
from reasoning_graphs.search.evaluator import RolloutEvaluator
from reasoning_graphs.step_format import CODE_PREFIX
from tests.helpers import ScriptedGenerator, SpyExecutor

CORRECT = """def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i"""
WRONG = "def two_sum(nums, target):\n    return [0, 1]"  # right on the first public test only


def code_step(code: str) -> str:
    """A model output that writes the code right away, so each rollout is a single call."""
    return f"{CODE_PREFIX}\n```python\n{code}\n```"


def make_evaluator(script: list[str], n: int = 4):
    generator = ScriptedGenerator(script)
    executor = SpyExecutor()
    evaluator = RolloutEvaluator(generator, SearchConfig(), executor, TWO_SUM, n=n)
    return evaluator, generator, executor


def make_step_node() -> ReasoningNode:
    return ReasoningNode(TWO_SUM.to_prompt()).add_child("Step 1: Use a hash map.")


@pytest.mark.parametrize("script, expected", [
    pytest.param([code_step(CORRECT)] * 4, 1.0, id="all rollouts pass"),
    pytest.param([code_step(WRONG)] * 4, 0.0, id="none passes"),
    pytest.param([code_step(c) for c in (CORRECT, WRONG, CORRECT, WRONG)], 0.5, id="half pass"),
    pytest.param([code_step(CORRECT), "", code_step(CORRECT), code_step(CORRECT)], 0.75,
                 id="a stuck rollout counts as a failure"),
])
def test_score_is_the_fraction_of_passing_rollouts(script, expected):
    evaluator, generator, _ = make_evaluator(script)
    assert evaluator.score(make_step_node()) == expected
    assert len(generator.calls) == 4  # one call per rollout here


def test_only_public_tests_are_used():
    evaluator, _, executor = make_evaluator([code_step(CORRECT)] * 4)
    evaluator.score(make_step_node())
    assert executor.runs and all(tests == TWO_SUM.public_tests for tests in executor.runs)


def test_stuck_rollouts_are_not_executed():
    evaluator, _, executor = make_evaluator(["", "", code_step(CORRECT), ""])
    assert evaluator.score(make_step_node()) == 0.25
    assert len(executor.runs) == 1


@pytest.mark.parametrize("code, expected", [
    pytest.param(CORRECT, 1.0, id="passing code"),
    pytest.param(WRONG, 0.0, id="failing code"),
])
def test_code_node_is_scored_by_its_own_tests(code, expected):
    leaf = make_step_node().add_child(code_step(code))
    leaf.is_terminal = True
    leaf.code = code
    evaluator, generator, executor = make_evaluator([])

    assert evaluator.score(leaf) == expected
    assert generator.calls == [] and len(executor.runs) == 1  # no rollout, one execution


def test_real_tree_is_unchanged():
    node = make_step_node()
    evaluator, _, _ = make_evaluator([code_step(CORRECT)] * 4)
    evaluator.score(node)
    assert node.children == []


def test_n_must_be_positive():
    with pytest.raises(ValueError):
        make_evaluator([], n=0)


def test_passing_rollout_solutions_are_kept():
    node = make_step_node()
    evaluator, _, _ = make_evaluator([code_step(CORRECT), code_step(WRONG), code_step(CORRECT), ""] + [code_step(CORRECT)] * 4)

    evaluator.score(node)
    assert len(evaluator.rollout_leaves) == 2  # only the passing ones: not the wrong one, not the stuck one
    assert all(leaf.is_terminal and leaf.origin is node for leaf in evaluator.rollout_leaves)

    other = make_step_node()
    evaluator.score(other)
    assert len(evaluator.rollout_leaves) == 6  # the list grows across score calls
    assert [leaf.origin for leaf in evaluator.rollout_leaves[2:]] == [other] * 4


def test_code_nodes_of_the_tree_are_not_added():
    leaf = make_step_node().add_child(code_step(CORRECT))
    leaf.is_terminal = True
    leaf.code = CORRECT
    evaluator, _, _ = make_evaluator([])

    evaluator.score(leaf)
    assert evaluator.rollout_leaves == []  # already a candidate as a node of the tree
