import pytest

from reasoning_graphs.config import SearchConfig
from reasoning_graphs.execution.executor import ExecutionResult, TestResult as Result
from reasoning_graphs.execution.grading import pick_by_branch
from reasoning_graphs.reasoning_node import ReasoningNode
from reasoning_graphs.generators.counting_generator import CountingGenerator
from reasoning_graphs.search.baselines import best_of_n, best_within_budget
from reasoning_graphs.step_format import CODE_PREFIX
from tests.helpers import ScriptedGenerator


def labelled_code(name: str) -> str:
    return f"{CODE_PREFIX}\n```python\ndef {name}():\n    return 1\n```"


def test_n_independent_solutions_from_the_root():
    # Each solution here is written right away, so each rollout is a single call
    generator = ScriptedGenerator([labelled_code("a"), labelled_code("b"), labelled_code("c")])
    root = ReasoningNode("Problem: p.")

    leaves = best_of_n(root, generator, SearchConfig(), n=3)

    assert [leaf.code for leaf in leaves] == ["def a():\n    return 1", "def b():\n    return 1", "def c():\n    return 1"]
    assert all(leaf.is_terminal and leaf.score is None for leaf in leaves)  # no scores
    assert root.children == []  # rollouts work on detached copies


def test_stuck_rollouts_are_left_out():
    generator = ScriptedGenerator([labelled_code("a"), "", "Step 1: Think.", "", labelled_code("b")])

    leaves = best_of_n(ReasoningNode("Problem: p."), generator, SearchConfig(), n=4)

    assert [leaf.code for leaf in leaves] == ["def a():\n    return 1", "def b():\n    return 1"]
    assert len(generator.calls) == 5  # rollout 2 got stuck at once, rollout 3 after one step: they still cost calls


def test_single_generation_is_n_equal_1():
    generator = ScriptedGenerator(["Step 1: Think.", "Step 2: More.", "Step 3: Even more.", "```python\ndef f():\n    pass\n```"])

    leaves = best_of_n(ReasoningNode("Problem: p."), generator, SearchConfig(), n=1)

    assert len(leaves) == 1 and leaves[0].depth == 4  # one full rollout down to forced code
    assert [n for _, n in generator.calls] == [1, 1, 1, 1]


def test_without_scores_the_choice_is_the_first_passing_solution():
    leaves = best_of_n(ReasoningNode("Problem: p."), ScriptedGenerator([labelled_code(x) for x in "abc"]),
                       SearchConfig(), n=3)
    results = {leaf.id: ExecutionResult([Result(None, status)]) for leaf, status in
               zip(leaves, ["failed", "passed", "passed"])}
    assert pick_by_branch(leaves, results) is leaves[1]


def test_n_must_be_positive():
    with pytest.raises(ValueError):
        best_of_n(ReasoningNode("Problem: p."), ScriptedGenerator([]), SearchConfig(), n=0)


def test_best_within_budget_stops_once_the_budget_is_spent():
    # Each rollout here costs 2 outputs (one step, then the code): a budget of 5 needs 3 rollouts
    script = []
    for name in "abcd":
        script += ["Step 1: Think.", labelled_code(name)]
    counter = CountingGenerator(ScriptedGenerator(script))
    counter.outputs = 10  # outputs spent before: only the new ones count

    leaves = best_within_budget(ReasoningNode("Problem: p."), counter, SearchConfig(), budget=5)

    assert len(leaves) == 3 and counter.outputs == 16  # overshoots by 1 rather than stopping short


def test_best_within_budget_makes_at_least_one_rollout():
    counter = CountingGenerator(ScriptedGenerator([labelled_code("a")]))
    leaves = best_within_budget(ReasoningNode("Problem: p."), counter, SearchConfig(), budget=0)
    assert len(leaves) == 1
