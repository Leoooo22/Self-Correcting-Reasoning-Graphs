from reasoning_graphs.config import SearchConfig
from reasoning_graphs.reasoning_node import ReasoningNode
from reasoning_graphs.search.rollout import rollout
from reasoning_graphs.step_format import CODE_PREFIX
from tests.helpers import ScriptedGenerator

CODE = "```python\ndef f(a):\n    return a\n```"


def make_tree():
    root = ReasoningNode("Problem: return the input.")
    step_1 = root.add_child("Step 1: Think.")
    return root, step_1


def test_reaches_code_one_output_per_call():
    # From depth 1 with max_steps 3: Step 2, Step 3, then code is forced
    _, step_1 = make_tree()
    generator = ScriptedGenerator(["Step 2: More.", "Step 3: Even more.", CODE])

    result = rollout(step_1, generator, SearchConfig(branching=[4, 2, 2], code_branching=2))

    assert result.is_terminal and result.code == "def f(a):\n    return a"
    assert result.depth == 4
    assert result.get_full_text().startswith(step_1.get_full_text())
    assert [n for _, n in generator.calls] == [1, 1, 1]
    assert generator.calls[-1][0].endswith("\n" + CODE_PREFIX)  # code forced at the same depth as the search


def test_early_code_ends_the_rollout():
    _, step_1 = make_tree()
    generator = ScriptedGenerator([CODE_PREFIX + "\n" + CODE])

    result = rollout(step_1, generator, SearchConfig())

    assert result.is_terminal and result.depth == 2
    assert len(generator.calls) == 1


def test_invalid_output_means_stuck():
    _, step_1 = make_tree()
    generator = ScriptedGenerator(["Step 2: More.", ""])

    assert rollout(step_1, generator, SearchConfig()) is None
    assert len(generator.calls) == 2


def test_code_node_is_returned_as_is():
    _, step_1 = make_tree()
    leaf = step_1.add_child(CODE_PREFIX + "\n" + CODE)
    leaf.is_terminal = True
    generator = ScriptedGenerator([])

    assert rollout(leaf, generator, SearchConfig()) is leaf
    assert generator.calls == []  # the generator is not even called
    assert leaf.origin is None  # a node of the real tree: it does not come from a rollout


def test_code_node_remembers_the_real_node_it_came_from():
    _, step_1 = make_tree()
    generator = ScriptedGenerator(["Step 2: More.", "Step 3: Even more.", CODE])

    result = rollout(step_1, generator, SearchConfig())

    assert result.origin is step_1  # the real node, not its detached copy
    assert result.parent.score is None  # the copies along the way carry no score: origin is the only link
    assert step_1.origin is None  # nodes of the real tree have no origin


def test_real_tree_is_unchanged():
    root, step_1 = make_tree()
    generator = ScriptedGenerator(["Step 2: More.", "Step 3: Even more.", CODE])

    rollout(step_1, generator, SearchConfig())

    assert root.children == [step_1] and step_1.children == []
