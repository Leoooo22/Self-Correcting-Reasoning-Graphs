from reasoning_graphs.config import SearchConfig
from reasoning_graphs.reasoning_node import ReasoningNode
from reasoning_graphs.search.beam import beam_search
from reasoning_graphs.step_format import CODE_PREFIX
from tests.helpers import FixedScoreEvaluator, ScriptedGenerator


def code(name: str) -> str:
    """Unlabelled code block, as the model writes it when code is forced."""
    return f"```python\ndef {name}():\n    return 1\n```"


def labelled_code(name: str) -> str:
    return CODE_PREFIX + "\n" + code(name)


def by_text(nodes: list[ReasoningNode]) -> dict[str, ReasoningNode]:
    return {node.step_text: node for node in nodes}


def test_keeps_only_the_best_nodes_at_each_level():
    # max_steps 2, beam width 2: 4 first steps, keep 2; 4 second steps, keep 2; then forced code
    config = SearchConfig(branching=[4, 2], code_branching=1, beam_width=2)
    generator = ScriptedGenerator([
        ["Step 1: A", "Step 1: B", "Step 1: C", "Step 1: D"],  # root
        ["Step 2: B1", "Step 2: B2"],                          # B (best first)
        ["Step 2: A1", "Step 2: A2"],                          # A
        code("b2"),                                            # B2: code forced
        code("a1"),                                            # A1: code forced
    ])
    evaluator = FixedScoreEvaluator({
        "Step 1: A": 0.5, "Step 1: B": 0.75, "Step 1: C": 0.125, "Step 1: D": 0.0,
        "Step 2: B1": 0.25, "Step 2: B2": 1.0, "Step 2: A1": 0.75, "Step 2: A2": 0.0,
    }, default=1.0)
    root = ReasoningNode("Problem: p.")

    nodes = beam_search(root, generator, config, evaluator)
    node = by_text(nodes)

    assert len(nodes) == 1 + 4 + 4 + 2
    assert len(generator.calls) == 5  # root, B, A, B2, A1: nothing else was expanded
    # Pruned nodes stay in the tree, with their score, but get no children
    for pruned in ("Step 1: C", "Step 1: D", "Step 2: B1", "Step 2: A2"):
        assert node[pruned].children == [] and node[pruned].score is not None
    assert [n.code for n in nodes if n.is_terminal] == ["def b2():\n    return 1", "def a1():\n    return 1"]


def test_every_new_node_is_scored_but_not_the_root():
    config = SearchConfig(branching=[2], code_branching=1, beam_width=2)
    generator = ScriptedGenerator([["Step 1: A", "Step 1: B"], code("a"), code("b")])
    evaluator = FixedScoreEvaluator({}, default=0.5)
    root = ReasoningNode("Problem: p.")

    nodes = beam_search(root, generator, config, evaluator)

    assert root.score is None and root not in evaluator.scored
    assert all(n.score == 0.5 for n in nodes if not n.is_root())
    assert len(evaluator.scored) == len(nodes) - 1  # each node scored exactly once


def test_code_nodes_do_not_take_beam_slots():
    # Two code children with the top score and one reasoning step: with beam width 1, the step is kept
    config = SearchConfig(branching=[3], code_branching=1, beam_width=1)
    generator = ScriptedGenerator([
        [labelled_code("x"), labelled_code("y"), "Step 1: A"],
        code("a"),
    ])
    evaluator = FixedScoreEvaluator({"Step 1: A": 0.25}, default=1.0)

    nodes = beam_search(ReasoningNode("Problem: p."), generator, config, evaluator)
    node = by_text(nodes)

    assert len(node["Step 1: A"].children) == 1  # expanded although the code nodes scored higher
    assert sum(n.is_terminal for n in nodes) == 3


def test_stops_when_every_child_is_code():
    config = SearchConfig(branching=[4, 2], code_branching=1, beam_width=2)
    generator = ScriptedGenerator([[labelled_code("x"), labelled_code("y")]])

    nodes = beam_search(ReasoningNode("Problem: p."), generator, config, FixedScoreEvaluator({}))

    assert len(generator.calls) == 1 and len(nodes) == 3


def test_ties_keep_creation_order():
    config = SearchConfig(branching=[3], code_branching=1, beam_width=1)
    generator = ScriptedGenerator([["Step 1: A", "Step 1: B", "Step 1: C"], code("a")])

    nodes = beam_search(ReasoningNode("Problem: p."), generator, config, FixedScoreEvaluator({}, default=0.5))
    node = by_text(nodes)

    assert len(node["Step 1: A"].children) == 1
    assert node["Step 1: B"].children == [] and node["Step 1: C"].children == []


def test_beam_wider_than_the_candidates():
    config = SearchConfig(branching=[2], code_branching=1, beam_width=10)
    generator = ScriptedGenerator([["Step 1: A", "Step 1: B"], code("a"), code("b")])

    nodes = beam_search(ReasoningNode("Problem: p."), generator, config, FixedScoreEvaluator({}))

    assert sum(n.is_terminal for n in nodes) == 2  # both kept and expanded
