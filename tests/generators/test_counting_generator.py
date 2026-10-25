from reasoning_graphs.config import SearchConfig
from reasoning_graphs.generators.counting_generator import CountingGenerator
from reasoning_graphs.reasoning_node import ReasoningNode
from reasoning_graphs.search.expand import expand
from reasoning_graphs.search.rollout import rollout
from tests.helpers import ScriptedGenerator


def test_counts_calls_and_outputs_and_passes_answers_through():
    inner = ScriptedGenerator([["a", "b", "c", "d"], ["e"]])
    counter = CountingGenerator(inner)

    assert counter.generate_steps("p", 4) == ["a", "b", "c", "d"]
    assert counter.generate_steps("q", 1) == ["e"]
    assert (counter.calls, counter.outputs) == (2, 5)
    assert inner.calls == [("p", 4), ("q", 1)]  # the real generator saw exactly the same requests


def test_counts_everything_a_search_asks():
    # A root expansion asks for 4 outputs in one call; a rollout from depth 1 makes 3 calls of 1 output
    config = SearchConfig()
    inner = ScriptedGenerator([
        ["Step 1: A", "Step 1: B", "Step 1: C", "Step 1: D"],
        "Step 2: More.", "Step 3: Even more.", "```python\ndef f():\n    pass\n```",
    ])
    counter = CountingGenerator(inner)
    root = ReasoningNode("Problem: p.")

    children = expand(root, counter, config)
    rollout(children[0], counter, config)

    assert (counter.calls, counter.outputs) == (4, 7)
