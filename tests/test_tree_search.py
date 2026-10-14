from config import SearchConfig
from reasoning_node import ReasoningNode
from step_format import CODE_PREFIX
from step_generator import StepGenerator
from tree_search import expand

VALID_CODE = "```python\ndef f(a):\n    return a\n```"
BROKEN_CODE = "```python\ndef f(:\n```"


class FixedGenerator(StepGenerator):
    """Returns fixed outputs and records what it was asked (test spy)."""

    def __init__(self, outputs: list[str]):
        self.outputs = outputs
        self.last_prompt = None
        self.last_n = None

    def generate_steps(self, prompt: str, n: int) -> list[str]:
        self.last_prompt = prompt
        self.last_n = n
        return self.outputs


def test_root_expansion_cleans_deduplicates_and_marks_code():
    config = SearchConfig()
    root = ReasoningNode("Problem: return the input.")
    generator = FixedGenerator(["Step 1: A", "step 9: A", "", "B", CODE_PREFIX + "\n" + VALID_CODE])

    children = expand(root, generator, config)

    assert generator.last_n == config.branching[0]
    assert [c.step_text for c in children] == ["Step 1: A", "Step 1: B", CODE_PREFIX + "\n" + VALID_CODE]
    assert [c.is_terminal for c in children] == [False, False, True]
    assert children[0].code is None and children[2].code == "def f(a):\n    return a"


def test_code_is_forced_at_maximum_depth():
    config = SearchConfig()
    node = ReasoningNode("Problem: return the input.")
    for i in range(config.max_steps):
        node = node.add_child(f"Step {i + 1}: reasoning")
    generator = FixedGenerator([VALID_CODE, "Step 4: more thinking", BROKEN_CODE])

    children = expand(node, generator, config)

    assert generator.last_prompt.endswith("\n" + CODE_PREFIX)
    assert generator.last_n == config.code_branching
    assert len(children) == 1 and children[0].is_terminal and children[0].code is not None
    assert children[0].depth == config.max_steps + 1


def test_terminal_node_is_not_expanded():
    terminal = ReasoningNode("Problem: return the input.").add_child(CODE_PREFIX + "\n" + VALID_CODE)
    terminal.is_terminal = True
    generator = FixedGenerator(["Step 1: A"])

    assert expand(terminal, generator, SearchConfig()) == []
    assert generator.last_prompt is None  # the generator was not even called


def test_only_invalid_outputs_give_no_children():
    leaf = ReasoningNode("Problem: nothing valid.")
    generator = FixedGenerator(["", "   ", "Code: no block", CODE_PREFIX + "\n" + BROKEN_CODE])

    assert expand(leaf, generator, SearchConfig()) == []
    assert leaf.children == []
