from reasoning_node import ReasoningNode
from step_generator import StepGenerator
from config import SearchConfig
from step_format import CODE_PREFIX, ParsedStep, parse_step, format_step


def expand(node: ReasoningNode, generator: StepGenerator, config: SearchConfig) -> list[ReasoningNode]:
    if node.is_terminal:
        return []

    force_code = node.depth >= config.max_steps
    if force_code:
        k = config.code_branching
    else:
        k = config.branching[node.depth]

    prompt = node.get_full_text()
    if force_code:
        prompt += "\n" + CODE_PREFIX

    raw_steps = generator.generate_steps(prompt, k)

    # Formatted text -> parsed step. Using the text as key also removes duplicates.
    unique_steps: dict[str, ParsedStep] = {}
    for raw in raw_steps:
        if force_code:
            # The model only continues after "Code:", so its output lacks the label
            raw = CODE_PREFIX + "\n" + raw
        parsed = parse_step(raw, force_code)
        if parsed is None:
            continue
        text = format_step(parsed, node.depth + 1)
        unique_steps[text] = parsed

    children = []
    for text, parsed in unique_steps.items():
        child = node.add_child(text)
        if parsed.is_code:
            child.is_terminal = True
            child.code = parsed.code
        children.append(child)

    return children


if __name__ == "__main__":
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

    config = SearchConfig()
    valid_code = "```python\ndef f(a):\n    return a\n```"
    broken_code = "```python\ndef f(:\n```"

    # Case 1: root node, mixed outputs
    root = ReasoningNode("Problem: return the input.")
    generator = FixedGenerator(["Step 1: A", "step 9: A", "", "B", CODE_PREFIX + "\n" + valid_code])
    children = expand(root, generator, config)
    for child in children:
        print(child)
    assert generator.last_n == config.branching[0]
    assert [c.step_text for c in children] == ["Step 1: A", "Step 1: B", CODE_PREFIX + "\n" + valid_code]
    assert [c.is_terminal for c in children] == [False, False, True]
    assert children[0].code is None and children[2].code == "def f(a):\n    return a"
    print("1. root expansion: OK\n")

    # Case 2: node at maximum depth, code is forced
    node = root
    for i in range(config.max_steps):
        node = node.add_child(f"Step {i + 1}: reasoning")
    generator = FixedGenerator([valid_code, "Step 4: more thinking", broken_code])
    children = expand(node, generator, config)
    for child in children:
        print(child)
    assert generator.last_prompt.endswith("\n" + CODE_PREFIX)
    assert generator.last_n == config.code_branching
    assert len(children) == 1 and children[0].is_terminal and children[0].code is not None
    assert children[0].depth == config.max_steps + 1
    print("2. forced code at maximum depth: OK\n")

    # Case 3: terminal nodes are not expanded, and the generator is not called
    generator = FixedGenerator(["Step 1: A"])
    assert expand(children[0], generator, config) == []
    assert generator.last_prompt is None
    print("3. terminal node not expanded: OK")

    # Case 4: only invalid outputs
    generator = FixedGenerator(["", "   ", "Code: no block", CODE_PREFIX + "\n" + broken_code])
    leaf = ReasoningNode("Problem: nothing valid.")
    assert expand(leaf, generator, config) == [] and leaf.children == []
    print("4. only invalid outputs: OK")

    print("\nAll checks passed")
