from reasoning_graphs.reasoning_node import ReasoningNode
from reasoning_graphs.generators.base import StepGenerator
from reasoning_graphs.config import SearchConfig
from reasoning_graphs.step_format import CODE_PREFIX, ParsedStep, parse_step, format_step


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
