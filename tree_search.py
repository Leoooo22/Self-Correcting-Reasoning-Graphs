from reasoning_node import ReasoningNode
from step_generator import StepGenerator, MockGenerator

ANSWER_PREFIX = "Answer:"


def expand(node: ReasoningNode, generator: StepGenerator, k: int) -> list[ReasoningNode]:
    if node.is_terminal:
        return []

    prompt = node.get_full_text()
    raw_steps = generator.generate_steps(prompt, k)

    clean_steps = []
    for step in raw_steps:
        step = step.strip()
        if step:
            clean_steps.append(step)

    unique_steps = list(dict.fromkeys(clean_steps))

    children = []
    for step in unique_steps:
        child = node.add_child(step)
        if step.startswith(ANSWER_PREFIX):
            child.is_terminal = True
        children.append(child)

    return children


if __name__ == "__main__":
    candidates = [
        ["3 * 4 = 12", "3 * 4 = 7", "4 * 5 = 20"],
        ["12 + 5 = 17", "12 + 5 = 18"],
        ["Answer: 17", "Answer: 18"],
    ]
    generator = MockGenerator(candidates, seed=42)
    root = ReasoningNode("What is 3 * 4 + 5?")

    # Expand the tree level by level: each level is the list of children of the previous one
    frontier = [root]
    for _ in range(4):
        next_frontier = []
        for node in frontier:
            next_frontier.extend(expand(node, generator, 3))
        frontier = next_frontier

    root.print_tree()
