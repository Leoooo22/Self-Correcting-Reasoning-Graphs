from reasoning_graphs.config import SearchConfig
from reasoning_graphs.generators.base import StepGenerator
from reasoning_graphs.reasoning_node import ReasoningNode
from reasoning_graphs.search.evaluator import Evaluator
from reasoning_graphs.search.expand import expand


def beam_search(root: ReasoningNode, generator: StepGenerator, config: SearchConfig,
                evaluator: Evaluator) -> list[ReasoningNode]:
    """Level by level, expands only the config.beam_width best-scored non-terminal nodes.

    Every new node gets its score in node.score. Returns all the nodes of the tree (root first);
    the code nodes among them are the candidate solutions.
    """
    all_nodes = [root]
    frontier = [root]

    while frontier:
        # 1. Expand every node of the frontier
        children = []
        for node in frontier:
            children.extend(expand(node, generator, config))
        all_nodes.extend(children)

        # 2. Score the new nodes; code nodes are final candidates and never enter the frontier
        candidates = []
        for node in children:
            node.score = evaluator.score(node)
            if not node.is_terminal:
                candidates.append(node)

        # 3. Keep the best ones (sort is stable: ties keep creation order)
        candidates.sort(key=lambda node: node.score, reverse=True)
        frontier = candidates[:config.beam_width]

    return all_nodes
