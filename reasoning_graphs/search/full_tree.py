from reasoning_graphs.config import SearchConfig
from reasoning_graphs.search.expand import expand
from reasoning_graphs.generators.base import StepGenerator
from reasoning_graphs.reasoning_node import ReasoningNode


def grow_full_tree(root: ReasoningNode, generator: StepGenerator, config: SearchConfig) -> list[ReasoningNode]:
    """Expands every node, level by level, and returns all the nodes of the tree (root first)."""
    all_nodes = [root]
    frontier = [root]
    # Always terminates: at max depth expand forces code, and code nodes are terminal
    while frontier:
        next_frontier = []
        for node in frontier:
            next_frontier.extend(expand(node, generator, config))
        all_nodes.extend(next_frontier)
        frontier = next_frontier
    return all_nodes
