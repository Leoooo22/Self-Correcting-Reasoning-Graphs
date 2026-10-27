from reasoning_graphs.config import SearchConfig
from reasoning_graphs.generators.base import StepGenerator
from reasoning_graphs.reasoning_node import ReasoningNode
from reasoning_graphs.search.expand import expand


def rollout(node: ReasoningNode, generator: StepGenerator, config: SearchConfig) -> ReasoningNode | None:
    """Continues a single branch from node until code. Returns the code node, or None if the model gets stuck.
    The real tree is left unchanged: the rollout grows on a detached copy of node.
    """
    if node.is_terminal:
        return node  # already code: nothing to continue

    # One output per step, but the same maximum depth as the real search
    rollout_config = SearchConfig(branching=[1] * config.max_steps, code_branching=1)
    origin = node
    node = node.detached_copy()

    # Always terminates: expand forces code at max depth, and code nodes are terminal
    while True:
        children = expand(node, generator, rollout_config)
        if not children:
            return None  # invalid output
        node = children[0]
        if node.is_terminal:
            node.origin = origin
            return node
