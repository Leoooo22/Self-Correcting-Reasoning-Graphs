from reasoning_graphs.config import SearchConfig
from reasoning_graphs.generators.base import StepGenerator
from reasoning_graphs.generators.counting_generator import CountingGenerator
from reasoning_graphs.reasoning_node import ReasoningNode
from reasoning_graphs.search.rollout import rollout


def best_of_n(root: ReasoningNode, generator: StepGenerator, config: SearchConfig, n: int) -> list[ReasoningNode]:
    """n independent complete solutions (rollouts from root), no scores. With n = 1: single generation.

    Returns the code nodes that were reached; stuck rollouts are left out (n - len(result) of them).
    The nodes are on detached copies: the root itself gets no children.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    leaves = []
    for _ in range(n):
        leaf = rollout(root, generator, config)
        if leaf is not None:
            leaves.append(leaf)
    return leaves


def best_within_budget(root: ReasoningNode, generator: CountingGenerator, config: SearchConfig,
                       budget: int) -> list[ReasoningNode]:
    """Best-of-N at equal compute: rollouts from root until `budget` more outputs have been requested.

    Used with budget = what another system spent on the same problem. The last rollout may overshoot the budget
    by a few outputs; at least one rollout is always made. Stuck rollouts are left out, as in best_of_n.
    """
    start = generator.outputs
    leaves = []
    while True:
        leaf = rollout(root, generator, config)
        if leaf is not None:
            leaves.append(leaf)
        if generator.outputs - start >= budget:
            return leaves
