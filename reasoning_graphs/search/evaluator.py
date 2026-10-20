from abc import ABC, abstractmethod

from reasoning_graphs.config import SearchConfig
from reasoning_graphs.execution.executor import CodeExecutor
from reasoning_graphs.generators.base import StepGenerator
from reasoning_graphs.problems.problem import Problem
from reasoning_graphs.reasoning_node import ReasoningNode
from reasoning_graphs.search.rollout import rollout


class Evaluator(ABC):
    @abstractmethod
    def score(self, node: ReasoningNode) -> float:
        """Estimated probability, in [0, 1], that continuing from node reaches code passing the tests."""
        pass


class RolloutEvaluator(Evaluator):
    """Score = fraction of n rollouts from the node whose code passes every PUBLIC test."""

    def __init__(self, generator: StepGenerator, config: SearchConfig, executor: CodeExecutor, problem: Problem,
                 n: int = 4):
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        self.generator = generator
        self.config = config
        self.executor = executor
        self.problem = problem
        self.n = n

    def score(self, node: ReasoningNode) -> float:
        # A code node has no future to sample: its own tests are its score
        if node.is_terminal:
            return 1.0 if self._passes(node.code) else 0.0

        successes = 0
        for _ in range(self.n):
            code_node = rollout(node, self.generator, self.config)
            # A stuck rollout (None) counts as a failure: it stays in the denominator
            if code_node is not None and self._passes(code_node.code):
                successes += 1
        return successes / self.n

    def _passes(self, code: str) -> bool:
        result = self.executor.run(code, self.problem.function_name, self.problem.public_tests)
        return result.all_passed
