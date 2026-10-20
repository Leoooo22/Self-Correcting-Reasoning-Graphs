"""Scores the first-level branches of Two Sum with rollouts of the real model.
Run: HF_HUB_OFFLINE=1 python hf_scores.py"""
import time

from reasoning_graphs.config import SearchConfig
from reasoning_graphs.execution.executor import CodeExecutor
from reasoning_graphs.generators.hf import HFGenerator
from reasoning_graphs.problems.two_sum import TWO_SUM
from reasoning_graphs.reasoning_node import ReasoningNode
from reasoning_graphs.search.evaluator import RolloutEvaluator
from reasoning_graphs.search.expand import expand

NUM_ROLLOUTS = 8  # scores in steps of 1/8; up to 8 × 3 = 24 generator calls per first-level node


def main():
    config = SearchConfig()
    generator = HFGenerator()
    executor = CodeExecutor()
    root = ReasoningNode(TWO_SUM.to_prompt())

    # Only the first level is needed: one expansion of the root (the same first call as hf_tree.py)
    first_level = expand(root, generator, config)
    evaluator = RolloutEvaluator(generator, config, executor, TWO_SUM, n=NUM_ROLLOUTS)

    total_start = time.perf_counter()
    for node in first_level:
        start = time.perf_counter()
        node.score = evaluator.score(node)
        elapsed = time.perf_counter() - start
        print(f"score {node.score:.3f}  ({elapsed:5.1f}s)  {node.step_text[:70]}")

    total = time.perf_counter() - total_start
    print(f"\n{len(first_level)} nodes x {NUM_ROLLOUTS} rollouts in {total:.0f}s")


if __name__ == "__main__":
    main()
