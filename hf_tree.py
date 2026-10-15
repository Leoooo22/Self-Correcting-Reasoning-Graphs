"""Grows a full tree on Two Sum with the real model and measures it. Run: HF_HUB_OFFLINE=1 python hf_tree.py"""
import time
from collections import Counter

from main import print_public_results
from reasoning_graphs.config import SearchConfig
from reasoning_graphs.execution.executor import CodeExecutor
from reasoning_graphs.execution.grading import grade_on_public, pick_solution
from reasoning_graphs.generators.hf import HFGenerator
from reasoning_graphs.problems.two_sum import TWO_SUM
from reasoning_graphs.reasoning_node import ReasoningNode
from reasoning_graphs.search.full_tree import grow_full_tree


def main():
    config = SearchConfig()
    generator = HFGenerator()
    root = ReasoningNode(TWO_SUM.to_prompt())

    # 1. Grow the whole tree with the model
    start = time.perf_counter()
    all_nodes = grow_full_tree(root, generator, config)
    growth_time = time.perf_counter() - start
    root.print_tree()

    # 2. Tree statistics. Every non-terminal node was expanded exactly once (one generator call)
    expanded = [node for node in all_nodes if not node.is_terminal]
    requested = sum(config.num_candidates(node.depth) for node in expanded)
    created = sum(len(node.children) for node in expanded)
    terminals = [node for node in all_nodes if node.is_terminal]
    dead_ends = [node for node in all_nodes if node.is_leaf() and not node.is_terminal]
    per_depth = Counter(node.depth for node in all_nodes)

    print(f"\nnodes per depth: {dict(sorted(per_depth.items()))}")
    print(f"generator calls: {len(expanded)}, {growth_time:.1f}s ({growth_time / len(expanded):.2f}s per call)")
    print(f"outputs requested: {requested}, became nodes: {created}, lost (invalid or duplicate): {requested - created}")
    print(f"code leaves: {len(terminals)}, dead ends (leaves without code): {len(dead_ends)}\n")

    # 3. Grade the code leaves on the PUBLIC tests and pick a solution
    executor = CodeExecutor()
    start = time.perf_counter()
    public_results = grade_on_public(terminals, TWO_SUM, executor)
    grading_time = time.perf_counter() - start
    print_public_results(terminals, public_results, TWO_SUM)
    passing = sum(result.all_passed for result in public_results.values())
    print(f"\n{passing}/{len(terminals)} code leaves pass the public tests (grading: {grading_time:.1f}s)")

    # 4. Measure the chosen solution on the HIDDEN tests
    chosen = pick_solution(terminals, public_results)
    if chosen is None:
        print("No code passes the public tests")
    else:
        hidden = TWO_SUM.hidden_tests
        hidden_result = executor.run(chosen.code, TWO_SUM.function_name, hidden)
        print(f"Chosen: node {chosen.id} -> hidden tests: {hidden_result.num_passed}/{len(hidden)} passed")
        print(chosen.code)

    # Invariants: true whatever the generator does, because our code decides them
    for node in all_nodes:
        assert node.depth <= config.max_steps + 1, f"node too deep: {node}"
        if not node.is_root() and not node.is_terminal:
            assert node.step_text.startswith(f"Step {node.depth}:"), f"wrong numbering: {node}"


if __name__ == "__main__":
    main()
