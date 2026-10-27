from reasoning_graphs.execution.executor import CodeExecutor, ExecutionResult
from reasoning_graphs.problems.problem import Problem
from reasoning_graphs.reasoning_node import ReasoningNode


def grade_on_public(leaves: list[ReasoningNode], problem: Problem, executor: CodeExecutor) -> dict[int, ExecutionResult]:
    """Runs each code leaf on the PUBLIC tests only (the hidden ones are for the final measurement)."""
    public_results = {}
    for node in leaves:
        public_results[node.id] = executor.run(node.code, problem.function_name, problem.public_tests)
    return public_results


def pick_solution(leaves: list[ReasoningNode], public_results: dict[int, ExecutionResult]) -> ReasoningNode | None:
    """The first leaf that passes every public test, or None."""
    return next((node for node in leaves if public_results[node.id].all_passed), None)


def branch_value(leaf: ReasoningNode) -> float:
    """Score of the last scored node on the solution's path: the node a rollout started from (origin) for rollout
    solutions, whose own path is made of unscored copies, otherwise the parent. -1 (below any real score) if it has
    no score, e.g. the root."""
    anchor = leaf.origin if leaf.origin is not None else leaf.parent
    return anchor.score if anchor.score is not None else -1.0


def pick_by_branch(leaves: list[ReasoningNode], public_results: dict[int, ExecutionResult]) -> ReasoningNode | None:
    """Among the leaves passing every public test, the one whose branch scored best (ties: first one), or None."""
    valid_leaves = [node for node in leaves if public_results[node.id].all_passed]
    return max(valid_leaves, key=branch_value, default=None)