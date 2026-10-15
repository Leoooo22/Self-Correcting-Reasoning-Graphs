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
