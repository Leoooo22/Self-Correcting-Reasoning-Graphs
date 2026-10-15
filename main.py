from reasoning_graphs.reasoning_node import ReasoningNode
from reasoning_graphs.generators.base import MockGenerator
from reasoning_graphs.config import SearchConfig
from reasoning_graphs.problems.two_sum import TWO_SUM
from reasoning_graphs.execution.executor import CodeExecutor
from reasoning_graphs.step_format import extract_code
from reasoning_graphs.search.full_tree import grow_full_tree
from reasoning_graphs.execution.grading import pick_solution, grade_on_public

HASH_MAP_CODE = """```python
def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
```"""

BRUTE_FORCE_CODE = """```python
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
```"""

# Syntactically valid but wrong: returns the values instead of the indices
WRONG_CODE = """```python
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [nums[i], nums[j]]
```"""

BROKEN_LINE = "def two_sum(nums, target:"
BROKEN_CODE = f"""```python
{BROKEN_LINE}
    return []
```"""

STEP_CANDIDATES = [
    [
        "Step 1: Use a hash map from value to index, so each lookup is O(1).",
        "Step 1: Try every pair of indices with two nested loops.",
        "Step 1: Sort the array and use two pointers.",
        "",
        "Code:\n" + HASH_MAP_CODE,
    ],
    [
        "Step 2: Store each value with its index while scanning the array.",
        "Step 2: For each element, compute the complement target - x.",
        "Step 2: Compare the sum of the pair with target.",
    ],
    [
        "Step 3: If the complement was already seen, return both indices.",
        "Step 3: Return the two indices as soon as the sum matches.",
    ],
]

CODE_CANDIDATES = [HASH_MAP_CODE, BRUTE_FORCE_CODE, WRONG_CODE, BROKEN_CODE]


def print_public_results(leaves, public_results, problem):
    """One line per code leaf (verdict + first step of its branch), plus the tests it failed."""
    for node in leaves:
        result = public_results[node.id]
        first_step = node.get_path()[1]
        strategy = "(code written directly)" if first_step is node else first_step.step_text[:45]
        verdict = "PASS" if result.all_passed else "FAIL"
        print(f"node {node.id:>2}  {verdict}  {result.num_passed}/{len(result.results)} public  | {strategy}")
        for r in result.results:
            if r.status != "passed":
                print(f"           {r.status}: {problem.function_name}{tuple(r.test.args)} -> {r.actual!r}, expected {r.test.expected!r}")


def main():
    config = SearchConfig()
    # Seed chosen so that the root sees an empty step and an early code step
    generator = MockGenerator(STEP_CANDIDATES, CODE_CANDIDATES, seed=6)
    root = ReasoningNode(TWO_SUM.to_prompt())

    all_nodes = grow_full_tree(root, generator, config)

    root.print_tree()

    terminals = [node for node in all_nodes if node.is_terminal]
    print(f"\n{len(all_nodes)} nodes, {len(terminals)} terminal (code) nodes\n")

    # Grade every code leaf on the PUBLIC tests only, as the search will do
    executor = CodeExecutor()
    public_results = grade_on_public(terminals, TWO_SUM, executor)
    print_public_results(terminals, public_results, TWO_SUM)

    # Pick a solution using the public tests, then measure it on the HIDDEN tests
    chosen = pick_solution(terminals, public_results)
    if chosen is None:
        print("\nNo code passes the public tests")
    else:
        hidden = executor.run(chosen.code, TWO_SUM.function_name, TWO_SUM.hidden_tests)
        print(f"\nChosen: node {chosen.id} -> hidden tests: {hidden.num_passed}/{len(hidden.results)} passed")
        print(chosen.code)

    # Sanity checks on the whole tree
    for node in all_nodes:
        if node.is_leaf():
            assert node.is_terminal, f"branch ending without code: {node}"
        assert node.depth <= config.max_steps + 1, f"node too deep: {node}"
        if not node.is_root() and not node.is_terminal:
            assert node.step_text.startswith(f"Step {node.depth}:"), f"wrong numbering: {node}"
        if node.is_terminal:
            assert BROKEN_LINE not in node.code, f"syntax error accepted: {node}"

    # The executor must reject exactly the wrong-but-valid code
    wrong = extract_code(WRONG_CODE)
    for node in terminals:
        assert public_results[node.id].all_passed == (node.code != wrong), f"wrong verdict: {node}"
    assert chosen is not None and hidden.all_passed
    print("\nAll checks passed")


if __name__ == "__main__":
    main()
