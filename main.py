from reasoning_node import ReasoningNode
from step_generator import MockGenerator
from config import SearchConfig
from tree_search import expand

PROBLEM = """Problem: Given an array of integers nums and an integer target,
return the indices of the two numbers that add up to target.
You may assume that each input has exactly one solution.
def two_sum(nums: list[int], target: int) -> list[int]:"""

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


def main():
    config = SearchConfig()
    # Seed chosen so that the root sees an empty step and an early code step
    generator = MockGenerator(STEP_CANDIDATES, CODE_CANDIDATES, seed=6)
    root = ReasoningNode(PROBLEM)

    # Expand level by level until no node is left to expand.
    # This always terminates: at max depth expand forces code, and code nodes are terminal.
    all_nodes = [root]
    frontier = [root]
    while frontier:
        next_frontier = []
        for node in frontier:
            next_frontier.extend(expand(node, generator, config))
        all_nodes.extend(next_frontier)
        frontier = next_frontier

    root.print_tree()

    terminals = [node for node in all_nodes if node.is_terminal]
    print(f"\n{len(all_nodes)} nodes, {len(terminals)} terminal (code) nodes\n")
    for node in terminals:
        print(f"--- code of node {node.id} (depth {node.depth}) ---")
        print(node.code)

    # Sanity checks on the whole tree
    for node in all_nodes:
        if node.is_leaf():
            assert node.is_terminal, f"branch ending without code: {node}"
        assert node.depth <= config.max_steps + 1, f"node too deep: {node}"
        if not node.is_root() and not node.is_terminal:
            assert node.step_text.startswith(f"Step {node.depth}:"), f"wrong numbering: {node}"
        if node.is_terminal:
            assert BROKEN_LINE not in node.code, f"syntax error accepted: {node}"

    codes = {node.code for node in terminals}
    wrong = WRONG_CODE.removeprefix("```python\n").removesuffix("\n```")
    print("\nWrong-but-valid code present in the tree:", wrong in codes)
    print("All checks passed")


if __name__ == "__main__":
    main()
