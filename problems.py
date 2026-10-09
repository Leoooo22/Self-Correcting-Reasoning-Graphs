from problem import TestCase, Problem

TWO_SUM = Problem(
    name="two_sum",
    statement="""Problem: Given an array of integers nums and an integer target,
return the indices of the two numbers that add up to target.
You may assume that each input has exactly one solution, and you may not use the same element twice.
Return the two indices in increasing order.""",
    function_name="two_sum",
    signature="def two_sum(nums: list[int], target: int) -> list[int]:",
    public_tests=[
        TestCase([[2, 7, 11, 15], 9], [0, 1]),
        TestCase([[3, 2, 4], 6], [1, 2]),
    ],
    hidden_tests=[
        TestCase([[3, 3], 6], [0, 1]),                    # duplicates
        TestCase([[-1, -2, -3, -4, -5], -8], [2, 4]),     # negative numbers
        TestCase([[0, 4, 3, 0], 0], [0, 3]),              # zeros and zero target
        TestCase([[1, 2, 3, 4, 5, 6], 11], [4, 5]),       # pair at the end
        TestCase([[5, 75, 25], 100], [1, 2]),             # first element not in the pair
    ],
)


if __name__ == "__main__":
    def reference_two_sum(nums, target):
        seen = {}
        for i, x in enumerate(nums):
            if target - x in seen:
                return [seen[target - x], i]
            seen[x] = i

    print(f"{len(TWO_SUM.public_tests)} public tests, {len(TWO_SUM.hidden_tests)} hidden tests")
    assert len(TWO_SUM.public_tests) == 2 and len(TWO_SUM.hidden_tests) == 5

    # Check the hand-written tests against a known-correct solution
    for test in TWO_SUM.public_tests + TWO_SUM.hidden_tests:
        result = reference_two_sum(*test.args)
        assert result == test.expected, f"wrong test {test}: reference solution returned {result}"
    print("Reference solution passes all tests")

    prompt = TWO_SUM.to_prompt()
    print("\n" + prompt + "\n")

    # Hidden tests must never appear in the prompt
    for test in TWO_SUM.hidden_tests:
        args = ", ".join(repr(arg) for arg in test.args)
        assert f"{TWO_SUM.function_name}({args})" not in prompt, f"hidden test leaked: {test}"
    # No prompt line may start with "Step", or the mock would count it as a step
    assert not any(line.startswith("Step") for line in prompt.splitlines())
    print("All checks passed")
