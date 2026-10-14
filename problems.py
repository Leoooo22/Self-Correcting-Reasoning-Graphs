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
