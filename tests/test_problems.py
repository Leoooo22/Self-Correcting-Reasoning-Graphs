from problems import TWO_SUM


def reference_two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i


def test_two_sum_test_counts():
    assert len(TWO_SUM.public_tests) == 2
    assert len(TWO_SUM.hidden_tests) == 5


def test_hand_written_tests_match_a_reference_solution():
    for test in TWO_SUM.public_tests + TWO_SUM.hidden_tests:
        assert reference_two_sum(*test.args) == test.expected, f"wrong test: {test}"


def test_hidden_tests_never_reach_the_prompt():
    prompt = TWO_SUM.to_prompt()
    for test in TWO_SUM.hidden_tests:
        args = ", ".join(repr(arg) for arg in test.args)
        assert f"{TWO_SUM.function_name}({args})" not in prompt, f"hidden test leaked: {test}"


def test_no_prompt_line_starts_with_step():
    # Otherwise split_prompt and the mock would take that line for a step
    assert not any(line.startswith("Step") for line in TWO_SUM.to_prompt().splitlines())
