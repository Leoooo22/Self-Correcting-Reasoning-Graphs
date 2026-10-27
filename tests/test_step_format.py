import pytest

from reasoning_graphs.problems.two_sum import TWO_SUM
from reasoning_graphs.step_format import (
    FEW_SHOT_PROBLEM,
    FEW_SHOT_SOLUTION,
    build_messages,
    extract_code,
    format_step,
    parse_step,
    split_prompt,
)

# Already in canonical form, so format_step must reproduce it exactly
VALID_CODE = """Code:
```python
def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
```"""

BROKEN_CODE = """Code:
```python
def two_sum(nums, target:
    return []
```"""

ROOT = TWO_SUM.to_prompt()


# --- Parser: raw model output -> canonical step at depth 1, or None if discarded ---

@pytest.mark.parametrize("raw, force_code, expected", [
    pytest.param("Step 1: Use a hash map.", False, "Step 1: Use a hash map.", id="valid step"),
    pytest.param("Step 7: Use a hash map.", False, "Step 1: Use a hash map.", id="wrong step number"),
    pytest.param("Step 1: Use a hash map.\nStep 2: Loop over nums.", False, "Step 1: Use a hash map.",
                 id="two steps, keep the first"),
    pytest.param("Use a hash map.\n" + VALID_CODE, False, "Step 1: Use a hash map.", id="step followed by code"),
    pytest.param("Use a hash map.", False, "Step 1: Use a hash map.", id="no label"),
    pytest.param("   step 3 : Use a hash map.  ", False, "Step 1: Use a hash map.", id="spaces and lowercase label"),
    pytest.param("Step 1: Use a hash map\nfrom value to index.", False, "Step 1: Use a hash map\nfrom value to index.",
                 id="multi-line step"),
    pytest.param("", False, None, id="empty string"),
    pytest.param("Step 1:   ", False, None, id="label only"),
    pytest.param(VALID_CODE, False, VALID_CODE, id="valid code"),
    pytest.param(VALID_CODE + "\nThis works in O(n).", False, VALID_CODE, id="code with text after the block"),
    pytest.param("code:" + VALID_CODE[len("Code:"):], False, VALID_CODE, id="lowercase code label"),
    pytest.param(BROKEN_CODE, False, None, id="code with syntax error"),
    pytest.param("Code: def f(a): return a", False, None, id="code without a python block"),
    pytest.param("Code:\n```python\ndef f(a):\n    return a", False, None, id="unclosed code block"),
    pytest.param("Code:\n```python\n\n```", False, None, id="empty code block"),
    pytest.param("Step 1: Use a hash map.", True, None, id="reasoning step when code is forced"),
    pytest.param(VALID_CODE, True, VALID_CODE, id="code when code is forced"),
])
def test_parse_step(raw, force_code, expected):
    parsed = parse_step(raw, force_code)
    result = None if parsed is None else format_step(parsed, depth=1)
    assert result == expected


# --- Few-shot example: it must follow the same rules as everything else ---

def test_few_shot_problem_has_the_prompt_shape():
    # Same shape as Problem.to_prompt(): statement, "Examples:", "f(args) -> result" lines, signature
    lines = FEW_SHOT_PROBLEM.splitlines()
    examples_start = lines.index("Examples:") + 1
    assert lines[0].startswith("Problem:")
    assert all(" -> " in line for line in lines[examples_start:-1])
    assert lines[-1].startswith("def ") and lines[-1].endswith(":")
    assert ROOT.splitlines()[0].startswith("Problem:")


def test_few_shot_solution_is_valid_and_correct():
    code = extract_code(FEW_SHOT_SOLUTION)
    assert code is not None
    namespace = {}
    exec(code, namespace)
    assert namespace["largest"]([3, 1, 4]) == 4
    assert namespace["largest"]([-5, -2]) == -2


# --- Prompt builder ---

@pytest.mark.parametrize("full_text, expected_steps", [
    pytest.param(ROOT, "", id="root"),
    pytest.param(ROOT + "\nStep 1: A\nStep 2: B", "Step 1: A\nStep 2: B", id="with steps"),
    pytest.param(ROOT + "\nStep 1: A\nCode:", "Step 1: A\nCode:", id="forced code"),
])
def test_split_prompt(full_text, expected_steps):
    assert split_prompt(full_text) == (ROOT, expected_steps)


@pytest.mark.parametrize("full_text, expected_count, expected_last", [
    pytest.param(ROOT, 4, ROOT, id="root"),
    pytest.param(ROOT + "\nStep 1: A\nStep 2: B", 5, "Step 1: A\nStep 2: B\n", id="with steps"),
    pytest.param(ROOT + "\nStep 1: A\nCode:", 5, "Step 1: A\nCode:\n", id="forced code"),
])
def test_build_messages(full_text, expected_count, expected_last):
    # The problem always goes in the 4th message, the steps (if any) in a 5th one ending with "\n"
    messages = build_messages(full_text)
    assert len(messages) == expected_count
    assert messages[-1]["content"] == expected_last
    assert messages[3] == {"role": "user", "content": ROOT}
    assert [m["role"] for m in messages[:3]] == ["system", "user", "assistant"]
