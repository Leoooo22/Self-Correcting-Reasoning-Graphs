import ast
import re
from dataclasses import dataclass

STEP_PREFIX = "Step"
CODE_PREFIX = "Code:"

# Prompt for real generators: format rules + one solved example (few-shot)
SYSTEM_PROMPT = (
    "Solve the programming problem one step at a time. "
    "Write each reasoning step on a single line starting with 'Step N:', describing the plan in words, without code. "
    "When the plan is complete, write 'Code:' followed by a ```python block containing the complete function. "
    "Write nothing else: no explanations, no usage examples, no prints."
)

FEW_SHOT_PROBLEM = """Problem: Return the largest element of a non-empty list of integers.
Examples:
largest([3, 1, 4]) -> 4
largest([-5, -2]) -> -2
def largest(nums: list[int]) -> int:"""

FEW_SHOT_SOLUTION = """Step 1: Keep the largest value seen so far, starting from the first element.
Step 2: Scan the list and update that value whenever a larger element appears.
Code:
```python
def largest(nums):
    best = nums[0]
    for x in nums:
        if x > best:
            best = x
    return best
```"""


@dataclass
class ParsedStep:
    is_code: bool
    text: str
    code: str | None


def extract_code(text: str) -> str | None:
    match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if not match:
        return None
    code = match.group(1).strip()
    try:
        ast.parse(code)
        return code
    except Exception:
        return None


def parse_step(raw: str, force_code: bool) -> ParsedStep | None:
    clean = raw.strip()

    if clean.lower().startswith(CODE_PREFIX.lower()):
        code = extract_code(clean)
        if code is None:
            return None
        return ParsedStep(is_code=True, text=code, code=code)

    if force_code:
        return None

    label = rf"^{re.escape(STEP_PREFIX)}\s*\d*\s*:\s*"
    text = re.sub(label, "", clean, flags=re.IGNORECASE)

    next_step = rf"\n\s*({re.escape(STEP_PREFIX)}\b|{re.escape(CODE_PREFIX)})"
    match = re.search(next_step, text, flags=re.IGNORECASE)
    if match:
        text = text[:match.start()]

    text = text.strip()
    if not text:
        return None

    return ParsedStep(is_code=False, text=text, code=None)


def format_step(parsed: ParsedStep, depth: int) -> str:
    if parsed.is_code:
        return f"{CODE_PREFIX}\n```python\n{parsed.code}\n```"
    return f"{STEP_PREFIX} {depth}: {parsed.text}"


def split_prompt(full_text: str) -> tuple[str, str]:
    lines = full_text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith(STEP_PREFIX) or line.startswith(CODE_PREFIX):
            problem = "\n".join(lines[:i])
            steps = "\n".join(lines[i:])
            return problem, steps
    return full_text, ""


def build_messages(full_text: str) -> list[dict]:
    problem, steps = split_prompt(full_text)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": FEW_SHOT_PROBLEM},
        {"role": "assistant", "content": FEW_SHOT_SOLUTION},
        {"role": "user", "content": problem},
    ]

    if steps:
        messages.append({"role": "assistant", "content": steps + "\n"})
    return messages


if __name__ == "__main__":
    valid_code = """Code:
```python
def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
```"""

    # valid_code is already in canonical form, so format_step must reproduce it exactly
    expected_code = valid_code

    broken_code = """Code:
```python
def two_sum(nums, target:
    return []
```"""

    # (description, raw model output, force_code, expected formatted step at depth 1, or None)
    cases = [
        ("valid step", "Step 1: Use a hash map.", False, "Step 1: Use a hash map."),
        ("wrong step number", "Step 7: Use a hash map.", False, "Step 1: Use a hash map."),
        ("two steps, keep the first", "Step 1: Use a hash map.\nStep 2: Loop over nums.", False, "Step 1: Use a hash map."),
        ("step followed by code", "Use a hash map.\n" + valid_code, False, "Step 1: Use a hash map."),
        ("no label", "Use a hash map.", False, "Step 1: Use a hash map."),
        ("spaces and lowercase label", "   step 3 : Use a hash map.  ", False, "Step 1: Use a hash map."),
        ("multi-line step", "Step 1: Use a hash map\nfrom value to index.", False, "Step 1: Use a hash map\nfrom value to index."),
        ("empty string", "", False, None),
        ("label only", "Step 1:   ", False, None),
        ("valid code", valid_code, False, expected_code),
        ("code with text after the block", valid_code + "\nThis works in O(n).", False, expected_code),
        ("lowercase code label", "code:" + valid_code[len("Code:"):], False, expected_code),
        ("code with syntax error", broken_code, False, None),
        ("code without a python block", "Code: def f(a): return a", False, None),
        ("unclosed code block", "Code:\n```python\ndef f(a):\n    return a", False, None),
        ("reasoning step when code is forced", "Step 1: Use a hash map.", True, None),
        ("code when code is forced", valid_code, True, expected_code),
    ]

    failures = 0
    for description, raw, force_code, expected in cases:
        parsed = parse_step(raw, force_code)
        result = None if parsed is None else format_step(parsed, depth=1)
        ok = result == expected
        if not ok:
            failures += 1
        print(f"[{'PASS' if ok else 'FAIL'}] {description}")
        if not ok:
            print(f"    expected: {expected!r}")
            print(f"    got:      {result!r}")

    print(f"\n{len(cases) - failures}/{len(cases)} passed")

    from problems import TWO_SUM

    print(FEW_SHOT_PROBLEM)
    print("---")
    print(TWO_SUM.to_prompt())

    assert extract_code(FEW_SHOT_SOLUTION) is not None, "the few-shot solution has no valid code block"
    print("few-shot example: OK")

    # build_messages: the problem always goes in the 4th message, the steps (if any) in a 5th one
    root = TWO_SUM.to_prompt()
    # (description, full text, expected number of messages, expected content of the last message)
    message_cases = [
        ("root", root, 4, root),
        ("with steps", root + "\nStep 1: A\nStep 2: B", 5, "Step 1: A\nStep 2: B\n"),
        ("forced code", root + "\nStep 1: A\nCode:", 5, "Step 1: A\nCode:\n"),
    ]
    for description, text, expected_count, expected_last in message_cases:
        messages = build_messages(text)
        assert len(messages) == expected_count, f"{description}: wrong number of messages"
        assert messages[-1]["content"] == expected_last, f"{description}: wrong last message"
        assert messages[3]["content"] == root, f"{description}: the problem is not in the 4th message"
        print(f"[PASS] build_messages: {description}")
