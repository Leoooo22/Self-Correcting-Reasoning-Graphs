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
    if not code:
        return None  # an empty block parses fine but is no solution
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
