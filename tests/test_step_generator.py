from step_format import CODE_PREFIX, parse_step
from step_generator import MockGenerator

STEP_CANDIDATES = [
    ["Step 1: Use a hash map.", "Step 1: Try every pair."],
    ["Step 2: Store each value with its index.", "Step 2: Use two nested loops."],
]
CODE_CANDIDATES = [
    """```python
def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
```""",
    """```python
def two_sum(nums, target):
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[i] + nums[j] == target:
                return [i, j]
```""",
]
# Multi-line problem, mentioning "steps" on purpose: it must not be counted as a step
PROBLEM = "Problem: Two Sum, in as few steps as possible.\nReturn the indices.\ndef two_sum(nums, target):"


def make_generator(seed=42):
    return MockGenerator(STEP_CANDIDATES, CODE_CANDIDATES, seed=seed)


def test_problem_only_gives_level_0():
    outputs = make_generator().generate_steps(PROBLEM, 4)
    assert len(outputs) == 4
    assert all(o in STEP_CANDIDATES[0] for o in outputs)


def test_one_step_done_gives_level_1():
    outputs = make_generator().generate_steps(PROBLEM + "\nStep 1: Use a hash map.", 3)
    assert all(o in STEP_CANDIDATES[1] for o in outputs)


def test_exhausted_candidates_give_labelled_code():
    two_steps = PROBLEM + "\nStep 1: Use a hash map.\nStep 2: Store each value with its index."
    outputs = make_generator().generate_steps(two_steps, 2)
    assert all(o.startswith(CODE_PREFIX + "\n```python") for o in outputs)
    assert all(parse_step(o, force_code=False).is_code for o in outputs)


def test_forced_code_gives_unlabelled_code():
    outputs = make_generator().generate_steps(PROBLEM + "\nStep 1: Use a hash map.\nCode:", 2)
    assert all(o in CODE_CANDIDATES for o in outputs)
    assert all(parse_step(CODE_PREFIX + "\n" + o, force_code=True) is not None for o in outputs)


def test_same_seed_same_outputs():
    first = make_generator(seed=7)
    second = make_generator(seed=7)
    assert first.generate_steps(PROBLEM, 5) == second.generate_steps(PROBLEM, 5)
