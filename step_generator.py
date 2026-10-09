import random
from abc import ABC, abstractmethod

from step_format import STEP_PREFIX, CODE_PREFIX


class StepGenerator(ABC):
    @abstractmethod
    def generate_steps(self, prompt: str, n: int) -> list[str]:
        pass


class MockGenerator(StepGenerator):
    def __init__(self, step_candidates: list[list[str]], code_candidates: list[str], seed: int):
        self.step_candidates = step_candidates
        self.code_candidates = code_candidates
        self.rng = random.Random(seed)

    def generate_steps(self, prompt: str, n: int) -> list[str]:
        prompt = prompt.rstrip()
        if prompt.lower().endswith(CODE_PREFIX.lower()):
            return self.rng.choices(self.code_candidates, k=n)

        level = sum(line.lower().startswith(STEP_PREFIX.lower()) for line in prompt.splitlines())
        if level < len(self.step_candidates):
            return self.rng.choices(self.step_candidates[level], k=n)

        codes = self.rng.choices(self.code_candidates, k=n)
        return [CODE_PREFIX + "\n" + code for code in codes]


if __name__ == "__main__":
    from step_format import parse_step

    step_candidates = [
        ["Step 1: Use a hash map.", "Step 1: Try every pair."],
        ["Step 2: Store each value with its index.", "Step 2: Use two nested loops."],
    ]
    code_candidates = [
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
    problem = "Problem: Two Sum, in as few steps as possible.\nReturn the indices.\ndef two_sum(nums, target):"
    generator = MockGenerator(step_candidates, code_candidates, seed=42)

    # 1. Only the problem -> level 0
    outputs = generator.generate_steps(problem, 4)
    print("level 0:", outputs)
    assert len(outputs) == 4
    assert all(o in step_candidates[0] for o in outputs)

    # 2. One step done -> level 1
    outputs = generator.generate_steps(problem + "\nStep 1: Use a hash map.", 3)
    print("level 1:", outputs)
    assert all(o in step_candidates[1] for o in outputs)

    # 3. Candidates exhausted -> code written by the "model", label included
    two_steps = problem + "\nStep 1: Use a hash map.\nStep 2: Store each value with its index."
    outputs = generator.generate_steps(two_steps, 2)
    assert all(o.startswith(CODE_PREFIX + "\n```python") for o in outputs)
    assert all(parse_step(o, force_code=False).is_code for o in outputs)
    print("3. exhausted candidates -> labelled code: OK")

    # 4. Prompt ending with "Code:" -> continuation only, no label
    outputs = generator.generate_steps(problem + "\nStep 1: Use a hash map.\nCode:", 2)
    assert all(o in code_candidates for o in outputs)
    assert all(parse_step(CODE_PREFIX + "\n" + o, force_code=True) is not None for o in outputs)
    print("4. forced code -> unlabelled code: OK")

    # 5. Same seed, same calls -> same results
    first = MockGenerator(step_candidates, code_candidates, seed=7)
    second = MockGenerator(step_candidates, code_candidates, seed=7)
    assert first.generate_steps(problem, 5) == second.generate_steps(problem, 5)
    print("5. reproducibility: OK")

    print("\nAll checks passed")
