import random
from abc import ABC, abstractmethod

from reasoning_graphs.step_format import STEP_PREFIX, CODE_PREFIX


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
