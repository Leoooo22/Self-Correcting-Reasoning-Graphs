from abc import ABC, abstractmethod
import random


class StepGenerator(ABC):
    @abstractmethod
    def generate_steps(self, prompt: str, n: int) -> list[str]:
        pass
    

class MockGenerator(StepGenerator):
    def __init__(self, candidates, seed):
        self.candidates = candidates
        self.rng = random.Random(seed)
        
    def generate_steps(self, prompt: str, n: int) -> list[str]:
        rows = prompt.splitlines()
        level = len(rows) - 1

        if level >= len(self.candidates):
            return []

        return self.rng.choices(self.candidates[level], k=n)


if __name__ == "__main__":
    candidates = [
        ["3 * 4 = 12", "3 * 4 = 7", "4 * 5 = 20"],
        ["12 + 5 = 17", "12 + 5 = 18"],
        ["Answer: 17", "Answer: 18"],
    ]
    generator = MockGenerator(candidates, seed=42)

    print("Level 0:", generator.generate_steps("What is 3 * 4 + 5?", 3))
    print("Level 1:", generator.generate_steps("What is 3 * 4 + 5?\n3 * 4 = 12", 3))
    print("Level 2:", generator.generate_steps("What is 3 * 4 + 5?\n3 * 4 = 12\n12 + 5 = 17", 3))
    print("End:", generator.generate_steps("What is 3 * 4 + 5?\n3 * 4 = 12\n12 + 5 = 17\nAnswer: 17", 3))

