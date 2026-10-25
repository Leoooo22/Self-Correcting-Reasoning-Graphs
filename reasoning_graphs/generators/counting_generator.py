from reasoning_graphs.generators.base import StepGenerator


class CountingGenerator(StepGenerator):
    """Wraps any generator and counts what passes through it, without changing the outputs.

    calls: number of generate_steps calls; outputs: number of outputs requested (sum of n), the cost measure
    used to compare systems at equal compute, since one call with n outputs costs more than one with n = 1.
    """

    def __init__(self, generator: StepGenerator):
        self.generator = generator
        self.calls = 0
        self.outputs = 0

    def generate_steps(self, prompt: str, n: int) -> list[str]:
        self.calls += 1
        self.outputs += n
        return self.generator.generate_steps(prompt, n)
