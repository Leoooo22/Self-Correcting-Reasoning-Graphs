"""Test doubles shared by several test files (import with `from tests.helpers import ...`)."""
from reasoning_graphs.execution.executor import CodeExecutor
from reasoning_graphs.generators.base import StepGenerator


class ScriptedGenerator(StepGenerator):
    """Answers each call with the next output of a script, and records every call (test spy)."""

    def __init__(self, script: list[str]):
        self.script = list(script)
        self.calls = []  # (prompt, n) for every call

    def generate_steps(self, prompt: str, n: int) -> list[str]:
        self.calls.append((prompt, n))
        return [self.script.pop(0)]


class SpyExecutor(CodeExecutor):
    """A real CodeExecutor that also records the tests it was asked to run."""

    def __init__(self, timeout: float = 2.0):
        super().__init__(timeout)
        self.runs = []  # tests passed to every run

    def run(self, code, function_name, tests):
        self.runs.append(tests)
        return super().run(code, function_name, tests)
