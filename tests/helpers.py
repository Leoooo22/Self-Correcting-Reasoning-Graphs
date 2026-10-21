"""Test doubles shared by several test files (import with `from tests.helpers import ...`)."""
from reasoning_graphs.execution.executor import CodeExecutor
from reasoning_graphs.generators.base import StepGenerator
from reasoning_graphs.reasoning_node import ReasoningNode
from reasoning_graphs.search.evaluator import Evaluator


class ScriptedGenerator(StepGenerator):
    """Answers each call with the next entry of a script, and records every call (test spy).

    An entry is one output (a string) or several outputs (a list of strings).
    """

    def __init__(self, script: list):
        self.script = list(script)
        self.calls = []  # (prompt, n) for every call

    def generate_steps(self, prompt: str, n: int) -> list[str]:
        self.calls.append((prompt, n))
        entry = self.script.pop(0)
        return entry if isinstance(entry, list) else [entry]


class SpyExecutor(CodeExecutor):
    """A real CodeExecutor that also records the tests it was asked to run."""

    def __init__(self, timeout: float = 2.0):
        super().__init__(timeout)
        self.runs = []  # tests passed to every run

    def run(self, code, function_name, tests):
        self.runs.append(tests)
        return super().run(code, function_name, tests)


class FixedScoreEvaluator(Evaluator):
    """Scores from a table keyed by step_text (default for missing ones), and records what it scored."""

    def __init__(self, scores: dict[str, float], default: float = 0.0):
        self.scores = scores
        self.default = default
        self.scored = []  # nodes, in the order they were scored

    def score(self, node: ReasoningNode) -> float:
        self.scored.append(node)
        return self.scores.get(node.step_text, self.default)
