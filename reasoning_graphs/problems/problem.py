from dataclasses import dataclass, field
from typing import Any


@dataclass
class TestCase:
    args: list
    expected: Any


@dataclass
class Problem:
    name: str
    statement: str
    function_name: str
    signature: str
    public_tests: list[TestCase] = field(default_factory=list)
    hidden_tests: list[TestCase] = field(default_factory=list)

    def to_prompt(self) -> str:
        # Only public tests are shown: hidden tests must never reach the model
        lines = [self.statement, "Examples:"]
        for test in self.public_tests:
            args = ", ".join(repr(arg) for arg in test.args)
            lines.append(f"{self.function_name}({args}) -> {test.expected!r}")
        lines.append(self.signature)
        return "\n".join(lines)
