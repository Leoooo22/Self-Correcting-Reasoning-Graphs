from dataclasses import dataclass, field


@dataclass
class SearchConfig:
    branching: list[int] = field(default_factory=lambda: [4, 2, 2])
    code_branching: int = 2

    def __post_init__(self):
        if not self.branching:
            raise ValueError("branching cannot be empty")
        for b in self.branching:
            if b < 1:
                raise ValueError(f"branching values must be >= 1, got {self.branching}")
        if self.code_branching < 1:
            raise ValueError(f"code_branching must be >= 1, got {self.code_branching}")

    @property
    def max_steps(self) -> int:
        return len(self.branching)
