from dataclasses import dataclass, field


@dataclass
class SearchConfig:
    branching: list[int] = field(default_factory=lambda: [4, 2, 2])
    code_branching: int = 2
    beam_width: int = 2

    def __post_init__(self):
        if not self.branching:
            raise ValueError("branching cannot be empty")
        for b in self.branching:
            if b < 1:
                raise ValueError(f"branching values must be >= 1, got {self.branching}")
        if self.code_branching < 1:
            raise ValueError(f"code_branching must be >= 1, got {self.code_branching}")
        if self.beam_width < 1:
            raise ValueError(f"beam_width must be >= 1, got {self.beam_width}")

    @property
    def max_steps(self) -> int:
        return len(self.branching)

    def forces_code(self, depth: int) -> bool:
        """True if a node at this depth may only be followed by code."""
        return depth >= self.max_steps

    def num_candidates(self, depth: int) -> int:
        """How many outputs to request from the generator when expanding a node at this depth."""
        if self.forces_code(depth):
            return self.code_branching
        return self.branching[depth]
