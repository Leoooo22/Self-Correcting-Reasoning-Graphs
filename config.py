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


if __name__ == "__main__":
    default = SearchConfig()
    print("default:", default, "max_steps =", default.max_steps)
    assert default.branching == [4, 2, 2]
    assert default.code_branching == 2
    assert default.max_steps == 3

    custom = SearchConfig(branching=[6, 3], code_branching=1)
    print("custom: ", custom, "max_steps =", custom.max_steps)
    assert custom.max_steps == 2

    # Mutable default trap: each config must get its own list
    first = SearchConfig()
    second = SearchConfig()
    first.branching.append(5)
    print("after append on first ->", first.branching, second.branching)
    assert second.branching == [4, 2, 2]

    # Invalid values must be rejected at creation time
    for kwargs in ({"branching": []}, {"branching": [4, 0]}, {"code_branching": 0}):
        try:
            SearchConfig(**kwargs)
            raise AssertionError(f"{kwargs} should have raised ValueError")
        except ValueError as error:
            print(f"rejected {kwargs}: {error}")

    print("\nAll checks passed")
