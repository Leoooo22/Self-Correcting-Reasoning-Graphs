import pytest

from reasoning_graphs.config import SearchConfig


def test_defaults():
    config = SearchConfig()
    assert config.branching == [4, 2, 2]
    assert config.code_branching == 2
    assert config.beam_width == 2
    assert config.max_steps == 3


def test_custom_values():
    config = SearchConfig(branching=[6, 3], code_branching=1, beam_width=5)
    assert config.max_steps == 2
    assert config.code_branching == 1
    assert config.beam_width == 5


def test_each_config_has_its_own_list():
    # Mutable default trap: changing one config must not change the others
    first = SearchConfig()
    second = SearchConfig()
    first.branching.append(5)
    assert second.branching == [4, 2, 2]


@pytest.mark.parametrize("depth, forces_code, num_candidates", [
    pytest.param(0, False, 4, id="root"),
    pytest.param(1, False, 2, id="depth 1"),
    pytest.param(2, False, 2, id="depth 2"),
    pytest.param(3, True, 2, id="max depth: code only"),
])
def test_candidates_per_depth(depth, forces_code, num_candidates):
    config = SearchConfig(branching=[4, 2, 2], code_branching=2)
    assert config.forces_code(depth) == forces_code
    assert config.num_candidates(depth) == num_candidates


def test_code_branching_is_used_only_at_max_depth():
    # Different values, so a mix-up between branching and code_branching would show
    config = SearchConfig(branching=[5, 3], code_branching=7)
    assert [config.num_candidates(depth) for depth in range(3)] == [5, 3, 7]


@pytest.mark.parametrize("kwargs", [
    pytest.param({"branching": []}, id="empty branching"),
    pytest.param({"branching": [4, 0]}, id="zero in branching"),
    pytest.param({"code_branching": 0}, id="zero code branching"),
    pytest.param({"beam_width": 0}, id="zero beam width"),
])
def test_invalid_values_are_rejected(kwargs):
    with pytest.raises(ValueError):
        SearchConfig(**kwargs)
