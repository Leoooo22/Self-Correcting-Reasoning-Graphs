import pytest

from reasoning_graphs.config import SearchConfig


def test_defaults():
    config = SearchConfig()
    assert config.branching == [4, 2, 2]
    assert config.code_branching == 2
    assert config.max_steps == 3


def test_custom_values():
    config = SearchConfig(branching=[6, 3], code_branching=1)
    assert config.max_steps == 2
    assert config.code_branching == 1


def test_each_config_has_its_own_list():
    # Mutable default trap: changing one config must not change the others
    first = SearchConfig()
    second = SearchConfig()
    first.branching.append(5)
    assert second.branching == [4, 2, 2]


@pytest.mark.parametrize("kwargs", [
    pytest.param({"branching": []}, id="empty branching"),
    pytest.param({"branching": [4, 0]}, id="zero in branching"),
    pytest.param({"code_branching": 0}, id="zero code branching"),
])
def test_invalid_values_are_rejected(kwargs):
    with pytest.raises(ValueError):
        SearchConfig(**kwargs)
