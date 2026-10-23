import pytest

from reasoning_graphs.problems.mbpp import function_signature, mbpp_to_problem, parse_assert

REFERENCE = "def remove_Occ(s,ch): \n    return s.replace(ch, '')\n"


def make_record(test_list, code=REFERENCE, prompt="Write a function to remove a character."):
    """A record shaped like the sanitized MBPP ones (no download needed)."""
    return {"task_id": 11, "prompt": prompt, "code": code, "test_imports": [], "test_list": test_list}


@pytest.mark.parametrize("line, expected", [
    pytest.param('assert remove_Occ("hello","l") == "heo"', ("remove_Occ", ["hello", "l"], "heo"), id="strings"),
    pytest.param("assert f((1, 5, (4, 6)), [1, 2]) == {'a': 1}", ("f", [(1, 5, (4, 6)), [1, 2]], {"a": 1}),
                 id="tuples, lists, dicts"),
    pytest.param("assert f() == None", ("f", [], None), id="no arguments"),
    pytest.param("assert f({1, 2}) == 3", ("f", [{1, 2}], 3), id="a set argument is fine"),
])
def test_parse_assert_simple_shapes(line, expected):
    name, test = parse_assert(line)
    assert (name, test.args, test.expected) == expected


@pytest.mark.parametrize("line", [
    pytest.param("assert math.isclose(volume_sphere(10), 4188.79, rel_tol=0.001)", id="math.isclose"),
    pytest.param("assert set(f([1, 2])) == set([1, 2])", id="call on the left"),
    pytest.param("assert f([1, 2]) == set([1, 2])", id="non-literal expected value"),
    pytest.param("assert f(1) == {1, 2}", id="a set as expected value is not JSON-friendly"),
    pytest.param("assert f(x=1) == 2", id="keyword argument"),
    pytest.param("assert f(1) != 2", id="not an equality"),
    pytest.param("assert f(1) == 2 == 2", id="chained comparison"),
    pytest.param("assert f(1)", id="no comparison"),
    pytest.param("x = f(1)", id="not an assert"),
    pytest.param("assert f(1) == 2; assert f(2) == 3", id="two statements"),
    pytest.param("assert f(1 == 2", id="syntax error"),
])
def test_parse_assert_rejects_other_shapes(line):
    assert parse_assert(line) is None


def test_function_signature():
    code = "def helper(x):\n    return x\n\ndef target(a, b=2, *rest):\n    return helper(a)\n"
    assert function_signature(code, "target") == "def target(a, b=2, *rest):"
    assert function_signature(code, "missing") is None


def test_mbpp_to_problem():
    record = make_record([
        'assert remove_Occ("hello","l") == "heo"',
        'assert remove_Occ("abcda","a") == "bcd"',
        'assert remove_Occ("PHP","P") == "H"',
    ])
    problem = mbpp_to_problem(record)

    assert problem.name == "mbpp_11"
    assert problem.statement == "Problem: Write a function to remove a character."
    assert problem.function_name == "remove_Occ"
    assert problem.signature == "def remove_Occ(s, ch):"
    assert [t.args for t in problem.public_tests] == [["hello", "l"]]  # one public test...
    assert [t.expected for t in problem.hidden_tests] == ["bcd", "H"]  # ...the others hidden
    assert problem.to_prompt().splitlines()[-1] == "def remove_Occ(s, ch):"


@pytest.mark.parametrize("record", [
    pytest.param(make_record(['assert remove_Occ("a","a") == ""', "assert set(remove_Occ('a','a')) == set()"]),
                 id="one test in another shape"),
    pytest.param(make_record(['assert remove_Occ("a","a") == ""', 'assert other("a") == ""']),
                 id="tests calling different functions"),
    pytest.param(make_record(['assert missing("a") == ""']), id="function not in the reference code"),
    pytest.param(make_record(['assert remove_Occ("a","a") == ""'], prompt="Write a function.\nStep by step."),
                 id="prompt line starting with a label"),
])
def test_mbpp_to_problem_rejects(record):
    assert mbpp_to_problem(record) is None
