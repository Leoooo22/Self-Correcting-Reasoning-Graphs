"""Adapter from MBPP (sanitized) to our Problem / TestCase format."""
import ast
import json

from reasoning_graphs.problems.problem import Problem, TestCase
from reasoning_graphs.step_format import CODE_PREFIX, STEP_PREFIX

DATASET_NAME = "google-research-datasets/mbpp"
NUM_PUBLIC_TESTS = 1  # the first test is shown to the model, the others are hidden


def parse_assert(line: str) -> tuple[str, TestCase] | None:
    """`assert f(a, b) == expected` -> ("f", TestCase([a, b], expected)); None for any other shape."""
    try:
        module = ast.parse(line)
    except SyntaxError:
        return None

    # Walk down the tree, checking each level has exactly the expected shape
    if len(module.body) != 1 or not isinstance(module.body[0], ast.Assert):
        return None
    compare = module.body[0].test
    if not isinstance(compare, ast.Compare) or len(compare.ops) != 1 or not isinstance(compare.ops[0], ast.Eq):
        return None
    call = compare.left
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.keywords:
        return None

    # Arguments and expected value must be literals (no calls such as set([...]))...
    try:
        args = [ast.literal_eval(arg) for arg in call.args]
        expected = ast.literal_eval(compare.comparators[0])
    except (ValueError, SyntaxError, TypeError):
        return None
    # ...and the expected value JSON-friendly, since the runner compares results through JSON (a set is not)
    try:
        json.dumps(expected)
    except (TypeError, ValueError):
        return None

    return call.func.id, TestCase(args, expected)


def function_signature(code: str, name: str) -> str | None:
    """The `def name(params):` line of the function called `name` in the reference code, or None."""
    for node in ast.walk(ast.parse(code)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return f"def {name}({ast.unparse(node.args)}):"
    return None


def mbpp_to_problem(record: dict) -> Problem | None:
    """One MBPP record -> Problem, or None if it cannot be expressed in our format."""
    parsed = [parse_assert(line) for line in record["test_list"]]
    if any(p is None for p in parsed):
        return None  # at least one test is not `assert f(...) == literal`

    names = {name for name, _ in parsed}
    if len(names) != 1:
        return None  # the tests must all call the same function
    function_name = names.pop()

    signature = function_signature(record["code"], function_name)
    if signature is None:
        return None

    tests = [test for _, test in parsed]
    problem = Problem(
        name=f"mbpp_{record['task_id']}",
        statement="Problem: " + record["prompt"].strip(),
        function_name=function_name,
        signature=signature,
        public_tests=tests[:NUM_PUBLIC_TESTS],
        hidden_tests=tests[NUM_PUBLIC_TESTS:],
    )
    # split_prompt cuts the node text at the first line starting with a label: the prompt must have none
    if any(line.startswith((STEP_PREFIX, CODE_PREFIX)) for line in problem.to_prompt().splitlines()):
        return None
    return problem


def load_mbpp(split: str = "test") -> list[Problem]:
    """Downloads (first time) the sanitized MBPP split and returns the problems our format can express."""
    from datasets import load_dataset  # imported here: only this function needs the datasets library

    records = load_dataset(DATASET_NAME, "sanitized", split=split)
    problems = [mbpp_to_problem(record) for record in records]
    return [problem for problem in problems if problem is not None]
