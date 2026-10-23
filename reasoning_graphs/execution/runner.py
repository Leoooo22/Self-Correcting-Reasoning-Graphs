"""Runs model-generated code on test cases. Meant to be launched in a separate process.

Input (stdin, JSON):  {"code": str, "function_name": str, "tests": [{"args": str, "expected": any}]}
    where "args" is the Python literal of the argument list (e.g. "[(1, 2), 'a']"), so tuples stay tuples
Output (stdout, JSON): {"results": [{"status": "passed" | "failed" | "error", "actual": any, "error": str | None}]}
"""
import ast
import contextlib
import io
import json
import sys


def error_message(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def run_test(func, test: dict) -> dict:
    try:
        # The model's prints must not end up in our JSON output
        with contextlib.redirect_stdout(io.StringIO()):
            result = func(*ast.literal_eval(test["args"]))
    except (Exception, SystemExit) as error:
        return {"status": "error", "actual": None, "error": error_message(error)}

    # Normalize through JSON, so that e.g. a tuple compares equal to a list
    try:
        actual = json.loads(json.dumps(result))
    except (TypeError, ValueError):
        return {"status": "failed", "actual": None, "error": f"result is not JSON-serializable: {result!r}"}

    status = "passed" if actual == test["expected"] else "failed"
    return {"status": status, "actual": actual, "error": None}


def main():
    raw_input = sys.stdin.read()
    if not raw_input.strip():
        print(json.dumps({"results": []}))
        return

    data = json.loads(raw_input)
    code = data["code"]
    function_name = data["function_name"]
    tests = data["tests"]

    # Run the model's code in its own namespace, so it does not mix with the runner's variables
    namespace = {}
    setup_error = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code, namespace)
    except (Exception, SystemExit) as error:
        setup_error = error_message(error)

    func = namespace.get(function_name)
    if setup_error is None and not callable(func):
        setup_error = f"function {function_name} not found"

    # If the code cannot even be set up, every test is an error
    if setup_error is not None:
        results = [{"status": "error", "actual": None, "error": setup_error} for _ in tests]
    else:
        results = [run_test(func, test) for test in tests]

    print(json.dumps({"results": results}))


if __name__ == "__main__":
    main()
