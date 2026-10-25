"""Compares beam search, Best-of-N (same budget) and single generation on MBPP.

Run:        HF_HUB_OFFLINE=1 python -m scripts.run_experiment
Quick try:  HF_HUB_OFFLINE=1 python -m scripts.run_experiment --problems 1 --rollouts 1 --output results/smoke.jsonl

One JSON line per (problem, system) is appended to the output file. Problems already in the file are skipped,
so an interrupted run can be resumed by launching the same command again.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from reasoning_graphs.config import SearchConfig
from reasoning_graphs.execution.executor import CodeExecutor
from reasoning_graphs.execution.grading import grade_on_public, pick_by_branch
from reasoning_graphs.generators.counting_generator import CountingGenerator
from reasoning_graphs.generators.hf import HFGenerator
from reasoning_graphs.problems.mbpp import load_mbpp
from reasoning_graphs.problems.problem import Problem
from reasoning_graphs.reasoning_node import ReasoningNode
from reasoning_graphs.search.baselines import best_of_n, best_within_budget
from reasoning_graphs.search.beam import beam_search
from reasoning_graphs.search.evaluator import RolloutEvaluator

SYSTEMS = ["beam", "best_of_n", "single"]  # order matters: the beam's cost is Best-of-N's budget


def evaluate(leaves: list[ReasoningNode], problem: Problem, executor: CodeExecutor) -> dict:
    """The same grading for every system: public tests -> choose a solution -> hidden tests."""
    public_results = grade_on_public(leaves, problem, executor)
    chosen = pick_by_branch(leaves, public_results)
    if chosen is None:
        return {"passed_public": False, "solved": False}
    hidden = executor.run(chosen.code, problem.function_name, problem.hidden_tests)
    return {"passed_public": True, "solved": hidden.all_passed}


def run_problem(problem: Problem, model: HFGenerator, config: SearchConfig, executor: CodeExecutor,
                num_rollouts: int) -> list[dict]:
    """Runs every system on one problem and returns one result row per system."""
    rows = []
    beam_outputs = None
    for system in SYSTEMS:
        counter = CountingGenerator(model)  # a fresh counter per system, around the same model
        root = ReasoningNode(problem.to_prompt())
        start = time.perf_counter()

        if system == "beam":
            evaluator = RolloutEvaluator(counter, config, executor, problem, n=num_rollouts)
            leaves = [node for node in beam_search(root, counter, config, evaluator) if node.is_terminal]
        elif system == "best_of_n":
            leaves = best_within_budget(root, counter, config, budget=beam_outputs)
        else:
            leaves = best_of_n(root, counter, config, n=1)

        seconds = time.perf_counter() - start
        if system == "beam":
            beam_outputs = counter.outputs
        rows.append({
            "problem": problem.name,
            "system": system,
            **evaluate(leaves, problem, executor),
            "candidates": len(leaves),
            "outputs": counter.outputs,
            "calls": counter.calls,
            "seconds": round(seconds, 1),
        })
    return rows


def print_summary(rows: list[dict]) -> None:
    by_system = defaultdict(list)
    for row in rows:
        by_system[row["system"]].append(row)
    print(f"\n{'system':<10} {'solved':>10} {'passed public':>14} {'mean outputs':>13} {'mean seconds':>13}")
    for system in SYSTEMS:
        group = by_system[system]
        if not group:
            continue
        solved = sum(r["solved"] for r in group)
        public = sum(r["passed_public"] for r in group)
        outputs = sum(r["outputs"] for r in group) / len(group)
        seconds = sum(r["seconds"] for r in group) / len(group)
        print(f"{system:<10} {solved:>4}/{len(group):<5} {public:>8}/{len(group):<5} {outputs:>13.1f} {seconds:>13.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problems", type=int, default=20, help="how many MBPP test problems (the first ones)")
    parser.add_argument("--rollouts", type=int, default=4, help="rollouts per node for the beam search")
    parser.add_argument("--output", type=Path, default=Path("results/mbpp_pilot.jsonl"))
    args = parser.parse_args()

    # Results already on disk: those problems are skipped
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.output.exists():
        rows = [json.loads(line) for line in args.output.read_text().splitlines() if line.strip()]
    done = {row["problem"] for row in rows}

    problems = load_mbpp("test")[:args.problems]
    model = HFGenerator()  # loaded once, shared by every system through its own counter
    config = SearchConfig()
    executor = CodeExecutor(timeout=2.0)  # MBPP solutions finish in milliseconds: hanging code shouldn't cost 5 s

    for i, problem in enumerate(problems, start=1):
        if problem.name in done:
            continue
        problem_rows = run_problem(problem, model, config, executor, args.rollouts)
        # Written per problem, so an interruption loses at most the current problem
        with args.output.open("a") as file:
            for row in problem_rows:
                file.write(json.dumps(row) + "\n")
        rows.extend(problem_rows)
        verdicts = "  ".join(f"{r['system']}: {'✓' if r['solved'] else '✗'} ({r['outputs']} out)" for r in problem_rows)
        print(f"[{i}/{len(problems)}] {problem.name:<10} {verdicts}", flush=True)

    print_summary(rows)


if __name__ == "__main__":
    main()
