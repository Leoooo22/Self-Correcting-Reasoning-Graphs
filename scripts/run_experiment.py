"""Compares beam search, Best-of-N (same budget) and single generation on MBPP.

Run:        HF_HUB_OFFLINE=1 python -m scripts.run_experiment
Quick try:  HF_HUB_OFFLINE=1 python -m scripts.run_experiment --problems 1 --rollouts 1 --output results/smoke.jsonl
Only some:  HF_HUB_OFFLINE=1 python -m scripts.run_experiment --systems beam beam_rollouts --output results/x.jsonl

Systems: "beam" chooses among the code leaves of the tree; "beam_rollouts" is the same search, choosing among the
tree's code leaves plus the passing solutions written by its rollouts; "best_of_n" gets the beam's budget;
"single" is one solution.

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

SYSTEMS = ["beam", "beam_rollouts", "best_of_n", "single"]
BEAM_SYSTEMS = ("beam", "beam_rollouts")  # both come from one beam search


def evaluate(leaves: list[ReasoningNode], problem: Problem, executor: CodeExecutor) -> dict:
    """The same grading for every system: public tests -> choose a solution -> hidden tests."""
    public_results = grade_on_public(leaves, problem, executor)
    chosen = pick_by_branch(leaves, public_results)
    if chosen is None:
        return {"passed_public": False, "solved": False, "chosen_from": None}
    hidden = executor.run(chosen.code, problem.function_name, problem.hidden_tests)
    chosen_from = "rollout" if chosen.origin is not None else "tree"
    return {"passed_public": True, "solved": hidden.all_passed, "chosen_from": chosen_from}


def make_row(problem: Problem, system: str, leaves: list[ReasoningNode], counter: CountingGenerator,
             seconds: float, executor: CodeExecutor) -> dict:
    return {
        "problem": problem.name,
        "system": system,
        **evaluate(leaves, problem, executor),
        "candidates": len(leaves),
        "outputs": counter.outputs,
        "calls": counter.calls,
        "seconds": round(seconds, 1),
    }


def run_problem(problem: Problem, model: HFGenerator, config: SearchConfig, executor: CodeExecutor,
                num_rollouts: int, systems: list[str]) -> list[dict]:
    """Runs the chosen systems on one problem and returns one result row per system."""
    rows = []
    beam_outputs = None

    # One beam search serves both beam systems: same tree, same rollouts, same cost
    if any(system in systems for system in BEAM_SYSTEMS):
        counter = CountingGenerator(model)
        evaluator = RolloutEvaluator(counter, config, executor, problem, n=num_rollouts)
        start = time.perf_counter()
        nodes = beam_search(ReasoningNode(problem.to_prompt()), counter, config, evaluator)
        seconds = time.perf_counter() - start
        beam_outputs = counter.outputs
        tree_leaves = [node for node in nodes if node.is_terminal]
        if "beam" in systems:
            rows.append(make_row(problem, "beam", tree_leaves, counter, seconds, executor))
        if "beam_rollouts" in systems:
            rows.append(make_row(problem, "beam_rollouts", tree_leaves + evaluator.rollout_leaves,
                                 counter, seconds, executor))

    for system in ("best_of_n", "single"):
        if system not in systems:
            continue
        counter = CountingGenerator(model)  # a fresh counter per system, around the same model
        root = ReasoningNode(problem.to_prompt())
        start = time.perf_counter()
        if system == "best_of_n":
            leaves = best_within_budget(root, counter, config, budget=beam_outputs)
        else:
            leaves = best_of_n(root, counter, config, n=1)
        rows.append(make_row(problem, system, leaves, counter, time.perf_counter() - start, executor))
    return rows


def print_summary(rows: list[dict]) -> None:
    by_system = defaultdict(list)
    for row in rows:
        by_system[row["system"]].append(row)
    print(f"\n{'system':<14} {'solved':>10} {'passed public':>14} {'mean cand.':>11} {'mean outputs':>13} {'mean seconds':>13}")
    for system in SYSTEMS:
        group = by_system[system]
        if not group:
            continue
        n = len(group)
        solved = sum(r["solved"] for r in group)
        public = sum(r["passed_public"] for r in group)
        candidates = sum(r["candidates"] for r in group) / n
        outputs = sum(r["outputs"] for r in group) / n
        seconds = sum(r["seconds"] for r in group) / n
        print(f"{system:<14} {solved:>4}/{n:<5} {public:>8}/{n:<5} {candidates:>11.1f} {outputs:>13.1f} {seconds:>13.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problems", type=int, default=20, help="how many MBPP test problems (the first ones)")
    parser.add_argument("--rollouts", type=int, default=4, help="rollouts per node for the beam search")
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=SYSTEMS, help="which systems to run")
    parser.add_argument("--output", type=Path, default=Path("results/mbpp_pilot.jsonl"))
    args = parser.parse_args()
    # Best-of-N's budget is what the beam search spent: it cannot run without it
    if "best_of_n" in args.systems and not any(system in args.systems for system in BEAM_SYSTEMS):
        parser.error("best_of_n needs a beam system (beam or beam_rollouts) to set its budget")

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
        problem_rows = run_problem(problem, model, config, executor, args.rollouts, args.systems)
        # Written per problem, so an interruption loses at most the current problem
        with args.output.open("a") as file:
            for row in problem_rows:
                file.write(json.dumps(row) + "\n")
        rows.extend(problem_rows)
        verdicts = "  ".join(f"{r['system']}: {'✓' if r['solved'] else '✗'} ({r['candidates']} cand)" for r in problem_rows)
        print(f"[{i}/{len(problems)}] {problem.name:<10} {verdicts}", flush=True)

    print_summary(rows)


if __name__ == "__main__":
    main()
