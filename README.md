# Self-Correcting Reasoning Graphs

Solving programming problems with a small language model by **searching over a tree of reasoning steps**, instead of generating the solution in one shot.

The model writes its reasoning one step at a time (`Step 1: …`, `Step 2: …`) and ends each branch with code. Several alternative steps are sampled at every level, each partial solution gets a **score** estimating how likely it is to lead to correct code, and a **beam search** keeps only the most promising branches. The generated code is run against test cases in a separate process. Later phases add a **repair step** that fixes code from its test errors and a **process reward model (PRM)** trained to replace the expensive scoring.

Everything runs locally on a Mac with [Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct).

The search currently builds a tree (each step has exactly one parent); the name anticipates extensions that merge equivalent steps of different branches into a graph, and "self-correcting" the planned repair step.

## How it works

```
             ┌───────────── beam search ─────────────┐
 problem ──▶ │  expand: sample k next steps per node │ ──▶ code leaves ──▶ public test ──▶ chosen solution
             │  score: rollouts → public test        │                                       │
             │  keep the B best-scored nodes         │                                       ▼
             └───────────────────────────────────────┘                                hidden tests
```

- **Step format.** A branch is `Step N: <plan in words>` lines followed by `Code:` and a ```` ```python ```` block. Model output is never trusted: a parser validates every step (including a syntax check of the code with `ast`) and discards malformed ones. Step numbers come from the tree depth, not from the model.
- **Generator.** A Hugging Face model prompted with format rules, one solved example (few-shot) and the steps of the current branch as a **prefill**, so it continues *that* branch. A custom stopping criterion ends generation as soon as the model starts a further step.
- **Scores are values.** A node's score is the estimated probability that continuing from it reaches code passing the tests, measured with **rollouts**: from the node, the model completes N independent single-branch solutions, which are run on the public test; the score is the fraction that pass (Math-Shepherd style).
- **Public and hidden tests.** The search only ever sees the public test; the hidden tests decide the final measure, so the reported numbers cannot be gamed by the search.
- **Fair comparison.** Every system is charged for the outputs it requests from the model; Best-of-N gets exactly the budget the beam search spent on the same problem.
- **Safe execution.** Code runs in a separate Python process with a timeout, in a temporary directory.

## Results on MBPP

All 239 problems of the [MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) (sanitized) test split that fit the test format, with 1 public and 2+ hidden tests per problem. A problem counts as solved if the single solution the system returns passes every hidden test.

| System | Solved | Outputs per problem |
|---|---|---|
| Single generation | 87 / 239 (36.4%) | 3.4 |
| Beam search (B = 2, 4 rollouts per node), choosing among the tree's leaves | 149 / 239 (62.3%) | 81.7 |
| **Beam search, also keeping the solutions written by its rollouts** | **162 / 239 (67.8%)** | 81.7 |
| Best-of-N, same budget as the beam | 158 / 239 (66.1%) | 83.4 |

- Searching pays off: every search method solves far more problems than a single generation.
- Keeping the solutions the rollouts write, instead of using them only for scoring, gives the beam about 5× more candidates at no extra cost: on the same searches it wins 18 problems and loses 5 (sign test p ≈ 0.01).
- With that, the beam search is on par with Best-of-N at equal compute (17 problems won against 13, p ≈ 0.58): ahead by a margin well within run-to-run noise (two runs of the same beam search differ on 35 problems).
- One public test is a weak filter: 13–21% of the solutions that pass it fail the hidden tests, and solutions picked from rollouts are fooled more often than tree leaves.

Single runs, seed 0, a 0.5B model, about 21 hours on an M-series Mac in total. Details in [SPECIFICATIONS.md](SPECIFICATIONS.md), §6.2–6.3.

## Getting started

Requires Python 3.10+ (developed with 3.12).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run everything from the project root:

```bash
python -m scripts.demo     # whole pipeline on Two Sum with a scripted mock model: no download, a few seconds
pytest                     # 137 tests, about 5 seconds
```

With the real model (downloaded on first use, ~1 GB; afterwards `HF_HUB_OFFLINE=1` avoids network calls):

```bash
HF_HUB_OFFLINE=1 python -m scripts.hf_tree          # grow a full tree on Two Sum and grade it (~1 min)
HF_HUB_OFFLINE=1 python -m scripts.hf_scores        # rollout scores of the first-level branches (~3 min)
HF_HUB_OFFLINE=1 python -m scripts.run_experiment --problems 20   # MBPP comparison, ~3 min per problem
```

`run_experiment` appends one JSON line per (problem, system) to `results/mbpp_pilot.jsonl` and skips problems already there, so an interrupted run resumes with the same command. On a Mac, prefix it with `caffeinate -is` to keep the machine awake.

## Repository layout

```
reasoning_graphs/
├── reasoning_node.py      the tree
├── config.py              branching per level, maximum depth, beam width
├── step_format.py         step labels, parser, prompt builder (system prompt, few-shot example, prefill)
├── generators/            StepGenerator interface, mock, Hugging Face model, call counter
├── search/                expand, full tree, rollouts, evaluator, beam search, baselines
├── execution/             sandboxed runner, executor, grading and choice of the solution
└── problems/              Problem / TestCase, Two Sum, MBPP adapter
scripts/                   demo and experiments
tests/                     pytest suite, same layout as the package
SPECIFICATIONS.md          detailed design and current behavior of every component, with all measurements
```

## Status

| Phase | Status |
|---|---|
| Tree, step format, parser, mock generator | done |
| Safe code execution | done |
| Real model generator (prompt builder, stopping criterion) | done |
| Rollout evaluator | done |
| Beam search, choice of the solution by branch score | done |
| Baselines and MBPP comparison | done |
| Beam search keeping rollout solutions as candidates | done |
| Repair step from test errors | planned |
| Trained process reward model (data from rollouts, LoRA) | planned |
| vLLM on a cloud GPU, larger datasets | planned |

## References

- Lightman et al., *Let's Verify Step by Step* — process supervision.
- Yao et al., *Tree of Thoughts* — deliberate search over reasoning steps.
- Wang et al., *Math-Shepherd* — step labels from rollouts, without human annotation.
- Zhang et al., *Planning with Large Language Models for Code Generation* — tree search with tests as reward.
- Zhou et al., *Language Agent Tree Search* — tree search with feedback from execution.
- Chen et al., *Teaching Large Language Models to Self-Debug*; Shinn et al., *Reflexion* — fixing code from execution feedback.
- Austin et al., *Program Synthesis with Large Language Models* — the MBPP dataset.
