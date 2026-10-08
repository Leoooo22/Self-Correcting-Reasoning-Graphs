# Self-Correcting Reasoning Graphs: project specifications

*Version 2: programming domain (LeetCode style). Supersedes the initial version based on math problems.*

---

## 1. Goal

A system that solves **LeetCode-style** programming problems (problem statement + function signature + test cases) by exploring a **tree of reasoning paths** instead of generating the solution in one shot.

- A **generator** (LLM) proposes, step by step, several alternative reasoning steps in natural language. The last step of each branch is the **final code**.
- An **evaluator** gives each node a score that estimates how likely that branch is to lead to a correct solution.
- A **search algorithm** (beam search, possibly MCTS later) expands promising branches, prunes weak ones and, when the code fails the tests, **generates a repair step** from the error.
- Finally, the rollout-based evaluator is **distilled into a trained PRM** using automatically generated data.

Problems don't have to come from LeetCode. What matters is that they follow the same style: a function to implement, visible examples, hidden tests.

---

## 2. What changes compared to the math version

| Aspect | Math version (initial) | Programming version (current) |
|---|---|---|
| Problems | Arithmetic / math | LeetCode-style coding problems |
| Step | One calculation per line | One natural-language reasoning step; the last step is the code |
| End of branch | Line starting with `Answer:` | Step containing the final code (a ```` ```python ```` block) |
| Objective verification | Every step (only for explicit calculations) | Final code only, by running the tests |
| Judge of intermediate steps | Pretrained PRM (e.g. `Qwen2.5-Math-PRM-7B`) | Phase 1: **rollouts + tests**. Phase 2: **our own trained PRM** |
| Meaning of the score | "Is this step correct?" | "Starting from here, how likely is it to reach code that passes the tests?" (value model, Math-Shepherd style) |
| Correction | Pruning + backtracking only | Pruning + backtracking + **repair step** driven by test errors |
| New components | — | Safe code executor, public/hidden test handling, baselines |

`ReasoningNode`, the `StepGenerator` interface, `expand` and the search setup stay **unchanged**: they are domain-independent.

---

## 3. Scientific references

**Core (from the initial version)**
- *Let's Verify Step by Step* (OpenAI): process supervision beats outcome-only supervision.
- *Tree of Thoughts* (Yao et al.): tree search (BFS/DFS) with exploration, evaluation and backtracking.

**Added for the programming version**
- *Math-Shepherd*: intermediate step labels obtained automatically through rollouts; the basis for training our PRM.
- *Planning with Large Language Models for Code Generation* (PG-TD): tree search with tests as the reward.
- *Language Agent Tree Search* (LATS): tree + tests + reflection on errors.
- *Teaching Large Language Models to Self-Debug* and *Reflexion*: fixing code from execution feedback.
- *CodeT*: automatic test generation when no tests are available (future extension).

---

## 4. Architecture

```
                ┌──────────────────┐
  Problem  ───▶ │   Search         │ ◀──── Evaluator (rollout / PRM)
  + public      │ (beam search)    │              │
  tests         └──────┬───────────┘              ▼
                       │ expand            CodeExecutor (tests)
                       ▼
                 StepGenerator (mock / HF / vLLM)
                       │
                       ▼
                 ReasoningNode (tree)
```

### 4.1 `ReasoningNode`: tree node ✅ done
File: `reasoning_node.py`. Pure data structure, with no model or search logic.
- Attributes: `id`, `step_text`, `parent`, `children`, `depth`, `score` (`None` = not evaluated), `is_terminal`.
- Methods: `add_child`, `get_path`, `get_full_text`, `path_score` (product, skips the root and `None`), `is_root`, `is_leaf`, `print_tree`, `__repr__`.

### 4.2 `StepGenerator`: step generator
File: `step_generator.py`. Abstract interface: `generate_steps(prompt: str, n: int) -> list[str]`.
- `MockGenerator` ✅ done: candidates per level + seed. **Needs adapting**: it currently infers the level by counting lines, which breaks with multi-line problems and steps.
- `HFGenerator`: small model via `transformers` on the Mac (e.g. `Qwen2.5-Coder-0.5B/1.5B-Instruct`).
- `VLLMGenerator`: same model with vLLM on a cloud GPU (prefix caching for the tree's shared prefixes).

Every generator must produce diverse alternatives (sampling with temperature > 0) and **one step per call** (stop sequence).

### 4.3 `expand`: growing the tree ✅ done
File: `tree_search.py`. `expand(node, generator, k) -> list[ReasoningNode]`: skips terminal nodes, cleans, deduplicates, creates the children and marks terminal ones.
**Needs adapting**: terminal detection (currently `ANSWER_PREFIX = "Answer:"`) must become pluggable, because for code the terminal step is the one containing the final code.

### 4.4 Step format 🔲 to be decided
- Step = natural-language reasoning step (e.g. "Use a hash map from value to index").
- Last step = complete code in a ```` ```python ```` block.
- A step separator more robust than a newline (to be defined).
- **Small maximum depth** (2–3 steps + code), with more breadth at the first level, where the strategy is chosen.

### 4.5 Problems and test cases 🔲
- Problem structure: statement, function signature, **public tests** (2–3, like the examples in the statement), **hidden tests** (all the others).
- Test format: function call with arguments → expected output (LeetCode style).
- First toy problem: **Two Sum**, with hand-written tests.
- Then public datasets (from Hugging Face, `datasets` library): MBPP, HumanEval, then LeetCodeDataset / LiveCodeBench.

### 4.6 `CodeExecutor`: safe execution 🔲
- Runs the generated code in a **separate process** (`subprocess`) with a **timeout**.
- Returns, for each test: passed / failed / error, plus details (input, expected output, actual output, traceback). These are needed for the repair step.

### 4.7 `Evaluator`: node evaluation 🔲
Abstract interface that assigns a score in [0, 1] to a node.
- `RolloutEvaluator` (phase 1): completes the solution N times from the node, runs the **public tests**, score = success rate.
- `LearnedPRM` (phase 2): the trained PRM (see §5).
- Switching between the two = changing one line.

### 4.8 Search 🔲
- **Beam search**: like the frontier-based BFS already used in the `expand` test, but at each level only the `B` best-scoring nodes are kept.
- Output: the code of the best terminal node according to the public tests and the score.
- Possible extension: MCTS.

### 4.9 Repair step 🔲 (added value)
When a terminal node's code fails the public tests, the outcome (input, expected, actual, error) is added to the context and the generator produces a new step analysing the error + fixed code, as a **child of the failed node**.

---

## 5. The PRM (advanced phase)

**What it does.** Input: problem + steps up to a node. Output: a number in [0, 1] = estimated probability that this path leads to code that passes the tests. It replaces rollouts during search: a single model forward pass instead of N full generations.

**How it is built.**
1. **Data collection**: run the search with `RolloutEvaluator` on many **training** problems and save each node as `(text up to the node, rollout success rate)` in a JSONL file. All tests of training problems may be used for labelling. Order of magnitude: about 300 problems × about 30 nodes ≈ 10,000 examples.
2. **Model**: small pretrained LLM (e.g. `Qwen2.5-Coder-0.5B`) with a head that outputs a single number (`AutoModelForSequenceClassification`, `num_labels=1`).
3. **Training**: binary cross-entropy with soft labels, **LoRA** (`peft`), on the Mac or on a Colab GPU. Train/validation split **by problem**, not by node.
4. **Verification**: (a) the PRM on its own, on unseen problems (correlation with rollouts, AUC); (b) the PRM inside the search, compared with rollouts and baselines.

**Known risks.** Noisy labels (a small model failing even with a good plan); problems that are too easy or too hard (labels all 1 or all 0); superficial shortcuts (keywords); dependence on the generator (if the generator changes, collect new data).

---

## 6. System evaluation

- **Main metric**: percentage of problems solved on the **hidden tests** (pass@1).
- **Cost**: number of model calls or generated tokens.
- **Baselines**, compared **at equal compute**:
  1. single generation;
  2. **Best-of-N**: N independent complete solutions, keep the one that passes the public tests. A strong baseline for code.
- Systems to compare: search with rollouts, search with the PRM, with and without the repair step.

---

## 7. Methodological rules (anti-contamination)

1. The **search uses only the public tests**. Hidden tests are only for the final measurement.
2. **PRM training problems** are separate from evaluation problems.
3. The PRM train/validation split is **by problem**.
4. Systems are compared **at equal compute**.
5. Fixed seeds wherever possible, for reproducibility.

---

## 8. Tech stack

- **Python** 3.10+ (current venv: 3.12)
- **`torch`, `transformers`**: local models
- **`datasets`**: loading problem datasets
- **`peft`**: LoRA for PRM training
- **vLLM**: fast generation on Linux/cloud GPUs (not available on the Mac)
- **`subprocess`** (standard library): safe code execution
- **Models**: `Qwen2.5-Coder-0.5B/1.5B-Instruct` (generator and PRM base)

---

## 9. Roadmap

| # | Step | Status |
|---|---|---|
| 0 | `ReasoningNode`, `StepGenerator` + `MockGenerator`, `expand` | ✅ done |
| 1 | Step format + Two Sum with hand-written tests (public/hidden); adapt mock and terminal detection | 🔲 next |
| 2 | `CodeExecutor` (separate process + timeout) | 🔲 |
| 3 | `HFGenerator` with a small coder model | 🔲 |
| 4 | `Evaluator` + `RolloutEvaluator` | 🔲 |
| 5 | Beam search | 🔲 |
| 6 | Baselines (single, Best-of-N) and comparison on MBPP / HumanEval | 🔲 |
| 7 | Repair step from errors | 🔲 added value |
| 8 | Data collection + PRM training (`LearnedPRM`) | 🔲 advanced |
| 9 | `VLLMGenerator` on a cloud GPU, larger datasets | 🔲 scalability |

Steps 1–6 make up the complete project; 7–8 are the added value.

---

## 10. Open decisions

- Step separator and exact prompt format.
- Maximum depth and number of children per level.
- Criterion for detecting the final step (code block).
- Number of rollouts per node (cost vs noise).
- Number of public tests per problem.
- Starting dataset (MBPP or HumanEval).
