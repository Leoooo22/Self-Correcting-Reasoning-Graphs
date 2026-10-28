# Self-Correcting Reasoning Graphs: project specifications

*Describes the current behavior of the code. Last updated: 2026-10-14.*

---

## 1. Goal

A system that solves **LeetCode-style** programming problems (problem statement + function signature + test cases) by exploring a **tree of reasoning paths** instead of generating the solution in one shot.

- A **generator** (LLM) proposes, step by step, several alternative reasoning steps in natural language. The last step of each branch is the **final code**.
- An **evaluator** gives each node a score that estimates how likely that branch is to lead to a correct solution.
- A **search algorithm** (beam search, possibly MCTS later) expands promising branches, prunes weak ones and, when the code fails the tests, **generates a repair step** from the error.
- Finally, the rollout-based evaluator is **distilled into a trained PRM** using automatically generated data.

Problems don't have to come from LeetCode. What matters is that they follow the same style: a function to implement, visible examples, hidden tests.

**Meaning of a node's score.** Not "is this step correct?", but "starting from here, how likely is it to reach code that passes the tests?" (a value model, Math-Shepherd style). Only the final code can be checked objectively, by running it.

---

## 2. Scientific references

- *Let's Verify Step by Step* (OpenAI): process supervision beats outcome-only supervision.
- *Tree of Thoughts* (Yao et al.): tree search (BFS/DFS) with exploration, evaluation and backtracking.
- *Math-Shepherd*: intermediate step labels obtained automatically through rollouts; the basis for training our PRM.
- *Planning with Large Language Models for Code Generation* (PG-TD): tree search with tests as the reward.
- *Language Agent Tree Search* (LATS): tree + tests + reflection on errors.
- *Teaching Large Language Models to Self-Debug* and *Reflexion*: fixing code from execution feedback.
- *CodeT*: automatic test generation when no tests are available (future extension).

---

## 3. Architecture

### 3.1 Overview

```
                ┌──────────────────┐
  Problem  ───▶ │   Search         │ ◀──── Evaluator: rollouts ✅ / PRM 🔲
  + public      │ (beam search) ✅ │              │
  tests         └──────┬───────────┘              ▼
                       │ expand ✅          CodeExecutor (tests) ✅
                       ▼
                 StepGenerator: MockGenerator ✅ / HFGenerator ✅ / VLLMGenerator 🔲
                       │
                       ▼
                 ReasoningNode (tree) ✅
```

### 3.2 Files

The project code is the `reasoning_graphs` package, split into subpackages by role. Every folder has an (empty) `__init__.py`. Runnable scripts live in `scripts/`; tests, configuration and documentation stay at the project root.

```
Self-Correcting-Reasoning-Graphs/
├── reasoning_graphs/
│   ├── reasoning_node.py
│   ├── config.py
│   ├── step_format.py
│   ├── generators/    base.py, hf.py, counting_generator.py
│   ├── search/        expand.py, full_tree.py, rollout.py, evaluator.py, beam.py, baselines.py
│   ├── execution/     executor.py, runner.py, grading.py
│   └── problems/      problem.py, two_sum.py, mbpp.py
├── scripts/           demo.py, hf_tree.py, hf_scores.py, run_experiment.py
├── tests/             same layout as the package (§4.11)
├── results/           experiment outputs (not versioned)
├── pytest.ini
├── requirements.txt
├── README.md
└── SPECIFICATIONS.md
```

| Module (under `reasoning_graphs/`) | Contents | Depends on |
|---|---|---|
| `reasoning_node.py` | `ReasoningNode`: the tree | — |
| `config.py` | `SearchConfig`: branching per level, maximum depth | — |
| `step_format.py` | Step labels, parser, formatter, prompt builder (system prompt, few-shot example, `split_prompt`, `build_messages`) | — |
| `generators/base.py` | `StepGenerator` interface, `MockGenerator` | `step_format` |
| `generators/hf.py` | `HFGenerator` (real model via `transformers`), `StopAtNextStep` | `generators.base`, `step_format`, `torch`, `transformers` |
| `generators/counting_generator.py` | `CountingGenerator`: counts calls and requested outputs of any generator | `generators.base` |
| `search/expand.py` | `expand` | `reasoning_node`, `generators.base`, `config`, `step_format` |
| `search/full_tree.py` | `grow_full_tree`: expands every node | `search.expand` |
| `search/rollout.py` | `rollout`: one branch from a node down to code | `search.expand`, `config`, `reasoning_node` |
| `search/evaluator.py` | `Evaluator` interface, `RolloutEvaluator` | `search.rollout`, `execution.executor`, `problems.problem` |
| `search/beam.py` | `beam_search` | `search.expand`, `search.evaluator`, `config` |
| `search/baselines.py` | `best_of_n` (single generation is `n = 1`), `best_within_budget` | `search.rollout`, `generators.counting_generator` |
| `execution/runner.py` | Child process that executes model code on the tests | standard library only |
| `execution/executor.py` | `CodeExecutor`, `TestResult`, `ExecutionResult` | `problems.problem`, `runner.py` (as a file in the same folder) |
| `execution/grading.py` | `grade_on_public`, `pick_solution`, `branch_value`, `pick_by_branch`: grade code leaves and choose the solution | `execution.executor`, `problems.problem`, `reasoning_node` |
| `problems/problem.py` | `TestCase`, `Problem` | — |
| `problems/two_sum.py` | `TWO_SUM` | `problems.problem` |
| `problems/mbpp.py` | MBPP adapter: `parse_assert`, `function_signature`, `mbpp_to_problem`, `load_mbpp` | `problems.problem`, `step_format`, `datasets` (inside `load_mbpp` only) |

| Script (under `scripts/`, run with `python -m scripts.<name>`) | Contents |
|---|---|
| `demo.py` | End-to-end demo with the mock generator, no model needed (uses everything except `generators.hf`); also defines `print_public_results` |
| `hf_tree.py` | Full tree on Two Sum with the real model, with measurements (§4.10b) |
| `hf_scores.py` | Rollout scores of the first-level branches of Two Sum with the real model (§4.12) |
| `run_experiment.py` | MBPP comparison of beam search, Best-of-N at equal compute and single generation (§6.1) |

| At the project root | Contents |
|---|---|
| `README.md` | Project overview for readers of the repository |
| `tests/` | `pytest` tests (§4.11) |
| `pytest.ini` | `pytest` configuration |
| `results/` | JSONL outputs of the experiments, ignored by git |

**Imports** are absolute from the package root: `from reasoning_graphs.step_format import parse_step`, `from reasoning_graphs.generators.base import StepGenerator`. They work when Python runs from the project root: `pytest`, or a module run with `-m` (`python -m scripts.demo`, `python -m reasoning_graphs.generators.hf`). Running a file by its path (`python scripts/demo.py`) fails with `ModuleNotFoundError`, because Python then only sees that file's folder.

`generators/hf.py` is separate from `generators/base.py` so that the mock, the parser and their tests never need `torch` loaded.

New modules go in the subpackage matching their role: evaluators and beam search in `search/`, dataset adapters in `problems/`, `VLLMGenerator` in `generators/`.

### 3.3 One expansion, end to end

```
expand(node, generator, config)                                  search/expand.py
  │ force_code = node.depth >= config.max_steps
  │ k          = config.code_branching if force_code else config.branching[node.depth]
  │ prompt     = node.get_full_text()  (+ "\nCode:" if force_code)
  ▼
generator.generate_steps(prompt, k)  →  k raw strings
  │   HFGenerator: build_messages → chat template → tokenizer → generate (StopAtNextStep) → decode
  ▼
for each raw string:
  │ if force_code: raw = "Code:\n" + raw        (the label was in the prompt)
  │ parsed = parse_step(raw, force_code)        → None = discarded
  │ text   = format_step(parsed, node.depth + 1) → canonical text, also the dedup key
  ▼
one child per distinct canonical text; code children are terminal and store .code
```

---

## 4. Components (current behavior)

### 4.1 `ReasoningNode`: the tree ✅
File: `reasoning_graphs/reasoning_node.py`. Pure data structure, with no model or search logic.

**Attributes**

| Attribute | Type | Meaning |
|---|---|---|
| `id` | `int` | Unique, from a class-level counter (`itertools.count`), in creation order |
| `step_text` | `str` | Root: the problem text. Other nodes: the step **with its label**, in canonical form: `Step N: ...`, or `Code:` followed by the python block |
| `parent` | `ReasoningNode \| None` | `None` for the root |
| `children` | `list[ReasoningNode]` | In creation order |
| `depth` | `int` | 0 for the root, parent depth + 1 otherwise |
| `score` | `float \| None` | `None` = not evaluated yet (set by the future evaluator) |
| `code` | `str \| None` | Extracted code (without the fences), only on terminal nodes |
| `origin` | `ReasoningNode \| None` | Only on the code node a rollout returns: the real node the rollout started from (its own path is made of unscored detached copies). `None` everywhere else |
| `is_terminal` | `bool` | `True` for code nodes: they are never expanded |

**Methods**
- `add_child(step_text)`: creates the child, appends it to `children`, returns it.
- `get_path()`: list of nodes from the root to this node, both included.
- `detached_copy()`: a new chain of nodes, built from the root with the same `step_text`s as this node's path (the root with `ReasoningNode(...)`, the others with `add_child`), whose last node is returned. It has the same `get_full_text()` and `depth` as this node, its chain is linked both ways (`parent` and `children`), and it shares no node with this tree: adding children to it leaves the real tree unchanged. Used by rollouts, which must not attach anything to the tree. Scores, code and terminal flags are not copied.
- `get_full_text()`: the `step_text` of every node on the path joined with `"\n"`. This is what generators receive as the prompt. Since `step_text` stores the label, the result reads like a complete partial solution: problem, then `Step 1: ...`, `Step 2: ...`.
- `path_score()`: product of the scores along the path, skipping the root and nodes with `score is None`; `1.0` if nothing is scored.
- `is_root()`, `is_leaf()` (no children).
- `print_tree(indent=0)`: recursive print, 4 spaces per level.
- `__repr__`: `(id=…, depth=…, score=…, terminal=…, text="…")`, with the text on one line (`\n` shown as `⏎`) and cut at 40 characters.

### 4.2 Step format and parser ✅
File: `reasoning_graphs/step_format.py`. Guiding principle: **never trust the model's output** — parse and validate it, discard what is malformed. Everything about the format lives in this one module, shared by `expand`, the mock, `HFGenerator` and the checks: changing a label means changing one constant.

**Labels**: `STEP_PREFIX = "Step"`, `CODE_PREFIX = "Code:"`.
- Reasoning step: `Step N: <natural-language text>`.
- Final step: `Code:` on its own line, followed by a ```` ```python ```` block with the complete solution.
- The step number is **assigned by the system from the node depth**, never taken from the model.

Example of a complete branch (what `get_full_text()` returns on the code node):
````
Problem: Given an array of integers nums and an integer target,
...
def two_sum(nums: list[int], target: int) -> list[int]:
Step 1: Use a hash map from value to index, so each lookup is O(1).
Step 2: Store each value with its index while scanning the array.
Code:
```python
def two_sum(nums, target):
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
```
````

**`ParsedStep(is_code, text, code)`**: result of parsing. For a reasoning step `text` is the step without its label and `code` is `None`; for a code step both `text` and `code` hold the extracted code.

**`extract_code(text) -> str | None`**: finds the first ```` ```python ... ``` ```` block (non-greedy, across lines), strips it, and returns it only if `ast.parse` accepts it (syntax check, nothing is executed). No block, unclosed block, empty block (empty code parses fine but is no solution) or syntax error → `None`.

**`parse_step(raw, force_code) -> ParsedStep | None`** — rules, in order:
1. Strip surrounding whitespace.
2. If the text starts with `Code:` (case-insensitive): code step. Return it if `extract_code` succeeds, otherwise `None`. Anything after the closing fence is ignored.
3. If `force_code` is true and the text is not a code step: `None` (at maximum depth only code is valid).
4. Remove a leading `Step <optional number> :` label (case-insensitive, flexible spaces). A missing label is accepted: the text is taken as a step anyway.
5. Truncate at the first following line that starts (after optional spaces) with `Step` or `Code:`: only the **first** step is kept. Lines in between that don't start with a label stay, so a step can span several lines.
6. Empty after stripping → `None`.

**`format_step(parsed, depth) -> str`**: canonical text stored in the node: `"Step {depth}: {text}"` or `"Code:\n```python\n{code}\n```"`. `expand` uses it as the deduplication key, so outputs that differ only in label, numbering or spacing collapse into one child.

**Maximum depth.** Below `max_steps`, the prompt ends with no label and the generator chooses between another `Step` and `Code:`. At `max_steps`, `expand` appends `Code:` to the prompt: the model can only write code and any other output is discarded. Every branch therefore terminates, which rollouts require.

### 4.3 Prompt builder for real generators ✅
File: `reasoning_graphs/step_format.py`. Turns a node's text into the chat messages a real (instruct) model receives. Pure functions: no model needed to test them.

**Constants**
- `SYSTEM_PROMPT`: the format rules — solve one step at a time; each reasoning step on a single line starting with `Step N:`, describing the plan in words, without code; when the plan is complete, `Code:` followed by a ```` ```python ```` block with the complete function; nothing else (no explanations, usage examples or prints). Kept short: small models follow examples better than long instructions.
- `FEW_SHOT_PROBLEM`: a solved example problem ("return the largest element of a non-empty list of integers") written in **exactly** the shape of `Problem.to_prompt()` (`Problem:` statement, `Examples:`, `f(args) -> result` lines, signature). The second example uses negative numbers, so that a solution starting from 0 would be wrong. It is deliberately unrelated to the evaluation problems, so the model cannot copy a solution.
- `FEW_SHOT_SOLUTION`: its solution in the canonical format: two plan-level steps, then `Code:` and a clean function (no comments, prints or usage examples), nothing after the closing fence. The model imitates everything in it. Its code passes `extract_code` and returns the expected results on both examples.

**`split_prompt(full_text) -> (problem, steps)`**: splits the text at the **first line that starts with `Step` or `Code:`** (case-sensitive: tree text is always canonical). Everything before is the problem, everything from that line on is the steps, both re-joined with `"\n"`. No such line (root node) → `(full_text, "")`. This relies on no line of a problem statement starting with `Step`; `tests/problems/test_two_sum.py` checks it for `TWO_SUM`.

**`build_messages(full_text) -> list[dict]`**: messages with `role` and `content`:

| # | role | content |
|---|---|---|
| 1 | `system` | `SYSTEM_PROMPT` |
| 2 | `user` | `FEW_SHOT_PROBLEM` |
| 3 | `assistant` | `FEW_SHOT_SOLUTION` |
| 4 | `user` | the problem part of `full_text` |
| 5 | `assistant` | **only if there are steps**: the steps + `"\n"` (prefill) |

Messages 2–3 are the few-shot example: a fake earlier turn in which the model "sees itself" answer in the right format. Message 5 is the **prefill**: an assistant message that is already started and that the model continues, so it develops *this* branch instead of starting over. It always ends with `"\n"`: without it the model continues the last step's sentence instead of starting a new step.

The three kinds of node:

| Node | Messages | Last message | Model continues with |
|---|---|---|---|
| Root | 4 | `user`: the problem | `Step 1: ...` (or directly `Code:`) |
| Intermediate | 5 | `assistant`: `"Step 1: ...\n...Step k: ...\n"` | `Step k+1: ...` or `Code:` |
| Forced code | 5 | `assistant`: `"...Step k: ...\nCode:\n"` | the ```` ```python ```` block |

`build_messages` does not choose how to apply the chat template; the generator decides from the role of the last message (§4.6).

### 4.4 `SearchConfig` ✅
File: `reasoning_graphs/config.py`. Dataclass with defaults.
- `branching: list[int] = [4, 2, 2]`: how many candidates to request at each depth (index = depth of the node being expanded). Wide at the first level (strategy choice), narrow afterwards. Its length is the maximum number of reasoning steps.
- `code_branching: int = 2`: how many candidates to request when code is forced.
- `beam_width: int = 2`: how many non-terminal nodes the beam search keeps at each level (§4.13). Ignored by `grow_full_tree` and by rollouts.
- `max_steps` (property) = `len(branching)`.
- `forces_code(depth) -> bool`: `depth >= max_steps`, i.e. a node at this depth may only be followed by code.
- `num_candidates(depth) -> int`: how many outputs to request when expanding a node at this depth: `code_branching` if `forces_code(depth)`, else `branching[depth]`. With the defaults: 4 at the root, then 2, 2, and 2 (code) at depth 3.
- These two methods are the single place where the rule lives: `expand` uses them to decide what to ask the generator, and analysis code uses them to know, after the fact, how many outputs a node had requested (requested − `len(node.children)` = outputs lost to parsing or deduplication).
- Validation in `__post_init__`: `branching` not empty, every value ≥ 1, `code_branching` ≥ 1, `beam_width` ≥ 1, otherwise `ValueError`. Each instance gets its own list (`default_factory`).

The number of rollouts is a parameter of the `RolloutEvaluator`, the sampling temperature of the generator: they are not search settings.

### 4.5 `StepGenerator` interface and `MockGenerator` ✅
File: `reasoning_graphs/generators/base.py`.

**Interface** (abstract class): `generate_steps(prompt: str, n: int) -> list[str]`.
- `prompt` is a node's `get_full_text()`, possibly ending with `"\nCode:"` when code is forced.
- It returns `n` raw strings (fewer distinct ones are fine: `expand` deduplicates). They may be malformed or contain extra text: the parser handles that.
- When the prompt ends with `Code:`, each output is the **continuation only**: the code block **without** the label (`expand` adds it back).
- Every generator must produce diverse alternatives and, ideally, stop after one step.

**`MockGenerator(step_candidates, code_candidates, seed)`**: deterministic fake model for tests and demos.
- `step_candidates`: one list of candidate strings per level; `code_candidates`: code blocks without the label; `seed`: its own `random.Random`, so the same calls give the same outputs.
- Prompt ending with `Code:` (case-insensitive) → `n` random code candidates, unlabelled.
- Otherwise level = number of prompt lines starting with `Step` (case-insensitive) → `n` random choices (with repetition) from that level's candidates.
- Candidates exhausted (level ≥ number of levels) → `n` code candidates **with** the `Code:\n` label, as a model choosing to write code would do.

### 4.6 `HFGenerator`: real model ✅
File: `reasoning_graphs/generators/hf.py`. Implements `StepGenerator` with a Hugging Face causal language model.

**Constructor** `HFGenerator(model_name, temperature, top_p, max_new_tokens, seed)`:

| Parameter | Default | Role |
|---|---|---|
| `model_name` | `"Qwen/Qwen2.5-Coder-0.5B-Instruct"` | Model to load (~494M parameters, ~1 GB) |
| `temperature` | `0.8` | Sampling temperature: < 1 safer choices, > 1 more variety |
| `top_p` | `0.95` | Nucleus sampling: keep the most likely tokens covering 95% of the probability |
| `max_new_tokens` | `256` | Hard limit per answer; large enough for a complete function |
| `seed` | `0` | Passed to `transformers.set_seed` once |

- Device: `"mps"` (Apple GPU) if available, otherwise `"cpu"`.
- Tokenizer: `AutoTokenizer.from_pretrained(model_name)`. Model: `AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)`, moved to the device, put in `eval()` mode. Both are loaded **once**, in the constructor. The first run downloads the weights into `~/.cache/huggingface`; afterwards they load from the cache (`HF_HUB_OFFLINE=1` avoids any network call).
- `set_seed(seed)` is called once in the constructor, not per call: a whole run is reproducible, and successive calls still get different random numbers.

**`generate_steps(prompt, n)`**, in five phases:
1. **Messages**: `build_messages(prompt)` (§4.3).
2. **Chat template** (`apply_chat_template(..., tokenize=False)`), chosen from the last message:
   - last message `user` (root) → `add_generation_prompt=True`: opens an empty assistant turn (`...<|im_start|>assistant\n`);
   - last message `assistant` (prefill) → `continue_final_message=True`: leaves the prefilled message **open**, without the closing `<|im_end|>`, so the model continues it. (`add_generation_prompt` would close the prefill and the model would start a new answer from scratch.)
3. **Tokenization**: `tokenizer(text, return_tensors="pt")` on the device; `prompt_length` = number of prompt tokens (`input_ids.shape[1]`).
4. **Generation** under `torch.no_grad()`: `model.generate(**inputs, do_sample=True, temperature, top_p, max_new_tokens, num_return_sequences=n, stopping_criteria=StoppingCriteriaList([StopAtNextStep(tokenizer, prompt_length)]))`. All `n` answers are sampled in one call from the same prompt. Output shape: `[n, prompt_length + generated tokens]`; sequences that stop early are padded.
5. **Decoding**: for each row keep the tokens after `prompt_length` and `decode(..., skip_special_tokens=True)` (drops `<|im_end|>` and padding). Returns the list of `n` strings.

**`StopAtNextStep(tokenizer, prompt_length)`**: a `transformers.StoppingCriteria`. `generate` calls it after every generated token with the current `input_ids` (`[n, current length]`) and the scores; it returns a boolean tensor with one value per sequence (`True` = stop that sequence; the others continue).
- It decodes **only the new tokens** (`input_ids[:, prompt_length:]`, with `batch_decode`) and stops a sequence as soon as its new text contains `"\n" + STEP_PREFIX` or `"\n" + CODE_PREFIX`, i.e. the model has finished its step and started another one. The cut-off text (e.g. `"Step 2: ...\nStep"`) is removed later by `parse_step`.
- Only new tokens are checked because the prompt itself ends with `"\n"` (prefill or `assistant\n`): `transformers`' built-in `stop_strings` match across the prompt boundary and would stop every answer right after its first word `Step`.
- It works on decoded text rather than token ids because newlines are often merged with other characters into one token (e.g. `".\n"`).
- A new instance is created at every call, because `prompt_length` differs for every node.
- `scores` and `**kwargs` are unused but required by the `StoppingCriteria` call signature.

**Observed behavior** (Two Sum, 3 answers per call, M-series Mac):
- With few-shot + prefill the model follows the format: correct step numbers, steps consistent with the branch, `Code:` + block. No answer was discarded in the checks.
- With the stopping criterion each call takes ~1.5–1.7 s instead of ~4–4.5 s without it.
- In the forced-code case the model writes only the block and stops by itself (end of turn); the criterion never triggers there.
- The root produces genuinely different first steps (different strategies), which is what the tree needs.
- A step may span several lines when the following lines don't start with `Step`; generation then stops at the next `Step`/`Code:` and the parser keeps the multi-line step.
- Well-formatted code can still be wrong (e.g. `nums.index(complement)` reusing the same element): format is checked by the parser, correctness only by the tests.

### 4.6b `CountingGenerator`: measuring cost ✅
File: `reasoning_graphs/generators/counting_generator.py`. `CountingGenerator(generator)` is itself a `StepGenerator` that wraps another one: `generate_steps(prompt, n)` adds 1 to `calls` and `n` to `outputs`, then returns the wrapped generator's answer unchanged.
- Every generator call of every system (expansions, rollouts inside the evaluator, baselines) goes through `generate_steps` of the generator the caller passed in, so wrapping that one generator counts everything without touching any search code.
- `outputs` is the cost measure for comparisons at equal compute: one call with `n` outputs samples `n` sequences in parallel from the same prompt, costing more than a call with `n = 1` (though less than `n` separate calls). `calls` is kept as well, e.g. to relate cost to latency.
- Usage: one counter per (system, problem), around a model loaded once.

### 4.7 `expand`: growing the tree ✅
File: `reasoning_graphs/search/expand.py`. `expand(node, generator, config) -> list[ReasoningNode]`.
1. Terminal node → `[]`, and the generator is not called.
2. `force_code = config.forces_code(node.depth)`; `k = config.num_candidates(node.depth)` (§4.4).
3. `prompt = node.get_full_text()`, plus `"\n" + CODE_PREFIX` if forced.
4. `raw_steps = generator.generate_steps(prompt, k)`.
5. For each raw output: if forced, prepend `CODE_PREFIX + "\n"`; `parse_step(raw, force_code)`; skip `None`; key = `format_step(parsed, node.depth + 1)`. A dict keyed on the canonical text removes duplicates while keeping the first occurrence's order.
6. One child per distinct text (`add_child(text)`); code children get `is_terminal = True` and `code = parsed.code`.
7. Returns the new children (possibly fewer than `k`, or none if every output was invalid).

`expand` does not know which generator it is using: mock and real model are interchangeable.

### 4.7b `grow_full_tree`: exhaustive tree ✅
File: `reasoning_graphs/search/full_tree.py`. `grow_full_tree(root, generator, config) -> list[ReasoningNode]`.
- Breadth-first: the frontier starts as `[root]`; each round expands every frontier node with `expand` and the new children become the next frontier.
- Stops when a round produces no children. It always terminates: at `max_steps` `expand` forces code, and code nodes are terminal.
- Returns every node of the tree, root first, then level by level in creation order.
- No scoring and no pruning: every node is expanded. It is the baseline the beam search (§4.13) will restrict, and the way to measure a whole tree's size and cost.

### 4.8 Problems and test cases ✅
File: `reasoning_graphs/problems/problem.py`.
- `TestCase(args: list, expected)`: a call with **positional** arguments and the expected return value (LeetCode style). Positional, so that a correct solution with different parameter names still passes. Only JSON-friendly values (numbers, strings, lists, booleans, `None`).
- `Problem(name, statement, function_name, signature, public_tests, hidden_tests)`.
- `Problem.to_prompt()`: the statement, then `Examples:`, then one line `function_name(args) -> expected` per **public** test (values shown with `repr`), then the signature. Hidden tests never reach the model.

File: `reasoning_graphs/problems/two_sum.py`.
- `TWO_SUM`: statement starting with `Problem:`; ambiguity removed so tests can compare with `==` ("Return the two indices in increasing order"); signature `def two_sum(nums: list[int], target: int) -> list[int]:`.
- 2 public tests; 5 hidden tests, each targeting a typical bug: duplicates (`[3, 3]`), negative numbers, zeros with target 0, pair at the end, first element not in the pair.
- Checked by `tests/problems/test_two_sum.py`: a reference solution passes all 7 tests; no hidden test appears in the prompt; no prompt line starts with `Step`.

**MBPP** — file `reasoning_graphs/problems/mbpp.py`. Adapter from the sanitized MBPP dataset (`google-research-datasets/mbpp`, config `sanitized`, Hugging Face `datasets` library) to `Problem`.
- Splits: `train` (120 records), `validation` (43), `test` (257), `prompt` (7, meant as few-shot examples). **Evaluation uses `test`**; `train` is for collecting PRM data and `validation` for tuning, so that the problems a PRM is trained on are never the ones it is evaluated on (§7, rule 2), and results are comparable with the literature.
- A record has `task_id`, `prompt` (the statement), `code` (a reference solution), `test_list` (usually 3, sometimes 4–5 lines `assert ...`) and `test_imports` (non-empty only for the 13 test-split problems whose tests use `math.isclose`).
- `parse_assert(line) -> (function_name, TestCase) | None`: parses the line with `ast` and accepts exactly `assert name(arg, ...) == expected`: one statement, an `Assert` whose test is a `Compare` with a single `Eq`, a `Call` on a plain `Name` on the left, no keyword arguments; every argument and the expected value must be literals (`ast.literal_eval`: numbers, strings, tuples, lists, dicts, sets, booleans, `None`) and the expected value must be JSON-serializable (the runner compares through JSON, so a set is not allowed there). Anything else → `None`.
- `function_signature(code, name) -> str | None`: the `def name(params):` line of the function called `name` in the reference code (found with `ast.walk`, rebuilt with `ast.unparse`, so spacing is normalized), or `None`.
- `mbpp_to_problem(record) -> Problem | None`: every test must parse, all must call the same function, and that function must exist in the reference code; then `name = "mbpp_<task_id>"`, `statement = "Problem: " + prompt` (same shape as Two Sum and the few-shot example), the signature from the reference code, **the first test public and the others hidden** (`NUM_PUBLIC_TESTS = 1`). A problem whose prompt has a line starting with `Step` or `Code:` is rejected too, since `split_prompt` would cut there. Otherwise `None`.
- `load_mbpp(split="test") -> list[Problem]`: downloads the split (cached after the first time) and keeps the convertible problems. `datasets` is imported inside the function, so the rest of the package does not need it.
- **Kept**: 239 / 257 test problems, 112 / 120 train, 41 / 43 validation. All 18 excluded test problems have at least one test in another shape: `math.isclose(...)` on floats (rounding errors make `==` unreliable) or `set(...) == set(...)` for outputs whose order does not matter. Results on MBPP therefore refer to this subset, which under-represents floating-point problems. Supporting tolerances and sets in `TestCase` and the runner is a possible extension.
- Check (run once, not an automatic test, since it needs the download): the reference solution of every kept test problem passes all its parsed tests.
- **Why one public test**: the public test is shown in the prompt and guides the search; the hidden ones decide the final measure. One public / two hidden makes the search's filter weaker (a wrong program passes one test by chance more easily) but the final measure more reliable (a lucky program must also pass two unseen tests). Branch scores and `pick_by_branch` partly compensate for the weaker filter.

Later: HumanEval, then LeetCodeDataset / LiveCodeBench, each with its own adapter.

### 4.9 `CodeExecutor`: safe execution ✅
Two programs talking through stdin/stdout in JSON.

**`reasoning_graphs/execution/runner.py`** (child process):
- Input: `{"code", "function_name", "tests": [{"args", "expected"}]}`, where `args` is the **Python literal** of the argument list (`repr(list(args))`, read back with `ast.literal_eval`): JSON would turn tuples into lists, and a function that checks `isinstance(x, tuple)` would then fail on correct code. `expected` travels as JSON. Empty input → `{"results": []}`.
- `exec`s the code in its own namespace, with stdout redirected so model prints can't corrupt the JSON output. Any `Exception` or `SystemExit` during setup, or no callable named `function_name` → every test is `error` with that message.
- For each test: calls `func(*args)` (arguments rebuilt from their literal) with stdout redirected; an exception → `error` (`"TypeName: message"`). The result is normalized through JSON (`json.loads(json.dumps(result))`, so a tuple equals a list); not serializable → `failed`. Then `passed` if it equals `expected`, otherwise `failed`.
- Output: `{"results": [{"status", "actual", "error"}]}`, one per test, in order.

**`reasoning_graphs/execution/executor.py`**:
- `CodeExecutor(timeout=5.0).run(code, function_name, tests) -> ExecutionResult`. Launches `runner.py` with the same interpreter (`sys.executable`), in a fresh temporary working directory (files created by the code don't land in the project), all tests in **one** process.
- Timeout → every test `timeout`. Runner output that isn't valid JSON with `results` (e.g. `os._exit`, out of memory) → every test `error` with the exit code and the last 300 characters of stderr. Wrong number of results → every test `error`.
- `TestResult(test, status, actual, error)`, status ∈ `passed | failed | error | timeout`.
- `ExecutionResult.num_passed`; `.all_passed` (`False` when there are no tests).
- Security: a separate process protects against hangs and crashes, not against malicious file or network access; a real sandbox (container) is a possible later improvement.

### 4.9b Grading and choosing a solution ✅
File: `reasoning_graphs/execution/grading.py`. Shared by the demo and, later, by every system and baseline, so they all grade and choose in the same way.
- `grade_on_public(leaves, problem, executor) -> dict[int, ExecutionResult]`: runs each leaf's `code` on the problem's **public tests only** and returns the results keyed by `node.id`. It prints nothing: presentation is left to the caller.
- `pick_solution(leaves, public_results) -> ReasoningNode | None`: the **first** leaf, in the given order, that passes every public test; `None` if none does (or there are no leaves). Choosing by score instead of order will replace this rule once an evaluator exists.
- `branch_value(leaf) -> float`: the score of the **last scored node on the solution's path** — the most informed available estimate of its branch, since a score is a value that already looks ahead to the code. For a code node of the tree that is its parent (the last reasoning step); for a rollout solution, whose own path is made of unscored copies, it is its `origin` (the node the rollout started from, whose score is the pass rate of exactly the rollouts this solution is one of). `-1.0` if that node has no score (code written directly under the root, Best-of-N's rollouts from the root, or a tree grown without an evaluator): below any real score, so it never ties with a branch scored 0. The two kinds of anchor may sit at different depths (a rollout from depth 1 is judged by an estimate made two steps earlier than a depth-3 parent); whether that imbalance hurts the choice is an open question (§10).
- `pick_by_branch(leaves, public_results) -> ReasoningNode | None`: among the leaves passing every public test, the one with the highest `branch_value`; ties go to the first one in the given order; `None` if none passes. A failing leaf never wins, whatever its branch. Code written directly under the root (no evidence about its branch: it may pass the public tests by chance, like the adjacent-pairs solution) is not excluded but ranked last, so it is chosen only when nothing else passes. On a tree without scores every value is -1 and it behaves exactly like `pick_solution`.
- Why the parent's score and not a combination of the whole path: a sum or a product depends on the path's length (early code has fewer terms); a mean weighs equally the early, less informed estimates. The mean remains an alternative to test (§10).
- Measuring the chosen solution on the hidden tests stays with the caller (`executor.run(chosen.code, problem.function_name, problem.hidden_tests)`, only when `chosen` is not `None`).

### 4.10 Demo ✅
File: `scripts/demo.py` (run: `python -m scripts.demo`). End-to-end run with `MockGenerator` on Two Sum:
- Mock candidates include a valid hash-map solution, a valid brute-force solution, a **valid-but-wrong** solution (returns values instead of indices), a syntax error, an empty step and an early code step at the root. Seed 6 is chosen so these cases actually occur.
- Grows the full tree with `grow_full_tree` (§4.7b); prints it.
- Grades every code leaf on the **public tests** with `grade_on_public`, prints one line per leaf with `print_public_results` (verdict, first step of the branch, failed tests), picks a solution with `pick_solution`, and measures it on the **hidden tests**.
- Asserts: every leaf is terminal; no node deeper than `max_steps + 1`; reasoning steps numbered by depth; no syntax error accepted; the executor rejects exactly the wrong-but-valid code; the chosen code passes all hidden tests.

### 4.10b Real-model tree ✅
File: `scripts/hf_tree.py` (run: `HF_HUB_OFFLINE=1 python -m scripts.hf_tree`, about one minute). Same flow as the demo, with `HFGenerator` and no assumption about what the model produces:
1. `grow_full_tree` with the default `SearchConfig`, timed; prints the tree.
2. Statistics. Every non-terminal node was expanded exactly once, so: generator calls = number of non-terminal nodes; outputs requested = sum of `config.num_candidates(depth)` over them; outputs lost = requested − children created (invalid or duplicate); dead ends = leaves that are not terminal (every output of their expansion was discarded); nodes per depth.
3. `grade_on_public` (timed), `print_public_results`, number of code leaves passing the public tests.
4. `pick_solution`, then the chosen code on the hidden tests, or a message if no leaf passes.
5. Only the invariants that hold for any generator (depth ≤ `max_steps + 1`, reasoning steps numbered by depth). Whether a solution exists, and whether it passes the hidden tests, are results to report, not assertions.

**Measured (2026-10-16, Qwen2.5-Coder-0.5B, seed 0, M-series Mac)**:

| Measure | Value | Theoretical maximum |
|---|---|---|
| Nodes per depth | 1, 4, 8, 16, 18 | 1, 4, 8, 16, 32 |
| Generator calls | 23 (60 s, 2.6 s per call on average) | 29 |
| Outputs requested / became nodes / lost | 48 / 46 / 2 | — |
| Code leaves / dead ends | 24 / 0 | 32 / — |
| Code leaves passing the public tests | 11 / 24 (grading 0.4 s) | — |
| Chosen solution on the hidden tests | 3 / 5 | — |

Observations:
- Fewer calls and leaves than the maximum because 6 depth-3 nodes were already code (the model chose to write code after 2 steps), so they were not expanded further. Almost no output is lost: the format is followed reliably.
- Code calls are slower than step calls (a whole function vs one line), so the average is 2.6 s instead of ~1.5 s. Generation dominates the cost; grading is negligible.
- The chosen solution only checks **adjacent** pairs (`nums[i] + nums[i + 1]`): it passes both public tests, whose solutions happen to be adjacent, and fails 2 hidden tests. Two public tests are a weak filter, and choosing the first passing leaf takes no account of how promising its branch is. This is what node scores (evaluator) and better selection are for.
- Branch quality differs visibly: every leaf under "Start with the index 0" passes the public tests, every leaf under the two-pointer strategy fails (Two Sum asks for original indices, which sorting loses). This difference between branches is what a rollout score is meant to capture.

### 4.11 Tests ✅
Automatic tests live in `tests/` and run with `pytest` from the project root (`HF_HUB_OFFLINE=1 pytest` to avoid network calls). `pytest.ini` sets `pythonpath = .` (so tests can import the `reasoning_graphs` and `scripts` packages) and `testpaths = tests`. `tests/` mirrors the package layout: tests of `reasoning_graphs/<sub>/<module>.py` live in `tests/<sub>/`, tests of the top-level modules and of the demo in `tests/` itself; test file names are unique across folders. 145 tests, about 5 seconds. Modules contain only code: no test code in `if __name__ == "__main__":` blocks.

| Test file (under `tests/`) | What it checks |
|---|---|
| `test_reasoning_node.py` | `detached_copy`: same text and depth; a new chain linked both ways, sharing no node with the real tree, starting without children; adding children to the copy leaves the real tree unchanged; copy of the root |
| `test_config.py` | Defaults; custom values; each config gets its own list; `forces_code` / `num_candidates` at depths 0–3, and with distinct `branching` / `code_branching` values so a mix-up would show; empty branching, a 0 in branching and `code_branching = 0` raise `ValueError` |
| `test_step_format.py` | 17 parser cases, one test each (labels, numbering, truncation, multi-line steps, code validation, forced code); few-shot problem has the shape of `to_prompt()`; few-shot code is valid and correct on its examples; `split_prompt` and `build_messages` on root / intermediate / forced-code nodes (message count and roles, last message with trailing `\n`, problem in the 4th message) |
| `generators/test_mock.py` | Mock levels with a multi-line problem mentioning "steps", exhausted candidates → labelled code, forced code → unlabelled code, same seed → same outputs |
| `search/test_rollout.py` | With a scripted spy generator: reaches code with one output per call and code forced at the search's max depth; early code ends the rollout; an invalid output → `None`; a code node is returned as is without calling the generator (and keeps `origin = None`); the returned code node's `origin` is the real starting node while its parent is an unscored copy; the real tree is unchanged |
| `search/test_evaluator.py` | With a scripted generator and a spy executor: score is the fraction of passing rollouts (1, 0, 0.5, and 0.75 with a stuck rollout); only public tests are run; stuck rollouts are not executed; a code node is scored by its own tests with one execution and no generator call; the real tree is unchanged; `n` < 1 rejected; passing rollout solutions (not failing or stuck ones) are kept in `rollout_leaves` with their `origin`, accumulating across `score` calls; the tree's code nodes are not added |
| `search/test_beam.py` | With a scripted generator and a fixed-score evaluator: keeps exactly the best `beam_width` nodes at each level (only they are expanded; pruned nodes keep their score and get no children); every node except the root is scored once; code children don't take beam slots; stops when every child is code; ties keep creation order; a beam wider than the candidates keeps them all |
| `generators/test_counting_generator.py` | Counts calls and outputs and passes answers and requests through unchanged; counts everything an expansion and a rollout ask (1 call of 4 outputs + 3 calls of 1) |
| `search/test_baselines.py` | With a scripted generator: `n` independent solutions from the root, unscored, root left without children; stuck rollouts left out but their calls counted; `n = 1` is one full rollout down to forced code; without scores the choice is the first passing solution; `n` < 1 rejected; `best_within_budget` stops once the budget of new outputs is spent (overshooting rather than stopping short) and always makes at least one rollout |
| `search/test_expand.py` | With a spy generator: root expansion (cleaning, dedup, terminal code node), forced code at max depth (prompt ends with `Code:`, `code_branching` used), terminal node not expanded (generator not called), all-invalid outputs → no children |
| `problems/test_two_sum.py` | Test counts; every Two Sum test matches a reference solution; no hidden test appears in the prompt; no prompt line starts with `Step` |
| `execution/test_executor.py` | Correct, values instead of indices, partially right, exception, wrong function name, infinite loop (timeout within `timeout + 1` s), `os._exit`, no tests, all 7 Two Sum tests, verdicts on the demo's codes, tuple arguments (nested too) reach the function as tuples |
| `problems/test_mbpp.py` | With hand-made records (no download): `parse_assert` on simple shapes (strings; tuples, lists, dicts; no arguments; a set argument) and rejection of 11 other shapes (`math.isclose`, call on the left, non-literal or set expected value, keyword argument, `!=`, chained comparison, no comparison, not an assert, two statements, syntax error); `function_signature` (found among several functions, defaults and `*rest` kept, missing → `None`); `mbpp_to_problem` fields and the 1 public / rest hidden split; rejection for a test in another shape, tests calling different functions, function missing from the reference code, prompt line starting with a label |
| `execution/test_grading.py` | `grade_on_public` on a correct and a partially right leaf (keys are the leaves' ids, public tests only); `pick_solution` with hand-built results: first passing leaf when it isn't the first leaf, first of several passing, none passing, no leaves; `branch_value` (scored branch, 0 is a real score, -1 under the root); `pick_by_branch`: best branch, failing leaves never win, direct code loses even to a branch scored 0 but is kept when alone, ties, none passing, no leaves, same as `pick_solution` without scores; `branch_value` of a rollout solution comes from its origin (0.75, 0 and unscored origin → -1); a rollout solution from a better branch beats a tree leaf |
| `test_demo.py` | Runs `scripts.demo.main()`, whose own assertions check the whole demo |
| `generators/test_hf.py` | `StopAtNextStep` on a hand-built `input_ids` table (tokenizer only, no model): stops exactly the sequences whose new text started `Step`/`Code:`; a `"\nStep"` spanning the prompt boundary does not stop |

Conventions:
- Test doubles shared by several files live in `tests/helpers.py` (`ScriptedGenerator`: answers each call with the next entry of a script — one output or a list of outputs — and records the calls; `SpyExecutor`: a real `CodeExecutor` that records the tests it runs; `FixedScoreEvaluator`: scores from a table keyed by `step_text`, with a default, and records what it scored), imported with `from tests.helpers import ...`.
- One behavior per test function, named after what it checks.
- Tables of cases use `@pytest.mark.parametrize` with `pytest.param(..., id="...")`, so each case is reported separately by name.
- Expected errors use `with pytest.raises(...)`; shared setup uses fixtures (`executor` in `execution/test_executor.py`, a module-scoped `tokenizer` in `generators/test_hf.py`).
- Test modules don't bind `TestCase` / `TestResult` to their own names (pytest would try to collect them as test classes): import them under another name if needed (`TestResult as Result`).

**Manual check**: `generators/hf.py` keeps its `__main__` block (`HF_HUB_OFFLINE=1 python -m reasoning_graphs.generators.hf`). It loads the model and runs `generate_steps` on root / after Step 1 / forced code, printing raw and parsed outputs. Slow and stochastic, so it is not part of the automatic tests.

### 4.11b `rollout`: one branch down to code ✅
File: `reasoning_graphs/search/rollout.py`. `rollout(node, generator, config) -> ReasoningNode | None`.
- A terminal (code) node is returned as is: there is nothing to continue and the generator is not called.
- Otherwise it works on `node.detached_copy()`, so the real tree is never modified, and repeatedly calls `expand` with a **rollout config**: `SearchConfig(branching=[1] * config.max_steps, code_branching=1)`, i.e. one output per step and the same maximum depth as the search. The rollout therefore follows the same rules as the search it estimates: same step numbering, code forced at the same depth, same parsing.
- Each round takes the only child and continues from it. It returns the first code node reached, with `origin` set to the real node the rollout started from (its `code` is what gets tested; its `get_full_text()` is the whole rollout, useful as PRM data), or `None` if an expansion produced no valid child (the model got stuck).
- Always terminates: `expand` forces code at `max_steps`, and code nodes are terminal. From depth `d` it makes at most `max_steps − d + 1` generator calls, fewer if the model writes code early.
- Randomness comes from the generator's sampling: repeated rollouts from the same node follow different branches.

### 4.12 `Evaluator`: node evaluation ✅ (rollout version)
File: `reasoning_graphs/search/evaluator.py`.

**`Evaluator`** (abstract): `score(node) -> float`, the estimated probability, in [0, 1], that continuing from the node reaches code passing the tests (the node's *value*). Every evaluator has this one method, so switching between them is changing one line.

**`RolloutEvaluator(generator, config, executor, problem, n=4)`** (phase 1):
- Constructor arguments are the tools it needs, chosen by the caller: the generator that samples the rollouts (mock in tests, real model in experiments), the **same** `SearchConfig` as the search (so rollouts follow the search's rules and maximum depth), the `CodeExecutor` (and its timeout), the problem, and the number of rollouts `n` (≥ 1, otherwise `ValueError`).
- It receives the whole `Problem` and itself picks `problem.public_tests`, so hidden tests cannot be passed by mistake (§7, rule 1).
- `score(node)`:
  - code node → no future to sample: 1.0 if its own code passes every public test, else 0.0 (one execution, no generator call);
  - otherwise `n` independent rollouts from the node (§4.11b); a rollout is a success if it reaches code and that code passes **every** public test (all-or-nothing); score = successes / `n`. A stuck rollout (`None`) is a failure and stays in the denominator; it is not executed.
- The real tree is never modified (rollouts work on detached copies). `node.score` is not set: the caller decides where to store the score.
- **`rollout_leaves`**: the passing rollout solutions (each with `origin` = the scored node) are appended to this list instead of being discarded, so they can be candidate solutions too. Only passing ones are kept (a failing one could never be chosen), stuck rollouts and the tree's own code nodes never (those are already candidates as nodes of the tree). The list grows over every `score` call of the evaluator, so one evaluator is used per problem.
- Cost: up to `n × (max_steps − depth + 1)` generator calls and up to `n` executions per node.

**Real-model check** — `scripts/hf_scores.py` (run: `HF_HUB_OFFLINE=1 python -m scripts.hf_scores`, about 3 minutes): expands the root once (same seed and first call as `scripts/hf_tree.py`, so the same four first steps), then scores each first-level node with `RolloutEvaluator(n=8)` and stores it in `node.score`.

Measured (2026-10-20, Qwen2.5-Coder-0.5B, seed 0), next to the share of that branch's leaves passing the public tests in the full tree of §4.10b:

| Step 1 | Rollout score (n = 8) | Time | Leaves passing in the full tree |
|---|---|---|---|
| Start with the index 0 | 0.500 | 33 s | 7 / 7 |
| Find the complement of the target number | 0.750 | 31 s | 4 / 7 |
| Iterate through the list and check the complement | 0.125 | 39 s | 0 / 5 |
| Initialize two pointers `left` and `right` | 0.000 | 62 s | 0 / 5 |

Total 166 s for 32 rollouts (~5 s per rollout). Observations:
- The two weak branches are clearly separated from the two good ones, in the same order as in the full tree; the two-pointer branch, which cannot return original indices after sorting, scores 0.
- The two good branches swap places compared with the full tree. The two measures differ: a rollout samples single paths with repetitions, while the tree's leaves are deduplicated branches; and with n = 8 the typical error is about ±0.18, so 0.5 vs 0.75 is within noise. The complement branch is the hash-map idea, i.e. the algorithm that is correct on the hidden tests too, while many "index 0" leaves only checked adjacent pairs.
- The two-pointer branch took about twice as long: its code is more often wrong in ways that hang (loops that never end), and each hanging program costs the executor's full timeout (5 s). A shorter timeout would make rollouts cheaper.
- Scoring 4 nodes cost ~2.8 times the whole tree of §4.10b (166 s vs 60 s), as expected from the theory: evaluation, not expansion, dominates the cost.

**`LearnedPRM`** (phase 2) 🔲: the trained PRM (see §5), with the same interface.

### 4.13 Beam search ✅
File: `reasoning_graphs/search/beam.py`. `beam_search(root, generator, config, evaluator) -> list[ReasoningNode]`.

Same level-by-level loop as `grow_full_tree` (§4.7b), but the frontier is restricted to the best nodes. Each round has three phases:
1. **Expand** every frontier node with `expand`; all new children are added to the returned list.
2. **Score** every new child with `evaluator.score` and store it in `node.score`. Code children are candidate solutions: they are scored (for a `RolloutEvaluator`, by their own public tests) but never enter the frontier, so they never take a beam slot from a reasoning step.
3. **Select**: the non-terminal children sorted by score, highest first, cut to `config.beam_width`, become the next frontier. The sort is stable, so ties keep creation order (children of better-ranked parents first, then generation order).

- The root is not scored (it is the only node of its level and is always expanded): `root.score` stays `None`. Every other node is scored exactly once.
- Stops when a round leaves no non-terminal child (all code, or nothing valid). Always terminates, like `grow_full_tree`.
- Pruned nodes stay in the tree with their score and no children: the returned list records what was discarded. Since nothing grows under them, the code nodes of the returned list are exactly the candidate solutions.
- Returns all the nodes, root first, like `grow_full_tree`, so the two are interchangeable in scripts. Choosing the final solution is a separate step.

**Cost** (beam width B, `n` rollouts, no early code and no duplicates): every level expands at most B nodes (1 call each) and scores at most B × branching children, each with up to `n × (max_steps − depth + 1)` calls. With the defaults (`[4, 2, 2]`, code branching 2, B = 2) and n = 4: 7 expansions + 96 rollout calls ≈ 103 calls, examining 4 code leaves, against 29 calls and 32 leaves for the unscored full tree. On a tree this small exhaustive expansion is cheaper; the beam's cost grows linearly with depth while the full tree grows exponentially, and the scores buy selection quality rather than savings. With a learned PRM (one forward pass per score) most of the cost disappears. Systems must therefore be compared at equal compute (§6).

Possible extension: MCTS (reuses rollout nodes instead of discarding them).

### 4.13b Baselines ✅
File: `reasoning_graphs/search/baselines.py`. `best_of_n(root, generator, config, n) -> list[ReasoningNode]`.
- `n` independent complete solutions: `n` rollouts from the root (§4.11b), with the same `config` as the search, so the same step format, maximum depth and forced code. No evaluator and no scores: every returned node has `score = None`.
- Returns the code nodes reached; stuck rollouts (`None`) are left out, so the list holds only code nodes and can go straight to `grade_on_public`. Their number is `n − len(result)`; they still cost generator calls.
- The nodes live on detached copies: the root itself gets no children.
- **Single generation** is `best_of_n(..., n=1)`: one chain of reasoning steps down to code, no search. **Best-of-N** is `n > 1`, followed by choosing a solution that passes the public test.
- `n` < 1 → `ValueError`.
- `best_within_budget(root, generator, config, budget)`: Best-of-N at equal compute. `generator` must be a `CountingGenerator`; rollouts from the root continue until `budget` **more** outputs have been requested (counted from the counter's value at the start), so N adapts to what the other system spent on that problem. The last rollout may overshoot the budget by a few outputs; at least one rollout is always made; stuck rollouts are left out.
- Choosing the solution: the same `pick_by_branch` as for every system. Without scores every branch value is −1, so it picks the first solution that passes the public tests, exactly like `pick_solution`.
- Role in the evaluation (§6): Best-of-N is the strong baseline. It filters complete solutions only at the end, with the tests, while the beam search decides at every step which branches to continue; the comparison, at equal numbers of generator calls, measures whether those step-level decisions are worth the rollouts' cost.

### 4.14 Repair step 🔲 (added value)
When a terminal node's code fails the public tests, the outcome (input, expected, actual, error) is added to the context and the generator produces a new step analysing the error + fixed code, as a **child of the failed node**.

### 4.15 `VLLMGenerator` 🔲 (scalability)
Same model and same prompt builder as `HFGenerator`, served by vLLM on a cloud GPU (prefix caching for the tree's shared prefixes).

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

- **Main metric**: percentage of problems solved on the **hidden tests** (pass@1): the system returns one solution, chosen with the public tests only, and it counts as solved if it passes every hidden test.
- **Cost**: number of **outputs** requested from the generator (`CountingGenerator.outputs`, §4.6b); calls and seconds are recorded too.
- **Baselines**, compared **at equal compute**:
  1. single generation (`best_of_n(n=1)`);
  2. **Best-of-N** with the same budget as the beam search on the same problem (`best_within_budget`). A strong baseline for code.
- Systems to compare: search with rollouts, search with the PRM, with and without the repair step.

### 6.1 Experiment runner ✅
File: `scripts/run_experiment.py`. Run: `HF_HUB_OFFLINE=1 python -m scripts.run_experiment`; options `--problems` (default 20: the first MBPP test problems), `--rollouts` (default 4, rollouts per node for the beam search), `--output` (default `results/mbpp_pilot.jsonl`). Quick check: `--problems 1 --rollouts 1 --output results/smoke.jsonl` (about 80 s).
- Loads the model once (`HFGenerator`, seed 0) and the problems with `load_mbpp("test")`; `SearchConfig()` defaults (`[4, 2, 2]`, code branching 2, beam width 2); `CodeExecutor(timeout=2.0)` (MBPP solutions run in milliseconds, so hanging code should not cost 5 s per execution).
- Systems (option `--systems`, default all four): `beam`, `beam_rollouts`, `best_of_n`, `single`. For each problem:
  1. if `beam` or `beam_rollouts` is selected, **one** `beam_search` runs, with a fresh `CountingGenerator` around the shared model and a `RolloutEvaluator` built on **the same counter** (so rollouts are counted). Its result gives two rows with the same cost: `beam` chooses among the tree's code nodes; `beam_rollouts` among the tree's code nodes plus `evaluator.rollout_leaves`. The comparison between the two is therefore exact: same tree, same rollouts, only the set of candidates differs;
  2. `best_of_n`: `best_within_budget` with budget = the outputs the beam search just spent on this problem (so `best_of_n` requires a beam system; otherwise the script stops at once with an error);
  3. `single`: `best_of_n(n=1)`.
  Each of 2 and 3 has its own fresh counter and root.
- `evaluate(leaves, problem, executor)`, identical for every system: `grade_on_public` → `pick_by_branch` → the chosen code on the hidden tests. `passed_public` = some candidate passes the public test; `solved` = the chosen one passes every hidden test (`False` when nothing is chosen); `chosen_from` = `"rollout"` if the chosen solution has an `origin` (for `beam_rollouts` a rollout solution; Best-of-N's are all rollouts from the root), `"tree"` otherwise, `null` when nothing is chosen.
- One JSON line per (problem, system): `problem`, `system`, `passed_public`, `solved`, `chosen_from`, `candidates` (candidate solutions), `outputs`, `calls`, `seconds`. (The rows of §6.2 predate `chosen_from` and `beam_rollouts`.) Lines are appended after each problem, so an interruption loses at most the current problem; problems already present in the file are skipped, so the same command resumes a run. (The model's random state then differs from an uninterrupted run.) At the end, a summary per system: solved, passed public, mean candidates, mean outputs, mean seconds — over every row of the file.
- `results/` is in `.gitignore`.

Smoke run (2026-10-23, `--problems 1 --rollouts 1`, mbpp_11): beam 36 outputs, 38 s, no candidate passing the public test; Best-of-N with the same budget 37 outputs, 11 candidates, solved; single generation 4 outputs, not solved. A single problem with one rollout per node says nothing about the comparison; it only shows that the pipeline runs end to end.

### 6.2 Results: MBPP test split ✅
Run of 2026-10-23/25: `python -m scripts.run_experiment --problems 239` (then at the project root as `run_experiment.py`) (all 239 convertible MBPP test problems), defaults (`[4, 2, 2]`, code branching 2, beam width 2, 4 rollouts per node), Qwen2.5-Coder-0.5B-Instruct, seed 0, M-series Mac; results in `results/mbpp_pilot.jsonl` (the first 13 problems come from an earlier run of the same command, resumed). Total ≈ 13.5 h (beam 7.6 h, Best-of-N 5.7 h, single 0.2 h).

| System | Solved (hidden tests) | A candidate passes the public test | Passes public, fails hidden | Mean outputs | Mean candidates | Mean seconds |
|---|---|---|---|---|---|---|
| Single generation | 87 / 239 (36.4%) | 100 | 13 | 3.4 | 1.0 | 3.5 |
| Beam search (rollouts) | 146 / 239 (61.1%) | 178 | 32 | 82.0 | 4.2 | 113.9 |
| Best-of-N (same budget) | **158 / 239 (66.1%)** | 199 | 41 | 83.4 | 23.6 | 85.5 |

Paired comparison (same problems), with a two-sided exact sign test on the problems where the two systems disagree:

| Pair | Both solve | Neither | Only the first | Only the second | p |
|---|---|---|---|---|---|
| beam vs Best-of-N | 134 | 69 | 12 | 24 | 0.065 |
| beam vs single | 81 | 87 | 65 | 6 | < 0.0001 |
| Best-of-N vs single | 81 | 75 | 77 | 6 | < 0.0001 |

Findings:
- **Searching pays off**: both the beam search and Best-of-N solve far more problems than single generation (+25 and +30 points), clearly beyond chance.
- **Best-of-N beats the beam search at equal compute**: 158 vs 146, 24 problems won against 12. The difference is suggestive but not significant at the usual 5% level (p ≈ 0.065) with one run of 239 problems.
- The budget matching works: Best-of-N spent on average 1.3 outputs more than the beam (at most 3).
- **Main cause: the beam keeps almost none of the code it writes.** With the same budget, Best-of-N ends with 23.6 candidate solutions per problem, the beam with 4.2: about 90% of the beam's outputs go into rollouts, whose complete solutions are executed only to compute scores and then discarded. Of the 24 problems only Best-of-N solves, in 16 the beam has no candidate passing even the public test.
- **One public test is a weak filter**: 13–21% of the solutions passing the public test fail the hidden ones (single 13/100, beam 32/178, Best-of-N 41/199). The beam's scores rely on the same public test, so they are fooled in the same way.
- Limits: one seed, one run, a 0.5B model, the 239-problem subset of MBPP (floating-point problems excluded, §4.8).

Next improvement suggested by these results: keep the code nodes produced by the rollouts as candidate solutions of the beam search (they come from the most promising branches and are already tested), which changes the beam's candidates but not its cost, so the Best-of-N and single-generation rows stay valid and only the beam needs to be re-run.

### 6.3 Results: beam search keeping rollout solutions ✅
Run of 2026-10-27/28: `python -m scripts.run_experiment --problems 239 --systems beam beam_rollouts --output results/mbpp_beam_rollouts.jsonl` (same settings as §6.2). One new beam search per problem gives both `beam` (tree leaves only) and `beam_rollouts` (tree leaves + passing rollout solutions). Best-of-N and single generation are the rows of §6.2: their budget matches the new beam search on average (Best-of-N spent 1.6 outputs more per problem on average, 13 in absolute value per problem, since the new search is a new random draw), not problem by problem.

| System | Solved (hidden tests) | Passes public | Passes public, fails hidden | Mean candidates | Mean outputs |
|---|---|---|---|---|---|
| Single generation (§6.2) | 87 / 239 (36.4%) | 100 | 13 | 1.0 | 3.4 |
| Beam search, first run (§6.2) | 146 / 239 (61.1%) | 178 | 32 | 4.2 | 82.0 |
| Beam search, new run | 149 / 239 (62.3%) | 174 | 25 | 4.2 | 81.7 |
| **Beam search + rollout solutions** | **162 / 239 (67.8%)** | 204 | 42 | 21.6 | 81.7 |
| Best-of-N, same budget (§6.2) | 158 / 239 (66.1%) | 199 | 41 | 23.6 | 83.4 |

| Pair | Both | Neither | Only the first | Only the second | p (sign test) |
|---|---|---|---|---|---|
| beam (new) vs beam + rollouts — same search | 144 | 72 | 5 | 18 | 0.011 |
| beam + rollouts vs Best-of-N | 145 | 64 | 17 | 13 | 0.58 |
| beam, first run vs new run | 130 | 74 | 16 | 19 | 0.74 |

Findings:
- **Keeping the rollout solutions clearly helps**: on the very same searches, 18 problems gained against 5 lost (p ≈ 0.01), at no extra generation cost. In 30 problems the tree had no candidate passing the public test and a rollout solution did.
- **The beam search now matches Best-of-N** at equal compute: 162 vs 158 (17 vs 13 problems won). The difference is well within noise (p ≈ 0.58): the two are on par, no longer behind.
- **Run-to-run noise is large**: two runs of the same beam search differ on 35 problems (146 vs 149 solved). Differences of a few problems between systems are not meaningful on 239 problems.
- **Where the chosen solution comes from** (`chosen_from`, beam + rollouts): a tree leaf in 152 problems (132 solved, 87%), a rollout solution in 52 (30 solved, 58%), nothing in 35. Chosen rollout solutions fail the hidden tests far more often. In the 5 problems the change lost, a rollout solution was preferred to a tree leaf that was correct: consistent with the imbalance noted in §4.9b, rollout solutions being ranked by a shallower, less informed origin score than the tree leaves' parents.
- The beam search is slower in wall-clock time than Best-of-N for the same outputs (112 s vs 86 s per problem): many small calls (rollouts) instead of fewer calls of complete solutions.

Possible next refinements: rank rollout solutions below tree leaves at equal branch value, or use an anchor at the same depth for both; more rollouts per node (less noisy scores); several seeds for a reliable comparison.

---

## 7. Methodological rules (anti-contamination)

1. The **search uses only the public tests**. Hidden tests are only for the final measurement.
2. **PRM training problems** are separate from evaluation problems.
3. The PRM train/validation split is **by problem**.
4. Systems are compared **at equal compute**.
5. Fixed seeds wherever possible, for reproducibility.
6. The few-shot example must not come from, or resemble, an evaluation problem.

---

## 8. Tech stack

- **Python** 3.12 (venv in `.venv`)
- **`torch`** 2.14, **`transformers`** 5.19: local models. In this version `from_pretrained` takes `dtype=` (not the older `torch_dtype=`).
- **`datasets`** 5.1: loading MBPP. Dataset ids must be `namespace/name` (`google-research-datasets/mbpp`, not `mbpp`).
- **`peft`**: LoRA for PRM training (not used yet)
- **vLLM**: fast generation on Linux/cloud GPUs (not available on the Mac)
- **`subprocess`** (standard library): safe code execution
- **`pytest`** 9: automatic tests
- **Models**: `Qwen2.5-Coder-0.5B-Instruct` for development; `1.5B-Instruct` for experiments; also the PRM base.
- Running offline once the model is cached: `HF_HUB_OFFLINE=1 pytest`, `HF_HUB_OFFLINE=1 python -m reasoning_graphs.generators.hf`.

---

## 9. Roadmap

| # | Step | Status |
|---|---|---|
| 0 | `ReasoningNode`, `StepGenerator` + `MockGenerator`, `expand` | ✅ done |
| 1 | Step format and parser, `SearchConfig`, `Problem`/`TestCase` with Two Sum, demo | ✅ done |
| 2 | `CodeExecutor` (separate process + timeout) | ✅ done |
| 3 | `HFGenerator`: prompt builder (few-shot + prefill), `generate_steps`, `StopAtNextStep` | ✅ done |
| 3a | Move self-checks into `tests/` with `pytest`, add a `StopAtNextStep` test | ✅ done |
| 3a+ | Reorganize the modules into the `reasoning_graphs` package | ✅ done |
| 3b | First real tree on Two Sum: `grow_full_tree`, `grading.py`, `config.num_candidates`, `hf_tree.py` with measurements (§4.10b) | ✅ done |
| 4 | `Evaluator` + `RolloutEvaluator`, `rollout`, `ReasoningNode.detached_copy`, real-model check `hf_scores.py` | ✅ done |
| 5 | Beam search (`search/beam.py`, `beam_width` in `SearchConfig`), branch-based solution choice (`pick_by_branch`) | ✅ done (real-model run deferred to the step-6 comparison) |
| 6 | Baselines (single, Best-of-N) and comparison on MBPP (all 239 convertible test problems, §6.2) | ✅ done |
| 6+ | Beam search keeping rollout solutions as candidates (`origin`, `rollout_leaves`, `branch_value` via origin, `beam_rollouts` system) | ✅ done (§6.3) |
| 7 | Repair step from errors | 🔲 added value |
| 8 | Data collection + PRM training (`LearnedPRM`) | 🔲 advanced |
| 9 | `VLLMGenerator` on a cloud GPU, larger datasets | 🔲 scalability |

Steps 1–6 make up the complete project; 7–8 are the added value.

---

## 10. Open decisions

- Prompt tuning: the current system prompt and few-shot example work on Two Sum; to be re-checked on a dataset (rate of valid outputs, multi-line steps). Stopping at the first `\n` instead of the next label is an option if multi-line steps become a problem.
- Sampling defaults (`temperature`, `top_p`) and `max_new_tokens` for longer solutions.
- Choosing among passing leaves: last scored node on the path (current) vs mean of the path's scores; and whether rollout solutions, judged by a shallower origin than tree leaves' parents, are ranked fairly against them.
- Default values for the per-level branching and maximum depth (to be tuned by experiments; starting point `[4, 2, 2]`).
- Number of rollouts per node (cost vs noise); all-or-nothing vs partial credit on tests.
- Number of public tests per problem: 1 for MBPP (decided); revisit if the search's filter proves too weak.
- Supporting float tolerances and sets, to bring back the 18 excluded MBPP test problems.
