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
  Problem  ───▶ │   Search         │ ◀──── Evaluator (rollout / PRM)        🔲
  + public      │ (beam search) 🔲 │              │
  tests         └──────┬───────────┘              ▼
                       │ expand ✅          CodeExecutor (tests) ✅
                       ▼
                 StepGenerator: MockGenerator ✅ / HFGenerator ✅ / VLLMGenerator 🔲
                       │
                       ▼
                 ReasoningNode (tree) ✅
```

### 3.2 Files

The project code is the `reasoning_graphs` package, split into subpackages by role. Every folder has an (empty) `__init__.py`. The demo, the tests and the configuration stay at the project root.

```
Self-Correcting-Reasoning-Graphs/
├── reasoning_graphs/
│   ├── reasoning_node.py
│   ├── config.py
│   ├── step_format.py
│   ├── generators/    base.py, hf.py
│   ├── search/        expand.py, full_tree.py, rollout.py, evaluator.py
│   ├── execution/     executor.py, runner.py, grading.py
│   └── problems/      problem.py, two_sum.py
├── tests/             same layout as the package (§4.11)
├── main.py
├── hf_tree.py
├── hf_scores.py
├── pytest.ini
├── requirements.txt
└── SPECIFICATIONS.md
```

| Module (under `reasoning_graphs/`) | Contents | Depends on |
|---|---|---|
| `reasoning_node.py` | `ReasoningNode`: the tree | — |
| `config.py` | `SearchConfig`: branching per level, maximum depth | — |
| `step_format.py` | Step labels, parser, formatter, prompt builder (system prompt, few-shot example, `split_prompt`, `build_messages`) | — |
| `generators/base.py` | `StepGenerator` interface, `MockGenerator` | `step_format` |
| `generators/hf.py` | `HFGenerator` (real model via `transformers`), `StopAtNextStep` | `generators.base`, `step_format`, `torch`, `transformers` |
| `search/expand.py` | `expand` | `reasoning_node`, `generators.base`, `config`, `step_format` |
| `search/full_tree.py` | `grow_full_tree`: expands every node | `search.expand` |
| `search/rollout.py` | `rollout`: one branch from a node down to code | `search.expand`, `config`, `reasoning_node` |
| `search/evaluator.py` | `Evaluator` interface, `RolloutEvaluator` | `search.rollout`, `execution.executor`, `problems.problem` |
| `execution/runner.py` | Child process that executes model code on the tests | standard library only |
| `execution/executor.py` | `CodeExecutor`, `TestResult`, `ExecutionResult` | `problems.problem`, `runner.py` (as a file in the same folder) |
| `execution/grading.py` | `grade_on_public`, `pick_solution`: grade code leaves and choose the solution | `execution.executor`, `problems.problem`, `reasoning_node` |
| `problems/problem.py` | `TestCase`, `Problem` | — |
| `problems/two_sum.py` | `TWO_SUM` | `problems.problem` |

| At the project root | Contents |
|---|---|
| `main.py` | End-to-end demo with the mock generator (uses everything except `generators.hf`); also defines `print_public_results` |
| `hf_tree.py` | Full tree on Two Sum with the real model, with measurements (§4.10b) |
| `hf_scores.py` | Rollout scores of the first-level branches of Two Sum with the real model (§4.12) |
| `tests/` | `pytest` tests (§4.11) |
| `pytest.ini` | `pytest` configuration |

**Imports** are absolute from the package root: `from reasoning_graphs.step_format import parse_step`, `from reasoning_graphs.generators.base import StepGenerator`. They work when Python runs from the project root: `python main.py`, `pytest`, or a module run with `-m` (`python -m reasoning_graphs.generators.hf`). Running a package file by its path (`python reasoning_graphs/generators/hf.py`) fails with `ModuleNotFoundError`, because Python then only sees that file's folder.

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

**`extract_code(text) -> str | None`**: finds the first ```` ```python ... ``` ```` block (non-greedy, across lines), strips it, and returns it only if `ast.parse` accepts it (syntax check, nothing is executed). No block, unclosed block or syntax error → `None`.

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
- `max_steps` (property) = `len(branching)`.
- `forces_code(depth) -> bool`: `depth >= max_steps`, i.e. a node at this depth may only be followed by code.
- `num_candidates(depth) -> int`: how many outputs to request when expanding a node at this depth: `code_branching` if `forces_code(depth)`, else `branching[depth]`. With the defaults: 4 at the root, then 2, 2, and 2 (code) at depth 3.
- These two methods are the single place where the rule lives: `expand` uses them to decide what to ask the generator, and analysis code uses them to know, after the fact, how many outputs a node had requested (requested − `len(node.children)` = outputs lost to parsing or deduplication).
- Validation in `__post_init__`: `branching` not empty, every value ≥ 1, `code_branching` ≥ 1, otherwise `ValueError`. Each instance gets its own list (`default_factory`).

Beam width, number of rollouts and temperature will be added here.

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

Later: one adapter per dataset converts its format into `Problem`/`TestCase` with an automatic public/hidden split (`TestCase` may need a raw-`assert` kind for some datasets); datasets from Hugging Face (`datasets` library): MBPP, HumanEval, then LeetCodeDataset / LiveCodeBench.

### 4.9 `CodeExecutor`: safe execution ✅
Two programs talking through stdin/stdout in JSON.

**`reasoning_graphs/execution/runner.py`** (child process):
- Input: `{"code", "function_name", "tests": [{"args", "expected"}]}`. Empty input → `{"results": []}`.
- `exec`s the code in its own namespace, with stdout redirected so model prints can't corrupt the JSON output. Any `Exception` or `SystemExit` during setup, or no callable named `function_name` → every test is `error` with that message.
- For each test: calls `func(*args)` with stdout redirected; an exception → `error` (`"TypeName: message"`). The result is normalized through JSON (`json.loads(json.dumps(result))`, so a tuple equals a list); not serializable → `failed`. Then `passed` if it equals `expected`, otherwise `failed`.
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
- Measuring the chosen solution on the hidden tests stays with the caller (`executor.run(chosen.code, problem.function_name, problem.hidden_tests)`, only when `chosen` is not `None`).

### 4.10 Demo ✅
File: `main.py`. End-to-end run with `MockGenerator` on Two Sum:
- Mock candidates include a valid hash-map solution, a valid brute-force solution, a **valid-but-wrong** solution (returns values instead of indices), a syntax error, an empty step and an early code step at the root. Seed 6 is chosen so these cases actually occur.
- Grows the full tree with `grow_full_tree` (§4.7b); prints it.
- Grades every code leaf on the **public tests** with `grade_on_public`, prints one line per leaf with `print_public_results` (verdict, first step of the branch, failed tests), picks a solution with `pick_solution`, and measures it on the **hidden tests**.
- Asserts: every leaf is terminal; no node deeper than `max_steps + 1`; reasoning steps numbered by depth; no syntax error accepted; the executor rejects exactly the wrong-but-valid code; the chosen code passes all hidden tests.

### 4.10b Real-model tree ✅
File: `hf_tree.py` (run: `HF_HUB_OFFLINE=1 python hf_tree.py`, about one minute). Same flow as the demo, with `HFGenerator` and no assumption about what the model produces:
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
Automatic tests live in `tests/` and run with `pytest` from the project root (`HF_HUB_OFFLINE=1 pytest` to avoid network calls). `pytest.ini` sets `pythonpath = .` (so tests can import the `reasoning_graphs` package and `main`) and `testpaths = tests`. `tests/` mirrors the package layout: tests of `reasoning_graphs/<sub>/<module>.py` live in `tests/<sub>/`, tests of the top-level modules and of the demo in `tests/` itself; test file names are unique across folders. 88 tests, about 5 seconds. Modules contain only code: no test code in `if __name__ == "__main__":` blocks.

| Test file (under `tests/`) | What it checks |
|---|---|
| `test_reasoning_node.py` | `detached_copy`: same text and depth; a new chain linked both ways, sharing no node with the real tree, starting without children; adding children to the copy leaves the real tree unchanged; copy of the root |
| `test_config.py` | Defaults; custom values; each config gets its own list; `forces_code` / `num_candidates` at depths 0–3, and with distinct `branching` / `code_branching` values so a mix-up would show; empty branching, a 0 in branching and `code_branching = 0` raise `ValueError` |
| `test_step_format.py` | 17 parser cases, one test each (labels, numbering, truncation, multi-line steps, code validation, forced code); few-shot problem has the shape of `to_prompt()`; few-shot code is valid and correct on its examples; `split_prompt` and `build_messages` on root / intermediate / forced-code nodes (message count and roles, last message with trailing `\n`, problem in the 4th message) |
| `generators/test_mock.py` | Mock levels with a multi-line problem mentioning "steps", exhausted candidates → labelled code, forced code → unlabelled code, same seed → same outputs |
| `search/test_rollout.py` | With a scripted spy generator: reaches code with one output per call and code forced at the search's max depth; early code ends the rollout; an invalid output → `None`; a code node is returned as is without calling the generator; the real tree is unchanged |
| `search/test_evaluator.py` | With a scripted generator and a spy executor: score is the fraction of passing rollouts (1, 0, 0.5, and 0.75 with a stuck rollout); only public tests are run; stuck rollouts are not executed; a code node is scored by its own tests with one execution and no generator call; the real tree is unchanged; `n` < 1 rejected |
| `search/test_expand.py` | With a spy generator: root expansion (cleaning, dedup, terminal code node), forced code at max depth (prompt ends with `Code:`, `code_branching` used), terminal node not expanded (generator not called), all-invalid outputs → no children |
| `problems/test_two_sum.py` | Test counts; every Two Sum test matches a reference solution; no hidden test appears in the prompt; no prompt line starts with `Step` |
| `execution/test_executor.py` | Correct, values instead of indices, partially right, exception, wrong function name, infinite loop (timeout within `timeout + 1` s), `os._exit`, no tests, all 7 Two Sum tests, verdicts on the demo's codes |
| `execution/test_grading.py` | `grade_on_public` on a correct and a partially right leaf (keys are the leaves' ids, public tests only); `pick_solution` with hand-built results: first passing leaf when it isn't the first leaf, first of several passing, none passing, no leaves |
| `test_demo.py` | Runs `main.main()`, whose own assertions check the whole demo |
| `generators/test_hf.py` | `StopAtNextStep` on a hand-built `input_ids` table (tokenizer only, no model): stops exactly the sequences whose new text started `Step`/`Code:`; a `"\nStep"` spanning the prompt boundary does not stop |

Conventions:
- Test doubles shared by several files live in `tests/helpers.py` (`ScriptedGenerator`: answers each call with the next output of a script and records the calls; `SpyExecutor`: a real `CodeExecutor` that records the tests it runs), imported with `from tests.helpers import ...`.
- One behavior per test function, named after what it checks.
- Tables of cases use `@pytest.mark.parametrize` with `pytest.param(..., id="...")`, so each case is reported separately by name.
- Expected errors use `with pytest.raises(...)`; shared setup uses fixtures (`executor` in `execution/test_executor.py`, a module-scoped `tokenizer` in `generators/test_hf.py`).
- Test modules don't bind `TestCase` / `TestResult` to their own names (pytest would try to collect them as test classes): import them under another name if needed (`TestResult as Result`).

**Manual check**: `generators/hf.py` keeps its `__main__` block (`HF_HUB_OFFLINE=1 python -m reasoning_graphs.generators.hf`). It loads the model and runs `generate_steps` on root / after Step 1 / forced code, printing raw and parsed outputs. Slow and stochastic, so it is not part of the automatic tests.

### 4.11b `rollout`: one branch down to code ✅
File: `reasoning_graphs/search/rollout.py`. `rollout(node, generator, config) -> ReasoningNode | None`.
- A terminal (code) node is returned as is: there is nothing to continue and the generator is not called.
- Otherwise it works on `node.detached_copy()`, so the real tree is never modified, and repeatedly calls `expand` with a **rollout config**: `SearchConfig(branching=[1] * config.max_steps, code_branching=1)`, i.e. one output per step and the same maximum depth as the search. The rollout therefore follows the same rules as the search it estimates: same step numbering, code forced at the same depth, same parsing.
- Each round takes the only child and continues from it. It returns the first code node reached (its `code` is what gets tested; its `get_full_text()` is the whole rollout, useful as PRM data), or `None` if an expansion produced no valid child (the model got stuck).
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
- Cost: up to `n × (max_steps − depth + 1)` generator calls and up to `n` executions per node.

**Real-model check** — `hf_scores.py` (run: `HF_HUB_OFFLINE=1 python hf_scores.py`, about 3 minutes): expands the root once (same seed and first call as `hf_tree.py`, so the same four first steps), then scores each first-level node with `RolloutEvaluator(n=8)` and stores it in `node.score`.

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

### 4.13 Search 🔲
- **Beam search**: like `grow_full_tree`, but at each level only the `B` best-scoring nodes are kept.
- Output: the code of the best terminal node according to the public tests and the score.
- Possible extension: MCTS.

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
6. The few-shot example must not come from, or resemble, an evaluation problem.

---

## 8. Tech stack

- **Python** 3.12 (venv in `.venv`)
- **`torch`** 2.14, **`transformers`** 5.19: local models. In this version `from_pretrained` takes `dtype=` (not the older `torch_dtype=`).
- **`datasets`**: loading problem datasets (not used yet)
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
| 5 | Beam search | 🔲 next |
| 6 | Baselines (single, Best-of-N) and comparison on MBPP / HumanEval | 🔲 |
| 7 | Repair step from errors | 🔲 added value |
| 8 | Data collection + PRM training (`LearnedPRM`) | 🔲 advanced |
| 9 | `VLLMGenerator` on a cloud GPU, larger datasets | 🔲 scalability |

Steps 1–6 make up the complete project; 7–8 are the added value.

---

## 10. Open decisions

- Prompt tuning: the current system prompt and few-shot example work on Two Sum; to be re-checked on a dataset (rate of valid outputs, multi-line steps). Stopping at the first `\n` instead of the next label is an option if multi-line steps become a problem.
- Sampling defaults (`temperature`, `top_p`) and `max_new_tokens` for longer solutions.
- Default values for the per-level branching and maximum depth (to be tuned by experiments; starting point `[4, 2, 2]`).
- Number of rollouts per node (cost vs noise); all-or-nothing vs partial credit on tests.
- Number of public tests per problem.
- Starting dataset (MBPP or HumanEval).
