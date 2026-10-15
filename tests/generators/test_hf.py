import pytest
import torch
from transformers import AutoTokenizer

from reasoning_graphs.generators.hf import StopAtNextStep

# Only the tokenizer is loaded (fast, from the local cache): the model is not needed here
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
PROMPT_IDS = [151644, 77091, 198]  # "<|im_start|>assistant\n": the prompt ends with a newline


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_NAME)


def make_input_ids(tokenizer, new_texts: list[str]) -> torch.Tensor:
    """Fake generate state: the same prompt followed by each new text, padded to equal length."""
    rows = [tokenizer(text)["input_ids"] for text in new_texts]
    length = max(len(row) for row in rows)
    rows = [row + [tokenizer.pad_token_id] * (length - len(row)) for row in rows]
    return torch.tensor([PROMPT_IDS + row for row in rows])


def test_stops_only_the_sequences_that_started_another_step(tokenizer):
    input_ids = make_input_ids(tokenizer, [
        "Step 2: Scan the list.\nStep",     # started Step 3 -> stop
        "Step 2: Use a map.\nCode:",        # started the code -> stop
        "Step 2: For each element in the",  # still writing its step -> continue
    ])
    criterion = StopAtNextStep(tokenizer, len(PROMPT_IDS))
    assert criterion(input_ids, None).tolist() == [True, True, False]


def test_newline_at_the_end_of_the_prompt_does_not_stop(tokenizer):
    # The prompt ends with "\n" and the model has just written "Step": "\nStep" spans the
    # prompt boundary, which is exactly what transformers' stop_strings get wrong
    input_ids = make_input_ids(tokenizer, ["Step"])
    criterion = StopAtNextStep(tokenizer, len(PROMPT_IDS))
    assert criterion(input_ids, None).tolist() == [False]
