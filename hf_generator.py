import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from step_generator import StepGenerator
from step_format import build_messages


class HFGenerator(StepGenerator):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct", temperature: float = 0.8,
                 top_p: float = 0.95, max_new_tokens: int = 128, seed: int = 0):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"  # Apple GPU if available
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(self.device)
        self.model.eval()  # inference mode, no training

    def generate_steps(self, prompt: str, n: int) -> list[str]:
        raise NotImplementedError


if __name__ == "__main__":
    from problems import TWO_SUM

    generator = HFGenerator()
    print("device:", generator.device)
    print("parameters:", generator.model.num_parameters())

    # 1. Messages from the node text: rules, few-shot example, problem, steps so far
    full_text = TWO_SUM.to_prompt() + "\nStep 1: Use a hash map from value to index."
    messages = build_messages(full_text)

    # 2. Chat template: root -> open a new assistant turn, otherwise continue the prefilled steps
    if messages[-1]["role"] == "user":
        text = generator.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = generator.tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)
    print(repr(text[-150:]))

    # 3. Text -> token ids
    inputs = generator.tokenizer(text, return_tensors="pt").to(generator.device)
    #print("input shape:", inputs["input_ids"].shape)

    # 4. Sample two answers, one token at a time
    set_seed(generator.seed)
    with torch.no_grad():
        outputs = generator.model.generate(
            **inputs,
            do_sample=True,
            temperature=generator.temperature,
            top_p=generator.top_p,
            max_new_tokens=generator.max_new_tokens,
            num_return_sequences=2,
        )
    #print("output shape:", outputs.shape)

    # 5. Each row is prompt + answer: keep the answer and decode it
    prompt_length = inputs["input_ids"].shape[1]
    for sequence in outputs:
        new_tokens = sequence[prompt_length:]
        answer = generator.tokenizer.decode(new_tokens, skip_special_tokens=True)
        print("-----")
        print(answer)
