import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from step_generator import StepGenerator
from step_format import build_messages


class HFGenerator(StepGenerator):
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct", temperature: float = 0.8,
                 top_p: float = 0.95, max_new_tokens: int = 256, seed: int = 0):
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"  # Apple GPU if available
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(self.device)
        self.model.eval()  # inference mode, no training
        set_seed(seed)  # once per run: reproducible, but every call gets new random numbers

    def generate_steps(self, prompt: str, n: int) -> list[str]:
        # 1. Messages from the node text: rules, few-shot example, problem, steps so far
        messages = build_messages(prompt)

        # 2. Chat template: root -> open a new assistant turn, otherwise continue the prefilled steps
        if messages[-1]["role"] == "user":
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=True)

        # 3. Text -> token ids
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        # 4. Sample n answers, one token at a time
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=n,
            )

        # 5. Each row is prompt + answer: keep the answer and decode it
        prompt_length = inputs["input_ids"].shape[1]
        answers = []
        for sequence in outputs:
            new_tokens = sequence[prompt_length:]
            answer = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            answers.append(answer)
        return answers


if __name__ == "__main__":
    from problems import TWO_SUM
    from step_format import CODE_PREFIX, parse_step

    generator = HFGenerator()
    print("device:", generator.device)

    root = TWO_SUM.to_prompt()
    step_1 = root + "\nStep 1: Use a hash map from value to index."
    # (description, node text, is the code forced?) -> the three kinds of node in the tree
    cases = [
        ("root", root, False),
        ("after Step 1", step_1, False),
        ("forced code", step_1 + "\n" + CODE_PREFIX, True),
    ]
    for description, full_text, force_code in cases:
        print(f"\n===== {description} =====")
        answers = generator.generate_steps(full_text, 3)
        assert len(answers) == 3 and all(isinstance(a, str) for a in answers)
        for answer in answers:
            if force_code:
                answer = CODE_PREFIX + "\n" + answer  # the label was in the prompt, as in expand
            parsed = parse_step(answer, force_code)
            print("-----")
            print(answer)
            print(">> parsed:", "DISCARDED" if parsed is None else parsed)
