import argparse
import os
from os.path import join
from typing import Any

import pandas as pd
import torch
from tqdm.auto import trange
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


@torch.no_grad()
def run_extraction_attack(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    instructions: list[str],
    user_inputs: list[str],
    template: str,
    n_samples: int = 20,
    batch_size: int = 32,
    temperature: float = 1.0,
) -> list[dict[str, Any]]:
    results = []
    instructions = [i for i in instructions for _ in range(n_samples)]
    user_inputs = [i for i in user_inputs for _ in range(n_samples)]

    num_batches = len(instructions) // batch_size
    if len(instructions) % batch_size != 0:
        num_batches += 1

    for batch_idx in trange(num_batches):
        i_batch = instructions[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        a_batch = user_inputs[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        prompts = [
            template.format(instruction=i, attack=a) for i, a in zip(i_batch, a_batch)
        ]

        inputs = [tokenizer(x).input_ids for x in prompts]
        maxlen = max(map(len, inputs))
        inputs = [[tokenizer.eos_token_id] * (maxlen - len(x)) + x for x in inputs]
        iids = torch.tensor(inputs).to(model.device)
        mask = (iids != tokenizer.eos_token_id).float().to(model.device)

        out = model.generate(
            iids,
            max_new_tokens=384,
            attention_mask=mask,
            do_sample=(temperature > 0.0),
            temperature=temperature,
            top_p=1.0,
        )
        completions = [
            tokenizer.decode(x[maxlen:], skip_special_tokens=True) for x in out
        ]
        results.extend(completions)

    assert len(results) % n_samples == 0

    return [results[i : i + n_samples] for i in range(0, len(results), n_samples)]


ALPACA_TEMPLATE = """### Instruction:
{instruction}

### Input:
{attack}

### Response:"""

VICUNA_TEMPLATE = """{instruction}

USER: {attack}
ASSISTANT:"""


def determine_template(model: str) -> str:
    if "alpaca" in model:
        return ALPACA_TEMPLATE
    elif "vicuna" in model:
        assert "v1.3" in model or "v1.5" in model
        return VICUNA_TEMPLATE
    assert False, model


ALPACA_PATH = "/data/datasets/models/hf_cache/alpaca-7b"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="alpaca-7b")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/guess-model")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--field", choices=["instruction", "completion"])
    args = parser.parse_args()

    if args.model == "alpaca-7b":
        args.model = ALPACA_PATH

    data = pd.read_json(args.data, lines=True)
    instructions = data[args.field]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    template = determine_template(args.model)
    output = run_extraction_attack(
        model,
        tokenizer,
        instructions,
        data["input"].tolist(),
        template,
        n_samples=args.n_samples,
        temperature=args.temperature,
    )
    data["output"] = output

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.model.split("/")[-1]
    output_path = join(args.output_dir, f"{model_name}@{args.field}.jsonl")
    data.to_json(output_path, lines=True, orient="records")

    print("Done!")


if __name__ == "__main__":
    main()
