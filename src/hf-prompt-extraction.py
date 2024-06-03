import argparse
import functools
import itertools
import json
import os
from os.path import basename, join
from typing import Any, Callable

import torch
from tqdm.auto import trange
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from common import DATA, DEFENSES, undo_caesar, undo_interleave


@torch.no_grad()
def run_extraction_attack(
    model: AutoModelForCausalLM,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    instruction_set: list[str],
    attack_set: list[dict | str],
    template: str,
    defense: Callable,
    batch_size: int = 32,
    temperature: float = 0.0,
) -> list[dict[str, Any]]:
    results = []

    instructions = []
    attacks = []
    transforms = []

    for i, a in itertools.product(instruction_set, attack_set):
        instructions.append(i)

        def transform(s):
            return s

        if isinstance(a, dict):
            attack_str = a["attack-string"]
            if a["attack-type"] == "interleave":
                transform = functools.partial(undo_interleave, a["separator"])
            elif a["attack-type"] == "caesar":
                transform = functools.partial(undo_caesar, a["roll"])
            else:
                assert False
        else:
            attack_str = a
        attacks.append(attack_str)
        transforms.append(transform)

    num_batches = len(instructions) // batch_size
    if len(instructions) % batch_size != 0:
        num_batches += 1

    for batch_idx in trange(num_batches):
        i_batch = instructions[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        a_batch = attacks[batch_idx * batch_size : (batch_idx + 1) * batch_size]
        t_batch = transforms[batch_idx * batch_size : (batch_idx + 1) * batch_size]

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
        completion_raws = [
            tokenizer.decode(x[maxlen:], skip_special_tokens=True) for x in out
        ]
        for i, a, p, t, cr in zip(i_batch, a_batch, prompts, t_batch, completion_raws):
            c = t(defense(i, cr))
            results.append(
                {
                    "instruction": i,
                    "attack": a,
                    "prompt": p,
                    "completion-raw": cr,
                    "completion": c,
                }
            )

    return results


ALPACA_TEMPLATE = """### Instruction:
{instruction}

### Input:
{attack}

### Response:"""

VICUNA_TEMPLATE = """{instruction}

USER: {attack}
ASSISTANT:"""

FALCON_INSTRUCT_TEMPLATE = """{instruction}

User: {attack}
Assistant:"""


def determine_template(model: str) -> str:
    if "alpaca" in model:
        return ALPACA_TEMPLATE
    elif "vicuna" in model:
        assert "v1.3" in model or "v1.5" in model
        return VICUNA_TEMPLATE
    elif "falcon" in model and "instruct" in model:
        return FALCON_INSTRUCT_TEMPLATE

    assert False, model


ALPACA_PATH = "PATH OF ALPACA-7B (HF VERSION)"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="alpaca-7b")
    parser.add_argument("--data", choices=DATA.keys(), default="dev")
    parser.add_argument("--attack", type=str, default="attacks/attacks.json")
    parser.add_argument("--defense", choices=DEFENSES.keys(), default="no-defense")
    parser.add_argument("--output_dir", type=str, default="outputs/debug")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    if args.model == "alpaca-7b":
        args.model = ALPACA_PATH

    instruction_set = DATA[args.data]()

    with open(args.attack) as f:
        attack_set = json.load(f)

    if args.debug:
        instruction_set = instruction_set[:2]
        attack_set = attack_set[:2]

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    defense = DEFENSES[args.defense]
    template = determine_template(args.model)

    eval_results = run_extraction_attack(
        model,
        tokenizer,
        instruction_set,
        attack_set,
        template,
        defense,
        temperature=args.temperature,
    )
    attack_name = basename(args.attack).split(".")[0]

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.model.split("/")[-1]
    output_path = join(args.output_dir, f"{model_name}@{attack_name}.jsonl")
    with open(output_path, "w") as fo:
        for line in eval_results:
            json.dump(line, fo)
            fo.write("\n")

    print("Done!")


if __name__ == "__main__":
    main()
