import argparse
import functools
import itertools
import json
import os
from concurrent.futures import ThreadPoolExecutor
from os.path import basename, join
from typing import Any, Callable

import openai
from tqdm.auto import tqdm

from common import DATA, DEFENSES, undo_caesar, undo_interleave

api_key_file = "OPENAI-API-KEY-HERE"
openai.api_key_path = api_key_file


def gpt_x_extraction_attack(
    model: str,
    instruction: str,
    attack: str,
    temperature: float = 0.0,
) -> tuple[Any, str]:
    input_msg = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": attack},
    ]
    output = openai.ChatCompletion.create(
        model=model,
        messages=input_msg,
        temperature=temperature,
        max_tokens=384,
    )

    output_msg = output["choices"][0]["message"]
    assert output_msg["role"] == "assistant"
    return input_msg, output_msg["content"]


MODELS = {
    "gpt-3.5-turbo": lambda: functools.partial(
        gpt_x_extraction_attack, "gpt-3.5-turbo-0613"
    ),
    "gpt-4": lambda: functools.partial(gpt_x_extraction_attack, "gpt-4-0613"),
}


def run_extraction_attack(
    model: Callable,
    instruction_set: list[str],
    attack_set: list[dict | str],
    temperature: float,
    defense: Callable,
) -> list[dict[str, Any]]:
    transforms = []
    instructions = []
    attacks = []
    for instruction, attack in itertools.product(instruction_set, attack_set):

        def transform(s):
            return s

        if isinstance(attack, dict):
            attack_str = attack["attack-string"]
            if attack["attack-type"] == "interleave":
                transform = functools.partial(undo_interleave, attack["separator"])
            elif attack["attack-type"] == "caesar":
                transform = functools.partial(undo_caesar, attack["roll"])
            else:
                assert False
        else:
            attack_str = attack

        transforms.append(transform)
        instructions.append(instruction)
        attacks.append(attack_str)

    tpe = ThreadPoolExecutor(8)
    results = []
    for instruction, attack, (prompt, completion_raw), transform in zip(
        instructions,
        attacks,
        tqdm(
            tpe.map(
                functools.partial(model, temperature=temperature), instructions, attacks
            ),
            total=len(instructions),
        ),
        transforms,
    ):
        completion = transform(defense(instruction, completion_raw))

        results.append(
            {
                "instruction": instruction,
                "attack": attack,
                "prompt": prompt,
                "completion-raw": completion_raw,
                "completion": completion,
            }
        )
    tpe.shutdown()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS.keys(), default="gpt-3.5-turbo")
    parser.add_argument("--data", choices=DATA.keys(), default="dev")
    parser.add_argument("--attack", type=str, default="attacks/attacks.json")
    parser.add_argument("--defense", choices=DEFENSES.keys(), default="no-defense")
    parser.add_argument("--output_dir", type=str, default="./outputs/debug")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    instruction_set = DATA[args.data]()

    with open(args.attack) as f:
        attack_set = json.load(f)

    if args.debug:
        instruction_set = instruction_set[:2]
        attack_set = attack_set[:2]

    model = MODELS[args.model]()
    defense = DEFENSES[args.defense]

    eval_results = run_extraction_attack(
        model,
        instruction_set,
        attack_set,
        args.temperature,
        defense,
    )
    attack_name = basename(args.attack).split(".")[0]

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = join(args.output_dir, f"{args.model}@{attack_name}.jsonl")
    with open(output_path, "w") as fo:
        for line in eval_results:
            json.dump(line, fo)
            fo.write("\n")

    print("Done!")


if __name__ == "__main__":
    main()
