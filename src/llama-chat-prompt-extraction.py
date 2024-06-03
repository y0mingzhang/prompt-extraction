import argparse
import functools
import itertools
import json
import os
from os.path import basename, join
from typing import Any, Callable

import torch
from llama import Llama
from tqdm.auto import trange

from common import DATA, DEFENSES, undo_caesar, undo_interleave


@torch.no_grad()
def run_extraction_attack(
    model: Llama,
    instruction_set: list[str],
    attack_set: list[dict | str],
    defense: Callable,
    batch_size: int = 16,
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
            [
                {"role": "system", "content": i},
                {"role": "user", "content": a},
            ]
            for i, a in zip(i_batch, a_batch)
        ]
        completion_raws = [
            cr["generation"]["content"]
            for cr in model.chat_completion(
                prompts,
                max_gen_len=384,
                temperature=temperature,
                top_p=1.0,
            )
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=DATA.keys(), default="dev")
    parser.add_argument("--attack", type=str, default="attacks/attacks.json")
    parser.add_argument("--defense", choices=DEFENSES.keys(), default="no-defense")
    parser.add_argument("--output_dir", type=str, default="outputs/debug")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)

    args = parser.parse_args()

    instruction_set = DATA[args.data]()

    with open(args.attack) as f:
        attack_set = json.load(f)

    if args.debug:
        instruction_set = instruction_set[:2]
        attack_set = attack_set[:2]

    defense = DEFENSES[args.defense]

    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=640,
        max_batch_size=16,
    )

    eval_results = run_extraction_attack(
        generator, instruction_set, attack_set, defense, temperature=args.temperature
    )
    attack_name = basename(args.attack).split(".")[0]

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.ckpt_dir.split("/")[-1]
    output_path = join(args.output_dir, f"{model_name}@{attack_name}.jsonl")
    with open(output_path, "w") as fo:
        for line in eval_results:
            json.dump(line, fo)
            fo.write("\n")

    print("Done!")


if __name__ == "__main__":
    main()
