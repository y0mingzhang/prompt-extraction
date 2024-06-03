import argparse
import os
from os.path import join
from typing import Any

import pandas as pd
import torch
from llama import Llama
from tqdm.auto import trange


@torch.no_grad()
def run_extraction_attack(
    model: Llama,
    instructions: list[str],
    user_inputs: list[str],
    n_samples: int = 20,
    batch_size: int = 8,
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
            [
                {"role": "system", "content": i},
                {"role": "user", "content": a},
            ]
            for i, a in zip(i_batch, a_batch)
        ]
        completions = [
            cr["generation"]["content"]
            for cr in model.chat_completion(
                prompts,
                max_gen_len=384,
                temperature=temperature,
                top_p=1.0,
            )
        ]
        results.extend(completions)

    assert len(results) % n_samples == 0
    return [results[i : i + n_samples] for i in range(0, len(results), n_samples)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama-2-7b-chat")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/guess-model")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--field", choices=["instruction", "completion"])
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    args = parser.parse_args()

    data = pd.read_json(args.data, lines=True)
    instructions = data[args.field]

    generator = Llama.build(
        ckpt_dir=args.ckpt_dir,
        tokenizer_path=args.tokenizer_path,
        max_seq_len=640,
        max_batch_size=16,
    )
    output = run_extraction_attack(
        generator,
        instructions,
        data["input"].tolist(),
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
