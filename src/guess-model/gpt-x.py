import argparse
import os
from os.path import join
from typing import Any

import openai
import pandas as pd
from tqdm.auto import tqdm

api_key_file = "OPENAI-API-KEY-HERE"
openai.api_key_path = api_key_file


MODELS = {"gpt-3.5-turbo": "gpt-3.5-turbo-0613", "gpt-4": "gpt-4-0613"}


def run_extraction_attack(
    model: str,
    instructions: list[str],
    user_inputs: list[str],
    n_samples: int = 10,
    temperature: float = 1.0,
) -> list[dict[str, Any]]:
    results = []

    for i, a in zip(tqdm(instructions), user_inputs):
        input_msg = [
            {"role": "system", "content": i},
            {"role": "user", "content": a},
        ]
        completion = openai.ChatCompletion.create(
            model=model,
            messages=input_msg,
            n=n_samples,
            temperature=temperature,
        )

        assert len(completion["choices"]) == n_samples
        results.append([c["message"]["content"] for c in completion["choices"]])

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/guess-model")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--field", choices=["instruction", "completion"])
    args = parser.parse_args()

    data = pd.read_json(args.data, lines=True)
    instructions = data[args.field]

    model = MODELS[args.model]
    output = run_extraction_attack(
        model,
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
