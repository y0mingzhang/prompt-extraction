import functools

import pandas as pd
from datasets import load_dataset
from nltk import word_tokenize


def undo_interleave(sep: str, s: str) -> str:
    return s.replace(" " + sep, "")


def caesar_roll_char(c: str, roll: int) -> str:
    if c.isupper():
        delta = (ord(c) - ord("A") + roll) % 26
        return chr(ord("A") + delta)
    elif c.islower():
        delta = (ord(c) - ord("a") + roll) % 26
        return chr(ord("a") + delta)
    return c


def undo_caesar(roll: int, s: str) -> str:
    return "".join(caesar_roll_char(c, -roll) for c in s)


def n_gram_defense(instruction: str, completion: str, n_gram: int = 5):
    ins_tokens = word_tokenize(instruction)
    comp_tokens = word_tokenize(completion)

    # get the set of all n-grams
    ins_grams = set(
        tuple(ins_tokens[i : i + n_gram]) for i in range(len(ins_tokens) - n_gram + 1)
    )
    comp_grams = set(
        tuple(comp_tokens[i : i + n_gram]) for i in range(len(comp_tokens) - n_gram + 1)
    )

    # defense kicks in if an overlap in n-grams is detected!
    return "" if ins_grams & comp_grams else completion


DEFENSES = {
    "no-defense": lambda _, s: s,
    "5-gram": functools.partial(n_gram_defense, n_gram=5),
}

DATA = {
    "awesome": lambda: load_dataset("fka/awesome-chatgpt-prompts")["train"]["prompt"],
    "dev": lambda: pd.read_json("data/dev.jsonl", lines=True, orient="records")[
        "instruction"
    ].tolist(),
    "sharegpt": lambda: pd.read_json(
        "data/sharegpt-test.jsonl", lines=True, orient="records"
    )["instruction"].tolist(),
    "unnatural": lambda: pd.read_json(
        "data/unnatural-test.jsonl", lines=True, orient="records"
    )["instruction"].tolist(),
}
