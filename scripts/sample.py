import pandas as pd
import tiktoken
from nltk import sent_tokenize
from transformers import AutoTokenizer

gpt4_tokenizer = tiktoken.encoding_for_model("gpt-4")
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")


def prep_sharegpt():
    data = pd.read_json("data/sharegpt.json")

    data_filtered = data[
        data["id"].apply(lambda s: s.endswith("_0"))  # first message in convo
        & data["conversations"].apply(
            lambda c: len(c) >= 1
            and c[0]["from"] == "human"
            and len(sent_tokenize(c[0]["value"])) > 1  # at least 2 sentences
            and len(gpt4_tokenizer.encode(c[0]["value"])) <= 256  # at most 256 tokens
            and len(llama_tokenizer(c[0]["value"])["input_ids"])
            <= 256  # at most 256 tokens
        )
    ].copy()

    data_filtered["instruction"] = data_filtered["conversations"].apply(
        lambda c: c[0]["value"]
    )
    data_filtered_nodup = (
        data_filtered.drop_duplicates(subset="instruction")
        .drop(columns="conversations")
        .sample(frac=1.0, random_state=42)
    )

    data_val = data_filtered_nodup.iloc[:200]
    data_test = data_filtered_nodup.iloc[200:700]

    return data_val, data_test


def prep_unnatural():
    data = pd.read_json("data/unnatural.jsonl", lines=True)
    data["id"] = data.index.map(lambda i: f"unnatural-{i}")

    data_filtered = data[
        data["instruction"].apply(
            lambda s: len(gpt4_tokenizer.encode(s)) <= 256  # at most 256 tokens
            and len(llama_tokenizer(s)["input_ids"]) <= 256  # at most 256 tokens
        )
    ].copy()

    data_filtered_nodup = (
        data_filtered.drop_duplicates(subset="instruction")
        .drop(columns="instances")
        .sample(frac=1.0, random_state=42)
    )

    data_val = data_filtered_nodup.iloc[:200]
    data_test = data_filtered_nodup.iloc[200:700]

    return data_val, data_test


sharegpt_dev, sharegpt_test = prep_sharegpt()
unnatural_dev, unnatural_test = prep_unnatural()

dev = pd.concat((sharegpt_dev, unnatural_dev))
dev.to_json("data/dev.jsonl", lines=True, orient="records")
sharegpt_test.to_json("data/sharegpt-test.jsonl", lines=True, orient="records")
unnatural_test.to_json("data/unnatural-test.jsonl", lines=True, orient="records")
