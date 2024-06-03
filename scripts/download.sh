#!/bin/bash

mkdir -p data
curl -C - -L https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json -o data/sharegpt.json
curl -C - -L https://github.com/orhonovich/unnatural-instructions/raw/main/data/core_data.zip -s 2>&1 | zcat > data/unnatural.jsonl

python scripts/sample.py