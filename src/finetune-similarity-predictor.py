import argparse
import json
import math
import os
from collections import deque
from contextlib import nullcontext

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

parser = argparse.ArgumentParser()

OUTPUT_DIR = "outputs/rouge-predictor"
TRAIN_FILE = "data/success-train.jsonl"
VAL_FILE = "data/success-val.jsonl"
LABEL_COLUMN = "rougeL-recall"
WARMUP_STEPS = 50
TRAIN_STEPS = 2000
PRETRAINED_MODEL = "microsoft/deberta-v3-large"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
VAL_BATCH_SIZE = 16
MAX_LR = 2e-5
MIN_LR = 2e-6
MAX_ATTACK_LEN = 256
GRAD_ACC_STEPS = 8
MAX_EXTRACTIONS = 4
AUTOCAST = (
    torch.autocast(
        DEVICE,
        dtype=(torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32),
    )
    if DEVICE == "cuda"
    else nullcontext()
)


def get_lr(t: int) -> float:
    assert MAX_LR >= MIN_LR >= 0.0
    assert TRAIN_STEPS >= WARMUP_STEPS >= 0

    if t <= WARMUP_STEPS:
        return (t / WARMUP_STEPS) * MAX_LR

    elif t >= TRAIN_STEPS:
        return MIN_LR

    return (MAX_LR - MIN_LR) / 2 * math.cos(
        (t - WARMUP_STEPS) * math.pi / (TRAIN_STEPS - WARMUP_STEPS)
    ) + (MIN_LR + MAX_LR) / 2


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def hyperparameters():
    return {
        key: value
        for key, value in globals().items()
        if key
        in [
            "TRAIN_FILE",
            "VAL_FILE",
            "LABEL_COLUMN",
            "TRAIN_STEPS",
            "WARMUP_STEPS",
            "PRETRAINED_MODEL",
            "BATCH_SIZE",
            "MAX_LR",
            "MIN_LR",
            "MAX_ATTACK_LEN",
            "GRAD_ACC_STEPS",
            "MAX_EXTRACTIONS",
        ]
    }


def to_device(data):
    return {k: v.to(DEVICE) for k, v in data.items()}


class DictDataset(Dataset):
    def __init__(self, data: list[dict]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> dict:
        return self.data[i]


def get_dataloader(tokenizer, df, train):
    data = []
    for _, row in tqdm(df.iterrows(), "preping data..."):
        completion = row["completion"]
        extractions = [completion] + row["completions"][:MAX_EXTRACTIONS]
        input_ids = [tokenizer.cls_token_id]
        for a in extractions:
            input_ids.extend(
                tokenizer.encode(
                    a,
                    truncation=True,
                    max_length=MAX_ATTACK_LEN,
                    add_special_tokens=False,
                )
            )
            input_ids.append(tokenizer.sep_token_id)
            curr_attack_len = input_ids.index(tokenizer.sep_token_id) + 1
        token_type_ids = [0] * curr_attack_len + [1] * (
            len(input_ids) - curr_attack_len
        )
        data.append(
            {
                "input_ids": torch.LongTensor(input_ids),
                "token_type_ids": torch.LongTensor(token_type_ids),
                "labels": row[LABEL_COLUMN],
            }
        )

    dataset = DictDataset(data)
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE if train else VAL_BATCH_SIZE,
        shuffle=train,
        pin_memory=True,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )


def get_train_iter(dl):
    while True:
        for batch in dl:
            yield to_device(batch)


def main():
    os.makedirs(OUTPUT_DIR)
    with open(os.path.join(OUTPUT_DIR, "hyperparams.json"), "w") as f:
        json.dump(hyperparameters(), f)

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    train_data = pd.read_json(TRAIN_FILE, lines=True)
    val_data = pd.read_json(VAL_FILE, lines=True)
    train_dl = get_dataloader(tokenizer, train_data, True)
    val_dl = get_dataloader(tokenizer, val_data, False)
    train_iter = get_train_iter(train_dl)

    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL, num_labels=1
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0, fused=DEVICE == "cuda")
    loss_fn = torch.nn.L1Loss()
    train_losses = deque(maxlen=16)
    with tqdm(total=TRAIN_STEPS) as pbar:
        for i in range(TRAIN_STEPS):
            lr = get_lr(i)
            for g in opt.param_groups:
                g["lr"] = lr
            for _ in range(GRAD_ACC_STEPS):
                batch = next(train_iter)
                labels = batch.pop("labels")
                with AUTOCAST:
                    outputs = model(**batch)
                    preds = F.sigmoid(outputs["logits"].squeeze())
                    loss = loss_fn(preds, labels)
                loss.backward()
                train_losses.append((preds - labels).abs().mean().item())
                pbar.set_postfix(
                    {
                        "Train loss": f"{loss.item():.3f}",
                        "Mean absolute error": f"{sum(train_losses) / len(train_losses):.3f}",
                    }
                )
            opt.step()
            opt.zero_grad()
            pbar.update(1)

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pt"))

    model.eval()
    labels = val_data[LABEL_COLUMN].tolist()
    preds = []
    with torch.inference_mode():
        for batch in tqdm(val_dl, total=math.ceil(len(val_data) / VAL_BATCH_SIZE)):
            batch = to_device(batch)
            with AUTOCAST:
                outputs = model(**batch)
            preds_batch = F.sigmoid(outputs["logits"])
            preds.extend(preds_batch.flatten().tolist())
    assert len(labels) == len(preds)

    val_eval = {"MSE": sum((p - l) ** 2 for p, l in zip(preds, labels)) / len(preds)}
    val_data[LABEL_COLUMN + "-pred"] = preds

    print(json.dumps(val_eval, indent=2))
    with open(os.path.join(OUTPUT_DIR, "eval.json"), "w") as f:
        json.dump(val_eval, f)

    val_data.to_json(
        os.path.join(OUTPUT_DIR, "val.jsonl"), lines=True, orient="records"
    )


if __name__ == "__main__":
    main()
