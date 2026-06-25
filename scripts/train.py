"""
train.py — continued pretraining of Qwen2.5-0.5B on materials science abstracts.
Run on EC2 g4dn.xlarge via run_training.py.

Usage:
    .venv/bin/python scripts/train.py --config configs/lora_r16.json

Blocks:
  [x] load_data      — pull train/val JSONL from S3
  [x] load_model     — load Qwen2.5-0.5B from S3
  [x] prepare_model  — apply training technique from config (lora | ...)
  [x] tokenize       — pack texts into fixed-length chunks
  [x] train          — HuggingFace Trainer + loss logging
  [ ] save           — upload checkpoint + config + loss log to S3
"""

import json
import os
import tempfile

import boto3
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = os.environ["S3_BUCKET"]
S3_TRAIN_KEY = "materialLLM/data/arxiv/train.jsonl"
S3_VAL_KEY = "materialLLM/data/arxiv/val.jsonl"
S3_MODEL_PREFIX = "materialLLM/models/Qwen2.5-0.5B"


# ------------------------------------------------------------------
# Block 1: Data loading
# ------------------------------------------------------------------

def load_data() -> tuple[list[str], list[str]]:
    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-2"))

    def fetch(key):
        print(f"Loading s3://{S3_BUCKET}/{key} ...")
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        lines = obj["Body"].read().decode("utf-8").strip().splitlines()
        texts = [json.loads(l)["text"] for l in lines if l.strip()]
        print(f"  → {len(texts)} records")
        return texts

    return fetch(S3_TRAIN_KEY), fetch(S3_VAL_KEY)


# ------------------------------------------------------------------
# Block 2: Model loading
# ------------------------------------------------------------------

def load_model():
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-2"))

    print(f"Downloading model from s3://{S3_BUCKET}/{S3_MODEL_PREFIX}/")
    tmp_dir = tempfile.mkdtemp(prefix="qwen_")
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_MODEL_PREFIX + "/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            relative = key[len(S3_MODEL_PREFIX) + 1:]
            if not relative:
                continue
            local_path = os.path.join(tmp_dir, relative)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            s3.download_file(S3_BUCKET, key, local_path)

    tokenizer = AutoTokenizer.from_pretrained(tmp_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        tmp_dir,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model loaded — {n_params:.0f}M parameters")

    device = (
        "cuda" if torch.cuda.is_available() else
        "mps" if torch.backends.mps.is_available() else
        "cpu"
    )
    model = model.to(device)
    print(f"Device: {device}")

    return model, tokenizer, device


# ------------------------------------------------------------------
# Block 3: Prepare model (apply training technique)
# ------------------------------------------------------------------

def prepare_model(model, cfg: dict):
    technique = cfg["technique"]

    if technique == "lora":
        from peft import get_peft_model, LoraConfig, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=cfg["target_modules"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        raise ValueError(f"Unknown technique: {technique}")

    return model


# ------------------------------------------------------------------
# Block 4: Tokenize
# ------------------------------------------------------------------

def tokenize(texts: list[str], tokenizer, max_length: int = 512):
    from datasets import Dataset

    eos = tokenizer.eos_token_id
    print(f"Tokenizing {len(texts)} texts...")
    all_ids = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        all_ids.append(eos)

    n_chunks = len(all_ids) // max_length
    print(f"Total tokens: {len(all_ids):,} → {n_chunks} chunks of {max_length}")
    chunks = [all_ids[i * max_length: (i + 1) * max_length] for i in range(n_chunks)]

    return Dataset.from_dict({"input_ids": chunks, "labels": chunks})


# ------------------------------------------------------------------
# Block 5: Train
# ------------------------------------------------------------------

def train(model, tokenizer, train_dataset, val_dataset, cfg: dict):
    from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
    from transformers import TrainerCallback

    run_name = cfg["run_name"]
    log_path = f"/tmp/{run_name}_loss.jsonl"
    loss_records = []

    class LossLogger(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs:
                record = {"step": state.global_step, "loss": logs["loss"]}
                if "eval_loss" in logs:
                    record["eval_loss"] = logs["eval_loss"]
                loss_records.append(record)
                print(f"  step {state.global_step}: loss={logs['loss']:.4f}")

    training_args = TrainingArguments(
        output_dir=f"/tmp/{run_name}",
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=cfg["warmup_steps"],
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="no",
        report_to="none",
        dataloader_pin_memory=False,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=[LossLogger()],
    )

    print(f"\nStarting training run: {run_name}")
    trainer.train()

    with open(log_path, "w") as f:
        for record in loss_records:
            f.write(json.dumps(record) + "\n")
    print(f"Loss log saved to {log_path}")

    return trainer, log_path


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment config JSON")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    print(f"Config: {json.dumps(cfg, indent=2)}\n")

    train_texts, val_texts = load_data()
    model, tokenizer, device = load_model()
    model = prepare_model(model, cfg)
    train_dataset = tokenize(train_texts, tokenizer, max_length=cfg["max_length"])
    val_dataset = tokenize(val_texts, tokenizer, max_length=cfg["max_length"])

    print(f"\nTrain chunks: {len(train_dataset)}, Val chunks: {len(val_dataset)}")

    trainer, log_path = train(model, tokenizer, train_dataset, val_dataset, cfg)
