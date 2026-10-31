# This file is used to run your functions and print the results
#
# Usage (from any directory):
#   python main.py                  # evaluate on the test set the checkpoint stored in bin/
#   python main.py --train          # fine-tune BERT (5 runs) and save the best run on dev in bin/
import argparse
import os
from functools import partial

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from functions import (
    build_optimizer,
    eval_loop,
    linear_warmup_scheduler,
    load_checkpoint,
    sample_std,
    save_checkpoint,
    set_seed,
    train_model,
)
from model import JointBERT
from utils import DEVICE, IGNORE_INDEX, IntentsAndSlots, Lang, collate_fn, create_dev_set, load_data

DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(DIR, "dataset")
BIN_DIR = os.path.join(DIR, "bin")
CHECKPOINT = os.path.join(BIN_DIR, "bert.pt")

CONFIG = {
    "bert_model": "bert-base-uncased",
    "batch_size_train": 32,
    "batch_size_dev": 64,
    "batch_size_test": 64,
    "lr": 5e-5,
    "weight_decay": 0.01,
    "warmup": 0.1,     # fraction of the training steps with a linear warm-up of the learning rate
    "dropout": 0.1,
    "clip": 1.0,
    "n_epochs": 10,
    "patience": 3,     # epochs without improvement of the dev slot F1
    "runs": 5,         # independent runs (different seeds), we report mean and std on the test set
    "seed": 42,
}


def get_loader(raw, lang, tokenizer, batch_size, shuffle=False):
    dataset = IntentsAndSlots(raw, lang, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id))


def train(config, lang, tokenizer, train_loader, dev_loader, test_loader):
    print(f"\n=== Fine-tuning {config['bert_model']}: {config}")
    criterion_slots = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    criterion_intents = nn.CrossEntropyLoss()

    runs = []
    best_run = None
    for run in range(config["runs"]):
        # Every run is independent: new seed and a fresh copy of the pre-trained model. This is the only
        # place where the model is created; inside the epoch loop only the gradients are zeroed.
        set_seed(config["seed"] + run)
        model = JointBERT(config["bert_model"], len(lang.slot2id), len(lang.intent2id),
                          dropout=config["dropout"]).to(DEVICE)
        optimizer = build_optimizer(model, config["lr"], config["weight_decay"])
        total_steps = config["n_epochs"] * len(train_loader)
        scheduler = linear_warmup_scheduler(optimizer, int(config["warmup"] * total_steps), total_steps)

        best_dev_f1, history = train_model(model, optimizer, scheduler, train_loader, dev_loader, criterion_slots,
                                           criterion_intents, lang, config["n_epochs"], config["patience"],
                                           config["clip"])
        # model now holds the weights of the best epoch on dev
        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
        run_res = {"run": run, "dev_f1": float(best_dev_f1), "slot_f1": float(results_test['total']['f']),
                   "intent_acc": float(intent_test['accuracy']), "epochs": len(history["epoch"])}
        runs.append(run_res)
        print(f"run {run}: epochs {run_res['epochs']} | dev slot F1 {best_dev_f1:.4f} | "
              f"test slot F1 {run_res['slot_f1']:.4f} | test intent acc {run_res['intent_acc']:.4f}")

        # The checkpoint we keep is the run with the best DEV score (the test set is not used for selection)
        if best_run is None or best_dev_f1 > best_run["dev_f1"]:
            best_run = run_res
            os.makedirs(BIN_DIR, exist_ok=True)
            save_checkpoint(CHECKPOINT, model, config, lang, run_res)
        del model, optimizer  # free the GPU memory before the next run

    slot_f1s = np.asarray([r["slot_f1"] for r in runs])
    intent_acc = np.asarray([r["intent_acc"] for r in runs])
    print(f"\nBERT over {len(runs)} runs: slot F1 {slot_f1s.mean():.4f} +- {sample_std(slot_f1s):.4f} | "
          f"intent acc {intent_acc.mean():.4f} +- {sample_std(intent_acc):.4f} "
          f"(checkpoint saved: run {best_run['run']})")


def evaluate_checkpoint(test_raw):
    if not os.path.exists(CHECKPOINT):
        print(f"No checkpoint in {CHECKPOINT}, train the model with: python main.py --train")
        return
    checkpoint = load_checkpoint(CHECKPOINT, DEVICE)
    config = checkpoint["config"]

    # Recover the label vocabularies used at training time
    lang = Lang(intents=[], slots=[])
    lang.slot2id = checkpoint["slot2id"]
    lang.intent2id = checkpoint["intent2id"]
    lang.id2slot = {v: k for k, v in lang.slot2id.items()}
    lang.id2intent = {v: k for k, v in lang.intent2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(config["bert_model"])
    test_loader = get_loader(test_raw, lang, tokenizer, config["batch_size_test"])
    model = JointBERT(config["bert_model"], len(lang.slot2id), len(lang.intent2id), dropout=config["dropout"],
                      pretrained=False).to(DEVICE)
    model.load_state_dict(checkpoint["model"])
    results_test, intent_test, _ = eval_loop(test_loader, nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX),
                                             nn.CrossEntropyLoss(), model, lang)
    print(f"BERT ({config['bert_model']}): test slot F1 {results_test['total']['f']:.4f} | "
          f"test intent acc {intent_test['accuracy']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLU project - intent classification and slot filling, part B")
    parser.add_argument("--train", action="store_true", help="fine-tune the model (default: only evaluate bin/)")
    parser.add_argument("--bert_model", default=None, help="Hugging Face model name or local path")
    parser.add_argument("--lr", type=float, default=None, help="override the learning rate")
    parser.add_argument("--runs", type=int, default=None, help="override the number of runs")
    parser.add_argument("--epochs", type=int, default=None, help="override the maximum number of epochs")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    tmp_train_raw = load_data(os.path.join(DATASET_DIR, 'train.json'))
    test_raw = load_data(os.path.join(DATASET_DIR, 'test.json'))

    if not args.train:
        evaluate_checkpoint(test_raw)
        raise SystemExit(0)

    config = dict(CONFIG)
    for key in ("bert_model", "lr", "runs", "seed"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.epochs is not None:
        config["n_epochs"] = args.epochs

    train_raw, dev_raw = create_dev_set(tmp_train_raw)  # same dev split as part A

    corpus = train_raw + dev_raw + test_raw  # We do not want unk labels, as in part A
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])
    lang = Lang(intents, slots)

    tokenizer = AutoTokenizer.from_pretrained(config["bert_model"])
    train_loader = get_loader(train_raw, lang, tokenizer, config["batch_size_train"], shuffle=True)
    dev_loader = get_loader(dev_raw, lang, tokenizer, config["batch_size_dev"])
    test_loader = get_loader(test_raw, lang, tokenizer, config["batch_size_test"])

    train(config, lang, tokenizer, train_loader, dev_loader, test_loader)
