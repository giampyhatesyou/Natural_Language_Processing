# This file is used to run your functions and print the results
#
# Usage (from any directory):
#   python main.py                               # evaluate on the test set the checkpoints stored in bin/
#   python main.py --train                       # train (and evaluate) the three incremental experiments
#   python main.py --train --exp wt_vd_ntavsgd   # train a single experiment
import argparse
import os
from functools import partial

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from functions import *
from model import *
from utils import *

DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(DIR, "..", "dataset")
BIN_DIR = os.path.join(DIR, "bin")
PLOTS_DIR = os.path.join(DIR, "plots")

# Shared hyperparameters. NT-AvSGD needs SGD, so every experiment of part B uses SGD.
CONFIG = {
    "batch_size_train": 32,
    "batch_size_dev": 128,
    "batch_size_test": 128,
    "emb_size": 256,  # weight tying: emb_size == hid_size
    "hid_size": 256,
    "lr": 1.5,
    "clip": 5,
    "n_epochs": 100,
    "patience": 3,
    "seed": 42,
}

# Part 1.B is incremental: every experiment starts from the best model of the previous one
EXPERIMENTS = {
    # 1. LSTM + weight tying
    "wt": {"emb_dropout": 0.0, "out_dropout": 0.0, "nt_avsgd": False},
    # 2. + variational dropout on the embeddings and on the LSTM output
    "wt_vd": {"emb_dropout": 0.4, "out_dropout": 0.4, "nt_avsgd": False},
    # 3. + Non-monotonically Triggered AvSGD (patience counted only after the trigger, on the averaged weights)
    "wt_vd_ntavsgd": {"emb_dropout": 0.4, "out_dropout": 0.4, "nt_avsgd": True, "non_mono": 5, "patience": 5},
}


def build_model(config, vocab_len, pad_index):
    model = LM_LSTM_WT(config["emb_size"], config["hid_size"], vocab_len, pad_index=pad_index,
                       emb_dropout=config["emb_dropout"], out_dropout=config["out_dropout"])
    return model.to(DEVICE)


def train_experiment(name, config, lang, train_loader, dev_loader, test_loader):
    print(f"\n=== Training '{name}': {config}")
    set_seed(config["seed"])
    pad_index = lang.word2id["<pad>"]
    model = build_model(config, len(lang.word2id), pad_index)
    init_weights(model)  # once, before training
    optimizer = optim.SGD(model.parameters(), lr=config["lr"])
    ntasgd = NTAvSGD(model, optimizer, non_mono=config["non_mono"]) if config["nt_avsgd"] else None

    criterion_train = nn.CrossEntropyLoss(ignore_index=pad_index)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad_index, reduction='sum')

    best_dev_ppl, history = train_model(model, optimizer, train_loader, dev_loader, criterion_train, criterion_eval,
                                        config["n_epochs"], config["patience"], config["clip"], ntasgd=ntasgd)
    # The model now holds the best weights on dev (the averaged ones if NT-AvSGD triggered)
    test_ppl, _ = eval_loop(test_loader, criterion_eval, model)

    results = {"dev_ppl": best_dev_ppl, "test_ppl": test_ppl, "epochs": len(history["epoch"]), "history": history}
    os.makedirs(BIN_DIR, exist_ok=True)
    save_checkpoint(os.path.join(BIN_DIR, f"{name}.pt"), model, config, lang, results)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_history(history, f"LM part B - {name}", os.path.join(PLOTS_DIR, f"{name}.png"))
    print(f"'{name}': best dev PPL {best_dev_ppl:.2f} | test PPL {test_ppl:.2f}")
    return results


def evaluate_experiment(name, lang, test_loader):
    path = os.path.join(BIN_DIR, f"{name}.pt")
    if not os.path.exists(path):
        print(f"'{name}': no checkpoint in {path}, train it with: python main.py --train --exp {name}")
        return None
    checkpoint = load_checkpoint(path, DEVICE)
    if checkpoint["word2id"] != lang.word2id:
        raise ValueError(f"{path} was trained with a different vocabulary")
    pad_index = lang.word2id["<pad>"]
    model = build_model(checkpoint["config"], len(lang.word2id), pad_index)
    model.load_state_dict(checkpoint["model"])
    test_ppl, _ = eval_loop(test_loader, nn.CrossEntropyLoss(ignore_index=pad_index, reduction='sum'), model)
    print(f"'{name}': test PPL {test_ppl:.2f}")
    return {"test_ppl": test_ppl, "dev_ppl": checkpoint["results"]["dev_ppl"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLU project - LM part B")
    parser.add_argument("--train", action="store_true", help="train the models (default: only evaluate bin/)")
    parser.add_argument("--exp", default="all", choices=["all"] + list(EXPERIMENTS))
    parser.add_argument("--epochs", type=int, default=None, help="override the maximum number of epochs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--subset", type=int, default=None,
                        help="debug: use only the first N sentences of every split")
    args = parser.parse_args()

    train_raw = read_file(os.path.join(DATASET_DIR, "ptb.train.txt"))
    dev_raw = read_file(os.path.join(DATASET_DIR, "ptb.valid.txt"))
    test_raw = read_file(os.path.join(DATASET_DIR, "ptb.test.txt"))

    # The vocabulary is always computed on the whole training set
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    if args.subset:
        train_raw, dev_raw, test_raw = train_raw[:args.subset], dev_raw[:args.subset], test_raw[:args.subset]

    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)

    collate = partial(collate_fn, pad_token=lang.word2id["<pad>"])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size_train"], collate_fn=collate, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=CONFIG["batch_size_dev"], collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size_test"], collate_fn=collate)

    names = list(EXPERIMENTS) if args.exp == "all" else [args.exp]
    summary = {}
    for name in names:
        if args.train:
            config = {**CONFIG, **EXPERIMENTS[name]}
            if args.epochs is not None:
                config["n_epochs"] = args.epochs
            if args.seed is not None:
                config["seed"] = args.seed
            summary[name] = train_experiment(name, config, lang, train_loader, dev_loader, test_loader)
        else:
            summary[name] = evaluate_experiment(name, lang, test_loader)

    print("\nExperiment        dev PPL   test PPL")
    for name, res in summary.items():
        if res is not None:
            print(f"{name:<15} {res['dev_ppl']:>9.2f} {res['test_ppl']:>10.2f}")
