# This file is used to run your functions and print the results
#
# Usage (from any directory):
#   python main.py                                 # evaluate on the test set the checkpoints stored in bin/
#   python main.py --train                         # train the three incremental experiments (5 runs each)
#   python main.py --train --exp bilstm_dropout    # train a single experiment
import argparse
import os

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from functions import *
from model import *
from utils import *

DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(DIR, "dataset")
BIN_DIR = os.path.join(DIR, "bin")
PLOTS_DIR = os.path.join(DIR, "plots")

# Shared hyperparameters
CONFIG = {
    "batch_size_train": 128,
    "batch_size_dev": 64,
    "batch_size_test": 64,
    "lr": 0.001,
    "hid_size": 200,
    "emb_size": 300,
    "n_layer": 1,
    "clip": 5,
    "n_epochs": 100,
    "patience": 10,  # epochs without improvement of the dev slot F1 (evaluated every epoch)
    "runs": 5,       # independent runs (different seeds), we report mean and std on the test set
    "seed": 42,
}

# Part 2.A is incremental: every experiment starts from the best model of the previous one
EXPERIMENTS = {
    "lstm": {"bidirectional": False, "dropout": 0.0},
    "bilstm": {"bidirectional": True, "dropout": 0.0},
    "bilstm_dropout": {"bidirectional": True, "dropout": 0.3},
}


def build_model(config, lang):
    model = ModelIAS(config["hid_size"], len(lang.slot2id), len(lang.intent2id), config["emb_size"],
                     len(lang.word2id), n_layer=config["n_layer"], pad_index=PAD_TOKEN,
                     bidirectional=config["bidirectional"], dropout=config["dropout"])
    return model.to(DEVICE)


def train_experiment(name, config, lang, train_loader, dev_loader, test_loader):
    print(f"\n=== Training '{name}': {config}")
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()  # Because we do not have the pad token

    runs = []
    best_run = None
    for run in range(config["runs"]):
        # Every run is independent: new seed, new model, new optimizer (this is the ONLY place where
        # the weights are initialised; inside the epoch loop only the gradients are zeroed)
        set_seed(config["seed"] + run)
        model = build_model(config, lang)
        init_weights(model)
        optimizer = optim.Adam(model.parameters(), lr=config["lr"])

        best_dev_f1, history = train_model(model, optimizer, train_loader, dev_loader, criterion_slots,
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
            save_checkpoint(os.path.join(BIN_DIR, f"{name}.pt"), model, config, lang, run_res)
            os.makedirs(PLOTS_DIR, exist_ok=True)
            plot_history(history, f"NLU part A - {name} (run {run})", os.path.join(PLOTS_DIR, f"{name}.png"))

    slot_f1s = np.asarray([r["slot_f1"] for r in runs])
    intent_acc = np.asarray([r["intent_acc"] for r in runs])
    summary = {"slot_f1_mean": slot_f1s.mean(), "slot_f1_std": slot_f1s.std(),
               "intent_acc_mean": intent_acc.mean(), "intent_acc_std": intent_acc.std(), "runs": runs}
    print(f"'{name}' over {len(runs)} runs: slot F1 {slot_f1s.mean():.4f} +- {slot_f1s.std():.4f} | "
          f"intent acc {intent_acc.mean():.4f} +- {intent_acc.std():.4f} "
          f"(checkpoint saved: run {best_run['run']})")
    return summary


def evaluate_experiment(name, test_raw):
    path = os.path.join(BIN_DIR, f"{name}.pt")
    if not os.path.exists(path):
        print(f"'{name}': no checkpoint in {path}, train it with: python main.py --train --exp {name}")
        return None
    checkpoint = load_checkpoint(path, DEVICE)

    # Recover the vocabularies used at training time
    lang = Lang(words=[], intents=[], slots=[], cutoff=0)
    lang.word2id = checkpoint["w2id"]
    lang.slot2id = checkpoint["slot2id"]
    lang.intent2id = checkpoint["intent2id"]
    lang.id2word = {v: k for k, v in lang.word2id.items()}
    lang.id2slot = {v: k for k, v in lang.slot2id.items()}
    lang.id2intent = {v: k for k, v in lang.intent2id.items()}

    test_loader = DataLoader(IntentsAndSlots(test_raw, lang), batch_size=CONFIG['batch_size_test'],
                             collate_fn=collate_fn)
    model = build_model(checkpoint["config"], lang)  # same hyperparameters used for training
    model.load_state_dict(checkpoint["model"])
    results_test, intent_test, _ = eval_loop(test_loader, nn.CrossEntropyLoss(ignore_index=PAD_TOKEN),
                                             nn.CrossEntropyLoss(), model, lang)
    print(f"'{name}': test slot F1 {results_test['total']['f']:.4f} | "
          f"test intent acc {intent_test['accuracy']:.4f}")
    return {"slot_f1": results_test['total']['f'], "intent_acc": intent_test['accuracy']}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLU project - intent classification and slot filling, part A")
    parser.add_argument("--train", action="store_true", help="train the models (default: only evaluate bin/)")
    parser.add_argument("--exp", default="all", choices=["all"] + list(EXPERIMENTS))
    parser.add_argument("--runs", type=int, default=None, help="override the number of runs")
    parser.add_argument("--epochs", type=int, default=None, help="override the maximum number of epochs")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    tmp_train_raw = load_data(os.path.join(DATASET_DIR, 'train.json'))
    test_raw = load_data(os.path.join(DATASET_DIR, 'test.json'))
    names = list(EXPERIMENTS) if args.exp == "all" else [args.exp]

    if not args.train:
        for name in names:
            evaluate_experiment(name, test_raw)
        raise SystemExit(0)

    train_raw, dev_raw = create_dev_set(tmp_train_raw)

    words = sum([x['utterance'].split() for x in train_raw], [])  # No set() since we want to compute the cutoff
    corpus = train_raw + dev_raw + test_raw  # We do not want unk labels, however this depends on the research purpose
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])
    lang = Lang(words, intents, slots, cutoff=0)

    # Create our datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    # Dataloader instantiations
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size_train'], collate_fn=collate_fn,
                              shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=CONFIG['batch_size_dev'], collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size_test'], collate_fn=collate_fn)

    summary = {}
    for name in names:
        config = {**CONFIG, **EXPERIMENTS[name]}
        for key in ("runs", "seed"):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        if args.epochs is not None:
            config["n_epochs"] = args.epochs
        summary[name] = train_experiment(name, config, lang, train_loader, dev_loader, test_loader)

    print("\nExperiment       slot F1 (mean +- std)   intent acc (mean +- std)")
    for name, s in summary.items():
        print(f"{name:<16} {s['slot_f1_mean']:.4f} +- {s['slot_f1_std']:.4f}       "
              f"{s['intent_acc_mean']:.4f} +- {s['intent_acc_std']:.4f}")
