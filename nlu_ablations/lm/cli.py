"""Language modelling on the Penn Treebank: the six experiments of the incremental study.

    python -m nlu_ablations.lm                         # evaluate the checkpoints in checkpoints/lm/
    python -m nlu_ablations.lm --train                 # train every experiment
    python -m nlu_ablations.lm --train --exp wt wt_vd  # train some of them
"""
import argparse
import time
from functools import partial
from pathlib import Path

from torch import optim
from torch.utils.data import DataLoader

from .. import CHECKPOINTS_DIR, DATA_DIR, RESULTS_DIR
from ..common import DEVICE, init_weights, load_checkpoint, plot_history, sample_std, save_checkpoint, set_seed
from ..results import save_results
from .data import PennTreeBank, Vocabulary, collate_fn, read_file
from .models import LSTMLanguageModel
from .ntavsgd import NTAvSGD
from .training import evaluate, fit

DEFAULTS = {
    "emb_size": 256, "hid_size": 256, "emb_dropout": 0.0, "out_dropout": 0.0, "variational": False,
    "tie_weights": False, "optimizer": "sgd", "nt_avsgd": False, "batch_size_dev": 128, "batch_size_test": 128,
    "clip": 5, "n_epochs": 100, "patience": 3, "runs": 1, "seed": 42,
}

# Every experiment adds one technique to the best model of the previous one
EXPERIMENTS = {
    # Part A: the RNN of the course baseline replaced by an LSTM, then dropout, then AdamW instead of SGD
    "lstm": {"batch_size_train": 64, "lr": 2.0},
    "lstm_dropout": {"batch_size_train": 64, "lr": 2.0, "emb_dropout": 0.4, "out_dropout": 0.4},
    "lstm_dropout_adamw": {"batch_size_train": 64, "optimizer": "adamw", "lr": 1e-3, "weight_decay": 0.01,
                           "emb_dropout": 0.4, "out_dropout": 0.4},
    # Part B: the AWD-LSTM techniques, with SGD only because NT-AvSGD needs it
    "wt": {"batch_size_train": 32, "lr": 1.5, "tie_weights": True},
    "wt_vd": {"batch_size_train": 32, "lr": 1.5, "tie_weights": True, "variational": True,
              "emb_dropout": 0.4, "out_dropout": 0.4},
    "wt_vd_ntavsgd": {"batch_size_train": 32, "lr": 1.5, "tie_weights": True, "variational": True,
                      "emb_dropout": 0.4, "out_dropout": 0.4, "nt_avsgd": True, "non_mono": 5,
                      "patience": 5},  # patience counted after the trigger, on the averaged weights
}


def experiment_config(name, **overrides):
    config = {**DEFAULTS, **EXPERIMENTS[name]}
    config.update({key: value for key, value in overrides.items() if value is not None})
    return config


def model_config(config):
    """Model switches of a configuration. Checkpoints saved by the first layout of the project, with separate
    part A and part B models, have no switches: part B tied the weights and used variational dropout."""
    if "tie_weights" in config:
        return config
    part_b = "model" not in config
    return {**config, "tie_weights": part_b, "variational": part_b,
            "emb_dropout": config.get("emb_dropout", 0.0), "out_dropout": config.get("out_dropout", 0.0)}


def build_model(config, vocab):
    config = model_config(config)
    return LSTMLanguageModel(len(vocab), config["emb_size"], config["hid_size"], pad_index=vocab.pad_index,
                             emb_dropout=config["emb_dropout"], out_dropout=config["out_dropout"],
                             variational=config["variational"], tie_weights=config["tie_weights"]).to(DEVICE)


def build_optimizer(config, model):
    if config["optimizer"] == "adamw":
        return optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    return optim.SGD(model.parameters(), lr=config["lr"])


def load_corpus(data_dir, subset=None):
    splits = {split: read_file(data_dir / f"ptb.{name}.txt")
              for split, name in (("train", "train"), ("dev", "valid"), ("test", "test"))}
    vocab = Vocabulary(splits["train"])  # always built on the whole training set
    if subset:
        splits = {split: corpus[:subset] for split, corpus in splits.items()}
    return vocab, {split: PennTreeBank(corpus, vocab) for split, corpus in splits.items()}


def loader(dataset, batch_size, vocab, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      collate_fn=partial(collate_fn, pad_index=vocab.pad_index))


def train_experiment(name, config, vocab, datasets, checkpoints_dir, results_dir):
    print(f"\n=== Training '{name}': {config}")
    train = loader(datasets["train"], config["batch_size_train"], vocab, shuffle=True)
    dev = loader(datasets["dev"], config["batch_size_dev"], vocab)
    test = loader(datasets["test"], config["batch_size_test"], vocab)
    runs, best = [], None
    for run in range(config["runs"]):
        seed = config["seed"] + run
        set_seed(seed)
        model = build_model(config, vocab)
        init_weights(model)  # once per run: inside the epoch loop only the gradients are zeroed
        optimizer = build_optimizer(config, model)
        ntasgd = NTAvSGD(model, optimizer, non_mono=config["non_mono"]) if config["nt_avsgd"] else None
        start = time.time()
        stopper, history = fit(model, optimizer, train, dev, vocab.pad_index, config["n_epochs"],
                               config["patience"], config["clip"], ntasgd=ntasgd)
        test_ppl, _ = evaluate(model, test, vocab.pad_index)  # the model holds the selected weights
        result = {"seed": seed, "epochs": len(history["epoch"]), "best_epoch": stopper.best_epoch,
                  "dev_ppl": stopper.best_score, "test_ppl": test_ppl,
                  "trigger_epoch": ntasgd.trigger_epoch if ntasgd is not None else None,
                  "seconds": round(time.time() - start, 1), "history": history}
        runs.append(result)
        print(f"'{name}' run {run}: best dev PPL {result['dev_ppl']:.2f} (epoch {result['best_epoch']}) | "
              f"test PPL {test_ppl:.2f}")
        if best is None or result["dev_ppl"] < best["dev_ppl"]:  # keep the run selected on dev
            best = result
            save_checkpoint(checkpoints_dir / "lm" / f"{name}.pt", model, config=config, word2id=vocab.word2id,
                            results=result)
            marker = (result["trigger_epoch"], "NT-AvSGD trigger") if result["trigger_epoch"] else None
            plot_history(results_dir / "plots" / "lm" / f"{name}.png", f"Language modelling - {name}", history,
                         marker=marker)
    save_results("lm", name, config, runs, results_dir)
    return runs


def evaluate_checkpoint(name, vocab, datasets, checkpoints_dir):
    path = checkpoints_dir / "lm" / f"{name}.pt"
    if not path.exists():
        print(f"'{name}': no checkpoint in {path}, train it with: python -m nlu_ablations.lm --train --exp {name}")
        return None
    checkpoint = load_checkpoint(path)
    if checkpoint["word2id"] != vocab.word2id:
        raise ValueError(f"{path} was trained with a different vocabulary")
    model = build_model(checkpoint["config"], vocab)
    model.load_state_dict(checkpoint["model"])
    test_ppl, _ = evaluate(model, loader(datasets["test"], checkpoint["config"]["batch_size_test"], vocab),
                           vocab.pad_index)
    print(f"'{name}': test PPL {test_ppl:.2f}")
    return {"dev_ppl": checkpoint["results"]["dev_ppl"], "test_ppl": test_ppl}


def run(argv=None):
    """Run the command line and return the test scores of every experiment."""
    parser = argparse.ArgumentParser(prog="python -m nlu_ablations.lm", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", action="store_true", help="train the models (default: evaluate the checkpoints)")
    parser.add_argument("--exp", nargs="+", default=["all"], choices=["all", *EXPERIMENTS])
    parser.add_argument("--epochs", type=int, help="override the maximum number of epochs")
    parser.add_argument("--runs", type=int, help="override the number of runs (seeds)")
    parser.add_argument("--seed", type=int, help="override the seed of the first run")
    parser.add_argument("--subset", type=int, help="debug: use only the first N sentences of every split")
    parser.add_argument("--data", type=Path, default=DATA_DIR / "ptb")
    parser.add_argument("--checkpoints", type=Path, default=CHECKPOINTS_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    vocab, datasets = load_corpus(args.data, args.subset)
    names = list(EXPERIMENTS) if "all" in args.exp else args.exp
    summary = {}
    for name in names:
        if args.train:
            config = experiment_config(name, n_epochs=args.epochs, runs=args.runs, seed=args.seed)
            runs = train_experiment(name, config, vocab, datasets, args.checkpoints, args.results)
            summary[name] = {key: [run[key] for run in runs] for key in ("dev_ppl", "test_ppl")}
        else:
            result = evaluate_checkpoint(name, vocab, datasets, args.checkpoints)
            if result is not None:
                summary[name] = {key: [value] for key, value in result.items()}

    print("\nExperiment            dev PPL          test PPL")
    for name, scores in summary.items():
        cells = [f"{sum(v) / len(v):>7.2f} +- {sample_std(v):5.2f}" for v in (scores["dev_ppl"], scores["test_ppl"])]
        print(f"{name:<20} {cells[0]}  {cells[1]}")
    return summary


def main(argv=None):
    run(argv)
