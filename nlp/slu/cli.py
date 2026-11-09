"""Joint intent classification and slot filling on ATIS: the four models of the incremental study.

    python -m nlp.slu                      # evaluate the checkpoints in checkpoints/slu/
    python -m nlp.slu --train              # train every model (5 runs each)
    python -m nlp.slu --train --exp bert   # fine-tune BERT only
"""
import argparse
import time
from functools import partial
from pathlib import Path

from torch import optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .. import CHECKPOINTS_DIR, DATA_DIR, RESULTS_DIR
from ..common import DEVICE, init_weights, load_checkpoint, plot_history, sample_std, save_checkpoint, set_seed
from ..results import save_results
from . import bert_data
from .bert_data import BertIntentsAndSlots, LabelVocabulary
from .data import PAD_TOKEN, IntentsAndSlots, Vocabulary, collate_fn, create_dev_set, load_data
from .models import JointBERT, JointLSTM
from .training import bert_optimizer, evaluate, fit, linear_warmup

LSTM = {"model": "lstm", "emb_size": 300, "hid_size": 200, "n_layer": 1, "bidirectional": False, "dropout": 0.0,
        "lr": 0.001, "batch_size_train": 128, "batch_size_eval": 64, "clip": 5, "n_epochs": 100,
        "patience": 10, "runs": 5, "seed": 42}
BERT = {"model": "bert", "bert_model": "bert-base-uncased", "dropout": 0.1, "lr": 5e-5, "weight_decay": 0.01,
        "warmup": 0.1, "batch_size_train": 32, "batch_size_eval": 64, "clip": 1.0, "n_epochs": 10, "patience": 3,
        "runs": 5, "seed": 42}

# Every experiment adds one change to the best model of the previous one
EXPERIMENTS = {
    "lstm": LSTM,
    "bilstm": {**LSTM, "bidirectional": True},
    "bilstm_dropout": {**LSTM, "bidirectional": True, "dropout": 0.3},
    "bert": BERT,
}


def family(config):
    # Configurations saved by the first layout of the project have no "model" key
    return config.get("model") or ("bert" if "bert_model" in config else "lstm")


def experiment_config(name, bert_model=None, **overrides):
    config = {**EXPERIMENTS[name]}
    if family(config) == "bert" and bert_model is not None:
        config["bert_model"] = bert_model
    config.update({key: value for key, value in overrides.items() if value is not None})
    return config


def load_splits(data_dir):
    train, dev = create_dev_set(load_data(data_dir / "train.json"))
    return {"train": train, "dev": dev, "test": load_data(data_dir / "test.json")}


def prepare(config, splits):
    """Vocabulary, datasets and collate function of a model family."""
    corpus = [x for split in splits.values() for x in split]
    intents = {x["intent"] for x in corpus}
    slots = {slot for x in corpus for slot in x["slots"].split()}  # labels from every split: none is unknown
    if family(config) == "bert":
        tokenizer = AutoTokenizer.from_pretrained(config["bert_model"])
        vocab = LabelVocabulary(intents, slots)
        datasets = {split: BertIntentsAndSlots(raw, vocab, tokenizer) for split, raw in splits.items()}
        return vocab, datasets, partial(bert_data.collate_fn, pad_token_id=tokenizer.pad_token_id)
    words = [word for x in splits["train"] for word in x["utterance"].split()]
    vocab = Vocabulary(words, intents, slots)
    return vocab, {split: IntentsAndSlots(raw, vocab) for split, raw in splits.items()}, collate_fn


def build_model(config, vocab, pretrained=True):
    if family(config) == "bert":
        return JointBERT(config["bert_model"], len(vocab.slot2id), len(vocab.intent2id), dropout=config["dropout"],
                         pretrained=pretrained).to(DEVICE)
    return JointLSTM(len(vocab.word2id), config["emb_size"], config["hid_size"], len(vocab.slot2id),
                     len(vocab.intent2id), n_layers=config["n_layer"], pad_index=PAD_TOKEN,
                     bidirectional=config["bidirectional"], dropout=config["dropout"]).to(DEVICE)


def build_optimizer(config, model, steps_per_epoch):
    if family(config) == "bert":
        optimizer = bert_optimizer(model, config["lr"], config["weight_decay"])
        total_steps = config["n_epochs"] * steps_per_epoch
        return optimizer, linear_warmup(optimizer, int(config["warmup"] * total_steps), total_steps)
    return optim.Adam(model.parameters(), lr=config["lr"]), None


def vocabulary_payload(vocab):
    payload = {"slot2id": vocab.slot2id, "intent2id": vocab.intent2id}
    if hasattr(vocab, "word2id"):
        payload["word2id"] = vocab.word2id
    return payload


def train_experiment(name, config, vocab, datasets, collate, checkpoints_dir, results_dir):
    print(f"\n=== Training '{name}': {config}")
    train = DataLoader(datasets["train"], batch_size=config["batch_size_train"], shuffle=True, collate_fn=collate)
    dev = DataLoader(datasets["dev"], batch_size=config["batch_size_eval"], collate_fn=collate)
    test = DataLoader(datasets["test"], batch_size=config["batch_size_eval"], collate_fn=collate)
    runs, best = [], None
    for run in range(config["runs"]):
        # Every run is independent: new seed, new model, new optimizer. This is the only place where the weights
        # are created; inside the epoch loop only the gradients are zeroed.
        seed = config["seed"] + run
        set_seed(seed)
        model = build_model(config, vocab)
        if family(config) == "lstm":
            init_weights(model)  # never on BERT: it would overwrite the pre-trained weights
        optimizer, scheduler = build_optimizer(config, model, len(train))
        start = time.time()
        stopper, history = fit(model, optimizer, train, dev, vocab.id2slot, config["n_epochs"], config["patience"],
                               config["clip"], scheduler)
        scores = evaluate(model, test, vocab.id2slot)  # the model holds the weights of the best dev epoch
        result = {"seed": seed, "epochs": len(history["epoch"]), "best_epoch": stopper.best_epoch,
                  "dev_slot_f1": stopper.best_score, "test_slot_f1": scores["slot_f1"],
                  "test_intent_acc": scores["intent_acc"], "seconds": round(time.time() - start, 1),
                  "history": history}
        runs.append(result)
        print(f"run {run}: epochs {result['epochs']} | dev slot F1 {result['dev_slot_f1']:.4f} | "
              f"test slot F1 {result['test_slot_f1']:.4f} | test intent acc {result['test_intent_acc']:.4f}")
        if best is None or result["dev_slot_f1"] > best["dev_slot_f1"]:  # the run selected on dev is kept
            best = result
            save_checkpoint(checkpoints_dir / "slu" / f"{name}.pt", model, config=config,
                            **vocabulary_payload(vocab), results=result)
            plot_history(results_dir / "plots" / "slu" / f"{name}.png", f"Intent and slot filling - {name}",
                         history, metric=("dev_slot_f1", "dev slot F1"))
        del model, optimizer, scheduler  # free the GPU memory before the next run
    save_results("slu", name, config, runs, results_dir)
    return runs


def evaluate_checkpoint(name, test_raw, checkpoints_dir):
    path = checkpoints_dir / "slu" / f"{name}.pt"
    if not path.exists():
        print(f"'{name}': no checkpoint in {path}, train it with: python -m nlp.slu --train --exp {name}")
        return None
    checkpoint = load_checkpoint(path)
    config = checkpoint["config"]
    if family(config) == "bert":
        vocab = LabelVocabulary.from_dicts(checkpoint["slot2id"], checkpoint["intent2id"])
        tokenizer = AutoTokenizer.from_pretrained(config["bert_model"])
        dataset = BertIntentsAndSlots(test_raw, vocab, tokenizer)
        collate = partial(bert_data.collate_fn, pad_token_id=tokenizer.pad_token_id)
        model = build_model(config, vocab, pretrained=False)  # the fine-tuned weights come from the checkpoint
    else:
        word2id = checkpoint["word2id"] if "word2id" in checkpoint else checkpoint["w2id"]  # w2id: first layout
        vocab = Vocabulary.from_dicts(word2id, checkpoint["slot2id"], checkpoint["intent2id"])
        dataset, collate = IntentsAndSlots(test_raw, vocab), collate_fn
        model = build_model(config, vocab)
    model.load_state_dict(checkpoint["model"])
    batch_size = config.get("batch_size_eval", config.get("batch_size_test", 64))
    scores = evaluate(model, DataLoader(dataset, batch_size=batch_size, collate_fn=collate), vocab.id2slot)
    print(f"'{name}': test slot F1 {scores['slot_f1']:.4f} | test intent acc {scores['intent_acc']:.4f}")
    return scores


def run(argv=None):
    """Run the command line and return the test scores of every experiment."""
    parser = argparse.ArgumentParser(prog="python -m nlp.slu", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", action="store_true", help="train the models (default: evaluate the checkpoints)")
    parser.add_argument("--exp", nargs="+", default=["all"], choices=["all", *EXPERIMENTS])
    parser.add_argument("--runs", type=int, help="override the number of runs (seeds)")
    parser.add_argument("--epochs", type=int, help="override the maximum number of epochs")
    parser.add_argument("--seed", type=int, help="override the seed of the first run")
    parser.add_argument("--lr", type=float, help="override the learning rate")
    parser.add_argument("--bert_model", help="Hugging Face name or local folder of the BERT model")
    parser.add_argument("--data", type=Path, default=DATA_DIR / "atis")
    parser.add_argument("--checkpoints", type=Path, default=CHECKPOINTS_DIR)
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    splits = load_splits(args.data)
    names = list(EXPERIMENTS) if "all" in args.exp else args.exp
    prepared, summary = {}, {}
    for name in names:
        if args.train:
            config = experiment_config(name, bert_model=args.bert_model, runs=args.runs, n_epochs=args.epochs,
                                       seed=args.seed, lr=args.lr)
            key = (family(config), config.get("bert_model"))
            if key not in prepared:  # the models of a family share vocabulary and datasets
                prepared[key] = prepare(config, splits)
            runs = train_experiment(name, config, *prepared[key], args.checkpoints, args.results)
            summary[name] = {key: [run[f"test_{key}"] for run in runs] for key in ("slot_f1", "intent_acc")}
        else:
            scores = evaluate_checkpoint(name, splits["test"], args.checkpoints)
            if scores is not None:
                summary[name] = {key: [scores[key]] for key in ("slot_f1", "intent_acc")}

    print("\nExperiment        test slot F1 (mean +- std)   test intent acc (mean +- std)")
    for name, scores in summary.items():
        cells = [f"{sum(v) / len(v):.4f} +- {sample_std(v):.4f}" for v in (scores["slot_f1"], scores["intent_acc"])]
        print(f"{name:<16}  {cells[0]}           {cells[1]}")
    return summary


def main(argv=None):
    run(argv)
