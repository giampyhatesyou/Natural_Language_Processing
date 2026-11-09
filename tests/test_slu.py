import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from nlp import DATA_DIR
from nlp.common import init_weights, sample_std, save_checkpoint
from nlp.slu import cli
from nlp.slu.data import PAD_TOKEN, IntentsAndSlots, collate_fn, create_dev_set, load_data
from nlp.slu.models import JointLSTM
from nlp.slu.training import evaluate, fit

ATIS = DATA_DIR / "atis"


@pytest.fixture(scope="module")
def atis():
    splits = cli.load_splits(ATIS)
    vocab, datasets, collate = cli.prepare(cli.experiment_config("lstm"), splits)
    return splits, vocab, datasets


def loader(dataset, shuffle=False):
    return DataLoader(dataset, batch_size=64, collate_fn=collate_fn, shuffle=shuffle)


def test_dev_split_is_stratified_and_reproducible(atis):
    splits = atis[0]
    assert (len(splits["train"]), len(splits["dev"])) == (4480, 498)
    assert create_dev_set(load_data(ATIS / "train.json"))[1] == splits["dev"]
    assert {x["intent"] for x in splits["dev"]} <= {x["intent"] for x in splits["train"]}


def test_label_ids_do_not_depend_on_the_order_of_a_set(atis):
    # Regression: ids built by iterating a set of strings changed from one Python process to the next
    vocab = atis[1]
    slots = [slot for slot in vocab.slot2id if slot != "pad"]
    assert slots == sorted(slots) and vocab.slot2id["pad"] == PAD_TOKEN
    assert list(vocab.intent2id) == sorted(vocab.intent2id)


def test_dropout_is_applied_before_both_output_layers():
    # Regression: the first version applied dropout to the last hidden state after computing the intent logits.
    # With p = 1 every input of the two output layers is zero, so their logits reduce to the biases.
    model = JointLSTM(20, 8, 8, n_slots=5, n_intents=3, bidirectional=True, dropout=1.0).train()
    slots, intent = model(torch.tensor([[3, 4, 5], [6, 7, 0]]), torch.tensor([3, 2]))
    assert torch.allclose(intent, model.intent_out.bias.expand(2, 3))
    assert torch.allclose(slots.permute(0, 2, 1), model.slot_out.bias.expand(2, 3, 5))


@pytest.mark.parametrize("before, after, changed", [
    ("lstm", "bilstm", {"bidirectional"}),
    ("bilstm", "bilstm_dropout", {"dropout"}),
])
def test_every_experiment_adds_one_technique(before, after, changed):
    a, b = cli.experiment_config(before), cli.experiment_config(after)
    assert {key for key in a.keys() | b.keys() if a.get(key) != b.get(key)} == changed


class Oracle(nn.Module):
    """Returns the gold labels of every batch as one-hot logits."""

    def __init__(self, data, n_slots, n_intents):
        super().__init__()
        self.batches, self.n_slots, self.n_intents = iter(data), n_slots, n_intents

    def forward(self, utterances, lengths):
        batch = next(self.batches)
        labels = batch["slot_labels"].clamp(min=0)  # any class on the padding, which is not evaluated
        slots = nn.functional.one_hot(labels, self.n_slots).float().permute(0, 2, 1)
        return slots, nn.functional.one_hot(batch["intents"], self.n_intents).float()


def test_evaluation_of_an_oracle_scores_one(atis):
    _, vocab, datasets = atis
    data = loader(datasets["test"])
    scores = evaluate(Oracle(data, len(vocab.slot2id), len(vocab.intent2id)), data, vocab.id2slot)
    assert scores["slot_f1"] == 1.0 and scores["intent_acc"] == 1.0


def test_the_best_dev_epoch_is_restored_after_training(atis):
    splits, vocab, _ = atis
    dev = loader(IntentsAndSlots(splits["dev"][:128], vocab))
    torch.manual_seed(0)
    model = JointLSTM(len(vocab.word2id), 32, 32, len(vocab.slot2id), len(vocab.intent2id), bidirectional=True,
                      dropout=0.1)
    init_weights(model)
    stopper, history = fit(model, torch.optim.Adam(model.parameters(), lr=0.01),
                           loader(IntentsAndSlots(splits["train"][:256], vocab), shuffle=True), dev, vocab.id2slot,
                           n_epochs=6, patience=10, clip=5)
    assert stopper.best_score == max(history["dev_slot_f1"]) > 0
    # Regression: the first version evaluated the weights of the last epoch
    assert evaluate(model, dev, vocab.id2slot)["slot_f1"] == pytest.approx(stopper.best_score)


def test_sample_std_divides_by_n_minus_one():
    assert sample_std([1.0, 2.0, 3.0, 4.0]) == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert sample_std([5.0]) == 0.0


def test_command_line_trains_saves_and_reproduces(tmp_path):
    common = ["--exp", "bilstm_dropout", "--checkpoints", str(tmp_path), "--results", str(tmp_path)]
    trained = cli.run(["--train", "--runs", "2", "--epochs", "2", *common])
    record = json.loads((tmp_path / "slu" / "bilstm_dropout.json").read_text())
    assert [run["seed"] for run in record["runs"]] == [42, 43]
    assert trained["bilstm_dropout"]["slot_f1"] == [run["test_slot_f1"] for run in record["runs"]]
    # the checkpoint is the run with the best dev F1, and evaluating it gives back its test scores
    kept = max(record["runs"], key=lambda run: run["dev_slot_f1"])
    evaluated = cli.run(common)["bilstm_dropout"]
    assert evaluated["slot_f1"][0] == pytest.approx(kept["test_slot_f1"])
    assert evaluated["intent_acc"][0] == pytest.approx(kept["test_intent_acc"])


def test_checkpoints_of_the_first_layout_can_be_evaluated(atis, tmp_path):
    # The first version saved the vocabulary as w2id and had no "model" or "batch_size_eval" key in the config
    splits, vocab, datasets = atis
    old_config = {"hid_size": 16, "emb_size": 16, "n_layer": 1, "bidirectional": True, "dropout": 0.0,
                  "batch_size_test": 64}
    torch.manual_seed(0)
    model = cli.build_model({**old_config, "model": "lstm"}, vocab)
    save_checkpoint(tmp_path / "slu" / "bilstm.pt", model, config=old_config, w2id=vocab.word2id,
                    slot2id=vocab.slot2id, intent2id=vocab.intent2id)
    scores = cli.evaluate_checkpoint("bilstm", splits["test"], tmp_path)
    assert scores == evaluate(model, loader(datasets["test"]), vocab.id2slot)
