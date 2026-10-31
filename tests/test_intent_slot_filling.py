import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from helpers import ROOT, load_part

nlu = load_part("intent_slot_filling/part_A")
ATIS = ROOT / "intent_slot_filling" / "part_A" / "dataset"


@pytest.fixture(scope="module")
def atis():
    utils = nlu.utils
    train_raw, dev_raw = utils.create_dev_set(utils.load_data(ATIS / "train.json"))
    test_raw = utils.load_data(ATIS / "test.json")
    words = sum([x["utterance"].split() for x in train_raw], [])
    corpus = train_raw + dev_raw + test_raw
    lang = utils.Lang(words, {x["intent"] for x in corpus}, {s for x in corpus for s in x["slots"].split()})
    return train_raw, dev_raw, test_raw, lang


def loader(raw, lang, shuffle=False):
    return DataLoader(nlu.utils.IntentsAndSlots(raw, lang), batch_size=64, collate_fn=nlu.utils.collate_fn,
                      shuffle=shuffle)


def test_dev_split_is_stratified_and_reproducible(atis):
    train_raw, dev_raw, _, _ = atis
    assert (len(train_raw), len(dev_raw)) == (4480, 498)
    assert nlu.utils.create_dev_set(nlu.utils.load_data(ATIS / "train.json"))[1] == dev_raw
    assert {x["intent"] for x in dev_raw} <= {x["intent"] for x in train_raw}


def test_label_ids_do_not_depend_on_the_order_of_a_set(atis):
    # Regression: ids built by iterating a set of strings changed from one Python process to the next
    lang = atis[3]
    slots = [slot for slot in lang.slot2id if slot != "pad"]
    assert slots == sorted(slots) and lang.slot2id["pad"] == nlu.utils.PAD_TOKEN
    assert list(lang.intent2id) == sorted(lang.intent2id)


def test_dropout_is_applied_before_both_output_layers():
    # Regression: the first version applied dropout to the last hidden state after computing the intent logits.
    # With p = 1 every input of the two output layers is zero, so their logits reduce to the biases.
    model = nlu.model.ModelIAS(8, out_slot=5, out_int=3, emb_size=8, vocab_len=20, bidirectional=True,
                               dropout=1.0).train()
    slots, intent = model(torch.tensor([[3, 4, 5], [6, 7, 0]]), torch.tensor([3, 2]))
    assert torch.allclose(intent, model.intent_out.bias.expand(2, 3))
    assert torch.allclose(slots.permute(0, 2, 1), model.slot_out.bias.expand(2, 3, 5))


class Oracle(nn.Module):
    """Returns the gold labels of every batch as one-hot logits."""

    def __init__(self, data, n_slots, n_intents):
        super().__init__()
        self.batches, self.n_slots, self.n_intents = iter(data), n_slots, n_intents

    def forward(self, utterances, seq_lengths):
        batch = next(self.batches)
        slots = nn.functional.one_hot(batch["y_slots"], self.n_slots).float().permute(0, 2, 1)
        return slots, nn.functional.one_hot(batch["intents"], self.n_intents).float()


def test_evaluation_of_an_oracle_scores_one(atis):
    _, _, test_raw, lang = atis
    data = loader(test_raw, lang)
    oracle = Oracle(data, len(lang.slot2id), len(lang.intent2id))
    results, intents, _ = nlu.functions.eval_loop(data, nn.CrossEntropyLoss(ignore_index=nlu.utils.PAD_TOKEN),
                                                  nn.CrossEntropyLoss(), oracle, lang)
    assert results["total"]["f"] == 1.0 and intents["accuracy"] == 1.0


def test_the_best_dev_epoch_is_restored_after_training(atis):
    train_raw, dev_raw, _, lang = atis
    utils, f = nlu.utils, nlu.functions
    dev = loader(dev_raw[:128], lang)
    torch.manual_seed(0)
    model = nlu.model.ModelIAS(32, len(lang.slot2id), len(lang.intent2id), 32, len(lang.word2id),
                               bidirectional=True, dropout=0.1).to(utils.DEVICE)
    f.init_weights(model)
    criterion_slots, criterion_intents = nn.CrossEntropyLoss(ignore_index=utils.PAD_TOKEN), nn.CrossEntropyLoss()
    best_f1, history = f.train_model(model, torch.optim.Adam(model.parameters(), lr=0.01),
                                     loader(train_raw[:256], lang, shuffle=True), dev, criterion_slots,
                                     criterion_intents, lang, n_epochs=6, patience=10, clip=5)
    assert best_f1 == max(history["dev_f1"]) > 0
    # Regression: the first version evaluated the weights of the last epoch
    results, _, _ = f.eval_loop(dev, criterion_slots, criterion_intents, model, lang)
    assert results["total"]["f"] == pytest.approx(best_f1)


def test_sample_std_divides_by_n_minus_one():
    assert nlu.functions.sample_std([1.0, 2.0, 3.0, 4.0]) == pytest.approx(np.std([1, 2, 3, 4], ddof=1))
    assert nlu.functions.sample_std([5.0]) == 0.0
