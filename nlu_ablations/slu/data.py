"""ATIS: loading, dev split, and the word-level datasets of the LSTM models.

Every batch, for the LSTM and the BERT models alike, holds the model inputs under "inputs", the slot targets
("slot_labels", IGNORE_INDEX where a position has no word-level label), the intents, and the original words and
slot labels used by the evaluation.
"""
import json
from collections import Counter

import torch
from sklearn.model_selection import train_test_split
from torch.utils import data

from ..common import DEVICE

PAD_TOKEN = 0         # id of the padding word (and of the 'pad' slot class of the LSTM models)
IGNORE_INDEX = -100   # slot target of the positions without a word-level label, skipped by the loss


def load_data(path):
    with open(path) as f:
        return json.load(f)


def create_dev_set(train_raw, portion=0.10, seed=42):
    """Hold out a dev set stratified by intent; intents with a single example stay in the training set."""
    counts = Counter(x["intent"] for x in train_raw)
    stratifiable = [x for x in train_raw if counts[x["intent"]] > 1]
    singletons = [x for x in train_raw if counts[x["intent"]] == 1]
    train, dev = train_test_split(stratifiable, test_size=portion, random_state=seed, shuffle=True,
                                  stratify=[x["intent"] for x in stratifiable])
    return train + singletons, dev


class Vocabulary:
    """Word, slot and intent ids of the LSTM models. Words come from the training set (plus 'pad' and 'unk'),
    labels from every split so that no label is unknown. Label ids are sorted, hence the same in every execution."""

    def __init__(self, words, intents, slots, cutoff=0):
        word2id = {"pad": PAD_TOKEN, "unk": 1}
        for word, count in Counter(words).items():
            if count > cutoff:
                word2id[word] = len(word2id)
        slot2id = {"pad": PAD_TOKEN}
        for slot in sorted(slots):
            slot2id[slot] = len(slot2id)
        self.set(word2id, slot2id, {intent: i for i, intent in enumerate(sorted(intents))})

    @classmethod
    def from_dicts(cls, word2id, slot2id, intent2id):
        """Rebuild the vocabulary stored in a checkpoint."""
        vocab = cls.__new__(cls)
        vocab.set(word2id, slot2id, intent2id)
        return vocab

    def set(self, word2id, slot2id, intent2id):
        self.word2id, self.slot2id, self.intent2id = word2id, slot2id, intent2id
        self.id2word = {i: w for w, i in word2id.items()}
        self.id2slot = {i: s for s, i in slot2id.items()}
        self.id2intent = {i: x for x, i in intent2id.items()}


class IntentsAndSlots(data.Dataset):
    """Word-level samples: word ids, one slot id per word, intent id."""

    def __init__(self, dataset, vocab):
        unk = vocab.word2id["unk"]
        self.samples = []
        for x in dataset:
            words, slots = x["utterance"].split(), x["slots"].split()
            self.samples.append({"word_ids": [vocab.word2id.get(word, unk) for word in words],
                                 "slot_ids": [vocab.slot2id[slot] for slot in slots],
                                 "intent": vocab.intent2id[x["intent"]], "words": words, "slots": slots})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Pad to the longest utterance; sorted by length, as pack_padded_sequence requires."""
    batch = sorted(batch, key=lambda x: len(x["word_ids"]), reverse=True)
    lengths = [len(x["word_ids"]) for x in batch]
    utterances = torch.full((len(batch), max(1, lengths[0])), PAD_TOKEN, dtype=torch.long)
    slot_labels = torch.full_like(utterances, IGNORE_INDEX)
    for i, x in enumerate(batch):
        utterances[i, :lengths[i]] = torch.tensor(x["word_ids"])
        slot_labels[i, :lengths[i]] = torch.tensor(x["slot_ids"])
    return {"inputs": {"utterances": utterances.to(DEVICE), "lengths": torch.tensor(lengths)},
            "slot_labels": slot_labels.to(DEVICE),
            "intents": torch.tensor([x["intent"] for x in batch]).to(DEVICE),
            "words": [x["words"] for x in batch], "slots": [x["slots"] for x in batch]}
