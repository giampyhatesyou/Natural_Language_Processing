"""Penn Treebank: one sentence per sample, ending with an end-of-sentence token, padded per batch."""
import torch
from torch.utils import data

from ..common import DEVICE

PAD, EOS = "<pad>", "<eos>"


def read_file(path, eos_token=EOS):
    with open(path) as f:
        return [line.strip() + " " + eos_token for line in f]


class Vocabulary:
    """Word <-> id mapping of the training corpus, special tokens first."""

    def __init__(self, corpus, special_tokens=(PAD, EOS)):
        self.word2id = {}
        for token in special_tokens:
            self.word2id[token] = len(self.word2id)
        for sentence in corpus:
            for word in sentence.split():
                if word not in self.word2id:
                    self.word2id[word] = len(self.word2id)
        self.id2word = {i: word for word, i in self.word2id.items()}

    @property
    def pad_index(self):
        return self.word2id[PAD]

    def __len__(self):
        return len(self.word2id)


class PennTreeBank(data.Dataset):
    """The model reads every token but the last one and predicts every token but the first one."""

    def __init__(self, corpus, vocab):
        self.samples = []
        for sentence in corpus:
            ids = [self.word_id(word, vocab) for word in sentence.split()]
            self.samples.append((ids[:-1], ids[1:]))

    @staticmethod
    def word_id(word, vocab):
        if word not in vocab.word2id:
            # PTB is already preprocessed with <unk>: an unknown word means a wrong file or vocabulary
            raise ValueError(f"OOV word found: {word!r}")
        return vocab.word2id[word]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        source, target = self.samples[idx]
        return {"source": torch.LongTensor(source), "target": torch.LongTensor(target)}


def collate_fn(batch, pad_index):
    """Pad source and target to the longest sentence of the batch.

    Sorting by length does not change the loss, but it decides which dropout mask each sentence gets: it is kept so
    that a run with the same seed reproduces the reported results exactly."""
    batch = sorted(batch, key=lambda sample: len(sample["source"]), reverse=True)
    max_len = max(1, len(batch[0]["source"]))
    source = torch.full((len(batch), max_len), pad_index, dtype=torch.long)
    target = torch.full((len(batch), max_len), pad_index, dtype=torch.long)
    for i, sample in enumerate(batch):
        source[i, :len(sample["source"])] = sample["source"]
        target[i, :len(sample["target"])] = sample["target"]
    return {"source": source.to(DEVICE), "target": target.to(DEVICE),
            "number_tokens": sum(len(sample["target"]) for sample in batch)}
