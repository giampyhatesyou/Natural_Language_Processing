"""Sub-token datasets for BERT: the slot label of a word goes to its first WordPiece sub-token."""
import torch
from torch.utils import data

from ..common import DEVICE
from .data import IGNORE_INDEX


class LabelVocabulary:
    """Slot and intent ids of the BERT model; its tokenizer handles the words. Ids are sorted."""

    def __init__(self, intents, slots):
        self.set({slot: i for i, slot in enumerate(sorted(slots))},
                 {intent: i for i, intent in enumerate(sorted(intents))})

    @classmethod
    def from_dicts(cls, slot2id, intent2id):
        """Rebuild the vocabulary stored in a checkpoint."""
        vocab = cls.__new__(cls)
        vocab.set(slot2id, intent2id)
        return vocab

    def set(self, slot2id, intent2id):
        self.slot2id, self.intent2id = slot2id, intent2id
        self.id2slot = {i: s for s, i in slot2id.items()}
        self.id2intent = {i: x for x, i in intent2id.items()}


def align_slot_labels(word_ids, word_labels):
    """Move the word-level slot labels to the sub-token level.

    WordPiece can split a word into several sub-tokens (e.g. "burbank" -> "bu", "##rba", "##nk"). The label of the
    word goes to its first sub-token only; the other sub-tokens and the special tokens get IGNORE_INDEX, so the loss
    and the evaluation see exactly one prediction per original word. word_ids comes from the tokenizer: the index
    of the word of every sub-token (None for the special tokens).
    """
    labels = []
    previous = None
    for word_id in word_ids:
        if word_id is None or word_id == previous:
            labels.append(IGNORE_INDEX)
        else:
            labels.append(word_labels[word_id])
        previous = word_id
    return labels


class BertIntentsAndSlots(data.Dataset):
    def __init__(self, dataset, vocab, tokenizer):
        self.samples = []
        for x in dataset:
            words, slots = x["utterance"].split(), x["slots"].split()
            encoding = tokenizer(words, is_split_into_words=True)
            slot_labels = align_slot_labels(encoding.word_ids(), [vocab.slot2id[s] for s in slots])
            if sum(label != IGNORE_INDEX for label in slot_labels) != len(words):
                raise ValueError(f"Some words were lost by the tokenizer: {x['utterance']!r}")
            self.samples.append({
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
                "token_type_ids": encoding.get("token_type_ids", [0] * len(encoding["input_ids"])),
                "slot_labels": slot_labels, "intent": vocab.intent2id[x["intent"]], "words": words, "slots": slots,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, pad_token_id):
    """Pad the sub-token sequences; padding is masked out of the attention and of the slot loss."""
    max_len = max(len(x["input_ids"]) for x in batch)

    def pad(key, value):
        return torch.tensor([x[key] + [value] * (max_len - len(x[key])) for x in batch], dtype=torch.long).to(DEVICE)

    return {"inputs": {"input_ids": pad("input_ids", pad_token_id), "attention_mask": pad("attention_mask", 0),
                       "token_type_ids": pad("token_type_ids", 0)},
            "slot_labels": pad("slot_labels", IGNORE_INDEX),
            "intents": torch.tensor([x["intent"] for x in batch]).to(DEVICE),
            "words": [x["words"] for x in batch], "slots": [x["slots"] for x in batch]}
