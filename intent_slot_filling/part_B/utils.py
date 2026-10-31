# Add functions or classes used for data loading and preprocessing
import json
from collections import Counter

import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

# Device settings
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Label of the positions that must not contribute to the slot loss and to the slot evaluation:
# [CLS], [SEP], padding and every sub-token of a word except the first one.
# -100 is the default ignore_index of nn.CrossEntropyLoss.
IGNORE_INDEX = -100


# Label vocabularies: the words are handled by the BERT tokenizer
class Lang():
    def __init__(self, intents, slots):
        # sorted, so that the ids do not depend on the order of a set
        self.slot2id = {slot: i for i, slot in enumerate(sorted(slots))}
        self.intent2id = {intent: i for i, intent in enumerate(sorted(intents))}
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}


def align_slot_labels(word_ids, word_labels):
    """Move the word-level slot labels to the sub-token level.

    WordPiece can split a word into several sub-tokens (e.g. "burbank" -> "bu", "##rba", "##nk").
    The label of the word goes to its FIRST sub-token only; the other sub-tokens and the special tokens
    get IGNORE_INDEX, so the loss and the evaluation see exactly one prediction per original word.
    word_ids comes from the tokenizer: the index of the word of every sub-token (None for special tokens).
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


class IntentsAndSlots(data.Dataset):
    def __init__(self, dataset, lang, tokenizer):
        self.samples = []
        for x in dataset:
            words = x['utterance'].split()
            slots = x['slots'].split()
            enc = tokenizer(words, is_split_into_words=True)
            slot_labels = align_slot_labels(enc.word_ids(), [lang.slot2id[s] for s in slots])
            if sum(label != IGNORE_INDEX for label in slot_labels) != len(words):
                raise ValueError(f"Some words were lost by the tokenizer: {x['utterance']!r}")
            self.samples.append({
                'input_ids': enc['input_ids'],
                'attention_mask': enc['attention_mask'],
                'token_type_ids': enc.get('token_type_ids', [0] * len(enc['input_ids'])),
                'slot_labels': slot_labels,
                'intent': lang.intent2id[x['intent']],
                'words': words,  # kept for the word-level evaluation
                'slots': slots,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# Pads the sub-token sequences and batches them
def collate_fn(data, pad_token_id):
    max_len = max(len(x['input_ids']) for x in data)

    def pad(key, value):
        return torch.LongTensor([x[key] + [value] * (max_len - len(x[key])) for x in data]).to(DEVICE)

    return {
        'input_ids': pad('input_ids', pad_token_id),
        'attention_mask': pad('attention_mask', 0),  # 0 = padding, BERT does not attend to it
        'token_type_ids': pad('token_type_ids', 0),
        'slot_labels': pad('slot_labels', IGNORE_INDEX),
        'intents': torch.LongTensor([x['intent'] for x in data]).to(DEVICE),
        'words': [x['words'] for x in data],
        'slots': [x['slots'] for x in data],
    }


# Loads data from the provided path
def load_data(path):
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


# Create a dev set: 10% of the training set, stratified on the intents (same split as part A)
def create_dev_set(tmp_train_raw, portion=0.10, seed=42):
    labels = []
    inputs = []
    mini_train = []

    intents = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents)

    for id_y, y in enumerate(intents):
        if count_y[y] > 1:  # stratify needs at least 2 examples per class
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])

    X_train, X_dev, _, _ = train_test_split(inputs, labels, test_size=portion, random_state=seed,
                                            shuffle=True, stratify=labels)
    X_train.extend(mini_train)
    return X_train, X_dev
