# Add functions or classes used for data loading and preprocessing
import json
from collections import Counter

import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

# Device settings
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Pad token for vocabulary preparation
PAD_TOKEN = 0


# Computes and stores the vocabulary
class Lang():
    def __init__(self, words, intents, slots, cutoff=0):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True)
        self.slot2id = self.lab2id(slots)
        self.intent2id = self.lab2id(intents, pad=False)
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.id2slot = {v: k for k, v in self.slot2id.items()}
        self.id2intent = {v: k for k, v in self.intent2id.items()}

    # Create word2id dictionary
    def w2id(self, elements, cutoff=None, unk=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab

    # Create lab2id dictionary (sorted, so that the ids do not depend on the order of a set)
    def lab2id(self, elements, pad=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in sorted(elements):
            vocab[elem] = len(vocab)
        return vocab


# Provides ID versions of the datasets
class IntentsAndSlots(data.Dataset):
    def __init__(self, dataset, lang, unk='unk'):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk

        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.LongTensor(self.utt_ids[idx])
        slots = torch.LongTensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample

    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]

    def mapping_seq(self, data, mapper):  # Maps sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            res.append(tmp_seq)
        return res


# Pads sequences and batches them
def collate_fn(data):
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths) == 0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape
        # batch_size X maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences), max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq  # We copy each sequence into the matrix
        padded_seqs = padded_seqs.detach()  # We remove these tensors from the computational graph
        return padded_seqs, lengths

    # Sort data by seq lengths (needed by pack_padded_sequence)
    data.sort(key=lambda x: len(x['utterance']), reverse=True)
    new_item = {}
    for key in data[0].keys():
        new_item[key] = [d[key] for d in data]

    src_utt, _ = merge(new_item['utterance'])
    y_slots, y_lengths = merge(new_item["slots"])
    intent = torch.LongTensor(new_item["intent"])

    new_item["utterances"] = src_utt.to(DEVICE)
    new_item["intents"] = intent.to(DEVICE)
    new_item["y_slots"] = y_slots.to(DEVICE)
    new_item["slots_len"] = torch.LongTensor(y_lengths).to(DEVICE)
    return new_item


# Loads data from the provided path
def load_data(path):
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


# Create a dev set: 10% of the training set, stratified on the intents
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
