"""JointBERT pipeline, tested offline with a tiny WordPiece vocabulary and a randomly initialised two-layer BERT."""
from functools import partial

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from helpers import ROOT, load_part

bert = load_part("intent_slot_filling/part_B")
ATIS = ROOT / "intent_slot_filling" / "part_B" / "dataset"


def test_the_label_goes_to_the_first_sub_token_only():
    # [CLS] show bu ##rba ##nk [SEP]: two words, the second one split into three sub-tokens
    word_ids = [None, 0, 1, 1, 1, None]
    assert bert.utils.align_slot_labels(word_ids, [7, 9]) == [-100, 7, 9, -100, -100, -100]


@pytest.fixture(scope="module")
def tiny_bert(tmp_path_factory):
    from tokenizers import BertWordPieceTokenizer
    from transformers import BertConfig, BertModel, BertTokenizerFast

    path = tmp_path_factory.mktemp("tiny_bert")
    wordpiece = BertWordPieceTokenizer(lowercase=True)
    wordpiece.train_from_iterator([x["utterance"] for x in bert.utils.load_data(ATIS / "train.json")],
                                  vocab_size=600, min_frequency=2)
    wordpiece.save_model(str(path))
    tokenizer = BertTokenizerFast(str(path / "vocab.txt"), do_lower_case=True)
    config = BertConfig(vocab_size=tokenizer.vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=128)
    BertModel(config).save_pretrained(path)
    return str(path), tokenizer


@pytest.fixture(scope="module")
def atis_test(tiny_bert):
    test = bert.utils.load_data(ATIS / "test.json")
    lang = bert.utils.Lang({x["intent"] for x in test}, {s for x in test for s in x["slots"].split()})
    return lang, bert.utils.IntentsAndSlots(test, lang, tiny_bert[1])


def collate(tokenizer):
    return partial(bert.utils.collate_fn, pad_token_id=tokenizer.pad_token_id)


def test_every_word_keeps_exactly_its_label(atis_test):
    lang, dataset = atis_test
    split_utterances = 0
    for sample in dataset:
        labels = [lang.id2slot[i] for i in sample["slot_labels"] if i != bert.utils.IGNORE_INDEX]
        assert labels == sample["slots"]
        split_utterances += len(sample["input_ids"]) - 2 > len(sample["words"])
    assert split_utterances > 0  # the small vocabulary splits many words, so the alignment is really exercised


class Oracle(nn.Module):
    """Returns the gold labels as logits on the first sub-tokens and noise on the ignored positions."""

    def __init__(self, data, n_slots, n_intents):
        super().__init__()
        self.batches, self.n_slots, self.n_intents = iter(data), n_slots, n_intents

    def forward(self, input_ids, attention_mask, token_type_ids):
        batch = next(self.batches)
        labels = batch["slot_labels"].clone()
        ignored = labels == bert.utils.IGNORE_INDEX
        labels[ignored] = torch.randint(0, self.n_slots, (int(ignored.sum()),), device=labels.device)
        slots = nn.functional.one_hot(labels, self.n_slots).float().permute(0, 2, 1)
        return slots, nn.functional.one_hot(batch["intents"], self.n_intents).float()


def test_evaluation_of_an_oracle_scores_one(tiny_bert, atis_test):
    lang, dataset = atis_test
    data = DataLoader(dataset, batch_size=64, collate_fn=collate(tiny_bert[1]))
    oracle = Oracle(data, len(lang.slot2id), len(lang.intent2id))
    results, intents, _ = bert.functions.eval_loop(data, nn.CrossEntropyLoss(ignore_index=bert.utils.IGNORE_INDEX),
                                                   nn.CrossEntropyLoss(), oracle, lang)
    assert results["total"]["f"] == 1.0 and intents["accuracy"] == 1.0


def test_joint_bert_starts_from_the_pretrained_encoder(tiny_bert):
    # The pre-trained weights must reach the model untouched: only the two output layers are new
    from transformers import AutoModel

    reference = AutoModel.from_pretrained(tiny_bert[0]).state_dict()
    model = bert.model.JointBERT(tiny_bert[0], out_slot=5, out_int=3)
    for name, tensor in model.bert.state_dict().items():
        assert torch.equal(tensor, reference[name]), name


def test_fine_tuning_restores_the_best_dev_epoch(tiny_bert, atis_test):
    path, tokenizer = tiny_bert
    lang, dataset = atis_test
    utils, f = bert.utils, bert.functions
    train = DataLoader(Subset(dataset, range(64)), batch_size=16, collate_fn=collate(tokenizer), shuffle=True)
    dev = DataLoader(Subset(dataset, range(64, 128)), batch_size=32, collate_fn=collate(tokenizer))
    torch.manual_seed(0)
    model = bert.model.JointBERT(path, len(lang.slot2id), len(lang.intent2id)).to(utils.DEVICE)
    optimizer = f.build_optimizer(model, lr=1e-3, weight_decay=0.01)
    scheduler = f.linear_warmup_scheduler(optimizer, warmup_steps=2, total_steps=3 * len(train))
    criterion_slots, criterion_intents = nn.CrossEntropyLoss(ignore_index=utils.IGNORE_INDEX), nn.CrossEntropyLoss()
    best_f1, history = f.train_model(model, optimizer, scheduler, train, dev, criterion_slots, criterion_intents, lang,
                                     n_epochs=3, patience=3, clip=1.0)
    assert best_f1 == max(history["dev_f1"])
    results, _, _ = f.eval_loop(dev, criterion_slots, criterion_intents, model, lang)
    assert results["total"]["f"] == pytest.approx(best_f1)

    # Evaluation mode of main.py: same architecture built from the configuration, fine-tuned weights loaded
    reloaded = bert.model.JointBERT(path, len(lang.slot2id), len(lang.intent2id), pretrained=False).to(utils.DEVICE)
    reloaded.load_state_dict(model.state_dict())
    reloaded_results, _, _ = f.eval_loop(dev, criterion_slots, criterion_intents, reloaded, lang)
    assert reloaded_results["total"]["f"] == pytest.approx(best_f1)
