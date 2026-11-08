"""JointBERT pipeline, tested offline with a tiny WordPiece vocabulary and a randomly initialised two-layer BERT."""
import json
from functools import partial

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from nlu_ablations import DATA_DIR
from nlu_ablations.common import load_checkpoint
from nlu_ablations.slu import bert_data, cli
from nlu_ablations.slu.bert_data import BertIntentsAndSlots, LabelVocabulary, align_slot_labels
from nlu_ablations.slu.data import IGNORE_INDEX, load_data
from nlu_ablations.slu.models import JointBERT
from nlu_ablations.slu.training import bert_optimizer, evaluate, fit, linear_warmup

ATIS = DATA_DIR / "atis"


def test_the_label_goes_to_the_first_sub_token_only():
    # [CLS] show bu ##rba ##nk [SEP]: two words, the second one split into three sub-tokens
    word_ids = [None, 0, 1, 1, 1, None]
    assert align_slot_labels(word_ids, [7, 9]) == [-100, 7, 9, -100, -100, -100]


@pytest.fixture(scope="module")
def tiny_bert(tmp_path_factory):
    from tokenizers import BertWordPieceTokenizer
    from transformers import BertConfig, BertModel, BertTokenizerFast

    path = tmp_path_factory.mktemp("tiny_bert")
    wordpiece = BertWordPieceTokenizer(lowercase=True)
    wordpiece.train_from_iterator([x["utterance"] for x in load_data(ATIS / "train.json")], vocab_size=600,
                                  min_frequency=2)
    wordpiece.save_model(str(path))
    tokenizer = BertTokenizerFast(str(path / "vocab.txt"), do_lower_case=True)
    tokenizer.save_pretrained(path)
    config = BertConfig(vocab_size=tokenizer.vocab_size, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=64, max_position_embeddings=128)
    BertModel(config).save_pretrained(path)
    return str(path), tokenizer


@pytest.fixture(scope="module")
def atis_test(tiny_bert):
    test = load_data(ATIS / "test.json")
    vocab = LabelVocabulary({x["intent"] for x in test}, {s for x in test for s in x["slots"].split()})
    return vocab, BertIntentsAndSlots(test, vocab, tiny_bert[1])


def collate(tokenizer):
    return partial(bert_data.collate_fn, pad_token_id=tokenizer.pad_token_id)


def test_every_word_keeps_exactly_its_label(atis_test):
    vocab, dataset = atis_test
    split_utterances = 0
    for sample in dataset:
        labels = [vocab.id2slot[i] for i in sample["slot_labels"] if i != IGNORE_INDEX]
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
        ignored = labels == IGNORE_INDEX
        labels[ignored] = torch.randint(0, self.n_slots, (int(ignored.sum()),), device=labels.device)
        slots = nn.functional.one_hot(labels, self.n_slots).float().permute(0, 2, 1)
        return slots, nn.functional.one_hot(batch["intents"], self.n_intents).float()


def test_evaluation_of_an_oracle_scores_one(tiny_bert, atis_test):
    vocab, dataset = atis_test
    data = DataLoader(dataset, batch_size=64, collate_fn=collate(tiny_bert[1]))
    scores = evaluate(Oracle(data, len(vocab.slot2id), len(vocab.intent2id)), data, vocab.id2slot)
    assert scores["slot_f1"] == 1.0 and scores["intent_acc"] == 1.0


def test_joint_bert_starts_from_the_pretrained_encoder(tiny_bert):
    # The pre-trained weights must reach the model untouched: only the two output layers are new
    from transformers import AutoModel

    reference = AutoModel.from_pretrained(tiny_bert[0]).state_dict()
    model = JointBERT(tiny_bert[0], n_slots=5, n_intents=3)
    for name, tensor in model.bert.state_dict().items():
        assert torch.equal(tensor, reference[name]), name


def test_training_never_reinitialises_the_pretrained_encoder(tiny_bert, tmp_path):
    # The BERT of the first version scored like a randomly initialised model. With a learning rate of 0 the optimiser
    # cannot move the weights, so the encoder saved after training must still be the pre-trained one: nothing in the
    # training pipeline (such as the initialisation of the LSTM models) may overwrite it
    from transformers import AutoModel

    cli.run(["--train", "--exp", "bert", "--bert_model", tiny_bert[0], "--lr", "0", "--runs", "1", "--epochs", "2",
             "--checkpoints", str(tmp_path), "--results", str(tmp_path)])
    saved = load_checkpoint(tmp_path / "slu" / "bert.pt")["model"]
    for name, tensor in AutoModel.from_pretrained(tiny_bert[0]).state_dict().items():
        assert torch.equal(saved[f"bert.{name}"].cpu(), tensor), name


def test_fine_tuning_restores_the_best_dev_epoch(tiny_bert, atis_test):
    path, tokenizer = tiny_bert
    vocab, dataset = atis_test
    train = DataLoader(Subset(dataset, range(64)), batch_size=16, collate_fn=collate(tokenizer), shuffle=True)
    dev = DataLoader(Subset(dataset, range(64, 128)), batch_size=32, collate_fn=collate(tokenizer))
    torch.manual_seed(0)
    model = JointBERT(path, len(vocab.slot2id), len(vocab.intent2id))
    optimizer = bert_optimizer(model, lr=1e-3, weight_decay=0.01)
    scheduler = linear_warmup(optimizer, warmup_steps=2, total_steps=3 * len(train))
    stopper, history = fit(model, optimizer, train, dev, vocab.id2slot, n_epochs=3, patience=3, clip=1.0,
                           scheduler=scheduler)
    assert stopper.best_score == max(history["dev_slot_f1"])
    assert evaluate(model, dev, vocab.id2slot)["slot_f1"] == pytest.approx(stopper.best_score)

    # Evaluation mode: same architecture built from the configuration, fine-tuned weights loaded
    reloaded = JointBERT(path, len(vocab.slot2id), len(vocab.intent2id), pretrained=False)
    reloaded.load_state_dict(model.state_dict())
    assert evaluate(reloaded, dev, vocab.id2slot)["slot_f1"] == pytest.approx(stopper.best_score)


def test_command_line_fine_tunes_and_reproduces(tiny_bert, tmp_path):
    common = ["--exp", "bert", "--checkpoints", str(tmp_path), "--results", str(tmp_path)]
    cli.run(["--train", "--bert_model", tiny_bert[0], "--runs", "1", "--epochs", "1", *common])
    run = json.loads((tmp_path / "slu" / "bert.json").read_text())["runs"][0]
    evaluated = cli.run(common)["bert"]
    assert evaluated["slot_f1"][0] == pytest.approx(run["test_slot_f1"])
    assert evaluated["intent_acc"][0] == pytest.approx(run["test_intent_acc"])
