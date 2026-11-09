import json

import pytest
import torch
from torch import nn

from nlp import DATA_DIR
from nlp.common import EarlyStopping, init_weights, load_checkpoint
from nlp.lm import cli
from nlp.lm.data import Vocabulary, read_file
from nlp.lm.models import LSTMLanguageModel, VariationalDropout
from nlp.lm.ntavsgd import NTAvSGD
from nlp.lm.training import evaluate, fit

PTB = DATA_DIR / "ptb"


@pytest.fixture(scope="module")
def vocab():
    return Vocabulary(read_file(PTB / "ptb.train.txt"))


@pytest.fixture(scope="module")
def corpus():
    vocab, datasets = cli.load_corpus(PTB, subset=64)
    return vocab, cli.loader(datasets["train"], 16, vocab, shuffle=True), cli.loader(datasets["dev"], 64, vocab)


def small_model(vocab, **switches):
    torch.manual_seed(0)
    model = LSTMLanguageModel(len(vocab), 32, 32, pad_index=vocab.pad_index, **switches)
    init_weights(model)
    return model


def test_the_baseline_is_an_lstm(vocab):
    # Regression: the first version built the "LSTM" baseline with nn.RNN
    assert isinstance(cli.build_model(cli.experiment_config("lstm"), vocab).lstm, nn.LSTM)


@pytest.mark.parametrize("before, after, changed", [
    ("lstm", "lstm_dropout", {"emb_dropout", "out_dropout"}),
    ("lstm_dropout", "lstm_dropout_adamw", {"optimizer", "lr", "weight_decay"}),
    ("wt", "wt_vd", {"variational", "emb_dropout", "out_dropout"}),
    ("wt_vd", "wt_vd_ntavsgd", {"nt_avsgd", "non_mono", "patience"}),
])
def test_every_experiment_adds_one_technique(before, after, changed):
    # Regression: in the first version the dropout model also had weight tying and AdamW ran without dropout
    a, b = cli.experiment_config(before), cli.experiment_config(after)
    assert {key for key in a.keys() | b.keys() if a.get(key) != b.get(key)} == changed


def test_weight_tying_shares_one_matrix():
    model = LSTMLanguageModel(50, 16, 16, tie_weights=True)
    assert model.output.weight is model.embedding.weight
    assert LSTMLanguageModel(50, 16, 16).output.weight is not LSTMLanguageModel(50, 16, 16).embedding.weight
    with pytest.raises(ValueError):
        LSTMLanguageModel(50, 16, 32, tie_weights=True)


def test_variational_dropout_uses_one_mask_per_sequence():
    torch.manual_seed(0)
    dropout = VariationalDropout(0.5).train()
    out = dropout(torch.ones(4, 7, 10))
    for t in range(7):
        assert torch.equal(out[:, t], out[:, 0])
    assert set(out.unique().tolist()) <= {0.0, 2.0}  # the kept units are rescaled by 1 / (1 - p)
    x = torch.randn(4, 7, 10)
    assert torch.equal(dropout.eval()(x), x)


def make_ntavsgd(non_mono):
    model = nn.Linear(3, 1)
    return model, NTAvSGD(model, torch.optim.SGD(model.parameters(), lr=0.1), non_mono=non_mono)


def test_ntavsgd_triggers_only_when_worse_than_n_checks_before():
    _, ntasgd = make_ntavsgd(non_mono=3)
    trace = [100, 90, 85, 84, 83.9, 83.95, 84.5, 86]
    fired = [ntasgd.check(ppl, epoch) for epoch, ppl in enumerate(trace, start=1)]
    # Epoch 6 is already worse than the best value so far, where the rule of the first version triggered;
    # averaging starts at epoch 8, the first one worse than the best of more than 3 checks before
    assert fired == [False] * 7 + [True]
    assert ntasgd.trigger_epoch == 8


def test_ntavsgd_averages_every_iterate_after_the_trigger():
    torch.manual_seed(0)
    model, ntasgd = make_ntavsgd(non_mono=0)
    ntasgd.logs = [1.0]
    assert ntasgd.check(2.0, epoch=1)
    iterates = [[p.detach().clone() for p in model.parameters()]]
    for _ in range(5):
        ntasgd.zero_grad()
        model(torch.randn(4, 3)).pow(2).mean().backward()
        ntasgd.step()
        iterates.append([p.detach().clone() for p in model.parameters()])

    for i, average in enumerate(ntasgd.avg_params):
        assert torch.allclose(average, torch.stack([w[i] for w in iterates]).mean(0), atol=1e-6)
    sgd_weights = [p.detach().clone() for p in model.parameters()]
    with ntasgd.averaged_weights():
        for param, average in zip(model.parameters(), ntasgd.avg_params, strict=True):
            assert torch.equal(param, average)
    for param, saved in zip(model.parameters(), sgd_weights, strict=True):
        assert torch.equal(param, saved)  # the SGD weights are back after the evaluation


def test_the_best_dev_epoch_is_restored_after_training(corpus):
    # On 64 sentences Adam overfits after a few epochs: training stops after the best epoch, whose weights must be
    # the ones left in the model
    vocab, train, dev = corpus
    model = small_model(vocab)
    stopper, history = fit(model, torch.optim.Adam(model.parameters(), lr=0.01), train, dev, vocab.pad_index,
                           n_epochs=10, patience=2, clip=5)
    assert stopper.best_epoch < history["epoch"][-1]
    assert evaluate(model, dev, vocab.pad_index)[0] == pytest.approx(stopper.best_score, rel=1e-6)


def test_ntavsgd_selects_and_keeps_the_averaged_weights(corpus):
    # After the trigger the dev perplexity is the one of the averaged weights: those weights, not the SGD ones, must
    # be selected and left in the model
    vocab, train, dev = corpus
    model = small_model(vocab, tie_weights=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    ntasgd = NTAvSGD(model, optimizer, non_mono=0)
    ntasgd.logs = [1.0]  # an earlier check that no epoch can beat: averaging starts at the first epoch
    stopper, _ = fit(model, optimizer, train, dev, vocab.pad_index, n_epochs=4, patience=4, clip=5, ntasgd=ntasgd)
    assert ntasgd.trigger_epoch == 1 and stopper.best_epoch > 1  # the selected weights average several iterates
    assert evaluate(model, dev, vocab.pad_index)[0] == pytest.approx(stopper.best_score, rel=1e-6)


def test_early_stopping_keeps_the_best_weights():
    model = nn.Linear(2, 1)
    stopper = EarlyStopping(patience=2, mode="min")
    assert not stopper.step(3.0, model, epoch=1)
    best = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.weight.add_(1.0)
    assert not stopper.step(3.5, model, epoch=2, count=False)  # not counted: patience untouched
    assert not stopper.step(4.0, model, epoch=3)
    assert stopper.step(5.0, model, epoch=4)  # second counted epoch without improvement
    stopper.restore(model)
    assert stopper.best_epoch == 1 and torch.equal(model.weight, best["weight"])


def test_configurations_of_the_first_layout_build_the_same_models():
    # Checkpoints of the first version had separate part A and part B models instead of switches
    part_a = cli.model_config({"model": "LM_LSTM_dropout", "emb_dropout": 0.4, "out_dropout": 0.4})
    part_b = cli.model_config({"emb_dropout": 0.0, "out_dropout": 0.0, "nt_avsgd": False})
    assert (part_a["tie_weights"], part_a["variational"]) == (False, False)
    assert (part_b["tie_weights"], part_b["variational"]) == (True, True)


def test_command_line_trains_saves_and_reproduces(tmp_path):
    common = ["--subset", "64", "--checkpoints", str(tmp_path), "--results", str(tmp_path)]
    trained = cli.run(["--train", "--exp", "wt_vd_ntavsgd", "--epochs", "3", *common])
    record = json.loads((tmp_path / "lm" / "wt_vd_ntavsgd.json").read_text())
    run = record["runs"][0]
    assert run["dev_ppl"] == min(run["history"]["dev_ppl"]) == trained["wt_vd_ntavsgd"]["dev_ppl"][0]
    assert (tmp_path / "plots" / "lm" / "wt_vd_ntavsgd.png").exists()

    # the checkpoint holds the selected weights: dev and test perplexity are reproduced from it
    evaluated = cli.run(["--exp", "wt_vd_ntavsgd", *common])
    assert evaluated["wt_vd_ntavsgd"]["test_ppl"][0] == pytest.approx(run["test_ppl"], rel=1e-6)
    vocab, datasets = cli.load_corpus(PTB, subset=64)
    checkpoint = load_checkpoint(tmp_path / "lm" / "wt_vd_ntavsgd.pt")
    model = cli.build_model(checkpoint["config"], vocab)
    model.load_state_dict(checkpoint["model"])
    dev = cli.loader(datasets["dev"], 128, vocab)
    assert evaluate(model, dev, vocab.pad_index)[0] == pytest.approx(run["dev_ppl"], rel=1e-6)
