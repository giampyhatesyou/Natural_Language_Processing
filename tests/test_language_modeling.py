import math
from functools import partial

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from helpers import ROOT, load_part

lm_a = load_part("language_modeling/part_A")
lm_b = load_part("language_modeling/part_B")
PTB = ROOT / "language_modeling" / "dataset"


def test_part_a_baseline_is_an_lstm():
    # Regression: the first version built the "LSTM" baseline with nn.RNN
    assert isinstance(lm_a.model.LM_LSTM(16, 16, 50).lstm, nn.LSTM)


def test_part_a_dropout_model_has_no_weight_tying():
    # Regression: weight tying belongs to part B, the first version hid it in the dropout model of part A
    model = lm_a.model.LM_LSTM_dropout(16, 16, 50)
    assert model.output.weight.data_ptr() != model.embedding.weight.data_ptr()


def test_weight_tying_shares_one_matrix():
    model = lm_b.model.LM_LSTM_WT(16, 16, 50)
    assert model.output.weight is model.embedding.weight
    with pytest.raises(ValueError):
        lm_b.model.LM_LSTM_WT(16, 32, 50)


def test_variational_dropout_uses_one_mask_per_sequence():
    torch.manual_seed(0)
    dropout = lm_b.model.VariationalDropout(0.5).train()
    out = dropout(torch.ones(4, 7, 10))
    for t in range(7):
        assert torch.equal(out[:, t], out[:, 0])
    assert set(out.unique().tolist()) <= {0.0, 2.0}  # the kept units are rescaled by 1 / (1 - p)
    x = torch.randn(4, 7, 10)
    assert torch.equal(dropout.eval()(x), x)


def make_ntavsgd(non_mono):
    model = nn.Linear(3, 1)
    return model, lm_b.functions.NTAvSGD(model, torch.optim.SGD(model.parameters(), lr=0.1), non_mono=non_mono)


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


@pytest.fixture(scope="module")
def ptb():
    utils = lm_b.utils
    train = utils.read_file(PTB / "ptb.train.txt")
    lang = utils.Lang(train, ["<pad>", "<eos>"])
    collate = partial(utils.collate_fn, pad_token=lang.word2id["<pad>"])

    def loader(raw, shuffle=False):
        return DataLoader(utils.PennTreeBank(raw, lang), batch_size=16, collate_fn=collate, shuffle=shuffle)

    return lang, loader(train[:64], shuffle=True), loader(utils.read_file(PTB / "ptb.valid.txt")[:32])


def test_training_with_ntavsgd_keeps_the_selected_weights(ptb, tmp_path):
    lang, train_loader, dev_loader = ptb
    model_lib, f = lm_b.model, lm_b.functions
    pad = lang.word2id["<pad>"]
    torch.manual_seed(0)
    model = model_lib.LM_LSTM_WT(16, 16, len(lang.word2id), pad_index=pad, emb_dropout=0.2, out_dropout=0.2)
    model.to(lm_b.utils.DEVICE)
    f.init_weights(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=pad, reduction="sum")
    best_ppl, history = f.train_model(model, optimizer, train_loader, dev_loader, nn.CrossEntropyLoss(ignore_index=pad),
                                      criterion_eval, n_epochs=4, patience=2, clip=5,
                                      ntasgd=f.NTAvSGD(model, optimizer, non_mono=1))
    assert math.isfinite(best_ppl) and best_ppl == min(history["dev_ppl"])
    assert f.eval_loop(dev_loader, criterion_eval, model)[0] == pytest.approx(best_ppl, rel=1e-5)

    path = tmp_path / "model.pt"
    f.save_checkpoint(path, model, {"emb_size": 16}, lang, {"dev_ppl": best_ppl})
    reloaded = model_lib.LM_LSTM_WT(16, 16, len(lang.word2id), pad_index=pad).to(lm_b.utils.DEVICE)
    reloaded.load_state_dict(f.load_checkpoint(path, lm_b.utils.DEVICE)["model"])
    assert f.eval_loop(dev_loader, criterion_eval, reloaded)[0] == pytest.approx(best_ppl, rel=1e-5)
