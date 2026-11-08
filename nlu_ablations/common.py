"""Utilities shared by the two tasks: seeding, initialisation, early stopping, checkpoints, statistics, plots."""
import random

import numpy as np
import torch
from matplotlib.figure import Figure
from torch import nn

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_weights(model):
    """Initialisation of the recurrent models: Xavier for the input-to-hidden weights and orthogonal for the
    hidden-to-hidden ones (gate by gate), zero recurrent biases, small uniform linear layers. Called once, before
    training. Never on a pre-trained model, whose weights it would overwrite."""
    for module in model.modules():
        if type(module) in (nn.LSTM, nn.GRU, nn.RNN):
            gates = {nn.LSTM: 4, nn.GRU: 3, nn.RNN: 1}[type(module)]
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    for chunk in param.data.chunk(gates):
                        nn.init.xavier_uniform_(chunk)
                elif "weight_hh" in name:
                    for chunk in param.data.chunk(gates):
                        nn.init.orthogonal_(chunk)
                elif "bias" in name:
                    param.data.fill_(0)
        elif type(module) is nn.Linear:
            nn.init.uniform_(module.weight, -0.01, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.01)


class EarlyStopping:
    """Keeps a copy of the weights of the best epoch on the dev set and counts the epochs without improvement."""

    def __init__(self, patience, mode):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience, self.mode = patience, mode
        self.best_score = self.best_epoch = self.best_state = None
        self.bad_epochs = 0

    def improves(self, score):
        if self.best_score is None:
            return True
        return score < self.best_score if self.mode == "min" else score > self.best_score

    def step(self, score, model, epoch, count=True):
        """Record the dev score of an epoch; returns True when training should stop. With count=False an epoch
        without improvement does not use up patience."""
        if self.improves(score):
            self.best_score, self.best_epoch = score, epoch
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.bad_epochs = 0
        elif count:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def restore(self, model):
        """Load the weights of the best epoch: the test set is evaluated on them, not on the last epoch."""
        model.load_state_dict(self.best_state)


def save_checkpoint(path, model, **payload):
    # Only plain Python types and tensors, so that torch.load(weights_only=True) can read the file
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), **payload}, path)


def load_checkpoint(path, device=DEVICE):
    return torch.load(path, map_location=device)


def sample_std(values):
    """Sample standard deviation (n - 1 in the denominator), the right estimate for a handful of runs."""
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def plot_history(path, title, history, metric=None, marker=None):
    """Train and dev loss per epoch; optionally a dev metric on a second axis and a vertical line at an epoch.
    metric: (history key, label); marker: (epoch, label)."""
    fig = Figure(figsize=(6, 4))
    ax = fig.subplots()
    ax.plot(history["epoch"], history["train_loss"], label="train loss", marker="o", markersize=3)
    ax.plot(history["epoch"], history["dev_loss"], label="dev loss", marker="x", markersize=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    if marker is not None:
        ax.axvline(marker[0], color="gray", linestyle="--", label=marker[1])
    handles, labels = ax.get_legend_handles_labels()
    if metric is not None:
        key, label = metric
        right = ax.twinx()
        right.plot(history["epoch"], history[key], color="tab:green", linestyle="--", label=label)
        right.set_ylabel(label)
        more_handles, more_labels = right.get_legend_handles_labels()
        handles, labels = handles + more_handles, labels + more_labels
    ax.legend(handles, labels, loc="center right")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
