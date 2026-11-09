"""Training and evaluation of the language models."""
import math
from contextlib import nullcontext

import torch
from torch import nn
from tqdm import tqdm

from ..common import EarlyStopping


def train_epoch(model, data, optimizer, pad_index, clip=5.0):
    """One epoch; returns the average cross-entropy per token. `optimizer` can also be the NTAvSGD wrapper."""
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index)
    model.train()
    total_loss, total_tokens = 0.0, 0
    for batch in data:
        optimizer.zero_grad()  # zero the gradients, not the weights
        loss = criterion(model(batch["source"]), batch["target"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # avoid exploding gradients
        optimizer.step()
        total_loss += loss.item() * batch["number_tokens"]
        total_tokens += batch["number_tokens"]
    return total_loss / total_tokens


def evaluate(model, data, pad_index):
    """Perplexity: exponential of the mean cross-entropy over all the predicted tokens, padding excluded."""
    criterion = nn.CrossEntropyLoss(ignore_index=pad_index, reduction="sum")
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in data:
            total_loss += criterion(model(batch["source"]), batch["target"]).item()
            total_tokens += batch["number_tokens"]
    loss = total_loss / total_tokens
    return math.exp(loss), loss


def fit(model, optimizer, train, dev, pad_index, n_epochs, patience, clip, ntasgd=None):
    """Train with early stopping on the dev perplexity, then load the weights of the best epoch.

    With NT-AvSGD (`ntasgd` wraps `optimizer`) there is no early stopping before the trigger, which is itself a
    plateau detector; after it, the dev perplexity, the selected weights and the patience refer to the averaged
    weights. Returns the EarlyStopping object (best epoch and score) and the per-epoch history."""
    averaged = ntasgd.averaged_weights if ntasgd is not None else nullcontext
    stopper = EarlyStopping(patience, mode="min")
    history = {"epoch": [], "train_loss": [], "dev_loss": [], "dev_ppl": []}
    pbar = tqdm(range(1, n_epochs + 1))
    for epoch in pbar:
        train_loss = train_epoch(model, train, ntasgd if ntasgd is not None else optimizer, pad_index, clip)
        with averaged():  # no-op before the trigger
            dev_ppl, dev_loss = evaluate(model, dev, pad_index)
        if ntasgd is not None and not ntasgd.triggered and ntasgd.check(dev_ppl, epoch):
            tqdm.write(f"NT-AvSGD triggered at epoch {epoch} (dev PPL {dev_ppl:.2f}): averaging starts now")
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["dev_loss"].append(dev_loss)
        history["dev_ppl"].append(dev_ppl)
        pbar.set_description(f"Dev PPL: {dev_ppl:f}")
        with averaged():
            stop = stopper.step(dev_ppl, model, epoch, count=ntasgd is None or ntasgd.triggered)
        if stop:
            break
    if ntasgd is not None and not ntasgd.triggered:
        print("Warning: NT-AvSGD never triggered, the result is plain SGD")
    stopper.restore(model)
    return stopper, history
