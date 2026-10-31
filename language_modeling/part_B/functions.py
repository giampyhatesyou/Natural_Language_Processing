# Add the functions used to train and evaluate the models
import math
import random
from contextlib import contextmanager, nullcontext

import matplotlib
matplotlib.use("Agg")  # save the plots to file, no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_loop(data, optimizer, criterion, model, clip=5):
    # `optimizer` can be a torch optimizer or the NTAvSGD wrapper below (same zero_grad/step interface)
    model.train()
    loss_array = []
    number_of_tokens = []

    for sample in data:
        optimizer.zero_grad()  # Zeroing the gradient (not the weights!)
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward()  # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # Update the weights

    return sum(loss_array) / sum(number_of_tokens)


def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_array = []
    number_of_tokens = []
    with torch.no_grad():  # It is used to avoid the creation of the computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])  # reduction='sum' -> total loss of the batch
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])

    loss_to_return = sum(loss_array) / sum(number_of_tokens)  # average loss per token
    ppl = math.exp(loss_to_return)
    return ppl, loss_to_return


class NTAvSGD:
    """Non-monotonically Triggered AvSGD (Merity et al., 2018, Algorithm 1).

    It wraps an SGD optimizer and, until the trigger, it is exactly SGD. After every validation check
    (one per epoch, i.e. logging interval L = 1 epoch) the dev perplexity v is compared with the logs:
    averaging is triggered the first time that

        t > n  and  v > min(logs[:t - n])

    i.e. v is worse than the best perplexity obtained more than n checks ago (n = non-monotone interval),
    so a single bad epoch is not enough. From the trigger point T on, the weights w_T, ..., w_k produced
    by every SGD step are averaged with a running mean, and the AVERAGED weights are the ones used for
    validation, model selection and test.
    """

    def __init__(self, model, optimizer, non_mono=5):
        self.model = model
        self.optimizer = optimizer
        self.non_mono = non_mono
        self.logs = []             # dev perplexities collected before the trigger
        self.trigger_epoch = None
        self.avg_params = None     # running average of the weights, created at the trigger
        self.n_averaged = 0

    @property
    def triggered(self):
        return self.avg_params is not None

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()  # plain SGD step
        if self.triggered:
            self.n_averaged += 1
            with torch.no_grad():
                for avg, param in zip(self.avg_params, self.model.parameters()):
                    avg.add_(param - avg, alpha=1.0 / self.n_averaged)  # avg_k = avg_k-1 + (w_k - avg_k-1) / k

    def check(self, ppl_dev, epoch):
        """Validation check, called once per epoch before the trigger. Returns True if it triggers now."""
        t = len(self.logs)
        if not self.triggered and t > self.non_mono and ppl_dev > min(self.logs[:t - self.non_mono]):
            # The average starts from the current weights w_T
            self.avg_params = [param.detach().clone() for param in self.model.parameters()]
            self.n_averaged = 1
            self.trigger_epoch = epoch
        self.logs.append(ppl_dev)
        return self.trigger_epoch == epoch

    @contextmanager
    def averaged_weights(self):
        """Temporarily copy the averaged weights into the model (e.g. to evaluate or save them)."""
        if not self.triggered:
            yield
            return
        backup = [param.detach().clone() for param in self.model.parameters()]
        with torch.no_grad():
            for param, avg in zip(self.model.parameters(), self.avg_params):
                param.copy_(avg)
        try:
            yield
        finally:
            with torch.no_grad():
                for param, saved in zip(self.model.parameters(), backup):
                    param.copy_(saved)


def init_weights(mat):
    # It already iterates over all the sub-modules: call it once as init_weights(model)
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.xavier_uniform_(param[idx * mul:(idx + 1) * mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.orthogonal_(param[idx * mul:(idx + 1) * mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)


def train_model(model, optimizer, train_loader, dev_loader, criterion_train, criterion_eval,
                n_epochs, patience, clip, ntasgd=None):
    """Train with early stopping on the dev perplexity; at the end the best weights are loaded in the model.

    With NT-AvSGD (`ntasgd` is the wrapper of `optimizer`):
      - before the trigger there is no early stopping: the trigger itself is the plateau detector;
      - after the trigger the dev perplexity, the best checkpoint and the patience refer to the averaged weights.
    """
    averaged = ntasgd.averaged_weights if ntasgd is not None else nullcontext
    history = {"epoch": [], "train_loss": [], "dev_loss": [], "dev_ppl": [], "trigger_epoch": None}
    best_ppl = math.inf
    best_state = None
    patience_left = patience

    pbar = tqdm(range(1, n_epochs + 1))
    for epoch in pbar:
        loss = train_loop(train_loader, ntasgd if ntasgd is not None else optimizer, criterion_train, model, clip)
        with averaged():  # no-op before the trigger
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
        if ntasgd is not None and not ntasgd.triggered:
            if ntasgd.check(ppl_dev, epoch):
                history["trigger_epoch"] = epoch
                tqdm.write(f"NT-AvSGD triggered at epoch {epoch} (dev PPL {ppl_dev:.2f}): averaging starts now")

        history["epoch"].append(epoch)
        history["train_loss"].append(loss)
        history["dev_loss"].append(loss_dev)
        history["dev_ppl"].append(ppl_dev)
        pbar.set_description("Dev PPL: %f" % ppl_dev)

        if ppl_dev < best_ppl:  # the lower, the better
            best_ppl = ppl_dev
            with averaged():
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        elif ntasgd is None or ntasgd.triggered:
            patience_left -= 1
            if patience_left <= 0:  # Early stopping with patience
                break

    if ntasgd is not None and not ntasgd.triggered:
        print("Warning: NT-AvSGD never triggered, the result is plain SGD")
    model.load_state_dict(best_state)
    return best_ppl, history


def save_checkpoint(path, model, config, lang, results):
    # Only plain python types + tensors: loadable with torch.load(weights_only=True)
    torch.save({"model": model.state_dict(),
                "config": config,
                "word2id": lang.word2id,
                "results": results}, path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device)


def plot_history(history, title, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history["epoch"], history["train_loss"], label="train loss", marker="o", markersize=3)
    ax.plot(history["epoch"], history["dev_loss"], label="dev loss", marker="x", markersize=3)
    if history.get("trigger_epoch"):
        ax.axvline(history["trigger_epoch"], color="gray", linestyle="--", label="NT-AvSGD trigger")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy per token")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
