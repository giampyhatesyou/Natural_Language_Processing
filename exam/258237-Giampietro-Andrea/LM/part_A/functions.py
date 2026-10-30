# Add the functions used to train and evaluate the models
import math
import random

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
                n_epochs, patience, clip):
    """Train with early stopping on the dev perplexity.

    The model is initialised ONCE by the caller: inside the epoch loop we only zero the gradients.
    At the end the weights of the best epoch (lowest dev PPL) are loaded back into the model.
    """
    history = {"epoch": [], "train_loss": [], "dev_loss": [], "dev_ppl": []}
    best_ppl = math.inf
    best_state = None
    patience_left = patience

    pbar = tqdm(range(1, n_epochs + 1))
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)

        history["epoch"].append(epoch)
        history["train_loss"].append(loss)
        history["dev_loss"].append(loss_dev)
        history["dev_ppl"].append(ppl_dev)
        pbar.set_description("Dev PPL: %f" % ppl_dev)

        if ppl_dev < best_ppl:  # the lower, the better
            best_ppl = ppl_dev
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:  # Early stopping with patience
                break

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
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy per token")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
