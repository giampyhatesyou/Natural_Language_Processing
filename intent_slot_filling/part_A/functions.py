# Add the functions used to train and evaluate the models
import random

import matplotlib

matplotlib.use("Agg")  # save the plots to file, no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from tqdm import tqdm

from conll import evaluate


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_std(values):
    # Sample standard deviation (n - 1 in the denominator): the right estimate for a handful of runs
    values = np.asarray(values)
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad()  # Zeroing the gradient (not the weights!)
        slots, intent = model(sample['utterances'], sample['slots_len'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot  # In joint training we sum the losses
        loss_array.append(loss.item())
        loss.backward()  # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()  # Update the weights
    return loss_array


def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    with torch.no_grad():  # It is used to avoid the creation of the computational graph
        for sample in data:
            slots, intents = model(sample['utterances'], sample['slots_len'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())
            # Intent inference: get the highest probable class
            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference
            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                tmp_seq = []
                for id_el, elem in enumerate(to_decode):
                    tmp_seq.append((utterance[id_el], lang.id2slot[elem]))
                hyp_slots.append(tmp_seq)

    results = evaluate(ref_slots, hyp_slots)
    report_intent = classification_report(ref_intents, hyp_intents, zero_division=0, output_dict=True)
    return results, report_intent, loss_array


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


def train_model(model, optimizer, train_loader, dev_loader, criterion_slots, criterion_intents, lang,
                n_epochs, patience, clip):
    """One run: train with early stopping on the dev slot F1 (evaluated every epoch).

    The model is initialised ONCE by the caller; at the end the weights of the best epoch on dev are
    loaded back into the model, so the test set is evaluated on the best checkpoint, not on the last one.
    """
    history = {"epoch": [], "train_loss": [], "dev_loss": [], "dev_f1": []}
    best_f1 = -1
    best_state = None
    patience_left = patience

    for epoch in tqdm(range(1, n_epochs + 1), leave=False):
        loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=clip)
        results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
        f1 = results_dev['total']['f']

        history["epoch"].append(epoch)
        history["train_loss"].append(float(np.mean(loss)))
        history["dev_loss"].append(float(np.mean(loss_dev)))
        history["dev_f1"].append(f1)

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:  # Early stopping with patience
                break

    model.load_state_dict(best_state)
    return best_f1, history


def save_checkpoint(path, model, config, lang, results):
    # Only plain python types + tensors: loadable with torch.load(weights_only=True)
    torch.save({"model": model.state_dict(),
                "config": config,
                "w2id": lang.word2id,
                "slot2id": lang.slot2id,
                "intent2id": lang.intent2id,
                "results": results}, path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device)


def plot_history(history, title, path):
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(history["epoch"], history["train_loss"], label="train loss", marker="o", markersize=3)
    ax1.plot(history["epoch"], history["dev_loss"], label="dev loss", marker="x", markersize=3)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss (intent + slots)")
    ax2 = ax1.twinx()
    ax2.plot(history["epoch"], history["dev_f1"], label="dev slot F1", color="tab:green", linestyle="--")
    ax2.set_ylabel("slot F1")
    lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax1.legend(lines, labels, loc="center right")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
