"""Training and evaluation shared by the LSTM and BERT models."""
import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from ..common import EarlyStopping
from .conll import evaluate as conll_evaluate
from .data import IGNORE_INDEX


def joint_loss(slot_logits, intent_logits, batch):
    # Joint training: sum of the slot and intent cross-entropies; IGNORE_INDEX positions are skipped
    slot_loss = nn.functional.cross_entropy(slot_logits, batch["slot_labels"], ignore_index=IGNORE_INDEX)
    return slot_loss + nn.functional.cross_entropy(intent_logits, batch["intents"])


def train_epoch(model, data, optimizer, scheduler=None, clip=5.0):
    model.train()
    losses = []
    for batch in data:
        optimizer.zero_grad()  # zero the gradients, not the weights
        loss = joint_loss(*model(**batch["inputs"]), batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # avoid exploding gradients
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def evaluate(model, data, id2slot):
    """Slot F1 (chunk level, conlleval, on the original words), intent accuracy and loss. The prediction of a word
    is read where its label is: at every token for the LSTM, at the first sub-token for BERT."""
    # The 'pad' slot class of the LSTM models is never the label of a word: it is excluded from the predictions
    # (an undertrained model could otherwise predict it, and conlleval cannot score a label without IOB prefix)
    pad_classes = [i for i, slot in id2slot.items() if slot == "pad"]
    model.eval()
    losses, ref_slots, hyp_slots = [], [], []
    correct_intents = total_intents = 0
    with torch.no_grad():
        for batch in data:
            slot_logits, intent_logits = model(**batch["inputs"])
            losses.append(joint_loss(slot_logits, intent_logits, batch).item())
            correct_intents += (intent_logits.argmax(dim=1) == batch["intents"]).sum().item()
            total_intents += len(batch["intents"])
            slot_logits[:, pad_classes] = float("-inf")
            predictions = slot_logits.argmax(dim=1)
            labelled = batch["slot_labels"] != IGNORE_INDEX
            for i, (words, gold) in enumerate(zip(batch["words"], batch["slots"], strict=True)):
                predicted = [id2slot[p] for p in predictions[i][labelled[i]].tolist()]
                ref_slots.append(list(zip(words, gold, strict=True)))
                hyp_slots.append(list(zip(words, predicted, strict=True)))
    return {"slot_f1": conll_evaluate(ref_slots, hyp_slots)["total"]["f"],
            "intent_acc": correct_intents / total_intents, "loss": float(np.mean(losses))}


def fit(model, optimizer, train, dev, id2slot, n_epochs, patience, clip, scheduler=None):
    """Train with early stopping on the dev slot F1 (evaluated after every epoch), then load the weights of the best
    epoch. Returns the EarlyStopping object (best epoch and score) and the per-epoch history."""
    stopper = EarlyStopping(patience, mode="max")
    history = {"epoch": [], "train_loss": [], "dev_loss": [], "dev_slot_f1": [], "dev_intent_acc": []}
    for epoch in tqdm(range(1, n_epochs + 1), leave=False):
        train_loss = train_epoch(model, train, optimizer, scheduler, clip)
        scores = evaluate(model, dev, id2slot)
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["dev_loss"].append(scores["loss"])
        history["dev_slot_f1"].append(scores["slot_f1"])
        history["dev_intent_acc"].append(scores["intent_acc"])
        if stopper.step(scores["slot_f1"], model, epoch):
            break
    stopper.restore(model)
    return stopper, history


def bert_optimizer(model, lr, weight_decay):
    """Standard BERT fine-tuning: AdamW, no weight decay on biases and LayerNorm weights (the 1-D tensors)."""
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def linear_warmup(optimizer, warmup_steps, total_steps):
    """The learning rate grows linearly for warmup_steps, then decays linearly to 0 at total_steps."""
    def factor(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return LambdaLR(optimizer, factor)
