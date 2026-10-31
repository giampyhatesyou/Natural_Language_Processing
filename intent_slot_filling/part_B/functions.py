# Add the functions used to train and evaluate the models
import random

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from conll import evaluate
from utils import IGNORE_INDEX


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, lr, weight_decay):
    # Standard BERT fine-tuning: AdamW, no weight decay on biases and LayerNorm weights (the 1-D tensors)
    decay = [p for p in model.parameters() if p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.ndim < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def linear_warmup_scheduler(optimizer, warmup_steps, total_steps):
    # The learning rate grows linearly for warmup_steps, then decays linearly to 0 at total_steps
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return LambdaLR(optimizer, lr_lambda)


def train_loop(data, optimizer, scheduler, criterion_slots, criterion_intents, model, clip=1.0):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad()  # Zeroing the gradient (not the weights!)
        slots, intent = model(sample['input_ids'], sample['attention_mask'], sample['token_type_ids'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['slot_labels'])  # IGNORE_INDEX positions are skipped
        loss = loss_intent + loss_slot  # In joint training we sum the losses
        loss_array.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        scheduler.step()
    return loss_array


def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    with torch.no_grad():
        for sample in data:
            slots, intents = model(sample['input_ids'], sample['attention_mask'], sample['token_type_ids'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['slot_labels'])
            loss_array.append((loss_intent + loss_slot).item())

            # Intent inference: get the highest probable class
            hyp_intents.extend([lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()])
            ref_intents.extend([lang.id2intent[x] for x in sample['intents'].tolist()])

            # Slot inference at WORD level: the prediction of a word is the one of its first sub-token,
            # i.e. of the positions whose label is not IGNORE_INDEX
            output_slots = torch.argmax(slots, dim=1)  # batch_size X seq_len
            first_subtoken = sample['slot_labels'] != IGNORE_INDEX
            for id_seq, (words, gt_slots) in enumerate(zip(sample['words'], sample['slots'])):
                pred_ids = output_slots[id_seq][first_subtoken[id_seq]].tolist()
                ref_slots.append(list(zip(words, gt_slots)))
                hyp_slots.append([(word, lang.id2slot[pred]) for word, pred in zip(words, pred_ids)])

    results = evaluate(ref_slots, hyp_slots)
    report_intent = classification_report(ref_intents, hyp_intents, zero_division=0, output_dict=True)
    return results, report_intent, loss_array


def train_model(model, optimizer, scheduler, train_loader, dev_loader, criterion_slots, criterion_intents, lang,
                n_epochs, patience, clip):
    """One run: fine-tune with early stopping on the dev slot F1 (evaluated every epoch).

    The model is created ONCE per run by the caller (pre-trained encoder + new heads): inside the epoch loop
    only the gradients are zeroed. At the end the weights of the best epoch on dev are loaded back.
    """
    history = {"epoch": [], "train_loss": [], "dev_loss": [], "dev_f1": [], "dev_intent_acc": []}
    best_f1 = -1
    best_state = None
    patience_left = patience

    for epoch in tqdm(range(1, n_epochs + 1), leave=False):
        loss = train_loop(train_loader, optimizer, scheduler, criterion_slots, criterion_intents, model, clip)
        results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
        f1 = results_dev['total']['f']

        history["epoch"].append(epoch)
        history["train_loss"].append(float(np.mean(loss)))
        history["dev_loss"].append(float(np.mean(loss_dev)))
        history["dev_f1"].append(f1)
        history["dev_intent_acc"].append(float(intent_res['accuracy']))

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
                "slot2id": lang.slot2id,
                "intent2id": lang.intent2id,
                "results": results}, path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device)
