# Review of the original submission and fixes

The first version of this project (exam submission, July/August 2025) had logic errors that changed the meaning
of several experiments. This file lists each problem, what it broke, and how the current code fixes it.
The folders `LM/` and `NLU/` of the submission are now `language_modeling/` and `intent_slot_filling/`.

## LM, part A (`language_modeling/part_A`)

| # | Problem in the original code | Effect | Fix |
|---|---|---|---|
| 1 | `LM_LSTM` was built with `nn.RNN` | Step 1 of the assignment (RNN → LSTM) was not implemented: the committed code never trained an LSTM baseline, and the 160 PPL reported for it is in line with the RNN baseline of the lab | `nn.LSTM` |
| 2 | The dropout model contained weight tying (`self.output.weight = self.embedding.weight`) | Weight tying is a part B technique: it inflated the part A "+ dropout" result and made part A and part B impossible to compare (part B "+ weight tying" looked worse than part A) | removed from part A |
| 3 | Mode 3 (AdamW) instantiated `LM_LSTM`, i.e. the RNN without dropout | The protocol is incremental: AdamW must be applied to the best previous model (LSTM + dropout) | AdamW on `LM_LSTM_dropout` |
| 4 | Dropout hard-coded to 0.3 (constructor arguments ignored), train batch size hard-coded to 64 (config said 32), patience hard-coded, emb/hid 200 in the code vs 256 in the report | The code did not run the configuration described in the report | one `CONFIG` plus per-experiment overrides, all of them actually used |
| 5 | Model saving commented out inside a string literal, no evaluation mode | Nothing in `bin/`, results not verifiable | `bin/<experiment>.pt`; `python main.py` evaluates the saved checkpoints |
| 6 | Dataset paths relative to the working directory | The script only worked when launched from the repository root | paths relative to `__file__` |
| 7 | Minor: `model.apply(init_weights)` (the function already loops over all modules, so every module was initialised several times), `range(1, n_epochs)` (99 epochs), unused duplicate `get_vocab`, OOV words silently truncating the sentence | — | `init_weights(model)` once, `n_epochs` epochs, OOV raises an error |

## LM, part B (`language_modeling/part_B`)

| # | Problem in the original code | Effect | Fix |
|---|---|---|---|
| 1 | NT-AvSGD trigger: `current_ppl > min(history[:-1])` | It compares with the best of *all* previous epochs, so the non-monotone interval `n` is ignored and averaging starts at the first epoch without improvement. Paper (Merity et al., 2018, Alg. 1): `t > n and v > min(logs[:t-n])` | rule of the paper |
| 2 | Averaging over a copy of the weights saved at the end of every epoch, all kept in memory | Not the AvSGD of the paper (average of the iterates of every SGD step after the trigger) and memory grows with the epochs | running mean updated after every optimizer step, O(1) memory (`NTAvSGD` wrapper in `functions.py`) |
| 3 | Early stopping and best-model tracking on the non-averaged weights, test on the average of all the snapshots from the trigger to the last epoch | The final model was never validated: it also averaged the epochs after the best one, and patience 5 stopped training a few epochs after the trigger | before the trigger no early stopping (the trigger is the plateau detector); after it, dev perplexity, checkpoint and patience all refer to the averaged weights |
| 4 | `torch.save` into `model_bin/`, a folder that does not exist | Every training crashed at the very end: no checkpoint and no final summary | `bin/` (created if missing) + evaluation mode |
| 5 | `LM_LSTM_nt_avsgd` identical to `LM_LSTM_variational_dropout`, `train_with_AvSGD` identical to `train_loop` | Duplicated code | one model `LM_LSTM_WT` (dropout 0 = plain weight tying) and NT-AvSGD as an optimizer wrapper |
| 6 | The report mentions a projection layer for different embedding/hidden sizes | Not in the code | weight tying now explicitly requires `emb_size == hid_size` |

## NLU, part A (`intent_slot_filling/part_A`)

| # | Problem in the original code | Effect | Fix |
|---|---|---|---|
| 1 | Dropout on `last_hidden` applied *after* the intent logits were computed | Dead code: the intent classifier never saw dropout (the report says dropout is applied before both heads) | dropout before both output layers, plus on the embeddings |
| 2 | After early stopping the test set was evaluated with the weights of the last epoch | The reported numbers came from a model 15 epochs past the best one (evaluation every 5 epochs, patience 3), while the report says the best model was tested | the best dev weights are restored before testing |
| 3 | Every run reset `best_f1 = 0` and saved to the same file | `bin/` contained the best epoch of the *last* run, and the test mode evaluated that single run while the report shows a 5-run mean | the saved checkpoint is the run with the best dev F1 |
| 4 | The third results line printed the intent accuracy mean with the slot F1 standard deviation | Wrong std in the logs | fixed, mean ± std printed once per experiment |
| 5 | The epoch loop reused the variable `x` of the runs loop; `patience_init` unused; `model_name` undefined for an invalid choice; `save_model` ignored its `path` argument | Fragile code (NameError on invalid input) | rewritten with `argparse` |
| 6 | Label ids created by iterating a `set` of strings | String hashing is randomised per Python process, so the ids changed from one execution to the next | sorted label vocabularies |
| 7 | No seeds; `CUDA_LAUNCH_BLOCKING=1` left on | Runs not reproducible; slower GPU training | seed = base seed + run index; debug flag removed |

## NLU, part B (`intent_slot_filling/part_B`)

| # | Problem | Effect | Fix |
|---|---|---|---|
| 1 | Only the empty template files were committed | The BERT results in the report could not be reproduced | full implementation (`JointBERT`) |
| 2 | Reported slot F1 0.890: far below what a fine-tuned `bert-base-uncased` reaches on ATIS (about 0.95), explained in the report as WordPiece noise | A tiny randomly initialised 2-layer BERT trained with the new code already reaches 0.887, so the original model was very likely not using its pre-trained weights (re-initialised inside the training loop, or the part A `init_weights` applied to BERT, which overwrites every `nn.Linear` of the encoder) and/or evaluated at sub-token level | the pre-trained model is created once per run and never re-initialised (only the two heads are new); the slot label goes to the first sub-token of each word (the other sub-tokens get -100 and are ignored by the loss); evaluation at word level with `conll.py`; AdamW with linear warm-up, early stopping on dev, 5 runs |
