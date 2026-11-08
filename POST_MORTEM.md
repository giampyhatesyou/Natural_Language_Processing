# Post-mortem of the first version

The first version of this project was written in July and August 2025 as the final assignment of the NLU course.
In October 2026 I reviewed it as if it were someone else's code. The review found errors that changed the meaning of
several experiments: an "LSTM" that was a vanilla RNN, ablations that changed two factors at once, an optimisation
algorithm that triggered and averaged the wrong way, models tested on the wrong epoch, and the code of one
experiment missing altogether. This document groups the problems by type, shows their effect on the reported
numbers and explains how each fix was verified.

The first version had one folder for each of the four parts of the assignment: language modelling (LM) part A and
part B, intent classification and slot filling (ATIS) part A and part B. The "Where" columns below refer to these
parts; the current code is the `nlu_ablations` package described in the [README](README.md).

## Effect on the reported numbers

| Experiment | First version | Current version | Why it changed |
| - | - | - | - |
| LM, LSTM baseline (test PPL) | 160.0 | 141.3 | the first version trained a vanilla RNN |
| LM, + dropout (test PPL) | 102.4 | 126.0 | the first version's dropout model also had weight tying |
| LM, + NT-AvSGD vs + variational dropout (test PPL) | 96.0 vs 95.8: worse | 92.3 vs 97.4: 5.1 better | trigger rule, averaging and model selection fixed |
| ATIS, + dropout vs BiLSTM (intent accuracy, %) | 94.96 vs 95.30: worse | 94.94 vs 94.54: +0.40, not significant | dropout never reached the intent head |
| ATIS, BERT (slot F1) | 89.0 | 95.49 ± 0.11 | the code was rewritten: word-level alignment, pre-trained weights kept |

The first-version numbers are the ones of its reports. Hyperparameters and protocol changed as well, so the
comparison is indicative; the point is that several conclusions of the first version were reversed.

## Algorithms implemented incorrectly

| Where | Problem | Effect | Fix | Verified by |
| - | - | - | - | - |
| LM part A | `LM_LSTM` was built with `nn.RNN` | The LSTM baseline was never trained; its 160 PPL is in line with the RNN of the lab | `nn.LSTM` | regression test |
| LM part B | NT-AvSGD triggered when `v > min(history[:-1])` | The non-monotone interval was ignored: averaging started at the first epoch without improvement (epoch 6 instead of 14 on the trace of a test run) | Rule of the paper: `t > n and v > min(logs[:t-n])` | unit test on a synthetic trace |
| LM part B | Averaging over copies of the weights saved at the end of every epoch, all kept in memory | Not the average of the iterates of the paper, and memory grew with the epochs | Running mean updated after every SGD step | unit test: the running mean equals the mean of the iterates |
| LM part B | Early stopping and checkpoint on the non-averaged weights, test on the average of all the snapshots after the trigger | The tested model had never been selected on the dev set | After the trigger, dev perplexity, checkpoint and patience use the averaged weights | regression test: after the trigger the averaged weights are selected and left in the model; GPU run (trigger at epoch 74) |

## Experimental protocol

| Where | Problem | Effect | Fix | Verified by |
| - | - | - | - | - |
| LM part A | The dropout model also had weight tying, a part B technique | The ablation changed two factors; the "+ dropout" result beat the weight-tying experiment of part B | Weight tying removed from part A | regression test |
| LM part A | The AdamW experiment used the model without dropout | Not incremental: AdamW was not applied to the best previous model | AdamW on the dropout model | regression test |
| ATIS part A | The test set was evaluated with the weights of the last epoch, 15 epochs after the best one | The reported numbers came from an overfitted model | The best dev weights are restored before testing | regression test: after training, the dev score equals the best one of the history |
| ATIS part A | Every run overwrote the same checkpoint | `bin/` held the best epoch of the last run, not a model selected on dev | The run with the best dev F1 is kept | end-to-end test: the checkpoint gives back the test scores of the run with the best dev F1 |
| ATIS | Spread over runs computed as a population standard deviation, no significance test | The spread was underestimated by about 10% and differences between models were never tested | Sample standard deviation and Welch's t-test | the dropout gain turns out not to be significant |

## Silent and dead code

| Where | Problem | Effect | Fix | Verified by |
| - | - | - | - | - |
| ATIS part A | Dropout on the last hidden state applied after the intent logits had been computed | The intent classifier never saw dropout, which then seemed to hurt intent accuracy | Dropout before both output layers | regression test: with p = 1 the logits reduce to the bias |
| LM part A | Dropout probability and batch size hard-coded, ignoring the configuration | The code did not run the configuration described in the report | Every value comes from the configuration | — |
| ATIS part A | Label ids created by iterating a `set` of strings | String hashing is randomised per process, so the ids changed from one execution to the next | Sorted vocabularies | unit test |

## Missing or unverifiable results

| Where | Problem | Effect | Fix | Verified by |
| - | - | - | - | - |
| ATIS part B | The BERT code was never committed; the report explained a slot F1 of 89.0 as WordPiece noise | Nothing to reproduce. With the current code, a tiny randomly initialised BERT reaches 88.7, so 89.0 is consistent with a model that did not use its pre-trained weights | New implementation: label on the first sub-token of every word, evaluation on words, pre-trained encoder loaded once per run and never re-initialised | alignment tests on all of ATIS, oracle evaluation, regression test: trained with a learning rate of 0, the saved encoder is still the pre-trained one; GPU run: 95.5 F1 |
| LM part B | `torch.save` into a folder that did not exist | Every training crashed at the end, so no checkpoint was ever saved | The checkpoint folder is created when missing | end-to-end test: trains, saves and reloads a checkpoint |
| All parts | No seeds, paths relative to the working directory, choices typed at an `input()` prompt | Runs were not reproducible and the scripts could not be automated | Seeds, paths relative to the package, command-line arguments | CI runs the tests on every push |

## Structure

| Where | Problem | Effect | Fix | Verified by |
| - | - | - | - | - |
| All parts | Three copies of the initialisation, of the training and evaluation loops, of the early stopping and of the checkpoint code (the fourth part had no code) | The copies had drifted apart: LM part A did not save its model, LM part B saved it into `model_bin/` and ATIS part A into `bin/` | One package: shared code in `nlu_ablations/common.py`, one subpackage per task | checkpoints saved by the code before the refactoring give the same scores with the package; on CPU the same seed gives identical training runs |
| All parts | Results existed only in the printed output of the training | The tables of the reports could not be regenerated from any file | Every training writes a JSON record; tables and figures are generated from the records by `reports/analysis.py` | the records of the GPU run were rebuilt from its logs, whose hyperparameters were checked against the registries |

## How the fixes were verified

- **Tests** in [`tests/`](tests): the regression tests above, the trigger and the running mean of NT-AvSGD, the
  variational dropout mask, the sub-token alignment on the whole ATIS test set and end-to-end runs of both command
  lines (train, save, reload, evaluate). GitHub Actions runs them, together with a linter, on every push.
- **Oracle evaluation**: a fake model that returns the gold labels must score exactly 1.0 with the evaluation code,
  both for the LSTM models and for BERT.
- **Sanity checks with a tiny model**: a randomly initialised two-layer BERT trained with the new pipeline learns
  the task (88.7 slot F1), so labels, loss masking and evaluation are wired correctly.
- **Reproducing the checkpoints**: evaluating a saved model gives the test scores recorded at the end of its
  training.
- **The full GPU run**, whose logs are in [`results/logs`](results/logs).

The lessons I took from this review are summarised in the [README](README.md#what-i-learned).
