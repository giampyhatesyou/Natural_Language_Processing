# Audit of the original implementation

This document is the evidence behind Sections 5 and 6 of the [paper](paper/trusting_ablations.pdf). The original
implementation of the two studies was written in July and August 2025 as my final project for the Natural
Language Understanding course at the University of Trento, and is preserved in the git history (commit `6dcb369`).
In October 2026 I audited it as if it were someone else's code. Every error found is listed below with its effect
on the reported study, the fix, and the test that now fails if the error comes back. The numbers that the original
study reported are in [`results/original_study.json`](results/original_study.json).

## Effect on the conclusions

Effect of each ablation step as reported by the original implementation and after the audit: reduction of the
test perplexity, or gain in points of slot F1 and intent accuracy (positive: the step helps). The audited ATIS
effects are differences of means over 5 seeds, with Welch's t-test; the language models were trained once.

| Step | Metric | Original | Audited | p | Outcome |
| - | - | - | - | - | - |
| + dropout | PPL | +57.6 | +15.2 | – | 3.8× overstated |
| + AdamW | PPL | +1.1 | +17.0 | – | 16× understated |
| + variational dropout | PPL | +22.2 | +19.5 | – | confirmed |
| + NT-AvSGD | PPL | -0.2 | +5.1 | – | sign reversed |
| BiLSTM | slot F1 | +1.34 | +1.89 | < 0.001 | confirmed |
| BiLSTM | intent acc. | +2.35 | +1.10 | 0.005 | 2.1× overstated |
| + dropout | slot F1 | +0.42 | +0.46 | 0.099 | not significant |
| + dropout | intent acc. | -0.34 | +0.40 | 0.059 | not significant |
| BERT | slot F1 | -5.32 | +0.90 | 0.001 | sign reversed |
| BERT | intent acc. | +1.94 | +2.60 | < 0.001 | confirmed |

The outcome follows a fixed rule: *not significant* if the audited difference is not significant (p ≥ 0.05);
otherwise *sign reversed* if the signs differ, *overstated* or *understated* if the two effects differ by more than
a factor of two, *confirmed* otherwise. Weight tying is left out: in both studies its baseline is trained with
other hyperparameters. The two implementations also differ in some hyperparameters (the paper lists them), so the
table measures the combined effect of the fixes and of those changes. `python paper/analysis.py` rebuilds it.

## Errors

"LM" is the language-modelling study, "ATIS" the intent-classification and slot-filling study; parts A and B are
the two halves of each study, as in the paper.

### Algorithms implemented incorrectly

| Where | Problem | Effect | Fix | Caught by |
| - | - | - | - | - |
| LM part A | The "LSTM" was built with `nn.RNN` | The LSTM baseline was never trained; its 160 PPL is in line with a vanilla RNN | `nn.LSTM` | `test_the_baseline_is_an_lstm` |
| LM part B | NT-AvSGD triggered when `v > min(history[:-1])` | The non-monotone interval was ignored: averaging started at the first epoch without improvement | Rule of the AWD-LSTM paper: `t > n and v > min(logs[:t-n])` | `test_ntavsgd_triggers_only_when_worse_than_n_checks_before` |
| LM part B | Average of copies of the weights saved at the end of every epoch, all kept in memory | Not the average of the iterates, and memory grew with the epochs | Running mean updated after every SGD step | `test_ntavsgd_averages_every_iterate_after_the_trigger` |
| LM part B | Early stopping and checkpoint on the non-averaged weights, test on the average of the snapshots | The tested model had never been selected on the dev set | After the trigger, dev perplexity, checkpoint and patience use the averaged weights | `test_ntavsgd_selects_and_keeps_the_averaged_weights` |

### Experimental protocol

| Where | Problem | Effect | Fix | Caught by |
| - | - | - | - | - |
| LM part A | The dropout model also tied the weights, a technique of part B | The step changed two factors; the "+ dropout" result even beat the weight-tying experiment of part B | Weight tying removed from part A | `test_every_experiment_adds_one_technique` |
| LM part A | The AdamW experiment used the model without dropout | Not incremental: AdamW was not applied to the best previous model | AdamW on the dropout model | `test_every_experiment_adds_one_technique` |
| ATIS part A | The test set was evaluated with the weights of the last epoch, 15 epochs after the best one | The reported numbers came from an overfitted model | The best dev weights are restored before testing | `test_the_best_dev_epoch_is_restored_after_training` |
| ATIS part A | Every run overwrote the same checkpoint | The saved model was the best epoch of the last run, not a model selected on dev | The run with the best dev F1 is kept | `test_command_line_trains_saves_and_reproduces` |
| ATIS | Spread over runs as a population standard deviation, no significance test | The spread was underestimated by about 10% and the differences between models were never tested | Sample standard deviation, Welch's t-test and confidence intervals | `test_sample_std_divides_by_n_minus_one`; the dropout gain is not significant |

### Dead code and determinism

| Where | Problem | Effect | Fix | Caught by |
| - | - | - | - | - |
| ATIS part A | Dropout on the last hidden state applied after the intent logits | The intent layer never saw dropout, which then seemed to hurt intent accuracy | Dropout before both output layers | `test_dropout_is_applied_before_both_output_layers` (with p = 1 the logits are the biases) |
| LM part A | Dropout probability and batch size hard-coded, ignoring the configuration | The code did not run the configuration described in the report | Every value comes from the configuration | — |
| ATIS part A | Label ids created by iterating a `set` of strings | String hashing is randomised per process, so the ids changed between executions | Sorted vocabularies | `test_label_ids_do_not_depend_on_the_order_of_a_set` |

### Lost code and infrastructure

| Where | Problem | Effect | Fix | Caught by |
| - | - | - | - | - |
| ATIS part B | The BERT code was never committed; the report described copying each label to every sub-token and explained a slot F1 of 89.0 as WordPiece noise | Nothing to reproduce. A randomly initialised two-layer BERT trained with the audited pipeline reaches 88.7, so 89.0 is consistent with a model that lost its pre-trained weights | Label on the first sub-token only, evaluation on words, pre-trained encoder loaded once per run and never re-initialised | `test_the_label_goes_to_the_first_sub_token_only`, `test_every_word_keeps_exactly_its_label`, `test_training_never_reinitialises_the_pretrained_encoder` |
| LM part B | `torch.save` into a folder that did not exist | Every training crashed at the end, so no checkpoint was ever saved | The checkpoint folder is created when missing | `test_command_line_trains_saves_and_reproduces` |
| All parts | No seeds, paths relative to the working directory, choices typed at an `input()` prompt | Runs were not reproducible and could not be automated | Seeds, paths relative to the package, command-line arguments | CI on every push |

### Structure

| Where | Problem | Effect | Fix | Verified by |
| - | - | - | - | - |
| All parts | Three copies of the initialisation, of the training and evaluation loops, of the early stopping and of the checkpoint code (the fourth part had no code) | The copies had drifted apart: LM part A did not save its model, LM part B saved it into `model_bin/` and ATIS part A into `bin/` | One package: shared code in `nlp/common.py`, one subpackage per task | checkpoints saved by the code before the refactoring give the same scores with the package; on CPU the same seed gives identical training runs |
| All parts | Results existed only in the printed output of the training | The tables of the reports could not be regenerated from any file | Every training writes a JSON record; tables and figures are generated from the records by `paper/analysis.py` | the records of the GPU run were rebuilt from its logs, whose hyperparameters were checked against the registries |

## Mutation analysis

[`scripts/mutation_test.py`](scripts/mutation_test.py) puts each error back into the code, one at a time, and runs
the tests: a mutant is killed when a test fails. "Audited" mutants re-create an error above, "suspected" is the
likely cause of the BERT result, "plausible" mutants are variants of the same errors in other places. CI runs the
analysis on every push.

| Kind | Mutant | Killed by |
| - | - | - |
| audited | the LSTM baseline is a vanilla RNN | `test_the_baseline_is_an_lstm` |
| audited | the dropout model also ties the weights | `test_every_experiment_adds_one_technique` |
| audited | AdamW is applied to the model without dropout | `test_every_experiment_adds_one_technique` |
| audited | NT-AvSGD triggers without the non-monotone interval | `test_ntavsgd_triggers_only_when_worse_than_n_checks_before` |
| audited | NT-AvSGD selects the SGD weights instead of the averaged ones | `test_ntavsgd_selects_and_keeps_the_averaged_weights` |
| audited | the intent-slot models are tested on their last epoch | `test_the_best_dev_epoch_is_restored_after_training` |
| audited | dropout is applied after the intent logits | `test_dropout_is_applied_before_both_output_layers` |
| audited | label ids follow the iteration order of a set | `test_label_ids_do_not_depend_on_the_order_of_a_set` |
| audited | the spread over seeds is a population standard deviation | `test_sample_std_divides_by_n_minus_one` |
| audited | the checkpoint folder is not created | `test_command_line_trains_saves_and_reproduces` |
| audited | every run overwrites the checkpoint of the best one | `test_command_line_trains_saves_and_reproduces` |
| audited | BERT labels every sub-token of a word | `test_the_label_goes_to_the_first_sub_token_only` |
| suspected | the initialisation of the LSTM models is applied to BERT | `test_training_never_reinitialises_the_pretrained_encoder` |
| plausible | the language models are tested on their last epoch | `test_the_best_dev_epoch_is_restored_after_training` |
| plausible | the weights are re-initialised at every epoch | `test_training_never_reinitialises_the_pretrained_encoder` |
| plausible | early stopping keeps a reference to the weights instead of a copy | `test_the_best_dev_epoch_is_restored_after_training` |

All 16 mutants are killed. The first run of the analysis left two mutants alive, the selection of the averaged
weights and the restore of the best epoch in the language-model training; the two tests that kill them were
written because of it.

## Other checks

- **Oracle evaluation**: a fake model that returns the gold labels must score exactly 1.0, both with the LSTM
  pipeline and with the BERT one.
- **A tiny model learns the task**: a randomly initialised two-layer BERT trained with the audited pipeline reaches
  88.7 slot F1, so labels, loss masking and evaluation are wired correctly.
- **Checkpoints reproduce their scores**: evaluating a saved model gives the test scores recorded at the end of its
  training.
- **The full GPU run**, whose logs are in [`results/logs`](results/logs).
