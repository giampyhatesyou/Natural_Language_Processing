# NLU part A: joint intent classification and slot filling with an LSTM

ATIS, one model with two heads: the slot head labels every token (IOB tags), the intent head classifies the
utterance from the last hidden state. The loss is the sum of the two cross-entropies.

| Experiment | Model |
| - | - |
| `lstm` | baseline `ModelIAS` (unidirectional LSTM) |
| `bilstm` | bidirectional LSTM (the intent head sees the last forward and backward states) |
| `bilstm_dropout` | + dropout on the embeddings and before both output layers |

Every experiment is repeated 5 times with different seeds; we report mean and standard deviation on the test
set. In every run the model is evaluated on dev after each epoch (early stopping on the slot F1, patience 10)
and the best dev weights are restored before testing. `bin/` keeps the run with the best dev F1.

```bash
python main.py --train                          # train all the experiments (5 runs each)
python main.py --train --exp bilstm_dropout     # a single experiment
python main.py                                  # evaluate the checkpoints in bin/ on the test set
```
