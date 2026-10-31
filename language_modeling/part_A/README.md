# LM part A: from RNN to LSTM, dropout, AdamW

Word-level language model on Penn Treebank (one sentence per sample, `<eos>` appended), evaluated with
perplexity. The three experiments are incremental, each one starts from the best model of the previous step:

| Experiment | Model | Optimizer |
| - | - | - |
| `lstm` | embedding → LSTM → linear | SGD |
| `lstm_dropout` | + dropout after the embedding and before the output layer | SGD |
| `lstm_dropout_adamw` | same model | AdamW |

```bash
python main.py --train                 # train all the experiments, checkpoints in bin/, loss curves in plots/
python main.py --train --exp lstm      # a single experiment
python main.py                         # evaluate the checkpoints in bin/ on the test set
```

Hyperparameters are in `CONFIG` and `EXPERIMENTS` at the top of `main.py`. Early stopping on the dev perplexity
(patience 3); the test set is evaluated with the weights of the best dev epoch.
