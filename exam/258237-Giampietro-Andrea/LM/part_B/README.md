# LM part B: weight tying, variational dropout, NT-AvSGD

Starting from the LSTM of part A, the regularisation techniques of Merity et al. (2018) are added one at a
time. NT-AvSGD needs SGD, so every experiment uses SGD.

| Experiment | What changes |
| - | - |
| `wt` | LSTM with the output layer tied to the embedding matrix (`emb_size == hid_size`) |
| `wt_vd` | + variational dropout: one mask per sequence, shared by all time steps, on the embeddings and on the LSTM output |
| `wt_vd_ntavsgd` | + Non-monotonically Triggered AvSGD |

NT-AvSGD (`NTAvSGD` in `functions.py`) wraps the SGD optimizer. After every epoch the dev perplexity `v` is
logged; averaging starts the first time that `t > n and v > min(logs[:t-n])` (worse than the best value of
more than `n = 5` checks ago). From then on a running mean of the weights is updated after every SGD step, and
the averaged weights are the ones validated, saved and tested. Early stopping is only active after the trigger.

```bash
python main.py --train                         # train all the experiments
python main.py --train --exp wt_vd_ntavsgd     # a single experiment
python main.py                                 # evaluate the checkpoints in bin/ on the test set
```
