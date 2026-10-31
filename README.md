# Natural Language Understanding: language modelling and joint intent/slot filling

Final project of the Natural Language Understanding course at the University of Trento (a.y. 2024/25).
It has two assignments, each split into two incremental parts:

- **Language modelling on Penn Treebank.** Starting from an LSTM language model, I add dropout and AdamW, then
  the regularisation recipe of AWD-LSTM (Merity et al., 2018): weight tying, variational dropout and
  Non-monotonically Triggered AvSGD. The metric is perplexity (lower is better).
- **Intent classification and slot filling on ATIS.** One model predicts the intent of an utterance and an IOB
  slot label for every word: first an LSTM with two heads, then a fine-tuned BERT. The metrics are slot F1
  (CoNLL, chunk level) and intent accuracy.

> **Version 2.** The first submission had logic errors that changed the meaning of several experiments: the
> "LSTM" baseline was a vanilla RNN, NT-AvSGD triggered and averaged the wrong way, the NLU models were tested
> on the last epoch instead of the best one, and the BERT code was missing. Everything was reviewed and
> rewritten; [CHANGES.md](CHANGES.md) lists every problem with its fix.

The two reports, written as short papers in the course template, describe methods and results:
[language modelling](language_modeling/LM_Report.pdf) and
[intent classification and slot filling](intent_slot_filling/NLU_Report.pdf).

## Results

All the models were trained on a Colab T4 GPU with [`run_on_colab.ipynb`](run_on_colab.ipynb). The raw training
logs are in [`results/logs`](results/logs) and the loss curves of every experiment in the `plots/` folder of each part.

### Language modelling (Penn Treebank, perplexity, lower is better)

| Part | Model | Dev PPL | Test PPL |
| - | - | - | - |
| A | LSTM | 146.4 | 141.3 |
| A | + dropout | 131.1 | 126.0 |
| A | + AdamW | 121.2 | **109.0** |
| B | LSTM + weight tying | 120.7 | 116.9 |
| B | + variational dropout | 99.4 | 97.4 |
| B | + NT-AvSGD | 94.9 | **92.3** |

![Dev perplexity per epoch](reports/figures/lm_dev_ppl.png)

Every technique lowers the perplexity, and part B, trained with plain SGD as NT-AvSGD requires, ends well below
the AdamW model of part A. The curves show where the NT-AvSGD gain comes from: without early stopping before the
trigger, SGD keeps improving after epoch 58, where the run without NT-AvSGD stopped, and reaches 97.6 dev PPL;
averaging, triggered at epoch 74, brings it to 94.9 and is still improving at the 100-epoch limit. Of the 4.5 dev
points gained over variational dropout, 1.8 come from the longer training and 2.7 from the averaging.

### Intent classification and slot filling (ATIS test set, 5 runs, mean ± std)

| Model | Slot F1 | Intent accuracy |
| - | - | - |
| LSTM | 0.922 ± 0.005 | 0.934 ± 0.004 |
| BiLSTM | 0.941 ± 0.004 | 0.945 ± 0.001 |
| BiLSTM + dropout | 0.946 ± 0.003 | 0.949 ± 0.003 |
| BERT (`bert-base-uncased`) | **0.955 ± 0.001** | **0.975 ± 0.002** |

In every run the test set is evaluated with the weights of the best dev epoch. Every step of part A improves both
metrics, and the fine-tuned BERT is the best model on both tasks, with the smallest spread on slot filling.
`python main.py` evaluates the checkpoint kept in `bin/`, the run with the best dev slot F1.

## What is implemented

**LM part A** (`language_modeling/part_A`): LSTM language model → + dropout after the embedding and before the output layer →
SGD replaced by AdamW. Early stopping on the dev perplexity, test on the best dev epoch.

**LM part B** (`language_modeling/part_B`): weight tying (the output layer reuses the embedding matrix) → + variational dropout
(one dropout mask per sequence, shared by all time steps) → + NT-AvSGD, implemented from Algorithm 1 of the
paper as a wrapper around SGD: after every epoch the dev perplexity is compared with the best value of more
than `n` epochs before; once the plateau is detected, a running average of the weights is updated after every
SGD step and the averaged weights are validated, saved and tested.

**NLU part A** (`intent_slot_filling/part_A`): LSTM encoder, intent from the last hidden state and one slot label per token,
trained on the sum of the two cross-entropy losses → bidirectional LSTM → + dropout. Five runs per model with
different seeds, early stopping on the dev slot F1.

**NLU part B** (`intent_slot_filling/part_B`): `bert-base-uncased` with an intent head on the pooled `[CLS]` representation and a
slot head on the token representations. WordPiece splits some words into several sub-tokens, so the slot label
of a word is assigned to its first sub-token and the others are ignored by the loss; predictions are mapped
back to words and scored with the same CoNLL script as part A.

## How to run

```bash
pip install -r requirements.txt
cd intent_slot_filling/part_A
python main.py --train                        # train every experiment of this part (checkpoints in bin/)
python main.py --train --exp bilstm_dropout   # or a single one
python main.py                                # evaluate the checkpoints in bin/ on the test set
```

The four parts work the same way and can be launched from any directory. Hyperparameters are at the top of each `main.py`. Model weights are not stored in git.

### On a GPU with Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/giampyhatesyou/Natural_Language_Processing/blob/main/run_on_colab.ipynb)

[`run_on_colab.ipynb`](run_on_colab.ipynb) clones the repository, trains the four parts (one cell each) and
evaluates the checkpoints. After every part, the checkpoints, the loss curves and the logs are copied to
Google Drive, so the results survive a disconnection of the runtime.

## Repository structure

```
├── language_modeling/
│   ├── dataset/          Penn Treebank (train / valid / test)
│   ├── part_A/           LSTM, dropout, AdamW
│   ├── part_B/           weight tying, variational dropout, NT-AvSGD
│   └── LM_Report.pdf
├── intent_slot_filling/
│   ├── part_A/           LSTM / BiLSTM joint model, ATIS data, CoNLL evaluation script
│   ├── part_B/           JointBERT
│   └── NLU_Report.pdf
├── reports/              LaTeX sources of the two reports (course template) and the figure script
├── results/logs/         training logs of the Colab run
├── run_on_colab.ipynb
└── CHANGES.md
```

Each part follows the layout required by the course: `main.py` (entry point), `model.py` (architectures),
`functions.py` (training and evaluation), `utils.py` (data loading and preprocessing). The last cell of the
Colab notebook packs the two folders as `LM/` and `NLU/`, the structure expected for the course submission.

To rebuild a report: `cd reports && pdflatex lm_report && bibtex lm_report && pdflatex lm_report && pdflatex lm_report`
(the same for `nlu_report`). `python reports/make_figures.py` redraws the perplexity curves from the logs.

## References

- Merity, Keskar and Socher. Regularizing and Optimizing LSTM Language Models. ICLR 2018.
- Gal and Ghahramani. A Theoretically Grounded Application of Dropout in Recurrent Neural Networks. NeurIPS 2016.
- Chen, Zhuo and Wang. BERT for Joint Intent Classification and Slot Filling. arXiv 2019.
- Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019.
