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
> rewritten; [CHANGES.md](CHANGES.md) lists every problem with its fix. The PDF reports in `exam/` describe
> the first submission.

## Results

### Intent classification and slot filling, LSTM (ATIS test set, 5 runs, mean ± std)

| Model | Slot F1 | Intent accuracy |
| - | - | - |
| LSTM | 0.921 ± 0.002 | 0.932 ± 0.007 |
| BiLSTM | 0.940 ± 0.002 | 0.945 ± 0.002 |
| BiLSTM + dropout | **0.947 ± 0.002** | **0.952 ± 0.004** |

Every step of the incremental protocol improves both metrics. In each run the test set is evaluated with the
weights of the best dev epoch (early stopping on the dev slot F1). The training curves of the run kept in
`bin/` are in [`NLU/part_A/plots`](exam/258237-Giampietro-Andrea/NLU/part_A/plots).

### Language modelling and BERT

These experiments are being re-run on a GPU with the fixed code (`python main.py --train` in `LM/part_A`,
`LM/part_B` and `NLU/part_B`). The numbers of the first submission are not reported because they were
produced by the code described in [CHANGES.md](CHANGES.md).

## What is implemented

**LM part A** (`LM/part_A`): LSTM language model → + dropout after the embedding and before the output layer →
SGD replaced by AdamW. Early stopping on the dev perplexity, test on the best dev epoch.

**LM part B** (`LM/part_B`): weight tying (the output layer reuses the embedding matrix) → + variational dropout
(one dropout mask per sequence, shared by all time steps) → + NT-AvSGD, implemented from Algorithm 1 of the
paper as a wrapper around SGD: after every epoch the dev perplexity is compared with the best value of more
than `n` epochs before; once the plateau is detected, a running average of the weights is updated after every
SGD step and the averaged weights are validated, saved and tested.

**NLU part A** (`NLU/part_A`): LSTM encoder, intent from the last hidden state and one slot label per token,
trained on the sum of the two cross-entropy losses → bidirectional LSTM → + dropout. Five runs per model with
different seeds, early stopping on the dev slot F1.

**NLU part B** (`NLU/part_B`): `bert-base-uncased` with an intent head on the pooled `[CLS]` representation and a
slot head on the token representations. WordPiece splits some words into several sub-tokens, so the slot label
of a word is assigned to its first sub-token and the others are ignored by the loss; predictions are mapped
back to words and scored with the same CoNLL script as part A.

## How to run

```bash
pip install -r requirements.txt
cd exam/258237-Giampietro-Andrea/NLU/part_A
python main.py --train                        # train every experiment of this part (checkpoints in bin/)
python main.py --train --exp bilstm_dropout   # or a single one
python main.py                                # evaluate the checkpoints in bin/ on the test set
```

Every part works the same way (`LM/part_A`, `LM/part_B`, `NLU/part_A`, `NLU/part_B`) and can be launched from
any directory. Hyperparameters are at the top of each `main.py`. Model weights are not stored in git.

### On a GPU with Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/giampyhatesyou/Natural_Language_Processing/blob/main/run_on_colab.ipynb)

[`run_on_colab.ipynb`](run_on_colab.ipynb) clones the repository, trains the four parts (one cell each) and
evaluates the checkpoints. After every part, the checkpoints, the loss curves and the logs are copied to
Google Drive, so the results survive a disconnection of the runtime.

## Repository structure

```
exam/258237-Giampietro-Andrea/
├── LM/
│   ├── dataset/          Penn Treebank (train / valid / test)
│   ├── part_A/           LSTM, dropout, AdamW
│   ├── part_B/           weight tying, variational dropout, NT-AvSGD
│   └── LM_Report.pdf
└── NLU/
    ├── part_A/           LSTM / BiLSTM joint model, ATIS data, CoNLL evaluation script
    ├── part_B/           JointBERT
    └── NLU_Report.pdf
```

Each part follows the layout required by the course: `main.py` (entry point), `model.py` (architectures),
`functions.py` (training and evaluation), `utils.py` (data loading and preprocessing).

## References

- Merity, Keskar and Socher. Regularizing and Optimizing LSTM Language Models. ICLR 2018.
- Gal and Ghahramani. A Theoretically Grounded Application of Dropout in Recurrent Neural Networks. NeurIPS 2016.
- Chen, Zhuo and Wang. BERT for Joint Intent Classification and Slot Filling. arXiv 2019.
- Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019.
