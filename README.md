# nlu-ablations

[![CI](https://github.com/giampyhatesyou/nlu-ablations/actions/workflows/ci.yml/badge.svg)](https://github.com/giampyhatesyou/nlu-ablations/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/giampyhatesyou/nlu-ablations/blob/main/run_on_colab.ipynb)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Two classic natural language understanding problems studied with an incremental protocol: every technique is added
on top of the best model of the previous step and evaluated on held-out data.

- **Language modelling on the Penn Treebank.** LSTM → dropout → AdamW, then the AWD-LSTM recipe: weight tying →
  variational dropout → NT-AvSGD. Test perplexity goes from 141.3 to 92.3.
- **Intent classification and slot filling on ATIS.** LSTM → BiLSTM → dropout → fine-tuned BERT. Slot F1 goes from
  92.2 to 95.5 and intent accuracy from 93.4% to 97.5% (means over 5 runs).

The project started as the final assignment of the NLU course at the University of Trento. A later review of that
first version found errors that changed the meaning of several experiments; the code was rewritten as a tested
Python package, and the review is documented in the [post-mortem](POST_MORTEM.md). Methods and results are
described in two short papers: [language modelling](reports/lm_report.pdf) and
[intent classification and slot filling](reports/slu_report.pdf).

## Results

### Language modelling (Penn Treebank, perplexity, lower is better)

| Part | Model | Dev PPL | Test PPL |
| - | - | - | - |
| A | LSTM | 146.4 | 141.3 |
| A | + dropout (0.4) | 131.1 | 126.0 |
| A | + AdamW | 121.2 | **109.0** |
| B | LSTM + weight tying | 120.7 | 116.9 |
| B | + variational dropout (0.4) | 99.4 | 97.4 |
| B | + NT-AvSGD (n = 5) | 94.9 | **92.3** |

![Dev perplexity per epoch](reports/figures/lm_dev_ppl.png)

Every technique lowers the perplexity, and part B, trained with plain SGD as NT-AvSGD requires, ends well below the
AdamW model of part A. The curves show where the NT-AvSGD gain comes from: without early stopping before the
trigger, SGD keeps improving after epoch 58, where the run without NT-AvSGD stopped, and reaches 97.6 dev PPL;
averaging, triggered at epoch 74, brings it to 94.9 and is still improving at the 100-epoch limit. Of the 4.5 dev
points gained over variational dropout, 1.8 come from the longer training and 2.7 from the averaging. Each
configuration was trained once, so these numbers carry no estimate of the variance across seeds.

### Intent classification and slot filling (ATIS test set, 5 runs)

| Model | Slot F1 | p | Intent accuracy (%) | p |
| - | - | - | - | - |
| LSTM | 92.24 ± 0.54 | – | 93.44 ± 0.47 | – |
| BiLSTM | 94.13 ± 0.45 | < 0.001 | 94.54 ± 0.15 | 0.005 |
| BiLSTM + dropout | 94.59 ± 0.30 | 0.099 | 94.94 ± 0.35 | 0.059 |
| BERT (`bert-base-uncased`) | **95.49 ± 0.11** | 0.001 | **97.54 ± 0.24** | < 0.001 |

Mean ± sample standard deviation over 5 seeds; p is Welch's t-test against the previous row. The bidirectional
encoder and BERT bring significant gains on both tasks; the gain of dropout is within the variation between seeds.
BERT is the best model on both tasks, by a wide margin on intents.

Both tables, the figure and the tables of the papers are generated from the JSON records in [`results/`](results)
by `python reports/analysis.py`.

## Methodology

- **Incremental protocol.** Within each part, every experiment adds one technique to the best model of the previous
  one, so each row of the tables is an ablation of the row below it.
- **Data.** Penn Treebank in the preprocessed version of Mikolov et al. (42,068 / 3,370 / 3,761 sentences, 10k-word
  vocabulary). ATIS with 4,978 training and 893 test utterances, 26 intents and 129 slot labels; 10% of the
  training set, stratified by intent, is held out as dev set.
- **Model selection.** Early stopping looks only at the dev set (perplexity for the language models, slot F1 for
  the intent/slot models). The weights of the best dev epoch are restored, and the test set is evaluated once per
  run. When a model is trained more than once, the checkpoint kept is the run with the best dev score.
- **Metrics.** Perplexity is the exponential of the mean cross-entropy over all the predicted tokens, `<eos>`
  included and padding excluded. Slot F1 is the chunk-level F1 of conlleval on the original words: for BERT, the
  prediction of a word is the one of its first WordPiece sub-token, so all the models are scored on the same units.
- **Variance and significance.** Every intent/slot model is trained 5 times with different seeds; tables report mean
  and sample standard deviation, and consecutive models are compared with Welch's t-test (α = 0.05).
- **Verification.** The algorithms and the evaluation code have their own tests (see below), including an oracle
  check: a model that returns the gold labels must score exactly 1.0.

## Reproducibility

```bash
git clone https://github.com/giampyhatesyou/nlu-ablations.git
cd nlu-ablations
pip install -e ".[dev]"    # Python 3.10 or later
```

```bash
python -m nlu_ablations.lm --train                  # train the six language models
python -m nlu_ablations.lm --train --exp wt wt_vd   # or some of them
python -m nlu_ablations.lm                          # evaluate the checkpoints on the test set
python -m nlu_ablations.slu --train --exp bert      # the same for intent classification and slot filling
python reports/analysis.py                          # tables and figure, from the JSON records
```

The two command lines are also installed as `nlu-lm` and `nlu-slu`, and `--help` lists their options: number of
runs and epochs, seed, data and output folders. The hyperparameters of every experiment are in a registry at the
top of [`lm/cli.py`](nlu_ablations/lm/cli.py) and [`slu/cli.py`](nlu_ablations/slu/cli.py), and a test checks that
within each part every experiment differs from the previous one only in the technique it adds. Seeds are fixed (42
plus the run index), so a re-run gives the same results up to the nondeterminism of GPU kernels.

Every training writes:

- `results/<task>/<experiment>.json`: configuration, software and hardware, and for every run the seed, the selected
  epoch, the dev and test scores and the per-epoch history;
- `results/plots/<task>/<experiment>.png`: the loss curves of the selected run;
- `checkpoints/<task>/<experiment>.pt`: the weights of the run with the best dev score, which are not stored in git
  because of their size.

The records in `results/` come from a run on a free Colab T4 GPU in October 2026, which took about 2 hours and
20 minutes: 48 minutes for language modelling part A, 63 for part B, 8 for the LSTM intent/slot models and 20 for
BERT. That run predates the JSON records: [`scripts/logs_to_results.py`](scripts/logs_to_results.py) converted its
logs, kept in [`results/logs`](results/logs), and checked that the hyperparameters they print are the ones of the
current registries. The run was made with the code before its refactoring into a package, and the package was
checked against that code: checkpoints saved by the old code give the same test scores when evaluated with the
package, and on CPU the two versions produce identical training runs from the same seed.

On a GPU, [`run_on_colab.ipynb`](run_on_colab.ipynb) clones the repository, trains every experiment, copies
checkpoints and results to Google Drive after each group of experiments and evaluates the checkpoints.

```bash
ruff check .    # lint
pytest          # unit, regression and end-to-end tests, under a minute on CPU
```

The tests download nothing: the BERT pipeline is tested with a tiny WordPiece vocabulary and a randomly initialised
two-layer BERT built on the fly.

## Limitations

- The language models were trained with a single seed, so their results have no variance estimate.
- Hyperparameters were not searched systematically: they come from the first version of the project and are shared
  by all the steps of a part.
- Part B uses a smaller batch and learning rate than part A, so the comparison between the LSTM of part A and the
  tied LSTM of part B is not a controlled one.
- NT-AvSGD hit the 100-epoch limit while still improving, and part of its gain comes from the longer training that
  its stopping rule allows.

## What I learned

- **Test the evaluation, not only the model.** An oracle that returns the gold labels must score exactly 1.0; it is
  the cheapest way to trust a metric.
- **Code that runs can still be dead.** A dropout applied after the logits never raised an error and never had an
  effect; only following the data flow, or a regression test, reveals it.
- **An ablation changes one factor at a time.** In the first version the "dropout" model also had weight tying and
  the AdamW run used the model without dropout, so neither row measured what its name said.
- **Implement the algorithm from its definition, then test the definition.** The first NT-AvSGD compared the dev
  perplexity with the wrong window and averaged the wrong weights.
- **Report variance and test significance.** Of the three steps of the intent/slot experiments, one (dropout) is
  within the noise between seeds.
- **Select on dev, report the mean.** The BERT run with the best dev F1 is the worst of the five on the test set.
- **One implementation per idea.** The first version had four copies of the seeding, initialisation, early-stopping
  and checkpoint code, one per part; in a single package there is one place to fix and one place to test.
- **Results are data.** Every training writes a JSON record, and every table and figure is generated from those
  records instead of being copied from a log.

## Repository structure

```
├── nlu_ablations/           the Python package
│   ├── common.py            seeding, initialisation, early stopping, checkpoints, statistics, plots
│   ├── results.py           JSON record of an experiment
│   ├── lm/                  language modelling: data, model, NT-AvSGD, training loop, command line
│   └── slu/                 intent classification and slot filling: data for the LSTMs and for BERT, models,
│                            conlleval, training loop, command line
├── data/                    Penn Treebank and ATIS
├── results/
│   ├── lm/, slu/            JSON record of every experiment
│   ├── plots/               loss curves of the selected runs
│   └── logs/                training logs of the Colab run
├── reports/                 the two papers (PDF and LaTeX sources), analysis.py, generated tables and figure
├── scripts/                 conversion of the logs of the Colab run into JSON records
├── tests/                   unit, regression and end-to-end tests
├── run_on_colab.ipynb       training and evaluation on a Colab GPU
├── POST_MORTEM.md           review of the first version
└── pyproject.toml           package metadata, dependencies, ruff and pytest configuration
```

To rebuild a paper: `cd reports && pdflatex lm_report && bibtex lm_report && pdflatex lm_report && pdflatex
lm_report` (the same for `slu_report`).

## License

The code is released under the [MIT license](LICENSE), with one exception: `nlu_ablations/slu/conll.py`, a Python
port of the conlleval script distributed with the course material. The Penn Treebank and ATIS data in `data/` are
not covered by the license either: they derive from corpora of the Linguistic Data Consortium, and the preprocessed
copies are the ones distributed with the course material.

## References

- Merity, Keskar and Socher. Regularizing and Optimizing LSTM Language Models. ICLR 2018.
- Gal and Ghahramani. A Theoretically Grounded Application of Dropout in Recurrent Neural Networks. NeurIPS 2016.
- Chen, Zhuo and Wang. BERT for Joint Intent Classification and Slot Filling. arXiv 2019.
- Devlin et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL 2019.
