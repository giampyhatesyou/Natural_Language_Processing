# NLU part B: joint intent classification and slot filling with BERT

`JointBERT` fine-tunes `bert-base-uncased` with two heads: the intent is predicted from the pooled `[CLS]`
representation, the slots from the representation of every sub-token.

WordPiece splits some words into several sub-tokens (`philadelphia` → `phil ##ade ##lphia`), while ATIS has
one slot label per word. The label of a word is assigned to its first sub-token; the other sub-tokens,
`[CLS]`, `[SEP]` and the padding get -100 and are ignored by the loss. At evaluation time the prediction of a
word is the one of its first sub-token, so the slot F1 is computed on the original words with `conll.py`,
exactly as in part A.

The pre-trained encoder is loaded once per run and never re-initialised: only the two linear heads start
from scratch. Optimisation: AdamW (lr 5e-5, no weight decay on biases and LayerNorm), linear warm-up on the
first 10% of the steps then linear decay, early stopping on the dev slot F1, 5 runs.

```bash
python main.py --train                           # fine-tune (needs the Hugging Face model, a GPU is recommended)
python main.py --train --bert_model <path>       # any BERT-like model name or local folder
python main.py                                   # evaluate the checkpoint in bin/ on the test set
```
