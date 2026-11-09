"""LSTM language model. The techniques of the incremental study are switches of the same model:

    lstm                  LSTMLanguageModel()
    + dropout             emb_dropout, out_dropout > 0 (a new mask for every token)
    + weight tying        tie_weights=True
    + variational dropout variational=True (one mask per sequence, shared by all the time steps)

NT-AvSGD changes the optimisation, not the model: see ntavsgd.py.
"""
from torch import nn


class VariationalDropout(nn.Module):
    """Dropout with one mask per sequence, reused at every time step (Gal and Ghahramani, 2016)."""

    def __init__(self, p=0.0):
        super().__init__()
        self.p = p

    def forward(self, x):  # x: batch_size X seq_len X features
        if not self.training or self.p == 0.0:
            return x
        # batch_size X 1 X features: broadcast over the time dimension
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p)
        return x * mask / (1 - self.p)  # rescale to keep the expected value unchanged


class LSTMLanguageModel(nn.Module):
    """Embedding -> dropout -> LSTM -> dropout -> linear layer over the vocabulary."""

    def __init__(self, vocab_size, emb_size, hidden_size, pad_index=0, emb_dropout=0.0, out_dropout=0.0,
                 variational=False, tie_weights=False, n_layers=1):
        super().__init__()
        if tie_weights and emb_size != hidden_size:
            # the output layer reuses the embedding matrix, so it maps hidden_size -> vocab with hidden == emb size
            raise ValueError("Weight tying requires emb_size == hidden_size")
        dropout = VariationalDropout if variational else nn.Dropout
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_index)
        self.emb_dropout = dropout(emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True)
        self.out_dropout = dropout(out_dropout)
        self.output = nn.Linear(hidden_size, vocab_size)
        if tie_weights:
            self.output.weight = self.embedding.weight  # one vocab_size X emb_size matrix for input and output

    def forward(self, input_sequence):
        emb = self.emb_dropout(self.embedding(input_sequence))
        lstm_out, _ = self.lstm(emb)
        logits = self.output(self.out_dropout(lstm_out))
        return logits.permute(0, 2, 1)  # batch_size X vocab_size X seq_len, as the loss expects
