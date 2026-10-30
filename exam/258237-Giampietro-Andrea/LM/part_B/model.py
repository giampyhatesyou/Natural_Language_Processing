import torch.nn as nn

# Part 1.B starts from the LSTM of part 1.A and adds, incrementally (Merity et al., 2018):
#   1. weight tying                                  -> LM_LSTM_WT with emb_dropout = out_dropout = 0
#   2. + variational dropout (no DropConnect)        -> LM_LSTM_WT with emb_dropout, out_dropout > 0
#   3. + Non-monotonically Triggered AvSGD            -> same model as 2, the change is in the optimisation (functions.py)


class VariationalDropout(nn.Module):
    """Variational (a.k.a. locked) dropout: the mask is sampled once per sequence and reused at every
    time step, instead of sampling a new mask for every token as nn.Dropout does."""

    def __init__(self, p=0.0):
        super(VariationalDropout, self).__init__()
        self.p = p

    def forward(self, x):  # x: batch_size X seq_len X features
        if not self.training or self.p == 0.0:
            return x
        # One mask per sequence: batch_size X 1 X features, broadcast over the time dimension
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p)
        return x * mask / (1 - self.p)  # rescale to keep the expected value unchanged


class LM_LSTM_WT(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, emb_dropout=0.0,
                 out_dropout=0.0, n_layers=1):
        super(LM_LSTM_WT, self).__init__()
        if emb_size != hidden_size:
            # The output layer reuses the embedding matrix, so it must map hidden_size -> vocab with
            # hidden_size == emb_size
            raise ValueError("Weight tying requires emb_size == hidden_size")
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        self.emb_dropout = VariationalDropout(emb_dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index
        self.out_dropout = VariationalDropout(out_dropout)
        self.output = nn.Linear(hidden_size, output_size)
        # Weight tying: the output projection and the embedding share the same matrix (vocab_size X emb_size)
        self.output.weight = self.embedding.weight

    def forward(self, input_sequence):
        emb = self.emb_dropout(self.embedding(input_sequence))
        lstm_out, _ = self.lstm(emb)
        lstm_out = self.out_dropout(lstm_out)
        output = self.output(lstm_out).permute(0, 2, 1)
        return output
