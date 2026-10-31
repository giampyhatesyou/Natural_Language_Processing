import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Part A is incremental:
#   1. baseline ModelIAS (LSTM encoder, one head for the slots and one for the intent)
#   2. bidirectional LSTM                           -> bidirectional=True
#   3. + dropout                                    -> dropout > 0


class ModelIAS(nn.Module):
    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len, n_layer=1, pad_index=0,
                 bidirectional=False, dropout=0.0):
        super(ModelIAS, self).__init__()
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)
        self.utt_encoder = nn.LSTM(input_size=emb_size, hidden_size=hid_size, num_layers=n_layer,
                                   bidirectional=bidirectional, batch_first=True)
        # With p = 0 dropout is the identity, so models 1 and 2 are not affected
        self.dropout = nn.Dropout(dropout)

        # The bidirectional encoder concatenates the forward and backward states
        out_size = hid_size * 2 if bidirectional else hid_size
        self.slot_out = nn.Linear(out_size, out_slot)
        self.intent_out = nn.Linear(out_size, out_int)

    def forward(self, utterance, seq_lengths):
        # utterance.size() = batch_size X seq_len
        utt_emb = self.dropout(self.embedding(utterance))  # batch_size X seq_len X emb_size

        # pack_padded_sequence avoids computation over pad tokens reducing the computational cost
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(), batch_first=True)
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)
        utt_encoded, input_sizes = pad_packed_sequence(packed_output, batch_first=True)

        # last_hidden: (n_layer * n_directions) X batch_size X hid_size, the last layer is at the end
        if self.bidirectional:
            # forward state at the last token + backward state at the first token
            last_hidden = torch.cat((last_hidden[-2], last_hidden[-1]), dim=1)
        else:
            last_hidden = last_hidden[-1]

        # Dropout must be applied BEFORE the output layers, otherwise it has no effect on the logits
        utt_encoded = self.dropout(utt_encoded)
        last_hidden = self.dropout(last_hidden)

        slots = self.slot_out(utt_encoded)       # batch_size X seq_len X classes
        intent = self.intent_out(last_hidden)    # batch_size X classes

        slots = slots.permute(0, 2, 1)  # batch_size X classes X seq_len, as the loss expects
        return slots, intent
