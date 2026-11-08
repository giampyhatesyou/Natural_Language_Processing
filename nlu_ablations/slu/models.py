"""Joint models: an intent classifier and a slot tagger on a shared encoder. Both return the slot logits of every
position (batch_size X classes X seq_len, as the loss expects) and the intent logits (batch_size X classes)."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoConfig, AutoModel


class JointLSTM(nn.Module):
    """LSTM encoder; slots from every hidden state, intent from the last one. The incremental steps are switches:
    bidirectional=True, then dropout > 0 (on the embeddings and before both output layers)."""

    def __init__(self, vocab_size, emb_size, hidden_size, n_slots, n_intents, n_layers=1, pad_index=0,
                 bidirectional=False, dropout=0.0):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=pad_index)
        self.utt_encoder = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)  # with p = 0 it is the identity
        out_size = 2 * hidden_size if bidirectional else hidden_size
        self.slot_out = nn.Linear(out_size, n_slots)
        self.intent_out = nn.Linear(out_size, n_intents)

    def forward(self, utterances, lengths):
        emb = self.dropout(self.embedding(utterances))
        # packing skips the computation over the padding
        packed, (last_hidden, _) = self.utt_encoder(pack_padded_sequence(emb, lengths.cpu(), batch_first=True))
        encoded, _ = pad_packed_sequence(packed, batch_first=True)
        # last_hidden: (n_layers * directions) X batch_size X hidden_size, the last layer at the end
        if self.bidirectional:
            # last forward state (last word) and last backward state (first word)
            last_hidden = torch.cat((last_hidden[-2], last_hidden[-1]), dim=1)
        else:
            last_hidden = last_hidden[-1]
        # dropout must act BEFORE the output layers, otherwise it does not change the logits
        slots = self.slot_out(self.dropout(encoded))
        intent = self.intent_out(self.dropout(last_hidden))
        return slots.permute(0, 2, 1), intent


class JointBERT(nn.Module):
    """Joint intent classification and slot filling on a pre-trained BERT (Chen et al., 2019): intent from the
    pooled [CLS] representation, slots from every sub-token (only the first sub-token of a word is labelled)."""

    def __init__(self, bert_name, n_slots, n_intents, dropout=0.1, pretrained=True):
        super().__init__()
        if pretrained:
            # Pre-trained weights: they must NOT be re-initialised (e.g. with common.init_weights, which would
            # overwrite every nn.Linear of the encoder); only the two output layers start from scratch
            self.bert = AutoModel.from_pretrained(bert_name)
        else:
            # Same architecture without downloading the weights, used when loading a fine-tuned checkpoint
            self.bert = AutoModel.from_config(AutoConfig.from_pretrained(bert_name))
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.slot_out = nn.Linear(hidden_size, n_slots)
        self.intent_out = nn.Linear(hidden_size, n_intents)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = outputs.pooler_output
        if pooled is None:  # encoders without a pooler: use the [CLS] state directly
            pooled = outputs.last_hidden_state[:, 0]
        slots = self.slot_out(self.dropout(outputs.last_hidden_state))
        intent = self.intent_out(self.dropout(pooled))
        return slots.permute(0, 2, 1), intent
