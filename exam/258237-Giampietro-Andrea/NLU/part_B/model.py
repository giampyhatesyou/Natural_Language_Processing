import torch.nn as nn
from transformers import AutoConfig, AutoModel


class JointBERT(nn.Module):
    """Joint intent classification and slot filling on top of a pre-trained BERT (Chen et al., 2019).

    - intent: linear layer on the pooled [CLS] representation (one label per utterance)
    - slots:  linear layer on every sub-token representation; only the first sub-token of each word
              is used in the loss and in the evaluation (see utils.align_slot_labels)
    """

    def __init__(self, bert_name, out_slot, out_int, dropout=0.1, pretrained=True):
        super(JointBERT, self).__init__()
        if pretrained:
            # Pre-trained weights: they must NOT be re-initialised (e.g. with the init_weights of part A,
            # which would overwrite every nn.Linear of the encoder): only the two heads below start from scratch
            self.bert = AutoModel.from_pretrained(bert_name)
        else:
            # Same architecture without downloading the weights, used when loading a fine-tuned checkpoint
            self.bert = AutoModel.from_config(AutoConfig.from_pretrained(bert_name))
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.slot_out = nn.Linear(hidden_size, out_slot)
        self.intent_out = nn.Linear(hidden_size, out_int)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs.last_hidden_state  # batch_size X seq_len X hidden_size
        pooled_output = outputs.pooler_output        # batch_size X hidden_size ([CLS] -> dense -> tanh)
        if pooled_output is None:  # encoders without a pooler: use the [CLS] state directly
            pooled_output = sequence_output[:, 0]

        slots = self.slot_out(self.dropout(sequence_output))    # batch_size X seq_len X classes
        intent = self.intent_out(self.dropout(pooled_output))   # batch_size X classes
        slots = slots.permute(0, 2, 1)  # batch_size X classes X seq_len, as the loss expects
        return slots, intent
