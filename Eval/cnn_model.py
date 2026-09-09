import torch
from torch import nn
from transformers import BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class BertCNN(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config, add_pooling_layer=False)
        # Apply convolutions with multiple window sizes and max pooling to BERT token representations.
        self.kernels = (3, 4, 5)
        self.convs = nn.ModuleList([nn.Conv1d(config.hidden_size, 128, k) for k in self.kernels])
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(128 * len(self.kernels), config.num_labels)
        self.post_init()

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        finetune = getattr(self.config, 'finetune_bert', False)
        if not finetune:
            self.bert.eval()
        with torch.set_grad_enabled(self.training and finetune):
            hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                               token_type_ids=token_type_ids).last_hidden_state
        hidden = hidden.transpose(1, 2)
        features = []
        for kernel, conv in zip(self.kernels, self.convs):
            values = torch.relu(conv(hidden))
            # Mask windows containing padding; use zero features if no full window is available.
            valid = attention_mask.unfold(1, kernel, 1).all(dim=-1)
            values = values.masked_fill(~valid.unsqueeze(1), float('-inf'))
            pooled = values.max(dim=-1).values
            features.append(torch.where(valid.any(dim=1, keepdim=True), pooled, torch.zeros_like(pooled)))
        logits = self.classifier(self.dropout(torch.cat(features, dim=1)))
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return SequenceClassifierOutput(loss=loss, logits=logits)
