import torch
from torch import nn
from distilbert_base import DistilBertBase, ModelInfo


class DistilBertCNN(DistilBertBase):
    def __init__(self, model_info=ModelInfo('distilbert-base-uncased'), alpha=0.5, alpha_step=0):
        super(DistilBertCNN, self).__init__(model_info, alpha, alpha_step)
        self.conv = nn.Conv2d(in_channels=self.info.embedding_dim, out_channels=self.info.embedding_dim,
                              kernel_size=(1, 2), stride=(1, 4))

    def _logits(self, x):
        x = self.encoder(input_ids=x[:, 0], attention_mask=x[:, 1], output_hidden_states=True)
        x_hidden_states = x["hidden_states"]
        x = torch.stack([x_hidden_state for x_hidden_state in x_hidden_states[-4:]], dim=3)
        x = x.transpose(1, 2)
        x = self.conv(x).squeeze(dim=3)
        x = x.transpose(1, 2)
        start = self.start_fc(x).squeeze(dim=2)
        end = self.end_fc(x).squeeze(dim=2)
        return start, end
