import torch
from torch import nn
from distilbert_base import ModelInfo
from models import DistilBertKnowledge


class DistilBertCNN(DistilBertKnowledge):
    def __init__(self, model_info=ModelInfo('distilbert-base-uncased'), alpha=0.75, alpha_step=0):
        super(DistilBertCNN, self).__init__(model_info, alpha, alpha_step)
        self.conv = nn.Conv1d(in_channels=self.info.embedding_dim, out_channels=self.info.embedding_dim, kernel_size=2)

    def _logits(self, x):
        x = self.encoder(input_ids=x[:, 0], attention_mask=x[:, 1], output_hidden_states=True)
        x_hidden_states = x["hidden_states"]
        x = torch.stack([x_hidden_state for x_hidden_state in x_hidden_states[-4:]], dim=3)
        batch_length = x.shape[0]
        x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        x = self.conv(x)
        x = torch.max(x, dim=2)[0]
        x = x.view(batch_length, self.info.max_length, self.info.embedding_dim)
        start = self.start_fc(x).squeeze(dim=2)
        end = self.end_fc(x).squeeze(dim=2)
        return start, end
