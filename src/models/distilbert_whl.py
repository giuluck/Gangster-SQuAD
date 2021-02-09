import torch
from torch import nn
from distilbert_base import ModelInfo
from models import DistilBertKnowledge, Highway


class DistilBertWHL(DistilBertKnowledge):
    """
    https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/default/15737384.pdf
    """
    def __init__(self, model_info=ModelInfo('distilbert-base-uncased'), alpha=0.75, alpha_step=0, k=4, highway=False):
        super(DistilBertWHL, self).__init__(model_info, alpha, alpha_step, False)
        self.k = k
        self.highway = Highway(in_size=self.info.embedding_dim * k) if highway else False
        self.dot_prod_attention = nn.Linear(k * self.info.embedding_dim, self.info.embedding_dim)

    def _logits(self, x):
        x = self.encoder(input_ids=x[:, 0], attention_mask=x[:, 1], output_hidden_states=True)
        x_hidden_states = x["hidden_states"]
        x = torch.stack([x_hidden_state for x_hidden_state in x_hidden_states[-self.k:]], dim=3)
        x = torch.flatten(x, start_dim=2, end_dim=3)
        if self.highway:
            x = self.highway(x)
        x = self.dot_prod_attention(x)
        start = self.start_fc(x).squeeze(dim=2)
        end = self.end_fc(x).squeeze(dim=2)
        return start, end
