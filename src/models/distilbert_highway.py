import torch
from torch import nn
import torch.nn.functional as F
from distilbert_base import DistilBertBase, ModelInfo


class Highway(nn.Module):
    def __init__(self, in_size, n_layers=1, act=F.relu):
        super(Highway, self).__init__()
        self.n_layers = n_layers
        self.act = act

        self.normal_layer = nn.ModuleList([nn.Linear(in_size, in_size) for _ in range(n_layers)])
        self.gate_layer = nn.ModuleList([nn.Linear(in_size, in_size) for _ in range(n_layers)])

    def forward(self, x):
        for i in range(self.n_layers):
            normal_layer_ret = self.act(self.normal_layer[i](x))
            gate = torch.sigmoid(self.gate_layer[i](x))

            x = torch.add(torch.mul(normal_layer_ret, gate), torch.mul((1.0 - gate), x))
        return x


class DistilBertWHL(DistilBertBase):
    """
    https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1194/reports/default/15737384.pdf
    """
    def __init__(self, model_info=ModelInfo('distilbert-base-uncased'), alpha=0.5, alpha_step=0, k=4, highway=False):
        super(DistilBertWHL, self).__init__(model_info, alpha, alpha_step)
        self.k = k
        self.highway = highway
        if highway:
            self.highway = Highway(in_size=self.info.embedding_dim * k)
        self.dot_prod_attention = nn.Linear(k * self.info.embedding_dim, self.info.embedding_dim)

    def _logits(self, x):
        x = self.encoder(input_ids=x[:, 0], attention_mask=x[:, 1], output_hidden_states=True)
        x_hidden_states = x["hidden_states"]
        x = torch.stack([x_hidden_state for x_hidden_state in x_hidden_states[-self.k:]], dim=3)
        x = x.flatten(start_dim=2, end_dim=3)
        if self.highway:
            x = self.highway(x)
        x = self.dot_prod_attention(x)
        start = self.start_fc(x).squeeze(dim=2)
        end = self.end_fc(x).squeeze(dim=2)
        return start, end
