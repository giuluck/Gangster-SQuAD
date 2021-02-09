import torch
from torch import nn
import torch.nn.functional as F
from transformers import DistilBertModel
import pytorch_lightning as pl


class ModelInfo:
    def __init__(self, pretrained_model, embedding_dim=768, max_length=512, cls_token=101, sep_token=102):
        self.pretrained_model = pretrained_model
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.cls_token = cls_token
        self.sep_token = sep_token


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


class DistilBertBase(pl.LightningModule):
    def __init__(self, model_info=ModelInfo('distilbert-base-uncased'), alpha=0.5, alpha_step=0, highway=False):
        super().__init__()
        # Define layers and loss function
        self.alpha = alpha
        self.alpha_step = alpha_step
        self.encoder = DistilBertModel.from_pretrained(model_info.pretrained_model)
        self.start_fc = nn.Linear(model_info.embedding_dim, 1)
        self.end_fc = nn.Linear(model_info.embedding_dim, 1)
        self.criterion = nn.CrossEntropyLoss()
        self.info = model_info
        self.highway = Highway(in_size=self.info.embedding_dim) if highway else False
        # W&B save hyperparameters
        self.save_hyperparameters({"criterion": self.criterion.__str__()})

    def _logits(self, x):
        x = self.encoder(input_ids=x[:, 0], attention_mask=x[:, 1])
        x = x["last_hidden_state"]
        if self.highway:
            x = self.highway(x)
        start = self.start_fc(x).squeeze(dim=2)
        end = self.end_fc(x).squeeze(dim=2)
        return start, end

    def forward(self, x):
        start, end = self._logits(x)
        _, start_idx = start.max(dim=1)
        _, end_idx = end.max(dim=1)
        _, ctxs_len = (x[:, 0] == self.info.sep_token).max(dim=1)
        start_idx = torch.minimum(start_idx, ctxs_len)
        end_idx = torch.minimum(end_idx, ctxs_len)
        return start_idx, end_idx

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred_start, pred_end = self._logits(x)
        loss_start = self.criterion(pred_start, y[:, 0])
        loss_end = self.criterion(pred_end, y[:, 1])
        # On training the default W&B behaviour only logs every step
        self.log('train/loss_start', loss_start, on_epoch=True, on_step=False,
                 prog_bar=True)
        self.log('train/loss_end', loss_end, on_epoch=True, on_step=False,
                 prog_bar=True)
        self.log('alpha', self.alpha, prog_bar=True)
        loss = self.alpha * loss_start + (1 - self.alpha) * loss_end
        if loss_end > loss_start:
            self.alpha -= self.alpha_step
        else:
            self.alpha += self.alpha_step
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer
