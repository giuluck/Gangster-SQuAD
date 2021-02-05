import torch
from torch import nn
from transformers import DistilBertModel
import pytorch_lightning as pl
from distilbert_base import ModelInfo


class DistilBertWithOutputKnowledge(pl.LightningModule):
    def __init__(self, model_info=ModelInfo('distilbert-base-uncased'), alpha=0.75):
        super().__init__()
        self.alpha = alpha
        self.encoder = DistilBertModel.from_pretrained(model_info.pretrained_model)
        self.start_fc = nn.Linear(model_info.embedding_dim, 1)
        self.end_fc = nn.Linear(model_info.embedding_dim, 1)
        self.criterion = nn.CrossEntropyLoss()
        self.info = model_info

    def _logits(self, x):
        x = self.encoder(input_ids=x[:, 0], attention_mask=x[:, 1])
        x = x["last_hidden_state"]
        start = self.start_fc(x).squeeze(dim=2)
        end = self.end_fc(x).squeeze(dim=2)
        return start, end

    def forward(self, x):
        # retrieve logits
        start, end = self._logits(x)
        # mask values after context lengths to -inf
        tensor_length = start.shape[1]
        _, contexts_length = (x[:, 0] == self.info.sep_token).max(dim=1)
        for i, ctx_len in enumerate(contexts_length):
            mask = torch.arange(tensor_length, device=self.device) > ctx_len
            start[i, mask] = -float('inf')
            end[i, mask] = -float('inf')
        # retrieve start indices
        _, start_indices = start.max(dim=1)
        # mask end tensor to assign -inf value to logits before the start idx
        for i, start_idx in enumerate(start_indices):
            mask = torch.arange(tensor_length, device=self.device) < start_idx
            end[i, mask] = -float('inf')
        # retrieve end indices and clip to maximal length
        _, end_indices = end.max(dim=1)
        return start_indices, end_indices

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred_start, pred_end = self._logits(x)
        loss_start = self.criterion(pred_start, y[:, 0])
        loss_end = self.criterion(pred_end, y[:, 1])
        self.log('loss_start', loss_start, prog_bar=True)
        self.log('loss_end', loss_end, prog_bar=True)
        loss = self.alpha * loss_start + (1 - self.alpha) * loss_end
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer
