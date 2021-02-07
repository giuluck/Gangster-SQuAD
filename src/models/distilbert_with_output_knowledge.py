import torch
from distilbert_base import ModelInfo, DistilBertBase


class DistilBertWithOutputKnowledge(DistilBertBase):
    def __init__(self, model_info=ModelInfo('distilbert-base-uncased'), alpha=0.75, alpha_step=0):
        super(DistilBertWithOutputKnowledge, self).__init__(model_info, alpha, alpha_step)

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
