import torch
from torch.utils.data import Dataset


class SquadDataset(Dataset):
    def __init__(self, dataframe, model_info, contain_answers=True):
        self.dataframe = dataframe
        self.max_len = model_info.max_length
        self.cls_tok = model_info.cls_token
        self.sep_tok = model_info.sep_token
        self.contain_answers = contain_answers

    def __getitem__(self, index):
        rec = self.dataframe.iloc[index]
        # retrieving paragraph and question tokens, limiting them to the maximal length
        qst_ids = rec['qst_ids'][:self.max_len - 3] + [self.sep_tok]
        ctx_ids = [self.cls_tok] + rec['ctx_ids'][:self.max_len - len(qst_ids) - 2] + [self.sep_tok]
        len_ids = len(ctx_ids) + len(qst_ids)
        # contexts and questions are used to build the input tensor
        ctx_ids = torch.tensor(ctx_ids)
        qst_ids = torch.tensor(qst_ids)
        input_ids = torch.cat((ctx_ids, qst_ids))
        input_masks = torch.ones_like(input_ids)
        input_tensor = torch.stack((input_ids, input_masks), dim=0)
        # the input tensor is padded to length 512
        pad_tensor = torch.zeros((2, self.max_len - len_ids), dtype=torch.long)
        input_tensor = torch.cat((input_tensor, pad_tensor), dim=1)
        if self.contain_answers:
            # an output tensor containing the two outputs is created as well
            output_tensor = torch.tensor([rec['start token'], rec['end token']])
            return input_tensor, output_tensor
        else:
            return input_tensor

    def __len__(self):
        return len(self.dataframe)
