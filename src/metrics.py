import numpy as np
import pandas as pd
import torch
from pytorch_lightning import Callback
from torch.utils.data import DataLoader

from dataframe import check_correctness
from preprocessing import retrieve_answer


def f1_score(answers, preds):
    """
    Computes the F1 score given two series:
    - one corresponding to the answers
    - one corresponding to the predictions
    """
    answers = answers.apply(lambda s: s.split())
    preds = preds.apply(lambda s: s.split())
    scores = []
    for ans, pred in zip(answers, preds):
        if len(pred) == 0 or len(ans) == 0:
            scores.append(1 if len(ans) == len(pred) else 0)
        else:
            intersection = [word for word in pred if word in ans]
            same = len(intersection)
            precision = 1.0 * same / len(pred)
            recall = 1.0 * same / len(ans)
            f1 = (2 * precision * recall) / (precision + recall)
            scores.append(f1)
    return np.mean(scores).item()


def compute_metrics(df, retrieving_procedure):
    """
    Given a DataFrame and a function to retrieve the answers from it,
    computes the Exact Match and the F1 score.
    """
    correct, wrong = check_correctness(df, retrieving_procedure)
    complete = pd.concat([correct, wrong])
    exact_match = len(correct) / len(complete)
    f1 = f1_score(complete['normalized answer'], complete['normalized retrieved'])
    return exact_match, f1


class MetricsCallback(Callback):
    def __init__(self, train_df, train_ds, val_df, val_ds):
        super(MetricsCallback, self).__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.train_dl = DataLoader(train_ds, batch_size=16, num_workers=4, pin_memory=True)
        self.val_dl = DataLoader(val_ds, batch_size=16, num_workers=4, pin_memory=True)

    @staticmethod
    def _log_metrics(name, pl_module, df, loader):
        starts = []
        ends = []
        for input, _ in loader:
            s, e = pl_module(input.to(pl_module.device))
            starts.append(s)
            ends.append(e)
        df['pred_start'] = [s.item() for ss in starts for s in ss]
        df['pred_end'] = [e.item() for ee in ends for e in ee]
        exact_match, f1 = compute_metrics(
            df,
            lambda rec: retrieve_answer(rec['pred_start'], rec['pred_end'], rec['offsets'], rec['context'])
        )
        pl_module.log(f'{name}/exact_match', exact_match)
        pl_module.log(f'{name}/f1_score', f1)

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        pl_module.eval()
        with torch.no_grad():
            self._log_metrics('train', pl_module, self.train_df, self.train_dl)
            self._log_metrics('val', pl_module, self.val_df, self.val_dl)
        pl_module.train()
