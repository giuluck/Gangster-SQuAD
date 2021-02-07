import re
import pandas as pd
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

    def split(s):
        return re.findall(r'[\w]+|[.,!?;\']', s)

    same = 0
    pred_len = 0
    ans_len = 0
    answers = answers.apply(lambda s: split(s))
    preds = preds.apply(lambda s: split(s))
    for ans, pred in zip(answers, preds):
        intersection = [word for word in pred if word in ans]
        same += len(intersection)
        pred_len += len(pred)
        ans_len += len(ans)
    precision = 1.0 * same / pred_len
    recall = 1.0 * same / ans_len
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


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
            s, e = pl_module(input)
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
        self._log_metrics('train', pl_module, self.train_df, self.train_dl)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_metrics('val', pl_module, self.val_df, self.val_dl)
