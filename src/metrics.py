import re
import pandas as pd
from preprocessing import check_correctness


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
