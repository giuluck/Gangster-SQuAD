import json
import pandas as pd

from evaluate import normalize_answer
from preprocessing import compute_boundaries


DEFAULT_EXCLUDED_CONTEXTS = {}
DEFAULT_EXCLUDED_QUESTIONS = {
    'k', 'j', 'n', 'b', 'v', 'dd', 'dd', 'dd', 'dd', 'd',
    "I couldn't could up with another question. But i need to fill this space because I can't submit the hit. "
}


def extract_data(data_path,
                 excluded_contexts=DEFAULT_EXCLUDED_CONTEXTS,
                 excluded_questions=DEFAULT_EXCLUDED_QUESTIONS,
                 contain_answers=True):
    """
    Extracts data from the given SQuAD dataset json file.
    It returns a pandas dataframe.
    """
    # opens the json file
    with open(data_path, 'r') as f:
        dataset = json.load(f)

    if contain_answers:
        columns = ['id', 'title', 'context', 'question', 'answer', 'start']
    else:
        columns = ['id', 'title', 'context', 'question']

    # stores each record in a list by exploring the levels of the json object
    samples = []
    for data in dataset['data']:
        title = data['title']
        for paragraph in data['paragraphs']:
            context = paragraph['context']
            if context not in excluded_contexts:
                for qas in paragraph['qas']:
                    question = qas['question']
                    if question not in excluded_questions:
                        record_id = qas['id']
                        if contain_answers:
                            answer = qas['answers'][0]['text']
                            start = qas['answers'][0]['answer_start']
                            sample = [record_id, title, context, question, answer, start]
                        else:
                            sample = [record_id, title, context, question]
                        samples.append(sample)
    # creates a dataframe from that list
    return pd.DataFrame(samples, columns=columns)


def process_dataframe(df, tokenizer):
    def process_record(record):
        # both context and question gets tokenized
        ctx_tokens = tokenizer.encode(record['context'])
        ctx_ids = ctx_tokens.ids[1:-1]                       # [CLS] and [SEP] tokens are discarded
        qst_tokens = tokenizer.encode(record['question'])
        qst_ids = qst_tokens.ids[1:-1]                       # [CLS] and [SEP] tokens are discarded
        # take all the context start chars then add a final index for the last character
        offsets = [s for s, _ in ctx_tokens.offsets[:-1]] + [len(record['context'])]
        # token boundaries to be used during training are computed
        start_token, end_token = compute_boundaries(offsets, record['start'], len(record['answer']))
        # input, output and utility data are returned to form the dataset
        return [ctx_ids, qst_ids, start_token, end_token, offsets]

    processed_df = pd.DataFrame(
        [[id] + process_record(record) for id, record in df.iterrows()],
        columns=['id', 'ctx_ids', 'qst_ids', 'start token', 'end token', 'offsets']
    ).set_index(['id'])
    return processed_df.join(df)


def get_dataframes(data_path,
                   excluded_contexts=DEFAULT_EXCLUDED_CONTEXTS,
                   excluded_questions=DEFAULT_EXCLUDED_QUESTIONS,
                   train_test_split=0.75,
                   train_val_split=0.75):
    """
    Extracts data from the given SQuAD dataset json file and splits it into three dataframes: train, val, and test.
    It returns the three dataframes, indexed with the record id.
    """
    df = extract_data(data_path, excluded_contexts, excluded_questions)
    train_df, test_df = title_based_split(df, train_test_split)
    train_df, val_df = title_based_split(train_df, train_val_split)
    return train_df.set_index(['id']), val_df.set_index(['id']), test_df.set_index(['id'])


def title_based_split(df, split_val):
    """
    Splits a dataframe into two disjoint dataframes basing on a splitting value
    and makes sure that the two dataframes do not share any 'title' attribute.
    """
    # retrieve split title and get the minimum id with that title
    split_title = df['title'].iloc[int(split_val * len(df))]
    split_index = df[df['title'] == split_title].index.min()
    return df.iloc[:split_index], df.iloc[split_index:]


def check_correctness(df, retrieving_procedure):
    """
    Given a dataframe (containing an 'answer' column) and a retrieving function to obtain an answer from each record of the
    dataframe itself, it checks if the real and the retrieved answers are equal and, if not, it appends the answer to a new
    dataframe of wrong answers which is returned.
    """
    correct_answers = []
    wrong_answers = []
    for record_id, record in df.iterrows():
        answer = record['answer']
        n_answer = normalize_answer(answer)
        retrieved = retrieving_procedure(record)
        n_retrieved = normalize_answer(retrieved)
        if n_answer != n_retrieved:
            wrong_answers.append((record_id, answer, n_answer, retrieved, n_retrieved))
        else:
            correct_answers.append((record_id, answer, n_answer, retrieved, n_retrieved))
    correct_df = pd.DataFrame(
        correct_answers,
        columns=['id', 'answer', 'normalized answer', 'retrieved', 'normalized retrieved']
    ).set_index(['id'])
    wrong_df = pd.DataFrame(
        wrong_answers,
        columns=['id', 'answer', 'normalized answer', 'retrieved', 'normalized retrieved']
    ).set_index(['id'])
    return correct_df, wrong_df
