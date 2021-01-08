import json
import pandas as pd

DEFAULT_EXCLUDED_CONTEXTS = {}
DEFAULT_EXCLUDED_QUESTIONS = {
    'k', 'j', 'n', 'b', 'v', 'dd', 'dd', 'dd', 'dd', 'd',
    "I couldn't could up with another question. But i need to fill this space because I can't submit the hit. "
}

"""
Extracts data from the given SQuAD dataset json file and splits it into three dataframes: train, val, and test.
It returns the three dataframes, indexed with the record id.
"""
def get_dataframes(
        data_path,
        excluded_contexts=DEFAULT_EXCLUDED_CONTEXTS,
        excluded_questions=DEFAULT_EXCLUDED_QUESTIONS,
        train_test_split=0.7,
        train_val_split=0.7
):
    df = extract_data(data_path, excluded_contexts, excluded_questions)
    train_df, test_df = title_based_split(df, train_test_split)
    train_df, val_df = title_based_split(train_df, train_val_split)
    return train_df.set_index(['id']), val_df.set_index(['id']), test_df.set_index(['id'])


"""
Extracts data from the given SQuAD dataset json file.
It returns a pandas dataframe.
"""
def extract_data(data_path, excluded_contexts, excluded_questions):
    # opens the json file
    with open(data_path, 'r') as f:
        dataset = json.load(f)
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
                        answer = qas['answers'][0]['text']
                        start = qas['answers'][0]['answer_start']
                        record_id = qas['id']
                        samples.append([record_id, title, context, question, answer, start])
    # creates a dataframe from that list
    return pd.DataFrame(samples, columns=['id', 'title', 'context', 'question', 'answer', 'start'])

"""
Splits a dataframe into two disjoint dataframes basing on a splitting value
and makes sure that the two dataframes do not share any 'title' attribute.
"""
def title_based_split(df, split_val):
    # retrieve split title and get the minimum id with that title
    split_title = df['title'].iloc[int(split_val * len(df))]
    split_index = df[df['title'] == split_title].index.min()
    return df.iloc[:split_index], df.iloc[split_index:]
