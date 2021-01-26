import pandas as pd
from bisect import bisect_left
from evaluate import normalize_answer


def get_offsets(tokens):
    """
    Given a list of strings, representing the tokens, returns a list of the char offsets.
    This list will be used when retrieving the answer, so that from the start/end token indices,
    the respective char indices could be computed back to extract the correct substring from the context.
    """
    offsets = [0]
    for token in tokens:
        offsets.append(len(token) + offsets[-1])
    return offsets


def compute_boundaries(offsets, start_char, answer_len):
    """
    Given the chars offsets list, the (char) index at which the answer starts in the context, and the length of the answer
    it computes the new start/end token indices representing the boundaries that must be computed by the neural model.
    """
    start_idx = bisect_left(offsets, start_char)
    end_idx = bisect_left(offsets, start_char + answer_len)
    return start_idx, end_idx


def retrieve_answer(start_token, end_token, offsets, context):
    """
    Given the two (token) boundaries, the list of (char) offsets, and the context paragraph that contains the answer,
    it returns the substring identifying the answer itself.
    """
    start_char = offsets[start_token]
    end_char = offsets[end_token]
    return context[start_char:end_char].strip()


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
