from bisect import bisect_left

"""
Given a list of strings, representing the tokens, returns a list of the char offsets.
This list will be used when retrieving the answer, so that from the start/end token indices,
the respective char indices could be computed back to extract the correct substring from the context.
"""
def get_offsets(tokens):
    offsets = [0]
    for token in tokens:
        offsets.append(len(token) + offsets[-1])
    return offsets

"""
Given the chars offsets list, the (char) index at which the answer starts in the context, and the length of the answer
it computes the new start/end token indices representing the boundaries that must be computed by the neural model.
"""
def compute_boundaries(offsets, start_char, answer_len):
    start_idx = bisect_left(offsets, start_char)
    end_idx = bisect_left(offsets, start_char + answer_len)
    return start_idx, end_idx

"""
Given the two (token) boundaries, the list of (char) offsets, and the context paragraph that contains the answer,
it returns the substring identifying the answer itself
"""
def retrieve_answer(start_token, end_token, offsets, context):
    start_char = offsets[start_token]
    end_char = offsets[end_token]
    return context[start_char:end_char].strip()
