import torch
import sys
import json
from torch.utils.data import DataLoader
from transformers import DistilBertTokenizer
from tokenizers import BertWordPieceTokenizer

sys.path.append('src')
sys.path.append('src/models')

from evaluate import normalize_answer
from dataframe import extract_data, process_dataframe
from models import DistilBertWHL
from dataset import SquadDataset
from preprocessing import retrieve_answer


def retrieving_procedure(rec):
    return retrieve_answer(rec['pred_start'], rec['pred_end'], rec['offsets'], rec['context'])


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(f"Using {device}.")

    print(f"Reading {sys.argv[1]}...")
    df = extract_data(sys.argv[1], contain_answers=False).set_index(['id'])
    print(f"DataFrame created.")

    print("Tokenizing the DataFrame...")
    model = DistilBertWHL(highway=True, alpha=0.5)
    DistilBertTokenizer.from_pretrained(model.info.pretrained_model).save_pretrained('slow_tokenizer/')
    tokenizer = BertWordPieceTokenizer('slow_tokenizer/vocab.txt', lowercase=True)
    df = process_dataframe(df, tokenizer, contain_answers=False)
    print("Tokenization complete.")

    dataset = SquadDataset(df, model.info, contain_answers=False)
    loader = DataLoader(dataset, batch_size=16, num_workers=4, pin_memory=True)

    print("Loading model weights...")
    model.load_state_dict(torch.load('model.pt'))
    model = model.to(device)
    print("Model loaded.")

    model.eval()
    print("Starting evaluation...")
    starts, ends = [], []
    num_batches = len(loader)
    for idx, input in enumerate(loader):
        if (idx + 1) % 100 == 0:
            print(f'Batch {idx + 1:{len(str(num_batches))}}/{num_batches}')
        with torch.no_grad():
            s, e = model(input.to(device))
        starts.append(s)
        ends.append(e)
    print("Evaluation completed.")

    df['pred_start'] = [s.item() for ss in starts for s in ss]
    df['pred_end'] = [e.item() for ee in ends for e in ee]

    print("Retrieving predictions...")
    predictions = {}
    for record_id, record in df.iterrows():
        retrieved = retrieving_procedure(record)
        n_retrieved = normalize_answer(retrieved)
        predictions[record_id] = n_retrieved
    print("Finish retrieving.")

    with open('predictions.json', 'w') as f:
        json.dump(predictions, f)
