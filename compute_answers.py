import torch
import sys
import json

from evaluate import normalize_answer
from src.dataframe import extract_data
from src.models import DistilBertWHL
from src.dataset import SquadDataset
from torch.utils.data import DataLoader

from src.preprocessing import retrieve_answer

if __name__ == '__main__':
    print(f"Reading ${sys.argv[1]}...")
    df = extract_data(sys.argv[1], contain_answers=False)
    print(f"DataFrame created.")
    model = DistilBertWHL(alpha=0.66, alpha_step=0.0001)
    dataset = SquadDataset(df, model.info)
    loader = DataLoader(dataset, batch_size=16, num_workers=4, pin_memory=True)

    print("Loading model...")
    model.load_state_dict(torch.load('model.pt'))
    model = model.cuda()
    print("Model loaded.")

    model.eval()
    print("Starting evaluation...")
    starts, ends = [], []
    num_batches = len(loader)
    for idx, (input, _) in enumerate(loader):
        if (idx + 1) % 100 == 0:
            print(f'Batch {idx + 1:{len(str(num_batches))}}/{num_batches}')
        with torch.no_grad():
            s, e = model(input.cuda())
        starts.append(s)
        ends.append(e)
    print("Evaluation completed.")

    df['pred_start'] = [s.item() for ss in starts for s in ss]
    df['pred_end'] = [e.item() for ee in ends for e in ee]
    # When the prediction (for both 'start' and 'end') points to a padding character
    # then it is placed at the last offset
    df['length_offsets'] = df['offsets'].apply(lambda x: len(x))
    df['pred_start'] = (df['pred_start'] < df['length_offsets']) * df['pred_start'] + (
                df['pred_start'] >= df['length_offsets']) * (df['length_offsets'] - 1)
    df['pred_end'] = (df['pred_end'] < df['length_offsets']) * df['pred_end'] + (
                df['pred_end'] >= df['length_offsets']) * (df['length_offsets'] - 1)

    print("Retrieving prediction...")
    predictions = []
    for record_id, record in df.iterrows():
        retrieved = retrieve_answer(df['pred_start'], df['pred_end'], df['offsets'], df['context'])
        n_retrieved = normalize_answer(retrieved)
        predictions.append({record_id: n_retrieved})
    print("Finish retrieving.")

    with open('prediction.json', 'w') as f:
        json.dump(predictions, f)
