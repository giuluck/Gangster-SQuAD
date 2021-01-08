from src.dataset import *

if __name__ == '__main__':
    dfs = get_dataframes('data/training_set.json')
    for key, val in dfs.items():
        print(key, len(val))
