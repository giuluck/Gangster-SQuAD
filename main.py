from src.dataframe import *

if __name__ == '__main__':
    train_df, val_df, test_df = get_dataframes('data/training_set.json')
    print('train:', len(train_df))
    print('  val:', len(val_df))
    print(' test:', len(test_df))
