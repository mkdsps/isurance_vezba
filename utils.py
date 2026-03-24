import pandas as pd
from typing import Tuple

def merge_train_test(df_train: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
    """
    Spaja train i test u jedan DataFrame i dodaje kolonu '_base'
    ('train' ili 'test')
    """

    df_train = df_train.copy()
    df_test = df_test.copy()

    df_train['_base'] = 'train'
    df_test['_base'] = 'test'

    df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)

    return df_all



def split_train_test(df_all: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Razdvaja DataFrame na train i test na osnovu kolone '_base'
    """
    df_train = df_all[df_all['_base'] == 'train'].copy()
    df_test = df_all[df_all['_base'] == 'test'].copy()

    df_train = df_train.drop(columns=['_base'])
    df_test = df_test.drop(columns=['_base'])

    return df_train, df_test

if __name__ == "__main__":
    main()