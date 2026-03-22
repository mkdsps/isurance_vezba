import pandas as pd
from i_clean import clean_i
from v_clean import clean_v
from a_clean import clean_a

def clean_all(df : pd.DataFrame) -> pd.DataFrame:
    """
        sluzi da spoji sve cleanere..
    """

    df = df.copy()
    
    df = clean_i(df)
    df = clean_a(df)
    df = clean_v(df)

    return df