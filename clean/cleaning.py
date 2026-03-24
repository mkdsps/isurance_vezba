import pandas as pd
from .a_clean import clean_a

def clean_all(df : pd.DataFrame) -> pd.DataFrame:
    """
        sluzi da spoji sve cleanere..
    """

    df = df.copy()
    
    df = clean_a(df)

    return df