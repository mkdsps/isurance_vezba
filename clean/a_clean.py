import pandas as pd

def clean_a(df : pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = dummy_clean(df)
    # df = kolona_klean()

    return df
    

def dummy_clean(df : pd.DataFrame) -> pd.DataFrame:
    """
        odvoji po kolonama ove dummy funk...
    """
    
    return df