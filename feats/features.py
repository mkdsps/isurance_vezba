import pandas as pd
from i_features import features_i
from a_features import features_a
from v_features import features_v

def features_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spaja sve feature engineering korake
    """

    df = df.copy()

    df = features_i(df)
    df = features_a(df)
    df = features_v(df)

    return df