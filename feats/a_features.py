import pandas as pd

def features_a(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = dummy_features(df)
    # df = neka_feature_i(df)

    return df


def dummy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering za 'a' deo
    """
    return df