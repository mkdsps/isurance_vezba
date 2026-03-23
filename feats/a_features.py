import pandas as pd
import numpy as np
def features_a(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Length_Power'] = df['Length'] * df['Power']
    df['log_Premium'] = np.log1p(df['Premium'])
    #df = dummy_features(df)
    # df = neka_feature_i(df)

    return df


def dummy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering za 'a' deo
    """
    return df
